#!/usr/bin/env python3
"""
Drop Detector - Finds the best moments in DJ sets and interviews
and generates trimmed video clips.

Sets are found by audio energy; interviews are found by transcribing the
speech and asking Claude which moments are worth clipping.

Usage:
    1. Place video files in the input/ folder
    2. Run: python detect.py            (asks what kind of content it is)
       or:  python detect.py --mode interview
    3. Clips are generated in output/
"""

import os
import re
import sys
import glob
import json
import argparse
import tempfile
import subprocess

import numpy as np
import librosa

MODE_SET = "set"
MODE_INTERVIEW = "interview"

LLM_CLI = "cli"        # Claude Code in headless mode; billed to your subscription
LLM_API = "api"        # Anthropic SDK; billed per token, needs ANTHROPIC_API_KEY

# --- Sets: absolute energy peaks ---
CLIP_BEFORE = 2        # Seconds before the drop to include in the clip
CLIP_AFTER = 28        # Seconds after the drop to include in the clip
MIN_DROP_DISTANCE = 30 # Minimum distance between drops (seconds)
ENERGY_THRESHOLD = 85  # Percentile to consider an energy peak as a drop

# --- Sets: bar alignment ---
BEAT_ALIGN = True      # Snap clip start/end to the musical grid
TARGET_DURATION = 30   # Desired clip length (seconds); rounded to whole phrases
BARS_BEFORE = 2        # Bars of lead-in before the drop
BEATS_PER_BAR = 4      # 4/4
BARS_PER_PHRASE = 16   # Clip length is quantized to whole phrases, not bars:
                       # ending mid-phrase lands on a beat but still hangs
EXPECTED_BPM = 174     # Tempo prior (drum and bass); without it librosa
                       # locks onto the half-time pulse (~87 BPM)

# --- Interviews: transcript + Claude ---
MOMENTS = 30           # How many highlights to ask Claude for
MIN_MOMENT = 15        # Reject anything shorter (seconds)
MAX_MOMENT = 90        # Reject anything longer (seconds)
WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"
WHISPER_LANGUAGE = None  # None autodetects
CLAUDE_MODEL = "claude-opus-5"
CLAUDE_TIMEOUT = 600   # Seconds to wait for the model to answer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "transcripts")

SYSTEM_PROMPT = """\
You are a video editor picking clips from an interview for social media.

You get the full transcript with timestamps. Choose the moments that work as \
standalone clips, and return the exact timestamps to cut.

A moment is worth clipping when it:
- stands on its own, with no need for what came before to make sense
- opens on something that makes you want to keep watching
- contains one complete thought: it starts where the idea starts and ends where it lands
- is memorable — a story, a strong opinion, a surprise, a joke, a confession

Skip pleasantries, throat-clearing, logistics, and anything that only makes \
sense with context you are cutting away.

Rules for the timestamps:
- start at the beginning of the sentence that opens the thought, not mid-sentence
- end where the thought finishes, not at a hard stop mid-sentence
- each clip must be between {min_moment} and {max_moment} seconds long
- clips must not overlap each other
- use the timestamps from the transcript; never invent one

Order the list best first, strongest clip at the top.

Rank on how well the fragment works *pulled out of the interview*, not on how \
important it was to the conversation: the strength of its opening, whether the \
idea closes on its own, and how memorable it is. A point that is central to the \
interview but needs ten minutes of setup ranks below a side story that lands in \
twenty seconds.

Write `title` and `reason` in the same language as the transcript. Keep `title` \
short — a few words, usable as a filename.

Return ONLY a JSON object, no prose and no code fence:
{{"moments": [{{"start": 123.4, "end": 156.7, "title": "...", "reason": "..."}}]}}\
"""


def extract_audio(video_path: str, audio_path: str, sample_rate: int = 22050) -> None:
    """Extract audio from a video file to mono WAV."""
    print(f"  Extracting audio from {os.path.basename(video_path)}...")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-ac", "1", "-ar", str(sample_rate), "-vn",
            audio_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def video_duration(video_path: str) -> float:
    """Length of a video in seconds, via ffprobe."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1",
            video_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
        text=True,
    )
    return float(out.stdout.strip())


# --------------------------------------------------------------------------
# Sets
# --------------------------------------------------------------------------


def detect_drops(y: np.ndarray, sr: int) -> list[float]:
    """
    Analyze audio and detect high-energy moments.

    Detects absolute energy peaks, not just buildup-to-drop transitions.
    This also captures chained drops where the DJ doesn't lower the energy.

    Returns a list of timestamps (in seconds) where drops occur.
    """
    print("  Analyzing energy...")
    hop_length = 2048
    rms = librosa.feature.rms(y=y, frame_length=4096, hop_length=hop_length)[0]

    # Smooth the energy curve (~1 second window)
    kernel_size = 11
    kernel = np.ones(kernel_size) / kernel_size
    rms_smooth = np.convolve(rms, kernel, mode="same")

    threshold = np.percentile(rms_smooth, ENERGY_THRESHOLD)
    high_energy_frames = np.where(rms_smooth > threshold)[0]
    if len(high_energy_frames) == 0:
        return []

    times = librosa.frames_to_time(high_energy_frames, sr=sr, hop_length=hop_length)

    # Group nearby detections and keep the peak with highest energy
    drops = []
    group_start = 0
    for i in range(1, len(times)):
        if times[i] - times[group_start] > MIN_DROP_DISTANCE:
            group = high_energy_frames[group_start:i]
            drops.append(times[group_start + int(np.argmax(rms_smooth[group]))])
            group_start = i

    group = high_energy_frames[group_start:]
    drops.append(times[group_start + int(np.argmax(rms_smooth[group]))])
    return drops


def track_beats(y: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    """Estimate tempo and beat times. Returns (bpm, beat_times)."""
    print("  Tracking beats...")
    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, units="frames", start_bpm=EXPECTED_BPM
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return float(np.atleast_1d(tempo)[0]), beat_times


def align_to_bars(drop_time: float, beats: np.ndarray) -> tuple[float, float]:
    """
    Snap a clip to the musical grid around a drop.

    The grid is anchored on the drop itself (drops land on a downbeat). The clip
    starts BARS_BEFORE bars earlier and ends a whole number of phrases *after
    the drop* — counting phrases from the start instead would push the end back
    into the middle of one. Returns (start, duration) in seconds.
    """
    fallback = (max(0.0, drop_time - CLIP_BEFORE), float(CLIP_BEFORE + CLIP_AFTER))
    if len(beats) < 2 * BEATS_PER_BAR:
        return fallback

    drop_i = int(np.argmin(np.abs(beats - drop_time)))

    # Median beat period is more robust than the reported BPM on long sets.
    beat_period = float(np.median(np.diff(beats)))
    bar = beat_period * BEATS_PER_BAR
    lead_in = BARS_BEFORE * bar
    phrases = max(1, round((TARGET_DURATION - lead_in) / (bar * BARS_PER_PHRASE)))

    start_i = drop_i - BARS_BEFORE * BEATS_PER_BAR
    end_i = drop_i + phrases * BARS_PER_PHRASE * BEATS_PER_BAR
    if start_i < 0 or end_i >= len(beats):
        return fallback

    start = float(beats[start_i])
    return start, float(beats[end_i]) - start


def plan_set_clips(video_path: str) -> list[dict]:
    """Find the drops in a set and the cut around each one."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name
    try:
        extract_audio(video_path, audio_path)
        print("  Loading audio...")
        y, sr = librosa.load(audio_path, sr=22050)
    finally:
        os.unlink(audio_path)

    drops = detect_drops(y, sr)
    if not drops or not BEAT_ALIGN:
        return [
            {"at": t, "start": max(0.0, t - CLIP_BEFORE),
             "duration": float(CLIP_BEFORE + CLIP_AFTER)}
            for t in drops
        ]

    bpm, beats = track_beats(y, sr)
    print(f"  Tempo: {bpm:.1f} BPM")

    clips = []
    for t in drops:
        start, duration = align_to_bars(t, beats)
        clips.append({"at": t, "start": start, "duration": duration})
    return clips


# --------------------------------------------------------------------------
# Interviews
# --------------------------------------------------------------------------


def transcribe(video_path: str, video_name: str) -> list[dict]:
    """
    Transcribe the speech in a video, as a list of {start, end, text} segments.

    Whisper runs locally, so nothing but the resulting text ever leaves the
    machine. Transcripts are cached in transcripts/: re-running to try a
    different prompt costs an API call, not another pass over the audio.
    """
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    cache_path = os.path.join(TRANSCRIPT_DIR, f"{video_name}.json")

    if os.path.exists(cache_path):
        print(f"  Using cached transcript ({os.path.basename(cache_path)})")
        with open(cache_path) as f:
            return json.load(f)["segments"]

    import mlx_whisper

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name
    try:
        extract_audio(video_path, audio_path, sample_rate=16000)
        print(f"  Transcribing with {WHISPER_MODEL} (this takes a while)...")
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=WHISPER_MODEL,
            language=WHISPER_LANGUAGE,
        )
    finally:
        os.unlink(audio_path)

    segments = [
        {"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()}
        for s in result["segments"]
    ]
    with open(cache_path, "w") as f:
        json.dump({"language": result.get("language"), "segments": segments}, f)

    print(f"  {len(segments)} segments, language: {result.get('language')}")
    return segments


def format_transcript(segments: list[dict]) -> str:
    """Render the transcript as timestamped lines for the model to read."""
    return "\n".join(
        f"[{s['start']:.1f} - {s['end']:.1f}] {s['text']}" for s in segments
    )


def call_claude_cli(system: str, prompt: str) -> str:
    """
    Ask Claude via Claude Code in headless mode, billed to the user's
    subscription rather than to an API key.

    Run from a scratch directory: started inside the project, Claude Code would
    load this repo's CLAUDE.md into the context of a question that has nothing
    to do with it.
    """
    with tempfile.TemporaryDirectory() as scratch:
        result = subprocess.run(
            [
                "claude", "-p",
                "--model", CLAUDE_MODEL,
                "--output-format", "json",
                "--append-system-prompt", system,
            ],
            input=prompt,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=scratch,
            timeout=CLAUDE_TIMEOUT,
        )

    if result.returncode != 0:
        raise RuntimeError(f"claude failed: {result.stderr.strip()[:500]}")

    envelope = json.loads(result.stdout)
    if envelope.get("is_error"):
        raise RuntimeError(f"claude returned an error: {envelope.get('result')}")
    return envelope["result"]


def call_claude_api(system: str, prompt: str) -> str:
    """Ask Claude through the Anthropic API, billed per token."""
    import anthropic

    response = anthropic.Anthropic().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in response.content if b.type == "text")


def parse_moments(raw: str) -> list[dict]:
    """Pull the moments out of the model's answer, tolerating a code fence."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    else:
        # Fall back to the outermost braces if the model added a sentence.
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]

    return json.loads(text)["moments"]


def snap_to_segments(
    start: float, end: float, segments: list[dict], total_duration: float
) -> tuple[float, float]:
    """
    Move a proposed cut onto the nearest transcript boundaries.

    Whisper's segment edges sit where the speaker actually paused, so snapping
    to them keeps the clip from opening or closing mid-word — the speech
    counterpart of align_to_bars. Also clamps to the video, since a timestamp
    past the end would silently produce a short or broken clip.
    """
    starts = np.array([s["start"] for s in segments])
    ends = np.array([s["end"] for s in segments])

    start = float(starts[int(np.argmin(np.abs(starts - start)))])
    end = float(ends[int(np.argmin(np.abs(ends - end)))])

    start = max(0.0, start)
    end = min(total_duration, end)
    return start, end - start


def plan_interview_clips(video_path: str, video_name: str, llm: str) -> list[dict]:
    """Transcribe an interview, ask Claude for the best moments, and vet them."""
    segments = transcribe(video_path, video_name)
    if not segments:
        return []

    system = SYSTEM_PROMPT.format(min_moment=MIN_MOMENT, max_moment=MAX_MOMENT)
    prompt = (
        f"Pick the {MOMENTS} best moments from this interview transcript.\n\n"
        f"{format_transcript(segments)}"
    )

    print(f"  Asking Claude for {MOMENTS} moments ({llm})...")
    raw = (call_claude_cli if llm == LLM_CLI else call_claude_api)(system, prompt)
    moments = parse_moments(raw)

    # The list arrives ranked best-first and is kept that way, so clip 01 is the
    # strongest one. When two overlap, the higher-ranked clip is the one to keep.
    total_duration = video_duration(video_path)
    clips = []
    for m in moments:
        start, duration = snap_to_segments(
            float(m["start"]), float(m["end"]), segments, total_duration
        )
        title = m.get("title", "?")

        # The model can propose a cut that snapping or clamping ruins; drop
        # those rather than hand ffmpeg a negative or absurd duration.
        if not MIN_MOMENT <= duration <= MAX_MOMENT:
            print(f"    Skipping {title!r}: {duration:.1f}s out of range")
            continue
        if any(start < c["start"] + c["duration"] and c["start"] < start + duration
               for c in clips):
            print(f"    Skipping {title!r}: overlaps a higher-ranked clip")
            continue

        clips.append(
            {
                "at": start,
                "start": start,
                "duration": duration,
                "title": title,
                "reason": m.get("reason", ""),
            }
        )

    save_notes(video_name, clips)
    return clips


def save_notes(video_name: str, clips: list[dict]) -> None:
    """Write why each moment was picked, next to the transcript."""
    path = os.path.join(TRANSCRIPT_DIR, f"{video_name}_moments.txt")
    with open(path, "w") as f:
        f.write(f"{video_name} - {len(clips)} moments, best first\n\n")
        for i, c in enumerate(clips, 1):
            f.write(
                f"{i:02d}. [{format_timestamp(c['start'])}] "
                f"{c['title']} ({c['duration']:.0f}s)\n    {c['reason']}\n\n"
            )
    print(f"  Notes: {os.path.relpath(path, BASE_DIR)}")


# --------------------------------------------------------------------------
# Cutting
# --------------------------------------------------------------------------


def format_timestamp(seconds: float) -> str:
    """Format seconds as MMmSSs."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}m{s:02d}s"


def cut_clip(
    video_path: str, start: float, duration: float, output_path: str, precise: bool
) -> None:
    """
    Cut a clip from the video.

    Stream copy can only cut on keyframes, which drifts the start by up to a GOP
    and would undo any alignment, so aligned clips are re-encoded instead.
    """
    if precise:
        codec_args = [
            "-c:v", "h264_videotoolbox", "-q:v", "80",
            "-c:a", "aac", "-b:a", "192k",
        ]
    else:
        codec_args = ["-c", "copy"]

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            *codec_args,
            output_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def process_video(video_path: str, mode: str, llm: str) -> None:
    """Process a video: find the good moments and generate clips."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\nProcessing: {video_name}")

    if mode == MODE_SET:
        clips = plan_set_clips(video_path)
        label, precise = "drop", BEAT_ALIGN
    else:
        clips = plan_interview_clips(video_path, video_name, llm)
        label, precise = "moment", True

    if not clips:
        print(f"  No {label}s detected.")
        return

    print(f"  Detected {len(clips)} {label}s:")
    for i, clip in enumerate(clips, 1):
        ts = format_timestamp(clip["at"])
        title = f" - {clip['title']}" if clip.get("title") else ""
        print(f"    {label.capitalize()} {i}: {ts} ({clip['duration']:.1f}s){title}")

        output_name = f"{video_name}_{label}_{i:02d}_{ts}.mp4"
        cut_clip(
            video_path,
            clip["start"],
            clip["duration"],
            os.path.join(OUTPUT_DIR, output_name),
            precise,
        )
        print(f"    -> {output_name}")


def ask_mode() -> str:
    """Ask what kind of content is in input/."""
    print("\nWhat kind of content is in input/?")
    print("  1) Music set - cuts on energy drops, aligned to the musical grid")
    print("  2) Interview - transcribes the speech and asks Claude what to keep")

    if not sys.stdin.isatty():
        print("\nNot running interactively. Pass --mode set or --mode interview.")
        sys.exit(1)

    while True:
        choice = input("Choose [1/2]: ").strip().lower()
        if choice in ("1", "set", "music"):
            return MODE_SET
        if choice in ("2", "interview"):
            return MODE_INTERVIEW
        print("  Please answer 1 or 2.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=[MODE_SET, MODE_INTERVIEW])
    parser.add_argument(
        "--llm",
        choices=[LLM_CLI, LLM_API],
        default=LLM_CLI,
        help="how to reach Claude for interviews (default: the claude CLI, "
        "billed to your Claude subscription)",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_extensions = ("*.mp4", "*.mov", "*.mkv", "*.avi")
    videos = sorted(
        v for ext in video_extensions for v in glob.glob(os.path.join(INPUT_DIR, ext))
    )
    if not videos:
        print(f"No videos found in {INPUT_DIR}/")
        print("Place your videos there and run again.")
        sys.exit(1)

    print(f"Found {len(videos)} video(s) in {INPUT_DIR}/")

    mode = args.mode or ask_mode()
    print(f"Mode: {mode}")

    for video_path in videos:
        process_video(video_path, mode, args.llm)

    print("\nDone! Clips are in output/")


if __name__ == "__main__":
    main()
