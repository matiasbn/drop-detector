#!/usr/bin/env python3
"""
Drop Detector - Detects the best moments (drops) in DJ sets
and generates trimmed video clips.

Usage:
    1. Place video files in the input/ folder
    2. Run: python detect.py
    3. Clips are generated in output/
"""

import os
import sys
import glob
import tempfile
import subprocess

import numpy as np
import librosa

# --- Configurable parameters ---
CLIP_BEFORE = 2        # Seconds before the drop to include in the clip
CLIP_AFTER = 28        # Seconds after the drop to include in the clip
MIN_DROP_DISTANCE = 30 # Minimum distance between drops (seconds)
ENERGY_THRESHOLD = 85  # Percentile to consider an energy peak as a drop

# --- Bar alignment ---
BEAT_ALIGN = True      # Snap clip start/end to the musical grid
TARGET_DURATION = 30   # Desired clip length (seconds); rounded to whole bars
BARS_BEFORE = 2        # Bars of lead-in before the drop
BEATS_PER_BAR = 4      # 4/4
BARS_PER_PHRASE = 16   # Clip length is quantized to whole phrases, not bars:
                       # ending mid-phrase lands on a beat but still hangs
EXPECTED_BPM = 174     # Tempo prior (drum and bass); without it librosa
                       # locks onto the half-time pulse (~87 BPM)

INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def extract_audio(video_path: str, audio_path: str) -> None:
    """Extract audio from a video file to mono WAV at 22050Hz."""
    print(f"  Extracting audio from {os.path.basename(video_path)}...")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-ac", "1", "-ar", "22050", "-vn",
            audio_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


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

    # Threshold: frames with energy above the configured percentile
    threshold = np.percentile(rms_smooth, ENERGY_THRESHOLD)
    high_energy_frames = np.where(rms_smooth > threshold)[0]

    if len(high_energy_frames) == 0:
        return []

    # Convert frames to seconds
    times = librosa.frames_to_time(high_energy_frames, sr=sr, hop_length=hop_length)

    # Group nearby detections and keep the peak with highest energy
    drops = []
    group_start = 0
    for i in range(1, len(times)):
        if times[i] - times[group_start] > MIN_DROP_DISTANCE:
            group_frames = high_energy_frames[group_start:i]
            best_idx = group_start + np.argmax(rms_smooth[group_frames])
            drops.append(times[best_idx])
            group_start = i

    # Last group
    group_frames = high_energy_frames[group_start:]
    best_idx = group_start + np.argmax(rms_smooth[group_frames])
    drops.append(times[best_idx])

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
    fallback = (max(0.0, drop_time - CLIP_BEFORE), CLIP_BEFORE + CLIP_AFTER)
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


def format_timestamp(seconds: float) -> str:
    """Format seconds as MMmSSs."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}m{s:02d}s"


def cut_clip(video_path: str, start: float, duration: float, output_path: str) -> None:
    """
    Cut a clip from the video.

    Stream copy can only cut on keyframes, which drifts the start by up to a GOP
    and would undo the bar alignment, so aligned clips are re-encoded instead.
    """
    if BEAT_ALIGN:
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


def process_video(video_path: str) -> None:
    """Process a video: detect drops and generate clips."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\nProcessing: {video_name}")

    # Extract audio to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        extract_audio(video_path, audio_path)
        print("  Loading audio...")
        y, sr = librosa.load(audio_path, sr=22050)
    finally:
        os.unlink(audio_path)

    drops = detect_drops(y, sr)

    if not drops:
        print("  No drops detected.")
        return

    if BEAT_ALIGN:
        bpm, beats = track_beats(y, sr)
        print(f"  Tempo: {bpm:.1f} BPM")
    else:
        beats = np.array([])

    print(f"  Detected {len(drops)} drops:")
    for i, t in enumerate(drops, 1):
        ts = format_timestamp(t)
        start, duration = align_to_bars(t, beats) if BEAT_ALIGN else (
            max(0.0, t - CLIP_BEFORE), float(CLIP_BEFORE + CLIP_AFTER)
        )
        print(f"    Drop {i}: {ts} ({duration:.1f}s)")

        output_name = f"{video_name}_drop_{i:02d}_{ts}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        cut_clip(video_path, start, duration, output_path)
        print(f"    -> {output_name}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_extensions = ("*.mp4", "*.mov", "*.mkv", "*.avi")
    videos = sorted(
        v for ext in video_extensions for v in glob.glob(os.path.join(INPUT_DIR, ext))
    )
    if not videos:
        print(f"No videos found in {INPUT_DIR}/")
        print("Place your DJ set videos there and run again.")
        sys.exit(1)

    print(f"Found {len(videos)} video(s) in {INPUT_DIR}/")

    for video_path in videos:
        process_video(video_path)

    print("\nDone! Clips are in output/")


if __name__ == "__main__":
    main()
