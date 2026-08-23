# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A three-stage pipeline that finds the good moments in a video and produces short, Instagram-ready clips. It handles two kinds of content, chosen by a prompt at startup (or `--mode`):

- **Music sets** — finds the high-energy drops. Fully offline: `ffmpeg` plus audio energy analysis via `librosa`.
- **Interviews** — transcribes the speech with Whisper (local) and asks Claude which moments are worth clipping. Only the transcript text leaves the machine; the video never does.

## Commands

The npm scripts are just thin wrappers that invoke the project's virtualenv Python. Run them in pipeline order:

```bash
npm run detect      # input/*.{mp4,mov,mkv,avi}  -> output/<name>_{drop,moment}_NN_MMmSSs.mp4
npm run downscale   # output/*                   -> downscaled/* (1080p, Instagram)
npm run rename      # output/* and downscaled/*  -> c1, c2, ... (sequential, in place, per folder)
npm run clean       # empties input/, output/ and downscaled/ (deletes all files in them)
```

Equivalent direct calls: `.venv/bin/python detect.py` (and `downscale.py`, `rename.py`).

`detect.py` asks whether `input/` holds a music set or an interview; `--mode set` / `--mode interview` skips the prompt (and is required when stdin isn't a TTY). Interview mode also takes `--llm`, covered below.

First-time setup:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Requires `ffmpeg` on PATH. Interview mode additionally needs the `claude` CLI on PATH (see `--llm` below). There are no tests, linters, or build step.

## Pipeline architecture

The three scripts are independent and chained only by directory convention — each reads one folder and writes the next. `input/`, `output/`, and `downscaled/` are all gitignored, so the repo ships empty.

```
input/ ──detect.py──▶ output/ ──downscale.py──▶ downscaled/
                         └──rename.py──▶ output/ (in place)
```

`detect.py` splits by mode only in *how it picks the moments* — `plan_set_clips` and `plan_interview_clips` both return the same list of `{at, start, duration}` dicts, and everything downstream (cutting, naming, the other two scripts) is shared.

- **detect.py** — the core. For each input video: extract mono 22050 Hz WAV (ffmpeg), compute RMS energy with librosa, smooth it (~1s moving average), then keep frames above the `ENERGY_THRESHOLD` percentile. Nearby detections are grouped within `MIN_DROP_DISTANCE` seconds and only the highest-energy peak per group survives. Each surviving drop is then cut from the **original** video. The tuning knobs are module-level constants at the top of the file — adjust detection behavior there. Note the design choice (docstring of `detect_drops`): it targets absolute energy peaks rather than buildup→drop transitions, so chained drops are captured even when the DJ never lowers energy.

  Cut boundaries are **bar-aligned** when `BEAT_ALIGN` is on (the default): `librosa.beat.beat_track` gives the beat grid, seeded with `EXPECTED_BPM` (174 — the sets are drum and bass; without that prior librosa locks onto the half-time pulse at ~87 BPM and the grid comes out half as fine). `align_to_bars` then anchors it on the drop itself — drops land on a downbeat, which sidesteps having to detect downbeats independently. The clip starts `BARS_BEFORE` bars earlier and ends a whole number of `BARS_PER_PHRASE`-bar phrases **after the drop** — counting phrases from the clip start instead would push the end back into the middle of one, which lands on a beat but still hangs. Note that phrase quantization, not `TARGET_DURATION`, decides the length: at 172 BPM a 16-bar phrase is 22.3s, so clips come out ~24.4s (one phrase plus lead-in) and the next step up is ~47s. `TARGET_DURATION` only picks which multiple is nearest. Because stream copy can only cut on keyframes — drifting the start by up to a GOP and undoing the alignment — aligned clips are **re-encoded** with `h264_videotoolbox` (~7s per clip; macOS-only, same caveat as `downscale.py`). Setting `BEAT_ALIGN = False` reverts to the old fast path: fixed `CLIP_BEFORE`+`CLIP_AFTER` seconds with `-c copy`. That fallback is also used per-clip whenever beat tracking is unusable (too few beats, or the clip would run past either end of the grid).
  **Interview mode** (`plan_interview_clips`) replaces detection entirely rather than re-tuning it. Absolute energy is meaningless on levelled speech, and an earlier attempt at scoring emphasis against a rolling-median baseline was dropped: it ranks *how* something was said, never *what*, so it can't tell a good story from a loud one. Instead:

  1. `transcribe` extracts 16 kHz mono WAV and runs Whisper locally via `mlx-whisper` (Apple Silicon GPU), with `verbose=False` so it draws a progress bar — Whisper's default prints nothing, and an hour of audio in silence reads as a hang. The first run also downloads ~3 GB of weights, whose Xet backend signs off with its own `Reconstruction complete` just before the quiet part starts. Results are cached in `transcripts/<name>.json` — re-running to try a different prompt should cost an API call, not another pass over the audio. Delete the cache file to force a re-transcribe.
  2. The timestamped transcript goes to Claude with `SYSTEM_PROMPT`, which asks for `MOMENTS` self-contained highlights as JSON, **ordered best first**. Ranking is on how well a fragment works pulled out of the interview — opening, whether the idea closes on its own, memorability — explicitly *not* on how central it was to the conversation. That order is preserved end to end, so clip `01` is the strongest and `rename.py` makes it `c1`; the clips are not in chronological order (the filename timestamp recovers that). Where two clips overlap, the higher-ranked one wins and the other is dropped.
  3. `snap_to_segments` moves each proposed cut onto the nearest Whisper segment boundary — those sit where the speaker actually paused, so this is the speech counterpart of `align_to_bars` — then clamps to the real video duration. Clips outside `MIN_MOMENT`..`MAX_MOMENT` are dropped: the model can return a plausible-looking span that snapping or clamping ruins, and ffmpeg would accept a broken one silently.

  Why each moment was picked is written to `transcripts/<name>_moments.txt`, not to `output/` — `rename.py` would otherwise renumber it alongside the clips. Interview clips are always re-encoded, for the same keyframe reason as bar-aligned ones.

  `--llm` picks how Claude is reached. The default `cli` shells out to `claude -p --output-format json`, which bills to the user's Claude Code subscription and needs no API key; it runs in a temp directory so Claude Code doesn't load this repo's CLAUDE.md into an unrelated question. `--llm api` uses the `anthropic` SDK and needs `ANTHROPIC_API_KEY` (~$0.10 per hour-long interview). The CLI path can't use structured outputs, so `parse_moments` tolerates a code fence or a stray sentence around the JSON.

- **downscale.py** — re-encodes to 1080p height using macOS hardware encoding (`h264_videotoolbox`); not portable to non-Mac without changing the codec.
- **rename.py** — renames everything in `output/` and `downscaled/` (each folder independently) to `c1`, `c2`, … using a natural sort, going through `_temp_c*` names first to avoid mid-rename collisions.

All scripts resolve their I/O directories relative to the script file (`__file__`), so they work regardless of the current working directory.
