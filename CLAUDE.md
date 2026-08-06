# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A three-stage local pipeline that finds the high-energy "drops" in DJ set videos and produces short, Instagram-ready clips. Everything runs offline by shelling out to `ffmpeg`; the only intelligence is audio energy analysis via `librosa`.

## Commands

The npm scripts are just thin wrappers that invoke the project's virtualenv Python. Run them in pipeline order:

```bash
npm run detect      # input/*.{mp4,mov,mkv,avi}  -> output/<name>_drop_NN_MMmSSs.mp4
npm run downscale   # output/*                   -> downscaled/* (1080p, Instagram)
npm run rename      # output/* and downscaled/*  -> c1, c2, ... (sequential, in place, per folder)
npm run clean       # empties input/, output/ and downscaled/ (deletes all files in them)
```

Equivalent direct calls: `.venv/bin/python detect.py` (and `downscale.py`, `rename.py`).

First-time setup:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Requires `ffmpeg` on PATH. There are no tests, linters, or build step.

## Pipeline architecture

The three scripts are independent and chained only by directory convention — each reads one folder and writes the next. `input/`, `output/`, and `downscaled/` are all gitignored, so the repo ships empty.

```
input/ ──detect.py──▶ output/ ──downscale.py──▶ downscaled/
                         └──rename.py──▶ output/ (in place)
```

- **detect.py** — the core. For each input video: extract mono 22050 Hz WAV (ffmpeg), compute RMS energy with librosa, smooth it (~1s moving average), then keep frames above the `ENERGY_THRESHOLD` percentile. Nearby detections are grouped within `MIN_DROP_DISTANCE` seconds and only the highest-energy peak per group survives. Each surviving drop is then cut from the **original** video. The tuning knobs are module-level constants at the top of the file — adjust detection behavior there. Note the design choice (docstring of `detect_drops`): it targets absolute energy peaks rather than buildup→drop transitions, so chained drops are captured even when the DJ never lowers energy.

  Cut boundaries are **bar-aligned** when `BEAT_ALIGN` is on (the default): `librosa.beat.beat_track` gives the beat grid, and `align_to_bars` anchors it on the drop itself — drops land on a downbeat, which sidesteps having to detect downbeats independently. The clip starts `BARS_BEFORE` bars earlier and runs the whole number of bars closest to `TARGET_DURATION`, so both ends fall on a beat (~30s in practice, varying with tempo). Because stream copy can only cut on keyframes — drifting the start by up to a GOP and undoing the alignment — aligned clips are **re-encoded** with `h264_videotoolbox` (~7s per clip; macOS-only, same caveat as `downscale.py`). Setting `BEAT_ALIGN = False` reverts to the old fast path: fixed `CLIP_BEFORE`+`CLIP_AFTER` seconds with `-c copy`. That fallback is also used per-clip whenever beat tracking is unusable (too few beats, or the clip would run past either end of the grid).
- **downscale.py** — re-encodes to 1080p height using macOS hardware encoding (`h264_videotoolbox`); not portable to non-Mac without changing the codec.
- **rename.py** — renames everything in `output/` and `downscaled/` (each folder independently) to `c1`, `c2`, … using a natural sort, going through `_temp_c*` names first to avoid mid-rename collisions.

All scripts resolve their I/O directories relative to the script file (`__file__`), so they work regardless of the current working directory.
