#!/usr/bin/env python3
"""
Rank - Keeps the clips with the strongest light show and sets the rest aside.

The sets are filmed in front of LED walls, and what makes a clip worth posting
is the lights going off, not the drop itself. Each clip is scored by how much
the picture changes from frame to frame (strobes, blackouts, colour hits), the
best KEEP are kept at least MIN_GAP seconds apart, and the rest are moved to
carrusel/descartados/ -- nothing is deleted.

Usage:
    python rank.py "<set folder>"     (holds video-audio.mov and carrusel/)

The clips in carrusel/ must be the untouched c1..cN from detect -> downscale ->
rename: their timestamps are recovered by re-running the drop planning on
video-audio.mov, which only works if nothing was added, removed or reordered.
"""

import os
import re
import sys
import glob
import json
import argparse
import subprocess

import numpy as np

import detect

KEEP = 15          # Clips left in carrusel/
MIN_GAP = 45       # Seconds between the starts of two kept clips. Clips last
                   # ~25s, so anything under that overlaps; 45 also keeps two
                   # back-to-back clips of the same stretch from both surviving
FPS = 12           # Frames sampled per second; strobes at 172 BPM are ~3 Hz
WIDTH, HEIGHT = 64, 36
WALL = 0.7         # Top fraction of the frame scored; the decks are below

DISCARD = "descartados"


def natural_key(path: str) -> int:
    return int(re.findall(r"\d+", os.path.basename(path))[0])


def light_score(path: str) -> float:
    """Mean frame-to-frame brightness change over the wall, 0..1."""
    raw = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", path,
            "-vf", f"fps={FPS},scale={WIDTH}:{HEIGHT}",
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, HEIGHT, WIDTH)
    wall = frames[:, : int(HEIGHT * WALL)].astype(np.float32) / 255
    return float(np.abs(np.diff(wall, axis=0)).mean())


def clip_times(video: str, carrusel: str, n: int) -> list[float]:
    """Start time of each clip, from the same planning detect.py used to cut them."""
    cache = os.path.join(carrusel, ".times.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    starts = [c["start"] for c in detect.plan_set_clips(video)]
    if len(starts) != n:
        raise SystemExit(
            f"Planning found {len(starts)} drops but carrusel/ has {n} clips; "
            "it was edited or cut with different settings."
        )
    with open(cache, "w") as f:
        json.dump(starts, f)
    return starts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder")
    parser.add_argument("--keep", type=int, default=KEEP)
    parser.add_argument("--gap", type=float, default=MIN_GAP)
    args = parser.parse_args()

    carrusel = os.path.join(args.folder, "carrusel")
    discard_dir = os.path.join(carrusel, DISCARD)
    if os.path.isdir(discard_dir):
        sys.exit(f"{DISCARD}/ already exists in {carrusel}; already ranked.")

    clips = sorted(glob.glob(os.path.join(carrusel, "c*.mp4")), key=natural_key)
    if not clips:
        sys.exit(f"No clips in {carrusel}")
    print(f"{len(clips)} clips in {carrusel}")

    starts = clip_times(os.path.join(args.folder, "video-audio.mov"), carrusel, len(clips))

    print("Scoring lights...")
    scored = []
    for path, start in zip(clips, starts):
        try:
            scored.append({"path": path, "start": start, "score": light_score(path)})
        except Exception as e:
            # An unreadable clip stays where it is, unranked, rather than stop the rest.
            print(f"  !! {os.path.basename(path)} skipped: {e}")

    # Best first; a clip too close to a better one already kept is dropped.
    scored.sort(key=lambda c: c["score"], reverse=True)
    kept, dropped = [], []
    for c in scored:
        close = any(abs(c["start"] - k["start"]) < args.gap for k in kept)
        (dropped if close or len(kept) >= args.keep else kept).append(c)

    # Kept clips are renamed by rank (c1 = best light show); discarded ones by
    # their time in the set. Temp names first, as in rename.py.
    os.makedirs(discard_dir)
    for c in dropped:
        name = f"{detect.format_timestamp(c['start'])}.mp4"
        os.rename(c["path"], os.path.join(discard_dir, name))
        c["new"] = f"{DISCARD}/{name}"
    for i, c in enumerate(kept, 1):
        tmp = os.path.join(carrusel, f"_temp_c{i}.mp4")
        os.rename(c["path"], tmp)
        c["path"] = tmp
    for i, c in enumerate(kept, 1):
        os.rename(c["path"], os.path.join(carrusel, f"c{i}.mp4"))
        c["new"] = f"c{i}.mp4"

    with open(os.path.join(carrusel, "ranking.txt"), "w") as f:
        f.write("score  time    file\n")
        for c in kept + dropped:
            f.write(f"{c['score']:.3f}  {detect.format_timestamp(c['start'])}  {c['new']}\n")

    print(f"\nKept {len(kept)} in carrusel/ (c1 = most lights), "
          f"moved {len(dropped)} to {DISCARD}/. See ranking.txt")


if __name__ == "__main__":
    main()
