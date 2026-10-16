#!/usr/bin/env python3
"""
Rank - Keeps the clips with the strongest light show and sets the rest aside.

The sets are filmed in front of LED walls, and what makes a clip worth posting
is the lights going off, not the drop itself. Each clip is scored by how much
the picture changes from frame to frame (strobes, blackouts, colour hits), the
best KEEP are kept at least MIN_GAP seconds apart, and the rest are moved --
nothing is deleted -- to carrusel/repetidos/ when they fall within MIN_GAP of a
kept clip (same stretch of the set) or to carrusel/descartados/ otherwise.

Run again on a folder ranked before repetidos/ existed, it only splits the
repeated clips out of descartados/, using ranking.txt.

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
REPEATED = "repetidos"


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


def seconds(ts: str) -> int:
    """Inverse of detect.format_timestamp: '14m33s' -> 873."""
    m, s = re.fullmatch(r"(\d+)m(\d+)s", ts).groups()
    return int(m) * 60 + int(s)


def write_ranking(carrusel: str, clips: list[dict]) -> None:
    with open(os.path.join(carrusel, "ranking.txt"), "w") as f:
        f.write("score  time    file\n")
        for c in clips:
            f.write(f"{c['score']:.3f}  {detect.format_timestamp(c['start'])}  {c['new']}\n")


def split_repeated(carrusel: str, gap: float) -> None:
    """Move the discarded clips that repeat a kept one out to repetidos/."""
    with open(os.path.join(carrusel, "ranking.txt")) as f:
        rows = [line.split() for line in f.readlines()[1:] if line.strip()]
    clips = [{"score": float(r[0]), "start": seconds(r[1]), "new": r[2]} for r in rows]
    kept = [c for c in clips if "/" not in c["new"]]

    os.makedirs(os.path.join(carrusel, REPEATED))
    moved = 0
    for c in clips:
        if c["new"].startswith(f"{DISCARD}/") and any(
            abs(c["start"] - k["start"]) < gap for k in kept
        ):
            new = f"{REPEATED}/{os.path.basename(c['new'])}"
            try:
                os.rename(os.path.join(carrusel, c["new"]), os.path.join(carrusel, new))
            except OSError as e:
                print(f"  !! {c['new']} not moved: {e}")
                continue
            c["new"] = new
            moved += 1
    write_ranking(carrusel, clips)
    print(f"Moved {moved} repeated clips from {DISCARD}/ to {REPEATED}/")


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
        if os.path.isdir(os.path.join(carrusel, REPEATED)):
            sys.exit(f"{carrusel} is already ranked.")
        split_repeated(carrusel, args.gap)
        return

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

    # Kept clips are renamed by rank (c1 = best light show); the others by
    # their time in the set, into repetidos/ if they repeat a kept clip's
    # stretch and descartados/ if they just didn't make the cut. Temp names
    # first, as in rename.py.
    os.makedirs(discard_dir)
    os.makedirs(os.path.join(carrusel, REPEATED))
    for c in dropped:
        repeated = any(abs(c["start"] - k["start"]) < args.gap for k in kept)
        folder = REPEATED if repeated else DISCARD
        name = f"{detect.format_timestamp(c['start'])}.mp4"
        os.rename(c["path"], os.path.join(carrusel, folder, name))
        c["new"] = f"{folder}/{name}"
    for i, c in enumerate(kept, 1):
        tmp = os.path.join(carrusel, f"_temp_c{i}.mp4")
        os.rename(c["path"], tmp)
        c["path"] = tmp
    for i, c in enumerate(kept, 1):
        os.rename(c["path"], os.path.join(carrusel, f"c{i}.mp4"))
        c["new"] = f"c{i}.mp4"

    write_ranking(carrusel, kept + dropped)

    n_rep = sum(c["new"].startswith(REPEATED) for c in dropped)
    print(f"\nKept {len(kept)} in carrusel/ (c1 = most lights), moved {n_rep} to "
          f"{REPEATED}/ and {len(dropped) - n_rep} to {DISCARD}/. See ranking.txt")


if __name__ == "__main__":
    main()
