#!/usr/bin/env python3
"""
Reel - Turns carousel clips into vertical 9:16 reels with a virtual camera
that follows the DJ, and closes them with the DNB Chile logo outro.

The sets are filmed horizontally in 4K from a fixed camera. A reel is a 9:16
window cut out of that frame that moves with the DJ like a camera operator
would: tight on the horizontal, softer on the vertical, banking a little into
each pan. It ends with a 0.2 s flash into assets/outro-dnbchile.mp4 (Signal
Glitch 2 transition, the logo, Black Hole mask out -- exported from CapCut for
the kapza reel), with the set's audio running under the logo and fading out
over the last second.

Usage:
    python reel.py "<set folder>" c15 [c3 ...]          # clips from carrusel/
    python reel.py "<set folder>" descartados/47m52s     # any clip in ranking.txt
    python reel.py "<set folder>" --start 1159.9 --duration 24.7

Writes <set folder>/reel-<clip>.mp4. The clip is cut again from the 4K
video-audio.mov, not upscaled from the 1080p carousel file.
"""

import os
import re
import sys
import json
import argparse
import tempfile
import subprocess

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

ZOOM = 1.35        # Crop height = source height / ZOOM. 1.35 frames the DJ
                   # from the waist up, like the kapza reel
FOLLOW_X = 0.12    # Seconds of smoothing on the horizontal follow. Low = the
                   # DJ stays centred, the camera reacts at once
FOLLOW_Y = 0.3     # Same on the vertical. 0.2 bobbed with every head move,
                   # 0.7 looked static; 0.3 was the one approved
TILT = 3.0         # Max camera roll (degrees), proportional to pan speed
HEAD_AT = 0.12     # Top of the DJ's box sits this far down the crop
DETECT_FPS = 12    # Person detection rate; positions are interpolated between
FLASH = 0.2        # Seconds of flash/blur/RGB split leading into the outro
FADE = 1.0         # Audio fade-out at the very end (seconds)
OUT_W, OUT_H = 1080, 1920
BITRATE = "14M"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTRO = os.path.join(BASE_DIR, "assets", "outro-dnbchile.mp4")


def probe(path: str) -> dict:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate:format=duration",
            "-of", "json", path,
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    s = info["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return {"w": s["width"], "h": s["height"], "fps": num / den,
            "duration": float(info["format"]["duration"])}


def ffmpeg(args: list[str]) -> None:
    """Run ffmpeg quietly, but keep the reason when it fails."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", *args],
                       stderr=subprocess.PIPE, text=True, errors="replace")
    if r.returncode != 0:
        tail = r.stderr.strip().splitlines()[-1:] or ["no output"]
        raise RuntimeError(f"ffmpeg failed: {tail[0]}")


def seconds(ts: str) -> int:
    m, s = re.fullmatch(r"(\d+)m(\d+)s", ts).groups()
    return int(m) * 60 + int(s)


def clip_span(set_dir: str, clip: str) -> tuple[float, float]:
    """
    Exact start and duration of a carousel clip in the long video.

    ranking.txt (written by rank.py) maps the clip's current name to its start,
    rounded to the second; .times.json holds the exact starts, so the nearest
    one is taken. The duration is the carousel file's own.
    """
    carrusel = os.path.join(set_dir, "carrusel")
    name = clip if clip.endswith(".mp4") else f"{clip}.mp4"
    with open(os.path.join(carrusel, "ranking.txt")) as f:
        rows = [line.split() for line in f.readlines()[1:] if line.strip()]
    match = [ts for _, ts, rel in rows if rel == name]
    if not match:
        raise ValueError(f"{name} is not in ranking.txt (run rank.py first)")
    approx = seconds(match[0])
    with open(os.path.join(carrusel, ".times.json")) as f:
        starts = json.load(f)
    start = min(starts, key=lambda s: abs(s - approx))
    return start, probe(os.path.join(carrusel, name))["duration"]


def load_detector():
    import torch
    from torchvision.models.detection import (
        fasterrcnn_mobilenet_v3_large_fpn,
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    )
    # First use downloads ~74 MB of weights to ~/.cache/torch.
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT, box_score_thresh=0.4
    ).eval()
    return torch, model


def track_dj(detector, video: str, start: float, duration: float) -> tuple[np.ndarray, ...]:
    """
    Where the DJ is, DETECT_FPS times a second: (times, centre x, box top),
    both as fractions of the frame. The DJ is the biggest person in frame;
    frames with nobody found (rare -- it even sees through blackouts) are
    filled in from their neighbours.
    """
    torch, model = detector
    W, H = 640, 360
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(start), "-i", video, "-t", str(duration),
         "-vf", f"fps={DETECT_FPS},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)

    cx, top = [], []
    with torch.no_grad():
        for i in range(0, len(frames), 16):
            batch = [torch.from_numpy(f.copy()).permute(2, 0, 1).float() / 255
                     for f in frames[i : i + 16]]
            for d in model(batch):
                people = [b.tolist() for b, label in zip(d["boxes"], d["labels"]) if label == 1]
                if not people:
                    cx.append(np.nan)
                    top.append(np.nan)
                    continue
                b = max(people, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                cx.append((b[0] + b[2]) / 2 / W)
                top.append(b[1] / H)

    t = np.arange(len(cx)) / DETECT_FPS
    cx, top = np.array(cx), np.array(top)
    ok = ~np.isnan(cx)
    if ok.sum() < 2:
        raise RuntimeError("no DJ found in the clip")
    print(f"    DJ found in {ok.sum()}/{len(cx)} frames")
    return t, np.interp(t, t[ok], cx[ok]), np.interp(t, t[ok], top[ok])


def camera_path(t, cx, top, src: dict, duration: float, opts) -> dict:
    """
    Per output frame: crop position (x, y) and roll, plus the crop sizes.

    The rotated crop is cut larger than the final one (by cos + 16/9 sin of the
    max tilt) and trimmed after rotating, so no black corners show.
    """
    SW, SH, fps = src["w"], src["h"], src["fps"]
    ch = int(SH / opts.zoom) // 2 * 2
    cw = int(ch * 9 / 16) // 2 * 2
    margin = np.cos(np.radians(opts.tilt)) + 16 / 9 * np.sin(np.radians(opts.tilt))
    ch2, cw2 = int(ch * margin) // 2 * 2, int(cw * margin) // 2 * 2

    # Median filter kills single-frame detector jumps; the Gaussian is the camera.
    cx = gaussian_filter1d(median_filter(cx, 3, mode="nearest"), DETECT_FPS * opts.follow_x, mode="nearest")
    top = gaussian_filter1d(median_filter(top, 3, mode="nearest"), DETECT_FPS * opts.follow_y, mode="nearest")

    tf = np.arange(int(duration * fps)) / fps
    xc = np.interp(tf, t, cx) * SW
    yc = np.interp(tf, t, top) * SH - HEAD_AT * ch + ch / 2

    # Bank into the pan like a camera operator: roll follows horizontal speed,
    # scaled so the fastest 5% of the clip reaches ~80% of the max tilt.
    v = np.gradient(gaussian_filter1d(xc, fps * 0.3)) * fps
    roll = np.clip(v / (np.percentile(np.abs(v), 95) + 1e-9) * opts.tilt * 0.8, -opts.tilt, opts.tilt)
    roll = gaussian_filter1d(roll, fps * 0.3) if opts.tilt else np.zeros_like(tf)

    x = np.clip(xc - cw2 / 2, 0, SW - cw2).astype(int) // 2 * 2
    y = np.clip(yc - ch2 / 2, 0, SH - ch2).astype(int) // 2 * 2
    return {"t": tf, "x": x, "y": y, "roll": roll, "cw": cw, "ch": ch, "cw2": cw2, "ch2": ch2}


def render(video: str, start: float, duration: float, path: dict, out: str, outro: bool, tmp: str) -> None:
    """Cut the moving crop with ffmpeg (sendcmd drives crop + rotate per frame), then add the outro."""
    cmd_file = os.path.join(tmp, "camera.cmd")
    with open(cmd_file, "w") as f:
        for t, x, y, r in zip(path["t"], path["x"], path["y"], path["roll"]):
            f.write(f"{t:.4f} crop x {x}, crop y {y}, rotate a {np.radians(r):.5f};\n")

    camera = (
        f"sendcmd=f='{cmd_file}',"
        f"crop@crop=w={path['cw2']}:h={path['ch2']}:x={path['x'][0]}:y={path['y'][0]},"
        f"rotate@rotate=a=0:ow={path['cw2']}:oh={path['ch2']},"
        f"crop={path['cw']}:{path['ch']},"
        f"scale={OUT_W}:{OUT_H}:flags=lanczos"
    )
    if not outro:
        ffmpeg(["-ss", str(start), "-i", video, "-t", str(duration), "-vf", camera,
                "-c:v", "h264_videotoolbox", "-b:v", BITRATE, "-c:a", "aac", "-b:a", "192k", out])
        return

    main = os.path.join(tmp, "main.mp4")
    ffmpeg(["-ss", str(start), "-i", video, "-t", str(duration), "-vf", camera, "-an",
            "-c:v", "h264_videotoolbox", "-b:v", BITRATE, main])

    # The outro is resampled to the clip's frame rate so concat doesn't stutter.
    src_fps = probe(video)["fps"]
    outro_clip = os.path.join(tmp, "outro.mp4")
    ffmpeg(["-i", OUTRO, "-an", "-vf", f"fps={src_fps},format=yuv420p",
            "-c:v", "h264_videotoolbox", "-b:v", BITRATE, outro_clip])
    total = duration + probe(outro_clip)["duration"]

    # The CapCut glitch started over the kapza footage; its first FLASH seconds
    # are rebuilt here over our last frames, then the clean part takes over.
    # The set's audio keeps playing under the logo, as in the original reel.
    cut = duration - FLASH
    graph = (
        f"[0:v]split[m][f];"
        f"[m]trim=0:{cut},setpts=PTS-STARTPTS[a];"
        f"[f]trim={cut},setpts=PTS-STARTPTS,"
        f"eq=brightness=0.45:contrast=0.7,gblur=sigma=10,rgbashift=rh=-24:bh=24[b];"
        f"[a][b][1:v]concat=n=3:v=1:a=0[v];"
        f"[2:a]afade=t=out:st={total - FADE}:d={FADE}[au]"
    )
    ffmpeg(["-i", main, "-i", outro_clip, "-ss", str(start), "-t", str(total), "-i", video,
            "-filter_complex", graph, "-map", "[v]", "-map", "[au]",
            "-c:v", "h264_videotoolbox", "-b:v", BITRATE, "-c:a", "aac", "-b:a", "192k", out])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="set folder (holds the long video and carrusel/)")
    parser.add_argument("clips", nargs="*", help="clip names from carrusel/ranking.txt, e.g. c15")
    parser.add_argument("--start", type=float, help="cut by time instead of by clip (seconds)")
    parser.add_argument("--duration", type=float, help="with --start (seconds)")
    parser.add_argument("--video", default="video-audio.mov", help="long video inside the folder")
    parser.add_argument("--zoom", type=float, default=ZOOM, help=f"default {ZOOM}")
    parser.add_argument("--follow-x", type=float, default=FOLLOW_X, help=f"seconds, default {FOLLOW_X}")
    parser.add_argument("--follow-y", type=float, default=FOLLOW_Y, help=f"seconds, default {FOLLOW_Y}")
    parser.add_argument("--tilt", type=float, default=TILT, help=f"max degrees, default {TILT}; 0 = none")
    parser.add_argument("--no-outro", action="store_true", help="skip the logo outro")
    parser.add_argument("--out", help="output path (single clip only)")
    opts = parser.parse_args()

    video = os.path.join(opts.folder, opts.video)
    if opts.start is not None:
        if opts.duration is None:
            sys.exit("--start needs --duration")
        jobs = [(f"{int(opts.start)}s", lambda: (opts.start, opts.duration))]
    elif opts.clips:
        jobs = [(c, lambda c=c: clip_span(opts.folder, c)) for c in opts.clips]
    else:
        sys.exit("Name at least one clip (e.g. c15) or pass --start/--duration")
    if opts.out and len(jobs) > 1:
        sys.exit("--out only works with a single clip")

    if not opts.no_outro and not os.path.exists(OUTRO):
        sys.exit(
            f"Missing {OUTRO} (gitignored, so it doesn't come with a clone). Rebuild it\n"
            "from the kapza reel, whose last 2.9 s are the glitch + logo + black hole:\n"
            '  ffmpeg -ss 17.2 -i ".../el-sotano/9-kapza/reel-promo.mp4" -an '
            '-vf "fps=24000/1001,format=yuv420p" -c:v h264_videotoolbox -b:v 14M '
            "assets/outro-dnbchile.mp4\nor pass --no-outro."
        )
    src = probe(video)
    detector = load_detector()
    failed = []
    for name, span in jobs:
        out = opts.out or os.path.join(opts.folder, f"reel-{name.replace('/', '-').removesuffix('.mp4')}.mp4")
        print(f"{name}:")
        try:
            start, duration = span()
            print(f"    {start:.1f}s + {duration:.1f}s from {opts.video}")
            t, cx, top = track_dj(detector, video, start, duration)
            path = camera_path(t, cx, top, src, duration, opts)
            with tempfile.TemporaryDirectory() as tmp:
                render(video, start, duration, path, out, not opts.no_outro, tmp)
            print(f"    -> {out}")
        except Exception as e:
            print(f"    !! failed: {e}")
            failed.append(f"{name}: {e}")

    if failed:
        print(f"\nDone with {len(failed)} failure(s):")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
