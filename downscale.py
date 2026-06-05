#!/usr/bin/env python3
"""
Downscale - Converts clips from output/ to 1080p for Instagram.
Results are saved in downscaled/
"""

import os
import sys
import glob
import subprocess

INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downscaled")


def downscale(video_path: str, output_path: str) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", "scale=-2:1080",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    extensions = ("*.mp4", "*.mov", "*.mkv", "*.avi")
    videos = sorted(
        v for ext in extensions for v in glob.glob(os.path.join(INPUT_DIR, ext))
    )

    if not videos:
        print(f"No videos found in {INPUT_DIR}/")
        sys.exit(1)

    print(f"Downscaling {len(videos)} video(s) to 1080p...")
    for i, video_path in enumerate(videos, 1):
        name = os.path.basename(video_path)
        print(f"  [{i}/{len(videos)}] {name}")
        output_path = os.path.join(OUTPUT_DIR, name)
        downscale(video_path, output_path)

    print(f"\nDone! Videos are in downscaled/")


if __name__ == "__main__":
    main()
