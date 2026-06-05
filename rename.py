#!/usr/bin/env python3
"""
Rename - Renames all video clips in output/ sequentially as c1.mp4, c2.mp4, etc.
"""

import os
import re
import sys
import glob

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def natural_sort_key(path: str):
    """Sort filenames naturally (c2 before c10)."""
    name = os.path.basename(path)
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", name)]


def main():
    extensions = ("*.mp4", "*.mov", "*.mkv", "*.avi")
    videos = sorted(
        (v for ext in extensions for v in glob.glob(os.path.join(OUTPUT_DIR, ext))),
        key=natural_sort_key,
    )

    if not videos:
        print(f"No videos found in {OUTPUT_DIR}/")
        sys.exit(1)

    # Rename to temp files first to avoid collisions
    temp_paths = []
    for i, video_path in enumerate(videos, 1):
        ext = os.path.splitext(video_path)[1]
        temp_path = os.path.join(OUTPUT_DIR, f"_temp_c{i}{ext}")
        os.rename(video_path, temp_path)
        temp_paths.append(temp_path)

    # Rename temp files to final names
    for i, temp_path in enumerate(temp_paths, 1):
        ext = os.path.splitext(temp_path)[1]
        final_path = os.path.join(OUTPUT_DIR, f"c{i}{ext}")
        os.rename(temp_path, final_path)
        print(f"  c{i}{ext}")

    print(f"\nDone! Renamed {len(temp_paths)} files.")


if __name__ == "__main__":
    main()
