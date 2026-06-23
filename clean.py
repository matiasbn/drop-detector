#!/usr/bin/env python3
"""
Clean - Empties the input/, output/ and downscaled/ folders.
"""

import os
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIRS = [
    os.path.join(BASE_DIR, "input"),
    os.path.join(BASE_DIR, "output"),
    os.path.join(BASE_DIR, "downscaled"),
]


def clean_dir(directory: str) -> int:
    """Delete every file inside a directory. Returns count removed."""
    if not os.path.isdir(directory):
        return 0

    removed = 0
    for path in glob.glob(os.path.join(directory, "*")):
        if os.path.isfile(path):
            os.remove(path)
            removed += 1

    print(f"  {os.path.basename(directory)}/: removed {removed} file(s)")
    return removed


def main():
    total = sum(clean_dir(d) for d in DIRS)
    print(f"\nDone! Removed {total} file(s).")


if __name__ == "__main__":
    main()
