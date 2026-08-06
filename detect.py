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


def detect_drops(audio_path: str) -> list[float]:
    """
    Analyze audio and detect high-energy moments.

    Detects absolute energy peaks, not just buildup-to-drop transitions.
    This also captures chained drops where the DJ doesn't lower the energy.

    Returns a list of timestamps (in seconds) where drops occur.
    """
    print("  Loading audio...")
    y, sr = librosa.load(audio_path, sr=22050)

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


def format_timestamp(seconds: float) -> str:
    """Format seconds as MMmSSs."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}m{s:02d}s"


def cut_clip(video_path: str, drop_time: float, output_path: str) -> None:
    """Cut a clip from the video around the drop."""
    start = max(0, drop_time - CLIP_BEFORE)
    duration = CLIP_BEFORE + CLIP_AFTER

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-c", "copy",
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
        drops = detect_drops(audio_path)
    finally:
        os.unlink(audio_path)

    if not drops:
        print("  No drops detected.")
        return

    print(f"  Detected {len(drops)} drops:")
    for i, t in enumerate(drops, 1):
        ts = format_timestamp(t)
        print(f"    Drop {i}: {ts}")

        output_name = f"{video_name}_drop_{i:02d}_{ts}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        cut_clip(video_path, t, output_path)
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
