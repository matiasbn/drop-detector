#!/usr/bin/env python3
"""
Drop Detector - Detecta los mejores momentos (drops) en DJ sets
y genera clips de video recortados.

Uso:
    1. Colocar videos MP4 en la carpeta input/
    2. Ejecutar: python detect.py
    3. Los clips se generan en output/
"""

import os
import sys
import glob
import tempfile
import subprocess

import numpy as np
import librosa

# --- Parámetros configurables ---
CLIP_BEFORE = 2        # Segundos antes del drop a incluir en el clip
CLIP_AFTER = 28        # Segundos después del drop a incluir en el clip
MIN_DROP_DISTANCE = 30 # Distancia mínima entre drops (segundos)
ENERGY_THRESHOLD = 85  # Percentil para considerar un pico de energía como drop

INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def extract_audio(video_path: str, audio_path: str) -> None:
    """Extrae el audio de un video MP4 a WAV mono 22050Hz."""
    print(f"  Extrayendo audio de {os.path.basename(video_path)}...")
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
    Analiza el audio y detecta los momentos de mayor energía.

    Detecta picos de energía absoluta, no solo transiciones buildup→drop.
    Esto captura también drops encadenados donde el DJ no baja la energía.

    Retorna una lista de timestamps (en segundos) donde ocurren los drops.
    """
    print("  Cargando audio...")
    y, sr = librosa.load(audio_path, sr=22050)

    print("  Analizando energía...")
    hop_length = 2048
    rms = librosa.feature.rms(y=y, frame_length=4096, hop_length=hop_length)[0]

    # Suavizar la curva de energía (~1 segundo de ventana)
    kernel_size = 11
    kernel = np.ones(kernel_size) / kernel_size
    rms_smooth = np.convolve(rms, kernel, mode="same")

    # Umbral: frames con energía por encima del percentil configurado
    threshold = np.percentile(rms_smooth, ENERGY_THRESHOLD)
    high_energy_frames = np.where(rms_smooth > threshold)[0]

    if len(high_energy_frames) == 0:
        return []

    # Convertir frames a segundos
    times = librosa.frames_to_time(high_energy_frames, sr=sr, hop_length=hop_length)

    # Agrupar detecciones cercanas y quedarse con el pico de mayor energía
    drops = []
    group_start = 0
    for i in range(1, len(times)):
        if times[i] - times[group_start] > MIN_DROP_DISTANCE:
            group_frames = high_energy_frames[group_start:i]
            best_idx = group_start + np.argmax(rms_smooth[group_frames])
            drops.append(times[best_idx])
            group_start = i

    # Último grupo
    group_frames = high_energy_frames[group_start:]
    best_idx = group_start + np.argmax(rms_smooth[group_frames])
    drops.append(times[best_idx])

    return drops


def format_timestamp(seconds: float) -> str:
    """Formatea segundos como MMmSSs."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}m{s:02d}s"


def cut_clip(video_path: str, drop_time: float, output_path: str) -> None:
    """Recorta un clip del video alrededor del drop."""
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
    """Procesa un video: detecta drops y genera clips."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\nProcesando: {video_name}")

    # Extraer audio a archivo temporal
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        extract_audio(video_path, audio_path)
        drops = detect_drops(audio_path)
    finally:
        os.unlink(audio_path)

    if not drops:
        print("  No se detectaron drops.")
        return

    print(f"  Se detectaron {len(drops)} drops:")
    for i, t in enumerate(drops, 1):
        ts = format_timestamp(t)
        print(f"    Drop {i}: {ts}")

        output_name = f"{video_name}_drop_{i:02d}_{ts}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        cut_clip(video_path, t, output_path)
        print(f"    → {output_name}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_extensions = ("*.mp4", "*.mov", "*.mkv", "*.avi")
    videos = sorted(
        v for ext in video_extensions for v in glob.glob(os.path.join(INPUT_DIR, ext))
    )
    if not videos:
        print(f"No se encontraron videos en {INPUT_DIR}/")
        print("Coloca tus videos de DJ sets ahí y vuelve a ejecutar.")
        sys.exit(1)

    print(f"Encontrados {len(videos)} video(s) en {INPUT_DIR}/")

    for video_path in videos:
        process_video(video_path)

    print("\nListo! Los clips están en output/")


if __name__ == "__main__":
    main()
