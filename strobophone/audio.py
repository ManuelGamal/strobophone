"""Frequencies from the video's audio track. A microphone frequency is a
measurement, so this is plain signal processing, no model."""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np


def extract_audio(video: Path, rate: int = 24000) -> np.ndarray:
    """Mono float32 samples in [-1, 1] from a video's audio track."""
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(rate),
        "-f",
        "s16le",
        "-",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def dominant_frequency(
    samples: np.ndarray, rate: float, fmin: float, fmax: float
) -> float:
    """Strongest frequency in [fmin, fmax], refined between FFT bins by parabolic
    interpolation of the log magnitude (Hann window, 8x zero padding)."""
    x = np.asarray(samples, float)
    x = (x - x.mean()) * np.hanning(len(x))
    n = 8 * len(x)
    spectrum = np.abs(np.fft.rfft(x, n))
    freqs = np.fft.rfftfreq(n, 1 / rate)
    band = np.nonzero((freqs >= fmin) & (freqs <= fmax))[0]
    k = band[np.argmax(spectrum[band])]
    a, b, c = np.log(spectrum[k - 1 : k + 2] + 1e-12)
    shift = 0.5 * (a - c) / (a - 2 * b + c)
    return float(freqs[k] + shift * rate / n)
