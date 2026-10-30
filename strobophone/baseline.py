"""Classical (non-AI) tracker: colour threshold + centroid. The OpenCV-style baseline
that isolates what SAM 3.1 adds."""

from __future__ import annotations

from pathlib import Path

import imageio_ffmpeg
import numpy as np

MIN_PIXELS = 5


def read_video(path: Path) -> tuple[np.ndarray, float]:
    """All frames as an (N, H, W, 3) uint8 RGB array, plus the container fps."""
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    width, height = meta["size"]
    frames = [np.frombuffer(buf, np.uint8).reshape(height, width, 3) for buf in reader]
    return np.stack(frames), float(meta["fps"])


def iter_frames(path: Path):
    """(fps, generator of (H, W, 3) uint8 RGB frames): one frame in memory at a time."""
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    width, height = meta["size"]
    return float(meta["fps"]), (np.frombuffer(buf, np.uint8).reshape(height, width, 3) for buf in reader)


def first_frame(path: Path) -> np.ndarray:
    _, frames = iter_frames(path)
    frame = next(frames).copy()
    frames.close()
    return frame


def track_colour(frames: np.ndarray, rgb: tuple[int, int, int], tol: float = 60.0) -> np.ndarray:
    """(N, 2) array of (x, y) centroids of pixels near `rgb`; NaN where the object is absent.

    Pixels are weighted by closeness to the target colour, so antialiased and
    chroma-subsampled edges pull the centroid symmetrically (sub-pixel accuracy).
    """
    out = np.full((len(frames), 2), np.nan)
    target = np.asarray(rgb, np.float32)
    for i, frame in enumerate(frames):
        dist = np.linalg.norm(frame.astype(np.float32) - target, axis=-1)
        weight = np.clip(1 - dist / tol, 0, None)
        if np.count_nonzero(weight) < MIN_PIXELS:
            continue
        ys, xs = np.nonzero(weight)
        w = weight[ys, xs]
        out[i] = (np.average(xs, weights=w), np.average(ys, weights=w))
    return out


def find_white_sheet(frame: np.ndarray, min_level: int = 238) -> tuple[float, float, float, float]:
    """Bounding box (left, top, width, height) of the near-white sheet in a frame."""
    white = (frame >= min_level).all(axis=-1)
    rows, cols = white.sum(axis=1), white.sum(axis=0)
    if rows.max() == 0:
        raise ValueError("no white sheet found")
    # Keep rows/columns that are at least half as white as the fullest one, so
    # stray bright pixels elsewhere do not stretch the box.
    r = np.nonzero(rows >= rows.max() / 2)[0]
    c = np.nonzero(cols >= cols.max() / 2)[0]
    return float(c[0]), float(r[0]), float(c[-1] - c[0] + 1), float(r[-1] - r[0] + 1)
