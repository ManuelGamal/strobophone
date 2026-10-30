"""Synthetic experiment videos with exactly known answers, for tests and sanity checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import numpy as np

A4_M = (0.210, 0.297)
BALL_RGB = (240, 110, 20)


@dataclass(frozen=True)
class FreeFallTruth:
    fps: int
    px_per_m: float
    g: float
    rest_frames: int
    ball_x_px: float
    ball_y_px: np.ndarray  # true centre per frame
    a4_box_px: tuple[float, float, float, float]  # left, top, width, height


def make_free_fall_video(
    path: Path,
    *,
    fps: int = 120,
    px_per_m: float = 300.0,
    g: float = 9.81,
    rest_frames: int = 15,
    fall_s: float = 0.5,
    size: tuple[int, int] = (480, 520),
    ball_radius_m: float = 0.04,
    y0_px: float = 60.0,
) -> FreeFallTruth:
    """Render an orange ball falling past an A4 sheet on a textured wall."""
    width, height = size
    t = np.arange(int(fall_s * fps) + 1) / fps
    ys = np.concatenate([np.full(rest_frames, y0_px), y0_px + 0.5 * g * t**2 * px_per_m])
    cx, r = width * 0.6, ball_radius_m * px_per_m
    yy, xx = np.mgrid[0:height, 0:width]

    wall = np.empty((height, width, 3), np.float32)
    wall[:] = (205, 200, 190)
    wall += np.random.default_rng(0).normal(0, 4, wall.shape)
    a4_w, a4_h = A4_M[0] * px_per_m, A4_M[1] * px_per_m
    ax, ay = round(width * 0.15), round(height * 0.35)
    a4_w_px, a4_h_px = round(a4_w), round(a4_h)
    wall[ay : ay + a4_h_px, ax : ax + a4_w_px] = (250, 250, 248)

    writer = imageio_ffmpeg.write_frames(
        str(path), (width, height), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
        quality=8, macro_block_size=8,
    )
    writer.send(None)
    for y in ys:
        d = np.sqrt((xx - cx) ** 2 + (yy - y) ** 2)
        alpha = np.clip(r - d + 0.5, 0, 1)[..., None]  # antialiased edge
        frame = wall * (1 - alpha) + np.array(BALL_RGB, np.float32) * alpha
        writer.send(np.clip(frame, 0, 255).astype(np.uint8).tobytes())
    writer.close()
    return FreeFallTruth(fps, px_per_m, g, rest_frames, cx, ys, (ax, ay, a4_w_px, a4_h_px))


@dataclass(frozen=True)
class PendulumTruth:
    fps: int
    g: float
    length_m: float  # pivot to bob centre
    pivot_px: tuple[float, float]
    theta: np.ndarray  # rad, per frame


def make_pendulum_video(
    path: Path,
    *,
    theta0_deg: float = 45,
    fps: int = 60,
    seconds: float = 8,
    length_m: float = 0.5,
    g: float = 9.81,
    damping: float = 0.02,
    px_per_m: float = 600.0,
    bob_radius_m: float = 0.025,
    size: tuple[int, int] = (720, 400),
) -> PendulumTruth:
    """Render an orange bob on a dark string swinging as a damped nonlinear pendulum."""
    from scipy.integrate import solve_ivp

    width, height = size
    t = np.arange(int(seconds * fps)) / fps
    sol = solve_ivp(
        lambda _, s: [s[1], -(g / length_m) * np.sin(s[0]) - damping * s[1]],
        (0, t[-1]), [np.radians(theta0_deg), 0.0], t_eval=t, rtol=1e-10, atol=1e-12,
    )
    theta = sol.y[0]
    px, py, L, r = width / 2, 40.0, length_m * px_per_m, bob_radius_m * px_per_m
    yy, xx = np.mgrid[0:height, 0:width]
    wall = np.empty((height, width, 3), np.float32)
    wall[:] = (205, 200, 190)
    wall += np.random.default_rng(0).normal(0, 4, wall.shape)

    writer = imageio_ffmpeg.write_frames(
        str(path), (width, height), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
        quality=8, macro_block_size=8,
    )
    writer.send(None)
    for th in theta:
        bx, by = px + L * np.sin(th), py + L * np.cos(th)
        # distance from each pixel to the string segment (pivot -> bob)
        u = np.clip(((xx - px) * (bx - px) + (yy - py) * (by - py)) / L**2, 0, 1)
        string = np.clip(1.5 - np.hypot(xx - px - u * (bx - px), yy - py - u * (by - py)), 0, 1)[..., None]
        bob = np.clip(r - np.hypot(xx - bx, yy - by) + 0.5, 0, 1)[..., None]
        frame = wall * (1 - string) + 40 * string
        frame = frame * (1 - bob) + np.array(BALL_RGB, np.float32) * bob
        writer.send(np.clip(frame, 0, 255).astype(np.uint8).tobytes())
    writer.close()
    return PendulumTruth(fps, g, length_m, (px, py), theta)
