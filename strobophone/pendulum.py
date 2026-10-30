"""Simple pendulum: g from the period, with no pixel scale.

The bob moves on a circle around the pivot, so a circle fit to its image track
gives the swing angle theta(t) directly. Each full swing gives a period and an
amplitude; the exact large-amplitude period T = (2/pi) K(sin^2(theta0/2)) T0
is inverted for the small-angle period T0, and g = 4 pi^2 L / T0^2. Only the
rope length L (pivot to bob centre) and the frame rate enter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats


@dataclass(frozen=True)
class PendulumResult:
    g: float
    ci: tuple[float, float]
    n_cycles: int
    periods: list[float]  # s, one per full swing
    amplitudes: list[float]  # rad
    g_small_angle: float  # what 4 pi^2 L / T^2 gives without the amplitude correction
    pivot_px: tuple[float, float]
    length_px: float


def fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """(cx, cy, r): algebraic fit refined by geometric least squares (an arc of a
    few tens of degrees biases the algebraic fit alone)."""
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = np.asarray(x, float)[ok], np.asarray(y, float)[ok]
    A = np.column_stack([x, y, np.ones_like(x)])
    D, E, F = np.linalg.lstsq(A, -(x**2 + y**2), rcond=None)[0]
    cx, cy = -D / 2, -E / 2
    r = np.sqrt(cx**2 + cy**2 - F)
    sol = optimize.least_squares(
        lambda p: np.hypot(x - p[0], y - p[1]) - p[2], [cx, cy, r]
    )
    return tuple(float(v) for v in sol.x)


def swings(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> list[tuple[float, float]]:
    """(period s, amplitude rad) for each full swing, from upward zero crossings of
    theta(t), interpolated between frames."""
    cx, cy, _ = fit_circle(x, y)
    ok = ~(np.isnan(x) | np.isnan(y))
    t, th = (
        np.asarray(t, float)[ok],
        np.arctan2(np.asarray(x)[ok] - cx, np.asarray(y)[ok] - cy),
    )
    up = np.nonzero((th[:-1] < 0) & (th[1:] >= 0))[0]
    crossings = t[up] - th[up] * (t[up + 1] - t[up]) / (th[up + 1] - th[up])
    out = []
    for a, b in zip(crossings, crossings[1:]):
        cycle = th[(t >= a) & (t <= b)]
        if len(cycle) >= 8:
            out.append((float(b - a), float((cycle.max() - cycle.min()) / 2)))
    return out


def measure_pendulum(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, length_m: float
) -> PendulumResult:
    s = swings(t, x, y)
    if len(s) < 3:
        raise ValueError(f"need at least 3 full swings, got {len(s)}")
    T = np.array([p for p, _ in s])
    amp = np.array([a for _, a in s])
    T0 = T * np.pi / (2 * special.ellipk(np.sin(amp / 2) ** 2))
    g_i = 4 * np.pi**2 * length_m / T0**2
    g = float(np.mean(g_i))
    half = (
        stats.t.ppf(0.975, len(g_i) - 1)
        * float(np.std(g_i, ddof=1))
        / np.sqrt(len(g_i))
    )
    cx, cy, r = fit_circle(x, y)
    return PendulumResult(
        g=g,
        ci=(g - half, g + half),
        n_cycles=len(s),
        periods=T.tolist(),
        amplitudes=amp.tolist(),
        g_small_angle=float(4 * np.pi**2 * length_m / np.mean(T) ** 2),
        pivot_px=(cx, cy),
        length_px=r,
    )
