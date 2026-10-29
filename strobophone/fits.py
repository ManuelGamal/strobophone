"""Deterministic model fits. No language model ever produces a number here."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class FreeFallResult:
    g: float
    ci_low: float
    ci_high: float
    n_points: int
    residual_rms_px: float
    coeffs: tuple[float, float, float] = (0.0, 0.0, 0.0)  # first arc: y(t) = c0 t^2 + c1 t + c2, px and s
    arc_coeffs: tuple[tuple[float, float, float], ...] = ()  # one (c0, c1, c2) per arc, shared c0


BOOTSTRAP_SAMPLES = 2000
MIN_POINTS = 4


def detect_release(y_px: np.ndarray, threshold_px: float = 2.0) -> int:
    """Index of the last at-rest frame before the object starts moving.

    Fitting from here is safe even if the true release was a frame or two
    earlier, because the free-fall fit has its own initial-velocity term.
    """
    y = np.asarray(y_px, float)
    moving = np.abs(y - np.median(y[:5])) > threshold_px
    if not moving.any():
        raise ValueError(f"no motion above {threshold_px} px in the track")
    return max(int(np.argmax(moving)) - 1, 0)


def trim_before_vertex(frames: np.ndarray, y: np.ndarray, max_passes: int = 8) -> np.ndarray:
    """Drop leading frames that come before the fitted parabola's vertex. For a drop
    from rest those are frames where the ball was still held: they are flat, not on
    the parabola, and bias the fit. Refits until no more frames are dropped."""
    use = np.asarray(frames)
    for _ in range(max_passes):
        c = np.polyfit(use, y[use], 2)
        if c[0] <= 0:
            break
        keep = use[use > -c[1] / (2 * c[0])]
        if len(keep) == len(use) or len(keep) < MIN_POINTS:
            break
        use = keep
    return use


def fit_free_fall(t_s: np.ndarray, y_px: np.ndarray, px_per_m: float, seed: int = 0) -> FreeFallResult:
    """Fit y = y0 + v0 t + g t^2 / 2 (image y grows downward) and bootstrap a 95% CI on g."""
    return fit_ballistic_arcs([(t_s, y_px)], px_per_m, seed)


def fit_ballistic_arcs(arcs: list[tuple[np.ndarray, np.ndarray]], px_per_m: float, seed: int = 0) -> FreeFallResult:
    """Joint fit of several ballistic arcs (the drop and each rebound between
    bounces): one shared curvature g/2, a separate velocity and offset per arc.

    The 95% CI is g +/- t(n-p) * residual-bootstrap SE, with residuals rescaled by
    sqrt(n/(n-p)) because raw residuals understate the noise; this keeps 95%
    coverage down to ~10 points (eval/ci_coverage.py)."""
    arcs = [(np.asarray(t, float), np.asarray(y, float)) for t, y in arcs]
    t = np.concatenate([a[0] for a in arcs])
    y = np.concatenate([a[1] for a in arcs])
    n, p = len(t), 1 + 2 * len(arcs)
    if n < max(MIN_POINTS, p + 1):
        raise ValueError(f"need at least {max(MIN_POINTS, p + 1)} tracked points, got {n}")
    which = np.concatenate([np.full(len(a[0]), k) for k, a in enumerate(arcs)])
    X = np.column_stack([t**2] + [col for k in range(len(arcs)) for col in (t * (which == k), (which == k).astype(float))])
    pinv = np.linalg.pinv(X)
    coef = pinv @ y
    fitted = X @ coef
    residuals = y - fitted

    rng = np.random.default_rng(seed)
    resampled = fitted + rng.choice(residuals * np.sqrt(n / (n - p)), size=(BOOTSTRAP_SAMPLES, n))
    g_samples = 2 * (resampled @ pinv[0]) / px_per_m
    g = 2 * coef[0] / px_per_m
    half = stats.t.ppf(0.975, n - p) * float(np.std(g_samples, ddof=1))
    arc_coeffs = tuple((float(coef[0]), float(coef[1 + 2 * k]), float(coef[2 + 2 * k])) for k in range(len(arcs)))
    return FreeFallResult(
        g=float(g),
        ci_low=float(g - half),
        ci_high=float(g + half),
        n_points=n,
        residual_rms_px=float(np.sqrt(np.mean(residuals**2))),
        coeffs=arc_coeffs[0],
        arc_coeffs=arc_coeffs,
    )
