"""Deterministic model fits. No language model ever produces a number here."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FreeFallResult:
    g: float
    ci_low: float
    ci_high: float
    n_points: int
    residual_rms_px: float
    coeffs: tuple[float, float, float] = (0.0, 0.0, 0.0)  # y(t) = c0 t^2 + c1 t + c2, px and s


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


def fit_free_fall(t_s: np.ndarray, y_px: np.ndarray, px_per_m: float, seed: int = 0) -> FreeFallResult:
    """Fit y = y0 + v0 t + g t^2 / 2 (image y grows downward) and bootstrap a 95% CI on g."""
    t, y = np.asarray(t_s, float), np.asarray(y_px, float)
    if len(t) < MIN_POINTS:
        raise ValueError(f"need at least {MIN_POINTS} tracked points, got {len(t)}")
    coeffs = np.polyfit(t, y, 2)
    fitted = np.polyval(coeffs, t)
    residuals = y - fitted

    # Residual bootstrap: refit on fitted + resampled residuals.
    rng = np.random.default_rng(seed)
    resampled = fitted + rng.choice(residuals, size=(BOOTSTRAP_SAMPLES, len(t)))
    a = np.polyfit(t, resampled.T, 2)[0]
    g_samples = 2 * a / px_per_m
    return FreeFallResult(
        g=2 * coeffs[0] / px_per_m,
        ci_low=float(np.percentile(g_samples, 2.5)),
        ci_high=float(np.percentile(g_samples, 97.5)),
        n_points=len(t),
        residual_rms_px=float(np.sqrt(np.mean(residuals**2))),
        coeffs=tuple(float(c) for c in coeffs),
    )
