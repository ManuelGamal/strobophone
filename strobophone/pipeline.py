"""Video -> track -> scale -> fit. Each stage is deterministic and swappable
(classical baseline today, SAM 3.1 tracks once the API is available)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from strobophone.bounce import find_bounces
from strobophone.calibration import Scale
from strobophone.fits import FreeFallResult, detect_release, fit_free_fall


@dataclass(frozen=True)
class FreeFallMeasurement:
    fit: FreeFallResult
    scale: Scale
    release_frame: int
    g_ci_with_scale: tuple[float, float]


def measure_free_fall(y_px: np.ndarray, fps: float, scale: Scale) -> FreeFallMeasurement:
    """g from a per-frame vertical track (NaN = not tracked in that frame).

    The reported CI combines the fit's bootstrap CI with the scale uncertainty
    (treated as one standard uncertainty), added in quadrature.
    """
    y = np.asarray(y_px, float)
    frames = np.nonzero(~np.isnan(y))[0]
    release = int(frames[detect_release(y[frames])])
    use = frames[frames >= release]
    bounces = find_bounces(y)
    if bounces:  # the lowest frame may already be after impact, so drop it too
        use = use[use < bounces[0]]
    fit = fit_free_fall((use - release) / fps, y[use], scale.px_per_m)

    fit_half_width = (fit.ci_high - fit.ci_low) / 2
    scale_half_width = 1.96 * scale.rel_uncertainty * fit.g
    half = float(np.hypot(fit_half_width, scale_half_width))
    return FreeFallMeasurement(fit, scale, release, (fit.g - half, fit.g + half))
