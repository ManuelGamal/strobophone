"""Video -> track -> scale -> fit. Each stage is deterministic and swappable
(classical baseline today, SAM 3.1 tracks once the API is available)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from strobophone.bounce import find_bounces
from strobophone.calibration import Scale
from strobophone.fits import MIN_POINTS, FreeFallResult, detect_release, fit_ballistic_arcs, trim_before_vertex


@dataclass(frozen=True)
class FreeFallMeasurement:
    fit: FreeFallResult
    scale: Scale
    release_frame: int
    g_ci_with_scale: tuple[float, float]
    fit_frames: np.ndarray  # all frame indices that entered the fit
    segments: list[np.ndarray]  # frame indices per ballistic arc (drop first, then rebounds)


def measure_free_fall(y_px: np.ndarray, fps: float, scale: Scale) -> FreeFallMeasurement:
    """g from a per-frame vertical track (NaN = not tracked in that frame).

    The reported CI combines the fit's bootstrap CI with the scale uncertainty
    (treated as one standard uncertainty), added in quadrature.
    """
    y = np.asarray(y_px, float)
    frames = np.nonzero(~np.isnan(y))[0]
    release = int(frames[detect_release(y[frames])])
    bounces = find_bounces(y)
    # The drop, from release to just before the first bounce (the lowest frame may
    # already be after impact), then every rebound arc closed by a later bounce.
    drop = frames[(frames >= release) & (frames < (bounces[0] if bounces else len(y)))]
    if len(drop) > MIN_POINTS:
        drop = trim_before_vertex(drop, y)
    segments = [drop] if len(drop) >= MIN_POINTS else []
    for b0, b1 in zip(bounces, bounces[1:]):
        arc = frames[(frames > b0) & (frames < b1)]
        if len(arc) >= MIN_POINTS:
            segments.append(arc)
    if not segments:
        raise ValueError(f"no ballistic stretch with {MIN_POINTS}+ tracked points")
    fit = fit_ballistic_arcs([(s / fps, y[s]) for s in segments], scale.px_per_m)
    release = int(segments[0][0])
    use = np.concatenate(segments)

    fit_half_width = (fit.ci_high - fit.ci_low) / 2
    scale_half_width = 1.96 * scale.rel_uncertainty * fit.g
    half = float(np.hypot(fit_half_width, scale_half_width))
    return FreeFallMeasurement(fit, scale, release, (fit.g - half, fit.g + half), use, segments)
