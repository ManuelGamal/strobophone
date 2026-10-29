"""Video -> track -> scale -> fit. Each stage is deterministic and swappable
(classical baseline today, SAM 3.1 tracks once the API is available)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from strobophone.bounce import find_bounces
from strobophone.calibration import Scale
from strobophone.fits import MIN_POINTS, FreeFallResult, detect_release, fit_ballistic_arcs, fit_free_fall, trim_before_vertex


@dataclass(frozen=True)
class FreeFallMeasurement:
    fit: FreeFallResult
    scale: Scale
    release_frame: int
    g_ci_with_scale: tuple[float, float]
    fit_frames: np.ndarray  # all frame indices that entered the fit
    segments: list[np.ndarray]  # frame indices per ballistic arc (drop first, then rebounds)
    arc_g: list[tuple[float, float]] = field(default_factory=list)  # (g, standard error) per arc with 5+ points
    birge_ratio: float = 1.0  # spread of the per-arc g relative to their errors; ~1 when they agree
    warnings: list[str] = field(default_factory=list)


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

    # Do the arcs agree? Each arc gives its own g; if they scatter more than their
    # errors allow (chi-square test), the joint fit's interval is too optimistic:
    # widen it by the Birge ratio and say why instead of reporting false precision.
    arc_g, warnings, birge = [], [], 1.0
    for seg in segments:
        if len(seg) >= 5:
            f = fit_free_fall(seg / fps, y[seg], scale.px_per_m)
            arc_g.append((f.g, (f.ci_high - f.ci_low) / 2 / stats.t.ppf(0.975, len(seg) - 3)))
    fit_half_width = (fit.ci_high - fit.ci_low) / 2
    if len(arc_g) >= 2:
        chi2 = sum(((g - fit.g) / se) ** 2 for g, se in arc_g)
        dof = len(arc_g) - 1
        if stats.chi2.sf(chi2, dof) < 0.01:
            birge = float(np.sqrt(chi2 / dof))
            fit_half_width *= birge
            gs = [g for g, _ in arc_g]
            warnings.append(
                f"arcs disagree: per-arc g ranges {min(gs):.2f} to {max(gs):.2f} m/s^2 (Birge ratio {birge:.1f}); "
                "the interval is widened accordingly. Likely causes: camera not side-on, the ball moving "
                "toward or away from the camera, or lens distortion near the frame edges."
            )
    scale_half_width = 1.96 * scale.rel_uncertainty * fit.g
    half = float(np.hypot(fit_half_width, scale_half_width))
    return FreeFallMeasurement(fit, scale, release, (fit.g - half, fit.g + half), use, segments, arc_g, birge, warnings)
