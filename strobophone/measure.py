"""One call from a raw free-fall clip to g: prepare the clip (field order, crop +
zoom), track with fallback concept phrases, map the track back to the original
pixels, fit. The web app and the CLI both use this."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from strobophone.baseline import read_video
from strobophone.bounce import RestitutionResult, measure_restitution
from strobophone.calibration import Scale
from strobophone.pipeline import FreeFallMeasurement, measure_free_fall
from strobophone.prep import Prep, prepare, track_with_fallbacks

CONCEPTS = ("ball", "white ball", "sphere", "round object", "circle")
PENDULUM_CONCEPTS = ("pendulum bob", "ball", "weight", "sphere", "round object")


@dataclass(frozen=True)
class VideoMeasurement:
    prep: Prep
    clip: Path  # the prepared clip SAM saw
    fps: float  # of the prepared clip (doubled when split into fields)
    concept: str
    attempts: list[tuple[str, str]]
    object_id: str
    boxes: np.ndarray  # (n, 4) l, t, r, b in original pixels, NaN where untracked
    y_px: np.ndarray  # box centre y in original pixels
    result: FreeFallMeasurement
    restitution: RestitutionResult | None  # first bounce, when the clip shows one


def _default_tracker(video, concept, n_frames):
    from strobophone import sam  # needs the API key; imported only when used

    return sam.track_boxes(video, concept, n_frames)


@dataclass(frozen=True)
class Track:
    prep: Prep
    clip: Path
    fps: float
    concept: str
    attempts: list[tuple[str, str]]
    object_id: str
    boxes: np.ndarray  # (n, 4) l, t, r, b in original pixels


def vertical_span(boxes: np.ndarray) -> float:
    return float(np.nanmax(boxes[:, 3]) - np.nanmin(boxes[:, 3]))


def horizontal_span(boxes: np.ndarray) -> float:
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    return float(np.nanmax(cx) - np.nanmin(cx))


def track_video(
    video,
    workdir,
    tracker=_default_tracker,
    concepts=CONCEPTS,
    log=lambda line: None,
    pick=vertical_span,
    fps_override: float | None = None,
) -> Track:
    """Prepare the clip, track with fallback concepts, and return the object that
    maximises `pick` (vertical span for a drop, horizontal for a pendulum), with
    its boxes mapped back to the original pixels."""
    log("checking field order and object size")
    prep, clip, fps = prepare(video, workdir)
    log(f"field order: {prep.parity or 'progressive'} -> {fps:g} fps")
    if fps_override:
        per_frame = 2 if prep.parity else 1  # deinterlacing doubled the file's rate
        log(f"capture rate set to {fps_override:g} fps (file says {fps / per_frame:g})")
        fps = fps_override * per_frame
    log(f"crop {prep.crop}, zoom {prep.zoom:g}x" if prep.crop else "no crop/zoom needed")
    n_frames = len(read_video(clip)[0])

    def logged_tracker(video, concept, n):
        log(f"asking SAM 3.1 for {concept!r}")
        return tracker(video, concept, n)

    concept, tracks, attempts = track_with_fallbacks(
        clip, concepts, n_frames, logged_tracker, on_attempt=lambda c, r: log(f"{c!r}: {r}")
    )
    if concept is None:
        raise ValueError("no moving object found: " + "; ".join(f"{c!r}: {r}" for c, r in attempts))
    oid, boxes = max(tracks.items(), key=lambda kv: pick(kv[1]))
    return Track(prep, clip, fps, concept, attempts, oid, prep.to_original(boxes))


def measure_video(
    video,
    workdir,
    scale: Scale,
    tracker=_default_tracker,
    concepts=CONCEPTS,
    log=lambda line: None,
    fps_override: float | None = None,
) -> VideoMeasurement:
    """g from a free-fall clip. `scale` is in pixels of the original video.
    `log` receives one human-readable line per step (shown live in the web app).
    `fps_override` is the true capture rate when the file's metadata is wrong
    (slow-motion exports often say 30 fps)."""
    tr = track_video(video, workdir, tracker, concepts, log, fps_override=fps_override)
    prep, clip, fps, concept, attempts, oid, boxes = (
        tr.prep, tr.clip, tr.fps, tr.concept, tr.attempts, tr.object_id, tr.boxes
    )
    y = (boxes[:, 1] + boxes[:, 3]) / 2
    result = measure_free_fall(y, fps, scale)
    lo, hi = result.g_ci_with_scale
    for w in result.warnings:
        log("warning: " + w)
    try:
        restitution = measure_restitution(y)
        log(f"restitution e = {restitution.e:.3f}, 95% CI [{restitution.ci[0]:.3f}, {restitution.ci[1]:.3f}]")
    except ValueError:
        restitution = None
    log(f"g = {result.fit.g:.3f} m/s^2, 95% CI [{lo:.3f}, {hi:.3f}] from {result.fit.n_points} points")
    return VideoMeasurement(
        prep,
        clip,
        fps,
        concept,
        attempts,
        oid,
        boxes,
        y,
        result,
        restitution,
    )
