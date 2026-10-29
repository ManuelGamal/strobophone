"""One call from a raw free-fall clip to g: prepare the clip (field order, crop +
zoom), track with fallback concept phrases, map the track back to the original
pixels, fit. The web app and the CLI both use this."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from strobophone.baseline import read_video
from strobophone.calibration import Scale
from strobophone.pipeline import FreeFallMeasurement, measure_free_fall
from strobophone.prep import Prep, prepare, track_with_fallbacks

CONCEPTS = ("ball", "white ball", "sphere", "round object", "circle")


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


def _default_tracker(video, concept, n_frames):
    from strobophone import sam  # needs the API key; imported only when used

    return sam.track_boxes(video, concept, n_frames)


def measure_video(
    video, workdir, scale: Scale, tracker=_default_tracker, concepts=CONCEPTS
) -> VideoMeasurement:
    """g from a free-fall clip. `scale` is in pixels of the original video."""
    prep, clip, fps = prepare(video, workdir)
    n_frames = len(read_video(clip)[0])
    concept, tracks, attempts = track_with_fallbacks(clip, concepts, n_frames, tracker)
    if concept is None:
        raise ValueError(
            "no falling object found: " + "; ".join(f"{c!r}: {r}" for c, r in attempts)
        )
    # The falling object is the one that moves furthest vertically.
    oid, boxes = max(
        tracks.items(), key=lambda kv: np.nanmax(kv[1][:, 3]) - np.nanmin(kv[1][:, 3])
    )
    boxes = prep.to_original(boxes)
    y = (boxes[:, 1] + boxes[:, 3]) / 2
    return VideoMeasurement(
        prep,
        clip,
        fps,
        concept,
        attempts,
        oid,
        boxes,
        y,
        measure_free_fall(y, fps, scale),
    )
