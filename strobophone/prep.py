"""Make a real clip trackable before SAM sees it: fallback concept phrases, a
motion-based crop + zoom for small objects, and the right field order for
interlaced footage. Every step is deterministic and recorded."""

from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from scipy import ndimage

from strobophone.baseline import iter_frames

DIFF_THRESHOLD = 25  # grey levels; above codec noise, below a moving object's contrast


@dataclass(frozen=True)
class Prep:
    crop: tuple[int, int, int, int] | None  # x, y, w, h in original pixels
    zoom: float
    parity: str | None  # "tff" / "bff" when deinterlaced into fields

    def to_original(self, boxes: np.ndarray) -> np.ndarray:
        """Map (left, top, right, bottom) boxes from the prepared clip back to the original."""
        # ffmpeg's scaler aligns pixel edges: a pixel-centre coordinate x in the zoomed
        # clip is (x + 0.5) / zoom - 0.5 in the cropped original.
        b = (np.asarray(boxes, float) + 0.5) / self.zoom - 0.5
        if self.crop:
            b = b + np.array([self.crop[0], self.crop[1], self.crop[0], self.crop[1]])
        return b


@dataclass(frozen=True)
class Motion:
    """What moves in a clip, gathered frame by frame (memory stays O(one frame))."""

    widths: list[int]  # per frame pair: width of the smallest real mover (px)
    moved: np.ndarray  # (H, W) bool: union of all frame-to-frame changes
    centroid_y: np.ndarray  # per frame pair: mean row of the changed pixels
    n_frames: int


def _smallest_mover(mask: np.ndarray) -> int | None:
    lab, _ = ndimage.label(mask)
    blobs = [
        (sl[1].stop - sl[1].start, np.count_nonzero(lab[sl] == i + 1))
        for i, sl in enumerate(ndimage.find_objects(lab))
    ]
    # Codec flicker makes specks of a few pixels; keep blobs that are at least
    # a tenth of the frame's largest mover so they don't pass for a tiny object.
    largest = max((area for _, area in blobs), default=0)
    sizes = [w for w, area in blobs if area >= max(4, 0.1 * largest)]
    return min(sizes) if sizes else None


def scan_motion(frames) -> Motion:
    """`frames`: an array or any iterable of (H, W, 3) frames."""
    prev, moved, widths, cy, n = None, None, [], [], 0
    for frame in frames:
        grey = frame.mean(axis=-1, dtype=np.float32)
        n += 1
        if prev is not None:
            mask = np.abs(grey - prev) > DIFF_THRESHOLD
            moved = mask if moved is None else moved | mask
            ys = np.nonzero(mask)[0]
            cy.append(ys.mean() if len(ys) else np.nan)
            w = _smallest_mover(mask)
            if w is not None:
                widths.append(w)
        prev = grey
    if moved is None:
        raise ValueError("the clip has fewer than two frames")
    return Motion(widths, moved, np.array(cy), n)


def _motion(frames) -> Motion:
    return frames if isinstance(frames, Motion) else scan_motion(frames)


def motion_roi(frames, pad_px: int = 20) -> tuple[int, int, int, int]:
    """Bounding box (x, y, w, h) of everything that moves during the clip, padded."""
    union = _motion(frames).moved
    moved = ndimage.binary_opening(union, iterations=1)
    ys, xs = np.nonzero(moved)
    if not len(xs):
        raise ValueError("nothing moves in the clip")
    height, width = union.shape
    # Even origin and size: ffmpeg rounds crops down to even pixels for 4:2:0 chroma,
    # and the box mapping back to the original must use the crop it really applied.
    x0, y0 = max(xs.min() - pad_px, 0) // 2 * 2, max(ys.min() - pad_px, 0) // 2 * 2
    x1, y1 = min(xs.max() + pad_px, width - 1), min(ys.max() + pad_px, height - 1)
    w, h = (x1 - x0 + 2) // 2 * 2, (y1 - y0 + 2) // 2 * 2
    w, h = min(w, (width - x0) // 2 * 2), min(h, (height - y0) // 2 * 2)
    return int(x0), int(y0), int(w), int(h)


def plan_zoom(
    frames, target_px: float = 32, max_zoom: float = 6, min_px: float = 16, max_side_px: float = 3840
) -> Prep:
    """Crop to the moving region and zoom so the smallest moving object is about
    `target_px` wide, but only when it is under `min_px`: SAM misses objects of a
    few pixels, while zooming objects it already sees costs accuracy and upload size.
    The zoomed clip's long side stays within `max_side_px` (x264 fails on huge frames)."""
    motion = _motion(frames)
    widths = motion.widths
    if not widths or np.median(widths) >= min_px:
        return Prep(None, 1.0, None)
    crop = motion_roi(motion)
    zoom = min(max_zoom, math.ceil(target_px / float(np.median(widths))), math.floor(max_side_px / max(crop[2:])))
    if zoom <= 1:
        return Prep(None, 1.0, None)
    return Prep(crop, float(zoom), None)


def parse_idet(log: str) -> dict[str, int]:
    """Field-order counts from ffmpeg's idet filter (multi-frame detection line)."""
    found = re.findall(
        r"Multi frame detection: TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*Progressive:\s*(\d+)",
        log,
    )
    if not found:
        raise ValueError("no idet multi-frame summary in the log")
    tff, bff, progressive = found[-1]  # ffmpeg prints an empty probe-pass summary first
    return {"tff": int(tff), "bff": int(bff), "progressive": int(progressive)}


def zigzag_score(y: np.ndarray) -> float:
    """Mean |third difference| of a track: ~0 for smooth (parabolic) motion, large
    when consecutive samples are out of order (wrong field order)."""
    v = np.asarray(y, float)
    v = v[~np.isnan(v)]
    return float(np.mean(np.abs(np.diff(v, 3)))) if len(v) > 3 else float("inf")


def _span(b: np.ndarray) -> float:
    """How far a tracked box moves: the larger of its centre's x and y ranges."""
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    return float(max(np.nanmax(cx) - np.nanmin(cx), np.nanmax(cy) - np.nanmin(cy)))


def track_with_fallbacks(video, concepts, n_frames, tracker, min_span_px: float = 30, on_attempt=None):
    """Try concept phrases in order until the tracker returns an object that moves
    at least `min_span_px` (horizontally or vertically). Returns (concept, boxes, attempts);
    `on_attempt(concept, outcome)` is called as each attempt finishes."""
    attempts = []

    def record(concept, outcome):
        attempts.append((concept, outcome))
        if on_attempt:
            on_attempt(concept, outcome)

    for concept in concepts:
        boxes = tracker(video, concept, n_frames)
        if not boxes:
            record(concept, "nothing found")
            continue
        if any(_span(b) >= min_span_px for b in boxes.values() if np.any(~np.isnan(b[:, 3]))):
            record(concept, "ok")
            return concept, boxes, attempts
        record(concept, "no moving object")
    return None, {}, attempts


def _ffmpeg(*args: str) -> str:
    """Run ffmpeg and return its stderr (where filters such as idet report)."""
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-nostats", "-y", *args]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stderr


def _filter(src: Path, dst: Path, filters: list[str]) -> Path:
    _ffmpeg(
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        ",".join(filters),
        "-c:v",
        "libx264",
        "-crf",
        "12",
        str(dst),
    )
    return dst


def is_interlaced(video: Path) -> bool:
    counts = parse_idet(
        _ffmpeg("-loglevel", "info", "-i", str(video), "-vf", "idet", "-f", "null", "-")
    )
    return counts["tff"] + counts["bff"] > counts["progressive"]


def prepare(video, workdir) -> tuple[Prep, Path, float]:
    """Deinterlace with the field order that gives a smooth track, then crop and
    zoom around the motion if the object is small. Returns (prep, clip, fps)."""
    video, workdir = Path(video), Path(workdir)
    parity, clip = None, video
    if is_interlaced(video):
        # idet's TFF/BFF guess is unreliable on re-encoded clips, so try both orders
        # and keep the one whose motion track is smoothest.
        scores = {}
        for p in ("tff", "bff"):
            out = _filter(
                video, workdir / f"{video.stem}_{p}.mp4", [f"yadif=mode=1:parity={p}"]
            )
            scores[p] = (zigzag_score(scan_motion(iter_frames(out)[1]).centroid_y), out)
        parity = min(scores, key=lambda p: scores[p][0])
        clip = scores[parity][1]
    fps, frames = iter_frames(clip)
    zoomed = plan_zoom(scan_motion(frames))
    prep = Prep(zoomed.crop, zoomed.zoom, parity)
    if prep.crop:
        x, y, w, h = prep.crop
        clip = _filter(
            clip,
            workdir / f"{video.stem}_zoom.mp4",
            [
                f"crop={w}:{h}:{x}:{y}",
                f"scale=iw*{prep.zoom:g}:ih*{prep.zoom:g}:flags=lanczos",
            ],
        )
    return prep, clip, fps
