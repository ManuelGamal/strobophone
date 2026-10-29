"""Helpers for real drop clips: dataset annotations, ground contact, and the
pixel scale implied by a known drop height."""

from __future__ import annotations

import re

import numpy as np

_FIELD = re.compile(r"(\w+)\s*=\s*(-?[\d.]+)")


def parse_apple_pi_annotation(text: str) -> tuple[float, list[dict]]:
    """(|g|, balls) from an Apple-PI case annotation. Each ball is
    {"name", "d", "e", "h", "v_0"} in SI units, in the order listed."""
    g = None
    balls = []
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        fields = {k: float(v) for k, v in _FIELD.findall(rest)}
        if name.strip() == "Environment":
            g = abs(fields["g"])
        elif name.strip() != "Ground" and "h" in fields:
            balls.append({"name": name.strip(), **fields})
    if g is None:
        raise ValueError("annotation has no Environment g")
    return g, balls


def contact_frame(y_bottom_px: np.ndarray, release: int) -> int:
    """First frame after `release` where the object stops descending (the
    bounce), or the last tracked frame if it is still falling."""
    y = np.asarray(y_bottom_px, float)
    frames = [i for i in range(release, len(y)) if not np.isnan(y[i])]
    for a, b in zip(frames, frames[1:]):
        if y[b] < y[a]:
            return a
    return frames[-1]


def scale_from_drop(y_release_px: float, y_contact_px: float, h_m: float) -> float:
    """px per metre from the image distance between release and ground contact."""
    drop_px = y_contact_px - y_release_px
    if drop_px <= 0:
        raise ValueError("contact point must be below the release point")
    return drop_px / h_m
