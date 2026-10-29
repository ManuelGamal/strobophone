"""Pixel-to-metre scale from a reference object of known size in the camera plane."""

from __future__ import annotations

from dataclasses import dataclass

# (short side, long side) in metres.
REFERENCES = {
    "A4": (0.210, 0.297),
    "letter": (0.2159, 0.2794),
}


@dataclass(frozen=True)
class Scale:
    px_per_m: float
    rel_uncertainty: float


def scale_from_reference(width_px: float, height_px: float, reference: str = "A4") -> Scale:
    """Scale from the reference's bounding box, in either orientation.

    Each side gives its own estimate; their disagreement (perspective tilt, a
    loose box) is reported as the relative uncertainty instead of hidden.
    """
    if reference not in REFERENCES:
        raise ValueError(f"unknown reference {reference!r}; known: {sorted(REFERENCES)}")
    short_m, long_m = REFERENCES[reference]
    short_px, long_px = sorted((width_px, height_px))
    from_long, from_short = long_px / long_m, short_px / short_m
    return Scale(
        px_per_m=(from_long + from_short) / 2,
        rel_uncertainty=abs(from_long - from_short) / (from_long + from_short),
    )
