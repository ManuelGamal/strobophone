"""Melde's experiment: loop length from the vibrating string's envelope, then
(lambda f)^2 against hanging mass gives the linear density and the pan mass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

BOOTSTRAP_SAMPLES = 2000


@dataclass(frozen=True)
class MeldeResult:
    mu: float
    mu_ci: tuple[float, float]
    pan_mass: float
    pan_mass_ci: tuple[float, float]


def wavelength_from_envelope(profile: np.ndarray, px_per_m: float) -> float:
    """lambda = 2 x node spacing. `profile` is the vibration amplitude along the
    string (e.g. per-column motion-blur width); nodes are its deep minima."""
    p = np.asarray(profile, float)
    span = p.max() - p.min()
    if span <= 0:
        raise ValueError("flat envelope: no nodes to measure")
    smooth = np.convolve(p, np.ones(5) / 5, mode="same")
    nodes, _ = find_peaks(-smooth, prominence=0.5 * span, distance=5)
    if len(nodes) < 2:
        raise ValueError(f"found {len(nodes)} nodes; need at least 2 to measure a loop")
    # Refine each node to sub-pixel with a parabola through its neighbours.
    refined = []
    for k in nodes:
        a, b, c = smooth[k - 1 : k + 2]
        denom = a - 2 * b + c
        refined.append(k + (0.5 * (a - c) / denom if denom else 0.0))
    return float(2 * np.median(np.diff(refined)) / px_per_m)


def fit_melde(
    masses_kg: np.ndarray,
    wavelengths_m: np.ndarray,
    freqs_hz: np.ndarray,
    g: float = 9.79,
    seed: int = 0,
) -> MeldeResult:
    """Wave speed v = lambda f and v^2 = (M + m0) g / mu, so v^2 against M is a line
    with slope g / mu and intercept m0 g / mu. Using each clip's own frequency
    keeps a driver that slows under load from biasing the result."""
    m = np.asarray(masses_kg, float)
    v2 = (np.asarray(wavelengths_m, float) * np.asarray(freqs_hz, float)) ** 2

    def solve(mm, vv):
        slope, intercept = np.polyfit(mm, vv, 1)
        return g / slope, intercept / slope

    mu, pan = solve(m, v2)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(m), len(m))
        if len(np.unique(idx)) >= 2:
            samples.append(solve(m[idx], v2[idx]))
    lo, hi = np.percentile(np.array(samples), [2.5, 97.5], axis=0)
    return MeldeResult(
        float(mu),
        (float(lo[0]), float(hi[0])),
        float(pan),
        (float(lo[1]), float(hi[1])),
    )
