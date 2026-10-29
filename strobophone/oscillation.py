"""Forced vibration and resonance: sinusoid fits per drive tempo, then the
resonance curve of a base-driven, viscously damped mass on a spring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

BOOTSTRAP_SAMPLES = 1000
G_CAIRO = 9.79


@dataclass(frozen=True)
class Sinusoid:
    amplitude: float
    frequency: float
    phase: float
    offset: float


@dataclass(frozen=True)
class ResonanceResult:
    f0: float
    f0_ci: tuple[float, float]
    zeta: float
    zeta_ci: tuple[float, float]


def _sine(t, amplitude, frequency, phase, offset):
    return amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset


def fit_sinusoid(t: np.ndarray, y: np.ndarray) -> Sinusoid:
    """Least-squares sinusoid, started from the track's FFT peak."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    dt = np.median(np.diff(t))
    n = 16 * len(y)
    spectrum = np.abs(np.fft.rfft(y - y.mean(), n))
    freqs = np.fft.rfftfreq(n, dt)
    f_guess = freqs[1:][np.argmax(spectrum[1:])]
    a_guess = np.sqrt(2) * np.std(y)
    best = None
    for phase in np.linspace(
        0, 2 * np.pi, 4, endpoint=False
    ):  # avoid a bad local minimum
        try:
            p, _ = curve_fit(_sine, t, y, p0=(a_guess, f_guess, phase, y.mean()))
        except RuntimeError:
            continue
        cost = np.sum((_sine(t, *p) - y) ** 2)
        if best is None or cost < best[0]:
            best = (cost, p)
    if best is None:
        raise ValueError("sinusoid fit did not converge")
    a, f, ph, off = best[1]
    if a < 0:
        a, ph = -a, ph + np.pi
    return Sinusoid(float(a), float(abs(f)), float(np.mod(ph, 2 * np.pi)), float(off))


def amplitude_ratio(
    t: np.ndarray, drive: np.ndarray, mass: np.ndarray
) -> tuple[float, float]:
    """(mass amplitude / drive amplitude, phase lag of the mass in radians, 0..2pi),
    both measured at the drive's frequency."""
    t = np.asarray(t, float)
    d = fit_sinusoid(t, drive)
    # Fit the mass as a linear model at the drive frequency: robust even when the
    # mass amplitude is small far above resonance.
    w = 2 * np.pi * d.frequency
    basis = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)])
    (s, c, _), *_ = np.linalg.lstsq(basis, np.asarray(mass, float), rcond=None)
    mass_amp, mass_phase = np.hypot(s, c), np.arctan2(c, s)
    return float(mass_amp / d.amplitude), float(np.mod(d.phase - mass_phase, 2 * np.pi))


def transmissibility(f: np.ndarray, f0: float, zeta: float) -> np.ndarray:
    """|X/Y| for m x'' + c x' + k (x - y) = 0: the hand moves the spring's top,
    and the damping (water around the mass) acts on the mass's absolute velocity."""
    r = np.asarray(f, float) / f0
    return 1 / np.sqrt((1 - r**2) ** 2 + (2 * zeta * r) ** 2)


def fit_resonance(
    freqs: np.ndarray, ratios: np.ndarray, seed: int = 0
) -> ResonanceResult:
    """Fit f0 and the damping ratio to the amplitude-ratio curve; pairs bootstrap for CIs."""
    freqs, ratios = np.asarray(freqs, float), np.asarray(ratios, float)
    p0 = (freqs[np.argmax(ratios)], max(1 / (2 * ratios.max()), 0.01))
    bounds = ((freqs.min() / 3, 1e-4), (freqs.max() * 3, 2.0))

    def fit(f, r):
        return curve_fit(transmissibility, f, r, p0=p0, bounds=bounds)[0]

    f0, zeta = fit(freqs, ratios)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(freqs), len(freqs))
        if len(np.unique(idx)) < 3:
            continue
        try:
            samples.append(fit(freqs[idx], ratios[idx]))
        except RuntimeError:
            continue
    lo, hi = np.percentile(np.array(samples), [2.5, 97.5], axis=0)
    return ResonanceResult(
        float(f0),
        (float(lo[0]), float(hi[0])),
        float(zeta),
        (float(lo[1]), float(hi[1])),
    )


def natural_frequency_from_static(
    mass_kg: float, extension_m: float, g: float = G_CAIRO
) -> float:
    """Expected f0 from part 1 of the lab: k = M g / x under the same mass."""
    k = mass_kg * g / extension_m
    return float(np.sqrt(k / mass_kg) / (2 * np.pi))
