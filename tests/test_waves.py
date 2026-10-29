import subprocess

import imageio_ffmpeg
import numpy as np
import pytest

from strobophone.audio import dominant_frequency, extract_audio
from strobophone.melde import fit_melde, wavelength_from_envelope
from strobophone.oscillation import (
    amplitude_ratio,
    fit_resonance,
    fit_sinusoid,
    natural_frequency_from_static,
    transmissibility,
)

RATE = 24000


# --- audio -------------------------------------------------------------------


def test_dominant_frequency_is_sub_bin_accurate():
    t = np.arange(int(0.5 * RATE)) / RATE  # 0.5 s -> 2 Hz FFT bins
    x = np.sin(2 * np.pi * 97.3 * t) + 0.3 * np.random.default_rng(0).normal(
        size=len(t)
    )
    assert dominant_frequency(x, RATE, 50, 200) == pytest.approx(97.3, abs=0.1)


def test_dominant_frequency_respects_band():
    t = np.arange(RATE) / RATE
    x = 2 * np.sin(2 * np.pi * 50 * t) + np.sin(
        2 * np.pi * 120 * t
    )  # mains hum is louder
    assert dominant_frequency(x, RATE, 80, 200) == pytest.approx(120, abs=0.1)


def test_audio_track_is_extracted_from_video(tmp_path):
    video = tmp_path / "tone.mp4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg,
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=64x64:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:duration=1",
            "-shortest",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(video),
        ],
        check=True,
    )
    samples = extract_audio(video, RATE)
    assert len(samples) == pytest.approx(RATE, rel=0.05)
    assert dominant_frequency(samples, RATE, 100, 1000) == pytest.approx(220, abs=0.5)


# --- forced vibration --------------------------------------------------------

F0, ZETA = 1.5, 0.08


def driven_tracks(f, n_periods=6, fps=60, hand_px=20.0, noise_px=0.5, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(n_periods / f * fps)) / fps
    w, r = 2 * np.pi * f, f / F0
    lag = np.arctan2(2 * ZETA * r, 1 - r**2)
    hand = 300 + hand_px * np.sin(w * t) + rng.normal(0, noise_px, len(t))
    mass = 400 + hand_px * transmissibility(f, F0, ZETA) * np.sin(w * t - lag)
    return t, hand, mass + rng.normal(0, noise_px, len(t))


def test_transmissibility_is_one_at_low_frequency_and_peaks_near_f0():
    assert transmissibility(np.array([0.01]), F0, ZETA)[0] == pytest.approx(1, abs=1e-3)
    f = np.linspace(0.5, 3, 2001)
    assert f[np.argmax(transmissibility(f, F0, ZETA))] == pytest.approx(F0, rel=0.02)
    assert transmissibility(np.array([F0]), F0, ZETA)[0] == pytest.approx(
        1 / (2 * ZETA)
    )


def test_sinusoid_fit_recovers_parameters():
    t = np.arange(240) / 60
    y = (
        5
        + 12 * np.sin(2 * np.pi * 1.3 * t + 0.7)
        + np.random.default_rng(1).normal(0, 0.3, len(t))
    )
    s = fit_sinusoid(t, y)
    assert s.frequency == pytest.approx(1.3, rel=0.01)
    assert s.amplitude == pytest.approx(12, rel=0.02)
    assert s.offset == pytest.approx(5, abs=0.2)


def test_amplitude_ratio_matches_theory_at_each_tempo():
    for f in (1.0, 1.4, 1.6, 2.2):
        t, hand, mass = driven_tracks(f)
        ratio, lag = amplitude_ratio(t, hand, mass)
        assert ratio == pytest.approx(
            transmissibility(np.array([f]), F0, ZETA)[0], rel=0.03
        )
        assert 0 < lag < np.pi


def test_resonance_fit_recovers_f0_and_damping_with_cis():
    freqs = np.linspace(1.0, 2.2, 13)
    true = transmissibility(freqs, F0, ZETA)
    noisy = true * (1 + np.random.default_rng(2).normal(0, 0.03, len(freqs)))
    r = fit_resonance(freqs, noisy)
    assert r.f0 == pytest.approx(F0, rel=0.02)
    assert r.zeta == pytest.approx(ZETA, rel=0.25)
    assert r.f0_ci[0] < F0 < r.f0_ci[1]
    assert r.zeta_ci[0] < ZETA < r.zeta_ci[1]


def test_natural_frequency_from_static_stretch():
    # k = m g / x; f0 = sqrt(k / m) / 2pi = sqrt(g / x) / 2pi, independent of mass.
    assert natural_frequency_from_static(0.5, 0.11) == pytest.approx(
        np.sqrt(9.79 / 0.11) / (2 * np.pi)
    )


# --- Melde -------------------------------------------------------------------


def test_wavelength_from_three_loop_envelope():
    px_per_m, loop_px = 500.0, 200.0  # lambda = 2 * 0.4 m
    x = np.arange(600)
    profile = 30 * np.abs(np.sin(np.pi * x / loop_px)) + 2
    profile += np.random.default_rng(3).normal(0, 1, len(x))
    assert wavelength_from_envelope(profile, px_per_m) == pytest.approx(0.8, rel=0.02)


def test_envelope_without_nodes_is_rejected():
    with pytest.raises(ValueError, match="nodes"):
        wavelength_from_envelope(np.full(300, 10.0), 500.0)


MU, PAN = 0.0005, 0.010


def melde_data(noise=0.0, seed=0):
    masses = np.array([0.02, 0.04, 0.06, 0.08, 0.10])
    freqs = np.array([100, 98, 96, 94, 92.0])  # toothbrush slows under load
    lam = np.sqrt((masses + PAN) * 9.79 / MU) / freqs
    return (
        masses,
        lam * (1 + np.random.default_rng(seed).normal(0, noise, len(lam))),
        freqs,
    )


def test_melde_exact_data_recovers_mu_and_pan_mass():
    r = fit_melde(*melde_data())
    assert r.mu == pytest.approx(MU, rel=1e-9)
    assert r.pan_mass == pytest.approx(PAN, rel=1e-6)


def test_melde_noisy_data_cis_contain_truth():
    r = fit_melde(*melde_data(noise=0.01))
    assert r.mu_ci[0] < MU < r.mu_ci[1]
    assert r.pan_mass_ci[0] < PAN < r.pan_mass_ci[1]
