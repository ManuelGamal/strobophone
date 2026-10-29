import numpy as np
import pytest

from strobophone.calibration import scale_from_reference
from strobophone.fits import detect_release, fit_free_fall

G = 9.81
PX_PER_M = 300.0
FPS = 120


def parabola(n=61, y0=60.0, v0=0.0, g=G):
    t = np.arange(n) / FPS
    return t, y0 + (v0 * t + 0.5 * g * t**2) * PX_PER_M


# --- calibration -------------------------------------------------------------


def test_a4_portrait_box_gives_exact_scale():
    s = scale_from_reference(0.210 * PX_PER_M, 0.297 * PX_PER_M, "A4")
    assert s.px_per_m == pytest.approx(PX_PER_M)
    assert s.rel_uncertainty == pytest.approx(0.0, abs=1e-12)


def test_a4_landscape_box_gives_same_scale():
    s = scale_from_reference(0.297 * PX_PER_M, 0.210 * PX_PER_M, "A4")
    assert s.px_per_m == pytest.approx(PX_PER_M)


def test_perspective_stretch_shows_up_as_uncertainty():
    # Height stretched 2% (camera tilted): the two sides disagree by 2%.
    s = scale_from_reference(0.210 * PX_PER_M, 0.297 * PX_PER_M * 1.02, "A4")
    assert s.rel_uncertainty == pytest.approx(0.01, rel=0.05)


def test_unknown_reference_is_rejected():
    with pytest.raises(ValueError, match="unknown reference"):
        scale_from_reference(100, 100, "banana")


# --- free fall ---------------------------------------------------------------


def test_exact_parabola_recovers_g():
    t, y = parabola()
    r = fit_free_fall(t, y, PX_PER_M)
    assert r.g == pytest.approx(G, rel=1e-9)
    assert r.n_points == len(t)


def test_initial_velocity_does_not_bias_g():
    t, y = parabola(v0=0.4)
    assert fit_free_fall(t, y, PX_PER_M).g == pytest.approx(G, rel=1e-9)


def test_noisy_track_ci_contains_true_g_and_is_tight():
    t, y = parabola()
    y = y + np.random.default_rng(1).normal(0, 1.0, len(y))  # 1 px tracking noise
    r = fit_free_fall(t, y, PX_PER_M)
    assert r.ci_low < G < r.ci_high
    assert r.ci_high - r.ci_low < 0.6
    assert 0.5 < r.residual_rms_px < 1.5


def test_ci_is_reproducible_for_a_seed():
    t, y = parabola()
    y = y + np.random.default_rng(2).normal(0, 1.0, len(y))
    a, b = fit_free_fall(t, y, PX_PER_M, seed=7), fit_free_fall(t, y, PX_PER_M, seed=7)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


def test_too_few_points_is_rejected():
    t, y = parabola(n=3)
    with pytest.raises(ValueError, match="at least"):
        fit_free_fall(t, y, PX_PER_M)


def test_release_detected_after_rest_frames():
    _, fall = parabola()
    y = np.concatenate([np.full(15, 60.0), fall])
    idx = detect_release(y)
    assert 15 <= idx <= 20
    t = (np.arange(len(y)) - idx) / FPS
    assert fit_free_fall(t[idx:], y[idx:], PX_PER_M).g == pytest.approx(G, rel=1e-6)


def test_release_detection_ignores_tracking_jitter_at_rest():
    _, fall = parabola()
    rest = 60.0 + np.random.default_rng(3).normal(0, 0.5, 15)
    idx = detect_release(np.concatenate([rest, fall]))
    assert 14 <= idx <= 20


def test_no_motion_is_rejected():
    with pytest.raises(ValueError, match="no motion"):
        detect_release(np.full(30, 60.0))


def test_ci_keeps_95_percent_coverage_with_few_points():
    # 30 fps, 1 m drop -> 13 points: the regime of ordinary phone video.
    rng = np.random.default_rng(1)
    t = np.arange(13) / 30
    hits = 0
    for k in range(300):
        y = 100 + 0.5 * 9.81 * 500 * t**2 + rng.normal(0, 1.0, len(t))
        r = fit_free_fall(t, y, 500.0, seed=k)
        hits += r.ci_low <= 9.81 <= r.ci_high
    assert hits / 300 >= 0.93
