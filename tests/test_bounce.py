import numpy as np
import pytest
from strobophone.bounce import find_bounces, measure_restitution

FPS = 30.0
G_PX = 9.81 * 600  # px/s^2 at 600 px/m


def bounce_track(
    e=0.7,
    y0=100.0,
    floor=900.0,
    fps=FPS,
    phase=0.37,
    n_bounces=2,
    rest=3,
    noise=0.0,
    seed=0,
):
    """Ball centre y (image coords, down = +) for a drop with bounces, sampled at
    times offset by `phase` frames so impacts fall between frames."""
    t_end, segs = 0.0, []
    h, v_up = floor - y0, 0.0
    t_fall = np.sqrt(2 * h / G_PX)
    segs.append((0.0, t_fall, lambda t: y0 + 0.5 * G_PX * t**2))
    t0, v = t_fall, G_PX * t_fall
    for _ in range(n_bounces):
        v_up = e * v
        dur = 2 * v_up / G_PX
        segs.append(
            (
                t0,
                t0 + dur,
                lambda t, t0=t0, v_up=v_up: (
                    floor - v_up * (t - t0) + 0.5 * G_PX * (t - t0) ** 2
                ),
            )
        )
        t0, v = t0 + dur, v_up
    t_end = t0
    times = (np.arange(int(t_end * fps)) + phase) / fps
    y = np.empty(len(times))
    for i, t in enumerate(times):
        for a, b, f in segs:
            if a <= t <= b:
                y[i] = f(t)
                break
    y = np.concatenate([np.full(rest, y0), y])
    return y + np.random.default_rng(seed).normal(0, noise, len(y))


def test_find_bounces_locates_each_lowest_point():
    y = bounce_track(n_bounces=2)
    b = find_bounces(y)
    assert len(b) == 2
    assert all(y[i] >= y[i - 1] and y[i] >= y[i + 1] for i in b)


def test_no_bounce_found_in_pure_fall():
    t = np.arange(20) / FPS
    assert find_bounces(100 + 0.5 * G_PX * t**2) == []


@pytest.mark.parametrize("e", [0.55, 0.7, 0.86])
def test_restitution_recovered_without_scale_or_fps(e):
    r = measure_restitution(bounce_track(e=e))
    assert r.e == pytest.approx(e, abs=0.01)


def test_restitution_with_tracking_noise_has_ci_containing_truth():
    r = measure_restitution(bounce_track(e=0.75, noise=0.7))
    assert r.e == pytest.approx(0.75, abs=0.03)
    assert r.ci[0] < 0.75 < r.ci[1]


def test_restitution_ignores_missing_frames():
    y = bounce_track(e=0.6)
    y[[5, 12]] = np.nan
    assert measure_restitution(y).e == pytest.approx(0.6, abs=0.01)


def test_restitution_needs_a_bounce():
    t = np.arange(20) / FPS
    with pytest.raises(ValueError, match="bounce"):
        measure_restitution(100 + 0.5 * G_PX * t**2)
