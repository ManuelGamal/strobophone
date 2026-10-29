import numpy as np
import pytest

from strobophone.baseline import find_white_sheet, read_video, track_colour
from strobophone.calibration import scale_from_reference
from strobophone.pipeline import measure_free_fall
from strobophone.synthetic import BALL_RGB, make_free_fall_video


@pytest.fixture(scope="module")
def fall(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "fall.mp4"
    truth = make_free_fall_video(path)
    frames, fps = read_video(path)
    return truth, frames, fps


def test_video_reads_back_with_frame_count_and_fps(fall):
    truth, frames, fps = fall
    assert fps == pytest.approx(truth.fps)
    assert len(frames) == len(truth.ball_y_px)


def test_colour_tracker_follows_ball_within_a_pixel(fall):
    truth, frames, _ = fall
    centres = track_colour(frames, BALL_RGB)
    assert np.nanmax(np.abs(centres[:, 1] - truth.ball_y_px)) < 1.0
    assert np.nanmax(np.abs(centres[:, 0] - truth.ball_x_px)) < 1.0


def test_colour_tracker_returns_nan_when_object_absent(fall):
    _, frames, _ = fall
    blank = np.full_like(frames[:2], 128)
    assert np.isnan(track_colour(blank, BALL_RGB)).all()


def test_white_sheet_box_matches_a4(fall):
    truth, frames, _ = fall
    _, _, w, h = find_white_sheet(frames[0])
    assert w == pytest.approx(truth.a4_box_px[2], abs=2)
    assert h == pytest.approx(truth.a4_box_px[3], abs=2)


def test_end_to_end_g_from_synthetic_video(fall):
    truth, frames, fps = fall
    _, _, w, h = find_white_sheet(frames[0])
    scale = scale_from_reference(w, h, "A4")
    m = measure_free_fall(track_colour(frames, BALL_RGB)[:, 1], fps, scale)
    assert m.fit.g == pytest.approx(truth.g, rel=0.01)
    lo, hi = m.g_ci_with_scale
    assert lo < truth.g < hi


def test_scale_uncertainty_widens_the_ci():
    fps, px = 120, 300.0
    t = np.arange(61) / fps
    y = np.concatenate([np.full(10, 60.0), 60 + 0.5 * 9.81 * t**2 * px])
    y = y + np.random.default_rng(0).normal(0, 0.5, len(y))
    exact = measure_free_fall(y, fps, scale_from_reference(0.210 * px, 0.297 * px))
    tilted = measure_free_fall(y, fps, scale_from_reference(0.210 * px, 0.297 * px * 1.04))
    assert (tilted.g_ci_with_scale[1] - tilted.g_ci_with_scale[0]) > 2 * (
        exact.g_ci_with_scale[1] - exact.g_ci_with_scale[0]
    )


def test_fit_stops_at_the_first_bounce():
    from strobophone.calibration import Scale
    from tests.test_bounce import FPS, G_PX, bounce_track

    m = measure_free_fall(bounce_track(e=0.7, noise=0.3), FPS, Scale(600.0, 0.0))
    assert m.fit.g == pytest.approx(G_PX / 600, rel=0.02)


def test_measurement_exposes_the_fitted_curve_and_frames_for_plotting():
    from strobophone.calibration import Scale
    from tests.test_bounce import FPS, bounce_track

    y = bounce_track(e=0.7)
    m = measure_free_fall(y, FPS, Scale(600.0, 0.0))
    assert m.fit_frames[0] == m.release_frame
    assert len(m.fit_frames) == m.fit.n_points
    t = (m.fit_frames - m.release_frame) / FPS
    np.testing.assert_allclose(np.polyval(m.fit.coeffs, t), y[m.fit_frames], atol=0.5)
