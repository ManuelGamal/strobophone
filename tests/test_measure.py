import numpy as np
import pytest
from strobophone.measure import measure_video

from strobophone.baseline import read_video, track_colour
from strobophone.calibration import Scale
from strobophone.synthetic import BALL_RGB, make_free_fall_video


def colour_tracker(works_for):
    """Stands in for SAM: tracks the synthetic ball by colour in the clip it is
    given, but only 'recognises' the concept phrases in `works_for`."""

    def tracker(video, concept, n_frames):
        if concept not in works_for:
            return {}
        xy = track_colour(read_video(video)[0], BALL_RGB)[:n_frames]
        return {"0": np.column_stack([xy - 3, xy + 3])}  # l, t, r, b

    return tracker


@pytest.fixture(scope="module")
def small_ball(tmp_path_factory):
    path = tmp_path_factory.mktemp("v") / "fall.mp4"
    return path, make_free_fall_video(path, ball_radius_m=0.012)


def test_small_ball_is_zoomed_and_g_comes_back_in_original_pixels(small_ball, tmp_path):
    path, truth = small_ball
    m = measure_video(
        path, tmp_path, Scale(truth.px_per_m, 0.0), colour_tracker({"ball"})
    )
    assert m.prep.zoom > 1
    assert m.result.fit.g == pytest.approx(truth.g, rel=0.01)
    np.testing.assert_allclose(
        m.y_px[truth.rest_frames + 5], truth.ball_y_px[truth.rest_frames + 5], atol=1.5
    )


def test_falls_back_to_later_concept_phrases(small_ball, tmp_path):
    path, truth = small_ball
    m = measure_video(
        path, tmp_path, Scale(truth.px_per_m, 0.0), colour_tracker({"sphere"})
    )
    assert m.concept == "sphere"
    assert [a[0] for a in m.attempts][-1] == "sphere"


def test_reports_every_attempt_when_nothing_is_found(small_ball, tmp_path):
    path, truth = small_ball
    with pytest.raises(ValueError, match="ball.*nothing found"):
        measure_video(path, tmp_path, Scale(truth.px_per_m, 0.0), colour_tracker(set()))


def test_progress_log_names_each_step(small_ball, tmp_path):
    path, truth = small_ball
    lines = []
    measure_video(path, tmp_path, Scale(truth.px_per_m, 0.0), colour_tracker({"sphere"}), log=lines.append)
    text = "\n".join(lines)
    for expected in ("progressive", "zoom", "'ball': nothing found", "'sphere'", "g ="):
        assert expected in text


def test_each_attempt_outcome_is_logged_before_the_next_attempt(small_ball, tmp_path):
    path, truth = small_ball
    lines = []
    measure_video(path, tmp_path, Scale(truth.px_per_m, 0.0), colour_tracker({"white ball"}), log=lines.append)
    assert lines.index("'ball': nothing found") < lines.index("asking SAM 3.1 for 'white ball'")


def test_zoomed_track_maps_back_without_offset(tmp_path):
    # ball at an odd x so a naive crop starts on an odd pixel (ffmpeg rounds it to even)
    path = tmp_path / "odd.mp4"
    truth = make_free_fall_video(path, ball_radius_m=0.012, size=(488, 520))
    m = measure_video(path, tmp_path, Scale(truth.px_per_m, 0.0), colour_tracker({"ball"}))
    assert m.prep.zoom > 1
    assert all(v % 2 == 0 for v in m.prep.crop)
    x = (m.boxes[:, 0] + m.boxes[:, 2]) / 2
    assert abs(np.nanmedian(x) - truth.ball_x_px) < 0.3


def test_restitution_is_reported_when_the_ball_bounces(tmp_path):
    from tests.test_bounce import bounce_track

    path = tmp_path / "big.mp4"
    make_free_fall_video(path, ball_radius_m=0.06)  # large ball: no crop/zoom, identity mapping

    def bouncing(video, concept, n_frames):
        y = np.full(n_frames, np.nan)
        track = bounce_track(e=0.7)[:n_frames]
        y[: len(track)] = track
        return {"0": np.column_stack([np.full(n_frames, 100.0), y - 5, np.full(n_frames, 110.0), y + 5])}

    m = measure_video(path, tmp_path, Scale(600.0, 0.0), bouncing)
    assert m.restitution.e == pytest.approx(0.7, abs=0.01)


def test_no_restitution_without_a_bounce(small_ball, tmp_path):
    path, truth = small_ball
    m = measure_video(path, tmp_path, Scale(truth.px_per_m, 0.0), colour_tracker({"ball"}))
    assert m.restitution is None


def test_track_video_returns_boxes_in_original_pixels(small_ball, tmp_path):
    from strobophone.measure import track_video

    path, truth = small_ball
    tr = track_video(path, tmp_path, colour_tracker({"ball"}))
    y = (tr.boxes[:, 1] + tr.boxes[:, 3]) / 2
    np.testing.assert_allclose(y[truth.rest_frames + 5], truth.ball_y_px[truth.rest_frames + 5], atol=1.5)
    assert tr.fps == pytest.approx(truth.fps)


def test_true_frame_rate_overrides_the_file(small_ball, tmp_path):
    """Slow-motion exports often say 30 fps while capturing at 120/240; the user's
    stated capture rate wins, and g follows it exactly (g scales as fps^2)."""
    path, truth = small_ball
    m = measure_video(
        path, tmp_path, Scale(truth.px_per_m, 0.0), colour_tracker({"ball"}), fps_override=truth.fps / 2
    )
    assert m.fps == pytest.approx(truth.fps / 2)
    assert m.result.fit.g == pytest.approx(truth.g / 4, rel=0.01)
