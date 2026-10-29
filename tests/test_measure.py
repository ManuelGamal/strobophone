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
