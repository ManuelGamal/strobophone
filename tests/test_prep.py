from pathlib import Path

import numpy as np
import pytest

from strobophone.baseline import read_video
from strobophone.prep import (
    Prep,
    motion_roi,
    parse_idet,
    plan_zoom,
    prepare,
    track_with_fallbacks,
    zigzag_score,
)
from strobophone.synthetic import make_free_fall_video


@pytest.fixture(scope="module")
def fall(tmp_path_factory):
    path = tmp_path_factory.mktemp("v") / "fall.mp4"
    truth = make_free_fall_video(path, ball_radius_m=0.012)  # ~7 px ball, like BallDrop
    frames, _ = read_video(path)
    return truth, frames


def test_motion_roi_contains_the_whole_trajectory(fall):
    truth, frames = fall
    x, y, w, h = motion_roi(frames)
    assert x <= truth.ball_x_px <= x + w
    assert y <= truth.ball_y_px[truth.rest_frames + 2] and truth.ball_y_px[-1] <= y + h


def test_motion_roi_excludes_static_background(fall):
    _, frames = fall
    _, _, w, _ = motion_roi(frames)
    assert w < frames.shape[2] / 3


def test_small_object_gets_zoomed_to_target_size(fall):
    _, frames = fall
    p = plan_zoom(frames, target_px=32)
    assert 3 <= p.zoom <= 6
    assert p.crop is not None


def test_large_object_is_not_zoomed(tmp_path):
    truth = make_free_fall_video(
        tmp_path / "big.mp4", ball_radius_m=0.06
    )  # ~36 px ball
    frames, _ = read_video(tmp_path / "big.mp4")
    assert plan_zoom(frames, target_px=32).zoom == 1


def test_prep_maps_boxes_back_to_original_pixels():
    p = Prep(crop=(150, 0, 100, 480), zoom=4.0, parity=None)
    boxes = np.array([[40.0, 400.0, 80.0, 440.0]])
    np.testing.assert_allclose(p.to_original(boxes), [[160.0, 100.0, 170.0, 110.0]])


def test_prep_identity_when_no_crop_or_zoom():
    p = Prep(crop=None, zoom=1.0, parity=None)
    b = np.array([[1.0, 2.0, 3.0, 4.0]])
    np.testing.assert_allclose(p.to_original(b), b)


def test_parse_idet_reads_multi_frame_counts():
    log = (
        "[Parsed_idet_0 @ 000001] Repeated Fields: Neither: 33 Top: 0 Bottom: 0\n"
        "[Parsed_idet_0 @ 000001] Single frame detection: TFF: 2 BFF: 9 Progressive: 10 Undetermined: 12\n"
        "[Parsed_idet_0 @ 000001] Multi frame detection: TFF: 0 BFF: 21 Progressive: 4 Undetermined: 8\n"
    )
    assert parse_idet(log) == {"tff": 0, "bff": 21, "progressive": 4}


def test_parse_idet_uses_the_last_summary_not_the_empty_probe_pass():
    empty = "[Parsed_idet_0 @ 1] Multi frame detection: TFF: 0 BFF: 0 Progressive: 0 Undetermined: 0\n"
    real = "[Parsed_idet_0 @ 2] Multi frame detection: TFF: 0 BFF: 18 Progressive: 14 Undetermined: 1\n"
    assert parse_idet(empty + real) == {"tff": 0, "bff": 18, "progressive": 14}


def test_zigzag_score_prefers_monotonic_fall():
    t = np.arange(30)
    smooth = 5 * t**2 / 10
    swapped = smooth.copy()
    swapped[0::2], swapped[1::2] = smooth[1::2], smooth[0::2]
    assert zigzag_score(smooth) < zigzag_score(swapped) / 10


def test_fallback_concepts_tried_in_order_until_a_mover_is_found():
    calls = []

    def fake_tracker(video, concept, n):
        calls.append(concept)
        if concept == "white ball":
            b = np.full((n, 4), np.nan)
            b[:, 1] = np.linspace(0, 300, n)
            b[:, 3] = b[:, 1] + 10
            b[:, 0], b[:, 2] = 0, 10
            return {"0": b}
        if concept == "ball":
            return {"0": np.tile([0.0, 5.0, 10.0, 15.0], (n, 1))}  # found, but static
        return {}

    concept, boxes, attempts = track_with_fallbacks(
        "v.mp4", ["ball", "white ball", "sphere"], 20, fake_tracker
    )
    assert concept == "white ball"
    assert calls == ["ball", "white ball"]
    assert attempts == [("ball", "no moving object"), ("white ball", "ok")]


def test_fallback_reports_total_failure():
    concept, boxes, attempts = track_with_fallbacks(
        "v.mp4", ["a", "b"], 10, lambda v, c, n: {}
    )
    assert concept is None and boxes == {}
    assert [a[1] for a in attempts] == ["nothing found", "nothing found"]


BALLDROP = Path(__file__).resolve().parents[1] / "data" / "dev" / "BallDrop.mp4"


@pytest.mark.skipif(not BALLDROP.exists(), reason="Tracker sample video not downloaded")
def test_balldrop_is_detected_as_interlaced_bottom_field_first_and_zoomed(tmp_path):
    prep, path, fps = prepare(BALLDROP, tmp_path)
    assert prep.parity == "bff"
    assert fps == pytest.approx(60)
    assert prep.zoom >= 3


def test_progressive_synthetic_clip_is_left_progressive(tmp_path):
    src = tmp_path / "fall.mp4"
    make_free_fall_video(src, ball_radius_m=0.06)
    prep, path, fps = prepare(src, tmp_path)
    assert prep.parity is None and prep.zoom == 1 and path == src
