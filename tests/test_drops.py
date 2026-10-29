import numpy as np
import pytest
from meta_sam_parser import FrameReference, SegmentationBoxRecord
from strobophone.drops import contact_frame, parse_apple_pi_annotation, scale_from_drop

from strobophone.sam import boxes_from_records


def box(oid, frame, l, t, r, b):
    return SegmentationBoxRecord(
        order=0,
        object_id=oid,
        left=l,
        top=t,
        right=r,
        bottom=b,
        frame=FrameReference(frame_index=frame),
    )


def test_boxes_from_records_groups_by_object_with_nan_gaps():
    out = boxes_from_records(
        [box("0", 0, 1, 2, 3, 4), box("0", 2, 5, 6, 7, 8), box("1", 1, 0, 0, 1, 1)], 3
    )
    assert set(out) == {"0", "1"}
    np.testing.assert_array_equal(out["0"][0], [1, 2, 3, 4])
    assert np.isnan(out["0"][1]).all()
    np.testing.assert_array_equal(out["0"][2], [5, 6, 7, 8])


def test_parse_two_ball_annotation():
    text = (
        "Environment: g=9.81, F_drag=!=0\nGround: e=0.5\n"
        "LeftSteelBall: d = 0.04, e = 0.6, h = 1.25, v_0 = 0.0\n"
        "RightSteelBall: d = 0.04, e = 0.6, h = 0.95, v_0 = 0.0"
    )
    g, balls = parse_apple_pi_annotation(text)
    assert g == pytest.approx(9.81)
    assert [b["name"] for b in balls] == ["LeftSteelBall", "RightSteelBall"]
    assert balls[1] == {
        "name": "RightSteelBall",
        "d": 0.04,
        "e": 0.6,
        "h": 0.95,
        "v_0": 0.0,
    }


def test_parse_negative_g_is_magnitude():
    g, _ = parse_apple_pi_annotation(
        "Environment: g=-9.81, F_drag=0.0\nClaySphere: d = 1.0, e = 0.2, h = 6.0, v_0 = 0.0"
    )
    assert g == pytest.approx(9.81)


def test_contact_frame_is_lowest_point_before_bounce():
    y = np.array([100, 100, 105, 120, 145, 180, 225, 280, 270, 262, 258, 262, 275])
    assert contact_frame(y, release=1) == 7


def test_contact_frame_ignores_nan_frames():
    y = np.array([100, 100, np.nan, 120, 160, np.nan, 230, 210])
    assert contact_frame(y, release=1) == 6


def test_contact_frame_requires_a_bounce_or_end():
    y = np.array([100, 100, 110, 130, 160, 200])  # still falling at the last frame
    assert contact_frame(y, release=1) == 5


def test_scale_from_drop_height():
    assert scale_from_drop(
        y_release_px=200.0, y_contact_px=1400.0, h_m=1.5
    ) == pytest.approx(800.0)


def test_scale_from_drop_rejects_upward_motion():
    with pytest.raises(ValueError, match="below"):
        scale_from_drop(y_release_px=900.0, y_contact_px=200.0, h_m=1.0)
