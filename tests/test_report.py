import pytest
from strobophone.report import Fact, render, untraced_numbers

FACTS = {
    "g": Fact("g", 9.675, "9.68 m/s²", "fit_free_fall", "measured g"),
    "g_ci_low": Fact("g_ci_low", 9.644, "9.64", "fit_free_fall", "95% CI low"),
    "n_points": Fact("n_points", 27, "27", "fit_free_fall", "points in fit"),
    "fps": Fact("fps", 60.0, "60 fps", "prepare", "frame rate"),
}


def test_placeholders_render_from_tool_facts():
    assert (
        render("We find g = [[g]] from [[n_points]] points.", FACTS)
        == "We find g = 9.68 m/s² from 27 points."
    )


def test_unknown_placeholder_is_an_error():
    with pytest.raises(KeyError, match="g_true"):
        render("Truth is [[g_true]].", FACTS)


def test_numbers_inside_placeholders_and_units_are_not_flagged():
    assert (
        untraced_numbers("g = [[g]] m/s^2, tracked by SAM 3.1 at [[fps]].", FACTS) == []
    )


def test_number_matching_a_fact_at_shown_precision_is_traced():
    assert untraced_numbers("about 9.7 m/s² from 27 points at 60 fps", FACTS) == []


def test_invented_number_is_flagged():
    assert untraced_numbers("close to the textbook 9.81 m/s², off by 1.4%", FACTS) == [
        "9.81",
        "1.4",
    ]


def test_rounded_value_must_round_correctly():
    # 9.675 rounds to 9.68 (or 9.7), never to 9.66
    assert untraced_numbers("g = 9.66", FACTS) == ["9.66"]


def test_facts_from_a_measurement_include_g_and_restitution(tmp_path):
    import numpy as np

    from strobophone.calibration import Scale
    from strobophone.measure import measure_video
    from strobophone.report import facts_for
    from tests.test_measure import make_free_fall_video
    from tests.test_bounce import bounce_track

    path = tmp_path / "big.mp4"
    make_free_fall_video(path, ball_radius_m=0.06)

    def bouncing(video, concept, n):
        y = np.full(n, np.nan)
        tr = bounce_track(e=0.7)[:n]
        y[: len(tr)] = tr
        return {"0": np.column_stack([np.full(n, 100.0), y - 5, np.full(n, 110.0), y + 5])}

    m = measure_video(path, tmp_path, Scale(600.0, 0.0), bouncing)
    f = facts_for(m, Scale(600.0, 0.0))
    assert f["g"].value == pytest.approx(m.result.fit.g)
    assert f["e"].shown == f"{m.restitution.e:.3f}"
    assert f["n_arcs"].value == len(m.result.segments)
