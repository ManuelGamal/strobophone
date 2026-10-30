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


def test_arabic_indic_digits_cannot_sneak_past_the_checker():
    # ٩٫٨١ is 9.81 in Arabic-Indic digits; ۹.۸۱ in Eastern Arabic (Persian) digits
    assert untraced_numbers("قيمة الكتاب ٩٫٨١ م/ث²", FACTS) == ["9.81"]
    assert untraced_numbers("قيمة ۹.۸۱", FACTS) == ["9.81"]
    assert untraced_numbers("من ٢٧ نقطة", FACTS) == []


class FakeClient:
    """Stands in for the Responses API: records the system prompt, returns drafts in order."""

    def __init__(self, *drafts):
        import json
        import types

        self.drafts, self.systems = list(drafts), []

        def create(**kw):
            self.systems.append(kw["input"][0]["content"][0]["text"])
            return types.SimpleNamespace(output_text=json.dumps({"report": self.drafts.pop(0)}))

        self.responses = types.SimpleNamespace(create=create)


def test_report_in_arabic_keeps_the_measured_numbers():
    from strobophone.report import write_report

    client = FakeClient("بلغ تسارع الجاذبية المقيس [[g]] من [[n_points]] نقطة.")
    r = write_report(FACTS, client, language="ar")
    assert r.accepted and "Arabic" in client.systems[0]
    assert r.text == "بلغ تسارع الجاذبية المقيس 9.68 m/s² من 27 نقطة."


def test_report_segments_mark_each_number_with_its_source():
    from strobophone.report import segments

    parts = segments("g = [[g]] at [[fps]].", FACTS)
    assert parts == [
        {"text": "g = "},
        {"text": "9.68 m/s²", "fact": "g", "source": "fit_free_fall", "label": "measured g"},
        {"text": " at "},
        {"text": "60 fps", "fact": "fps", "source": "prepare", "label": "frame rate"},
        {"text": "."},
    ]


def test_text_facts_are_translated_for_arabic_but_numbers_are_untouched():
    from strobophone.report import localize

    facts = {**FACTS, "ref_vs_ci": Fact("ref_vs_ci", "outside", "outside", "report.facts_for", "ref")}
    ar = localize(facts, "ar")
    assert ar["ref_vs_ci"].shown == "خارج"
    assert ar["g"] == FACTS["g"]
    assert localize(facts, "en") == facts
