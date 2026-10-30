import time

import pytest
from app.server import create_app
from fastapi.testclient import TestClient

from strobophone.synthetic import make_free_fall_video
from tests.test_measure import colour_tracker


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("v") / "fall.mp4"
    return path, make_free_fall_video(path, ball_radius_m=0.03)


def wait(client, job_id, timeout_s=60):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.2)
    raise TimeoutError(job)


def upload(client, path, **form):
    with open(path, "rb") as f:
        return client.post(
            "/api/measure", files={"video": ("fall.mp4", f, "video/mp4")}, data=form
        )


def test_upload_returns_g_with_track_and_fit_curve(clip, tmp_path):
    path, truth = clip
    client = TestClient(create_app(tracker=colour_tracker({"ball"}), workdir=tmp_path))
    r = upload(client, path, px_per_m=str(truth.px_per_m))
    assert r.status_code == 202
    job = wait(client, r.json()["id"])
    assert job["status"] == "done", job
    res = job["result"]
    assert res["g"] == pytest.approx(truth.g, rel=0.01)
    assert res["ci"][0] < res["g"] < res["ci"][1]
    assert (
        len(res["track"]["t"]) == len(res["track"]["y"]) == len(res["track"]["boxes"])
    )
    curve = res["fit"]["curves"][0]
    assert len(curve["t"]) == len(curve["y"]) > 10
    assert len(res["fit"]["residuals"]["r"]) == res["n_points"]
    assert any("g =" in line for line in job["log"])


def test_failure_is_reported_with_the_attempts(clip, tmp_path):
    path, truth = clip
    client = TestClient(create_app(tracker=colour_tracker(set()), workdir=tmp_path))
    job = wait(client, upload(client, path, px_per_m="300").json()["id"])
    assert job["status"] == "failed"
    assert "nothing found" in job["error"]


def test_daily_job_cap_protects_the_api_budget(clip, tmp_path):
    path, _ = clip
    client = TestClient(
        create_app(
            tracker=colour_tracker({"ball"}), workdir=tmp_path, max_jobs_per_day=1
        )
    )
    assert upload(client, path, px_per_m="300").status_code == 202
    assert upload(client, path, px_per_m="300").status_code == 429


def test_scale_is_required(clip, tmp_path):
    path, _ = clip
    client = TestClient(create_app(tracker=colour_tracker({"ball"}), workdir=tmp_path))
    assert upload(client, path).status_code == 422


@pytest.fixture(scope="module")
def pendulum_clip(tmp_path_factory):
    from strobophone.synthetic import make_pendulum_video

    path = tmp_path_factory.mktemp("p") / "pendulum.mp4"
    return path, make_pendulum_video(path, theta0_deg=30)


def test_pendulum_upload_needs_only_the_string_length(pendulum_clip, tmp_path):
    path, truth = pendulum_clip
    client = TestClient(create_app(tracker=colour_tracker({"ball"}), workdir=tmp_path))
    r = upload(client, path, experiment="pendulum", length_m=str(truth.length_m))
    assert r.status_code == 202
    job = wait(client, r.json()["id"], timeout_s=120)
    assert job["status"] == "done", job
    res = job["result"]
    assert res["experiment"] == "pendulum"
    assert res["g"] == pytest.approx(truth.g, rel=0.005)
    assert res["ci"][0] < truth.g < res["ci"][1]
    assert len(res["periods"]) == res["n_cycles"] >= 3
    assert len(res["track"]["t"]) == len(res["track"]["x"]) == len(res["track"]["boxes"])
    assert len(res["theta"]["t"]) == len(res["theta"]["deg"])


def test_pendulum_needs_a_length(pendulum_clip, tmp_path):
    path, _ = pendulum_clip
    client = TestClient(create_app(tracker=colour_tracker({"ball"}), workdir=tmp_path))
    assert upload(client, path, experiment="pendulum").status_code == 422


def test_stated_capture_rate_is_used(clip, tmp_path):
    path, truth = clip
    client = TestClient(create_app(tracker=colour_tracker({"ball"}), workdir=tmp_path))
    r = upload(client, path, px_per_m=str(truth.px_per_m), fps=str(truth.fps / 2))
    job = wait(client, r.json()["id"])
    assert job["status"] == "done", job
    assert job["result"]["g"] == pytest.approx(truth.g / 4, rel=0.01), job["log"]
    assert any("file says" in line for line in job["log"])


def test_lab_report_in_arabic_traces_every_number(clip, tmp_path):
    from tests.test_report import FakeClient

    path, truth = clip
    client = TestClient(
        create_app(
            tracker=colour_tracker({"ball"}),
            workdir=tmp_path,
            report_client=FakeClient("قيمة الجاذبية المقيسة [[g]] بفاصل ثقة من [[g_ci_low]] إلى [[g_ci_high]]."),
        )
    )
    job = wait(client, upload(client, path, px_per_m=str(truth.px_per_m)).json()["id"])
    r = client.post(f"/api/jobs/{job['id']}/report", json={"language": "ar"})
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["accepted"] and rep["language"] == "ar"
    facts = [s for s in rep["segments"] if "fact" in s]
    assert [s["fact"] for s in facts] == ["g", "g_ci_low", "g_ci_high"]
    assert all(s["source"] for s in facts)


def test_report_with_typed_numbers_is_blocked(clip, tmp_path):
    from tests.test_report import FakeClient

    path, truth = clip
    bad = "g is close to the value 12.34 found elsewhere."
    client = TestClient(
        create_app(tracker=colour_tracker({"ball"}), workdir=tmp_path, report_client=FakeClient(bad, bad, bad))
    )
    job = wait(client, upload(client, path, px_per_m=str(truth.px_per_m)).json()["id"])
    rep = client.post(f"/api/jobs/{job['id']}/report", json={"language": "en"}).json()
    assert not rep["accepted"] and rep["untraced"] == ["12.34"] and rep["segments"] == []
