"""StroboPhone web app: upload a free-fall clip, watch the pipeline run, get g
with its uncertainty, the tracked trajectory and the fitted curve.

    .venv/Scripts/python -m uvicorn app.server:app --port 8000

Jobs run one at a time in a background thread (SAM 3.1 calls cost money; a
daily cap bounds the spend of the public demo).
"""

from __future__ import annotations

import datetime as dt
import os
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from strobophone.baseline import find_white_sheet, first_frame
from strobophone.calibration import Scale, scale_from_reference
from strobophone.measure import (
    PENDULUM_CONCEPTS,
    _default_tracker,
    horizontal_span,
    measure_video,
    track_video,
)
from strobophone.pendulum import measure_pendulum
from strobophone.report import (
    LANGUAGES,
    facts_for,
    facts_for_pendulum,
    localize,
    segments,
    write_report,
)

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 60 * 1024 * 1024
MAX_SECONDS = 20.0


def _nan_to_none(a) -> list:
    return [
        None if v is None or (isinstance(v, float) and np.isnan(v)) else v
        for v in np.asarray(a, float).tolist()
    ]


def _result_json(m, scale: Scale) -> dict:
    r = m.result
    t = np.arange(len(m.y_px)) / m.fps
    return {
        "g": r.fit.g,
        "ci": list(r.g_ci_with_scale),
        "fit_ci": [r.fit.ci_low, r.fit.ci_high],
        "n_points": r.fit.n_points,
        "residual_rms_px": r.fit.residual_rms_px,
        "fps": m.fps,
        "scale": {"px_per_m": scale.px_per_m, "rel_uncertainty": scale.rel_uncertainty},
        "prep": {"parity": m.prep.parity, "crop": m.prep.crop, "zoom": m.prep.zoom},
        "concept": m.concept,
        "attempts": [list(a) for a in m.attempts],
        "release_frame": r.release_frame,
        "fit_frames": [int(i) for i in r.fit_frames],
        "track": {
            "t": t.tolist(),
            "y": _nan_to_none(m.y_px),
            "boxes": [
                _nan_to_none(b) if not np.isnan(b).all() else None for b in m.boxes
            ],
        },
        "fit": {
            # one curve per ballistic arc (drop, then rebounds), absolute time in s
            "curves": [
                {"t": (tt := np.linspace(seg[0], seg[-1], 40) / m.fps).tolist(), "y": np.polyval(c, tt).tolist()}
                for seg, c in zip(r.segments, r.fit.arc_coeffs)
            ],
            "residuals": {
                "t": (np.concatenate(r.segments) / m.fps).tolist(),
                "r": np.concatenate(
                    [m.y_px[seg] - np.polyval(c, seg / m.fps) for seg, c in zip(r.segments, r.fit.arc_coeffs)]
                ).tolist(),
            },
        },
        "warnings": r.warnings,
        "arc_g": [list(a) for a in r.arc_g],
        "birge_ratio": r.birge_ratio,
        "restitution": None
        if m.restitution is None
        else {"e": m.restitution.e, "ci": list(m.restitution.ci)},
    }


def _pendulum_json(tr, m) -> dict:
    t = np.arange(len(tr.boxes)) / tr.fps
    x, y = (tr.boxes[:, 0] + tr.boxes[:, 2]) / 2, (tr.boxes[:, 1] + tr.boxes[:, 3]) / 2
    cx, cy = m.pivot_px
    return {
        "experiment": "pendulum",
        "g": m.g,
        "ci": list(m.ci),
        "g_small_angle": m.g_small_angle,
        "n_cycles": m.n_cycles,
        "periods": m.periods,
        "amplitudes_deg": np.degrees(m.amplitudes).tolist(),
        "pivot_px": [cx, cy],
        "length_px": m.length_px,
        "fps": tr.fps,
        "prep": {"crop": tr.prep.crop, "zoom": tr.prep.zoom, "parity": tr.prep.parity},
        "concept": tr.concept,
        "attempts": tr.attempts,
        "track": {"t": t.tolist(), "x": _nan_to_none(x), "y": _nan_to_none(y), "boxes": _nan_to_none(tr.boxes)},
        "theta": {"t": t.tolist(), "deg": _nan_to_none(np.degrees(np.arctan2(x - cx, y - cy)))},
        "warnings": [],
    }


class ReportRequest(BaseModel):
    language: str = "en"


def _spark_client():
    from openai import OpenAI

    return OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])


def create_app(
    tracker=_default_tracker,
    workdir: Path | None = None,
    max_jobs_per_day: int = 40,
    report_client=None,
) -> FastAPI:
    app = FastAPI(title="StroboPhone")
    root = Path(workdir or tempfile.mkdtemp(prefix="strobophone_"))
    jobs: dict[str, dict] = {}
    runner = ThreadPoolExecutor(max_workers=1)
    quota = {"day": dt.date.today(), "used": 0}
    lock = threading.Lock()
    measured: dict[str, tuple] = {}  # job id -> (experiment name, report facts), for reports

    def take_quota() -> None:
        with lock:
            if quota["day"] != dt.date.today():
                quota.update(day=dt.date.today(), used=0)
            if quota["used"] >= max_jobs_per_day:
                raise HTTPException(
                    429,
                    "the demo's daily measurement budget is used up; try again tomorrow",
                )
            quota["used"] += 1

    def run(
        job_id: str,
        video: Path,
        px_per_m: float | None,
        reference: str | None,
        experiment: str = "free_fall",
        length_m: float | None = None,
        fps: float | None = None,
    ) -> None:
        job = jobs[job_id]
        job["status"] = "running"
        log = job["log"].append
        try:
            _, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
            if seconds > MAX_SECONDS:
                raise ValueError(f"clip is {seconds:.1f} s; trim it to under {MAX_SECONDS:g} s")
            if experiment == "pendulum":
                tr = track_video(
                    video, video.parent, tracker, PENDULUM_CONCEPTS, log, pick=horizontal_span, fps_override=fps
                )
                x, y = (tr.boxes[:, 0] + tr.boxes[:, 2]) / 2, (tr.boxes[:, 1] + tr.boxes[:, 3]) / 2
                m = measure_pendulum(np.arange(len(x)) / tr.fps, x, y, length_m)
                log(
                    f"{m.n_cycles} swings, mean period {np.mean(m.periods):.4f} s, first amplitude "
                    f"{np.degrees(m.amplitudes[0]):.1f} deg"
                )
                log(f"g = {m.g:.3f} m/s^2, 95% CI [{m.ci[0]:.3f}, {m.ci[1]:.3f}] (small-angle formula: {m.g_small_angle:.3f})")
                job["result"] = _pendulum_json(tr, m)
                measured[job_id] = ("pendulum", facts_for_pendulum(m, length_m, tr.fps, tr.concept))
                job["status"] = "done"
                return
            if reference:
                _, _, w, h = find_white_sheet(first_frame(video))
                scale = scale_from_reference(w, h, reference)
                log(
                    f"{reference} sheet {w:.0f} x {h:.0f} px -> {scale.px_per_m:.1f} px/m (+/-{100 * scale.rel_uncertainty:.1f}%)"
                )
            else:
                scale = Scale(px_per_m, 0.0)
            m = measure_video(video, video.parent, scale, tracker=tracker, log=log, fps_override=fps)
            job["result"] = _result_json(m, scale)
            measured[job_id] = ("free-fall", facts_for(m, scale))
            job["status"] = "done"
        except Exception as e:  # reported to the user, not swallowed
            job["error"] = str(e)
            job["status"] = "failed"
        finally:  # keep only the upload (served back to the page); intermediates can be large
            for f in video.parent.iterdir():
                if f != video and f.is_file():
                    f.unlink(missing_ok=True)

    @app.post("/api/measure", status_code=202)
    async def measure(
        video: UploadFile = File(...),
        px_per_m: float | None = Form(None),
        reference: str | None = Form(None),
        experiment: str = Form("free_fall"),
        length_m: float | None = Form(None),
        fps: float | None = Form(None),
    ):
        if fps is not None and not 1 <= fps <= 2000:
            raise HTTPException(422, "capture frame rate must be between 1 and 2000 fps")
        if experiment not in ("free_fall", "pendulum"):
            raise HTTPException(422, "experiment must be free_fall or pendulum")
        if experiment == "pendulum":
            if length_m is None or not 0.05 <= length_m <= 5:
                raise HTTPException(422, "give the string length in metres (pivot to bob centre), 0.05 to 5")
        elif px_per_m is None and reference is None:
            raise HTTPException(
                422, "give a scale: px_per_m, or a reference sheet (A4 or letter)"
            )
        if reference is not None and reference not in ("A4", "letter"):
            raise HTTPException(422, "reference must be A4 or letter")
        take_quota()
        job_id = uuid.uuid4().hex[:12]
        folder = root / job_id
        folder.mkdir(parents=True)
        path = folder / "upload.mp4"
        size = 0
        with path.open("wb") as out:
            while chunk := await video.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    out.close()
                    shutil.rmtree(folder, ignore_errors=True)
                    raise HTTPException(
                        413, f"upload is over {MAX_UPLOAD_BYTES >> 20} MB"
                    )
                out.write(chunk)
        jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "log": [],
            "result": None,
            "error": None,
        }
        runner.submit(run, job_id, path, px_per_m, reference, experiment, length_m, fps)
        return {"id": job_id}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "no such job")
        return jobs[job_id]

    @app.post("/api/jobs/{job_id}/report")
    def report(job_id: str, body: ReportRequest):
        """Muse Spark drafts the results section with [[fact]] placeholders; code fills
        in every number and blocks drafts with numbers it cannot trace to a tool."""
        if job_id not in measured:
            raise HTTPException(404, "no finished measurement with this id")
        if body.language not in LANGUAGES:
            raise HTTPException(422, f"language must be one of {', '.join(LANGUAGES)}")
        take_quota()
        experiment, facts = measured[job_id]
        facts = localize(facts, body.language)
        r = write_report(facts, report_client or _spark_client(), language=body.language, experiment=experiment)
        return {
            "language": body.language,
            "accepted": r.accepted,
            "revisions": r.revisions,
            "untraced": r.untraced,
            "segments": segments(r.template, facts) if r.accepted else [],
        }

    @app.get("/api/jobs/{job_id}/video")
    def job_video(job_id: str):
        path = root / job_id / "upload.mp4"
        if job_id not in jobs or not path.exists():
            raise HTTPException(404, "no such job")
        return FileResponse(path, media_type="video/mp4")

    if STATIC.exists():
        app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def _app_from_env() -> FastAPI:
    import os

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    return create_app(max_jobs_per_day=int(os.environ.get("MAX_JOBS_PER_DAY", "40")))


app = _app_from_env()
