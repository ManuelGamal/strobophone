"""StroboPhone web app: upload a free-fall clip, watch the pipeline run, get g
with its uncertainty, the tracked trajectory and the fitted curve.

    .venv/Scripts/python -m uvicorn app.server:app --port 8000

Jobs run one at a time in a background thread (SAM 3.1 calls cost money; a
daily cap bounds the spend of the public demo).
"""

from __future__ import annotations

import datetime as dt
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from strobophone.baseline import find_white_sheet, read_video
from strobophone.calibration import Scale, scale_from_reference
from strobophone.measure import _default_tracker, measure_video

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
    t_fit = np.linspace(0, (r.fit_frames[-1] - r.release_frame) / m.fps, 60)
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
            "t": (t_fit + r.release_frame / m.fps).tolist(),
            "y": np.polyval(r.fit.coeffs, t_fit).tolist(),
            "coeffs": list(r.fit.coeffs),  # y(t since release) in px
        },
    }


def create_app(
    tracker=_default_tracker, workdir: Path | None = None, max_jobs_per_day: int = 40
) -> FastAPI:
    app = FastAPI(title="StroboPhone")
    root = Path(workdir or tempfile.mkdtemp(prefix="strobophone_"))
    jobs: dict[str, dict] = {}
    runner = ThreadPoolExecutor(max_workers=1)
    quota = {"day": dt.date.today(), "used": 0}
    lock = threading.Lock()

    def run(
        job_id: str, video: Path, px_per_m: float | None, reference: str | None
    ) -> None:
        job = jobs[job_id]
        job["status"] = "running"
        log = job["log"].append
        try:
            frames, fps = read_video(video)
            if len(frames) / fps > MAX_SECONDS:
                raise ValueError(
                    f"clip is {len(frames) / fps:.1f} s; trim it to under {MAX_SECONDS:g} s around the drop"
                )
            if reference:
                _, _, w, h = find_white_sheet(frames[0])
                scale = scale_from_reference(w, h, reference)
                log(
                    f"{reference} sheet {w:.0f} x {h:.0f} px -> {scale.px_per_m:.1f} px/m (+/-{100 * scale.rel_uncertainty:.1f}%)"
                )
            else:
                scale = Scale(px_per_m, 0.0)
            del frames
            m = measure_video(video, video.parent, scale, tracker=tracker, log=log)
            job["result"] = _result_json(m, scale)
            job["status"] = "done"
        except Exception as e:  # reported to the user, not swallowed
            job["error"] = str(e)
            job["status"] = "failed"

    @app.post("/api/measure", status_code=202)
    async def measure(
        video: UploadFile = File(...),
        px_per_m: float | None = Form(None),
        reference: str | None = Form(None),
    ):
        if px_per_m is None and reference is None:
            raise HTTPException(
                422, "give a scale: px_per_m, or a reference sheet (A4 or letter)"
            )
        if reference is not None and reference not in ("A4", "letter"):
            raise HTTPException(422, "reference must be A4 or letter")
        with lock:
            if quota["day"] != dt.date.today():
                quota.update(day=dt.date.today(), used=0)
            if quota["used"] >= max_jobs_per_day:
                raise HTTPException(
                    429,
                    "the demo's daily measurement budget is used up; try again tomorrow",
                )
            quota["used"] += 1
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
        runner.submit(run, job_id, path, px_per_m, reference)
        return {"id": job_id}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "no such job")
        return jobs[job_id]

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
