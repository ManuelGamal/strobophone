"""Bake a measured example into the web app so visitors (and judges) can see a
full result without uploading anything or spending API credit.

    .venv/Scripts/python scripts/make_example.py 000046

Uses an Apple-PI FreeFall clip (CC BY 4.0, yaorunmao/Apple-PI-GT) and the cached
SAM 3.1 tracks from eval/apple_pi_auto.py; the scale comes from the annotated
ball diameter and the tracked box width (+/-5% assumed).
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

from apple_pi import DATA
from apple_pi_auto import AUTO, cached_tracker

from app.server import _result_json
from strobophone.calibration import Scale
from strobophone.drops import parse_apple_pi_annotation
from strobophone.measure import measure_video
from strobophone.pipeline import measure_free_fall

EXAMPLES = ROOT / "app" / "static" / "examples"


def main() -> None:
    case = DATA / sys.argv[1]
    md = json.loads((case / "metadata.json").read_text(encoding="utf-8"))
    _, balls = parse_apple_pi_annotation(md["annotation"])
    if len(balls) != 1:
        sys.exit("pick a single-ball clip")
    log: list[str] = []
    video = case / "rgb" / "video.mp4"
    m = measure_video(
        video,
        AUTO / case.name,
        Scale(1.0, 0.0),
        tracker=cached_tracker(case.name, False),
        log=log.append,
    )
    width = float(
        np.nanmedian(m.boxes[m.result.fit_frames, 2] - m.boxes[m.result.fit_frames, 0])
    )
    scale = Scale(width / balls[0]["d"], 0.05)
    m = dataclasses.replace(m, result=measure_free_fall(m.y_px, m.fps, scale))
    lo, hi = m.result.g_ci_with_scale
    log = [line for line in log if not line.startswith("g =")]
    log.append(
        f"scale from the {balls[0]['d'] * 1000:g} mm ball: {width:.1f} px wide -> {scale.px_per_m:.1f} px/m (+/-5%)"
    )
    log.append(
        f"g = {m.result.fit.g:.3f} m/s^2, 95% CI [{lo:.3f}, {hi:.3f}] from {m.result.fit.n_points} points"
    )
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    name = f"applepi_{case.name}"
    shutil.copyfile(video, EXAMPLES / f"{name}.mp4")
    payload = {
        "log": log,
        "result": _result_json(m, scale),
        "credit": f"Apple-π FreeFall clip {case.name} ({balls[0]['name']}), yaorunmao/Apple-PI-GT, CC BY 4.0",
        "video": f"examples/{name}.mp4",
    }
    (EXAMPLES / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    print(log[-1], "->", EXAMPLES / f"{name}.json")


if __name__ == "__main__":
    main()
