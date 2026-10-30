"""Free fall on IRIS real clips (rasulkhanbayov/IRIS, CC BY-NC 4.0; evaluation
only). Two independent routes to g, neither tuned on these clips:

- time of flight, g = 2h/T^2, from the annotated drop height and the frame rate;
- the one-command pipeline, with the scale taken from the tracked box width and
  the annotated ball radius (Falling_ball only; Dropping_ball lists no radius).

    .venv/Scripts/python eval/iris_fall.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

from apple_pi_auto import cached_tracker
from iris_pendulum import DATA, OUT, setting

from strobophone.bounce import g_from_time_of_flight
from strobophone.calibration import Scale
from strobophone.measure import measure_video
from strobophone.pipeline import measure_free_fall

CLASSES = {"Falling_ball": "falling_ball", "Dropping_ball": "dropping_ball"}
G_REF = 9.81


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")
    if DATA is None:
        sys.exit("pull the Kaggle output into data/iris_small first")
    root = DATA.parent
    params = json.loads(DATA.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for folder, pkey in CLASSES.items():
        for clip in sorted((root / folder).rglob("*.mp4")):
            if len(sys.argv) > 1 and not any(k in clip.as_posix() for k in sys.argv[1:]):
                continue  # optional filters, e.g. pendulum_45/01
            rel = clip.relative_to(root)
            key = setting(rel, params[pkey])
            row = {"clip": str(rel), "setting": key}
            if key is None:
                rows.append({**row, "status": "unknown setting"})
                print(rows[-1], flush=True)
                continue
            p = {k: v["mean"] for k, v in params[pkey][key].items()}
            name = "iris_" + "_".join(rel.with_suffix("").parts)
            work = OUT / name
            work.mkdir(parents=True, exist_ok=True)
            try:
                m = measure_video(
                    clip, work, Scale(1.0, 0.0), tracker=cached_tracker(name, False)
                )
            except Exception as e:
                rows.append({**row, "status": f"failed: {e}"})
                print(rows[-1], flush=True)
                continue
            try:
                g_tof = g_from_time_of_flight(m.y_px, p["drop_height"], m.fps)
            except ValueError:
                g_tof = float("nan")
            g_scale = lo = hi = float("nan")
            if "ball_radius" in p:
                f = m.result.fit_frames
                width = float(np.nanmedian(m.boxes[f, 2] - m.boxes[f, 0]))
                r = measure_free_fall(
                    m.y_px, m.fps, Scale(width / (2 * p["ball_radius"]), 0.05)
                )
                g_scale, (lo, hi) = r.fit.g, r.g_ci_with_scale
            rows.append(
                {
                    **row,
                    "status": "ok",
                    "fps": m.fps,
                    "concept": m.concept,
                    "zoom": m.prep.zoom,
                    "h_m": p["drop_height"],
                    "g_tof": round(g_tof, 3),
                    "g_scale": round(g_scale, 3),
                    "ci_low": round(lo, 3),
                    "ci_high": round(hi, 3),
                    "arcs": len(m.result.segments),
                    "warnings": len(m.result.warnings),
                }
            )
            print(rows[-1], flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / "fall.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"\n{len(ok)}/{len(rows)} clips tracked")
    for col in ("g_tof", "g_scale"):
        v = np.array([r[col] for r in ok], float)
        v = v[np.isfinite(v)]
        if len(v):
            print(
                f"{col}: n={len(v)} median {np.median(v):.3f}, median |error| "
                f"{100 * np.median(np.abs(v - G_REF)) / G_REF:.2f}%"
            )


if __name__ == "__main__":
    main()
