"""Projectile motion on Apple-PI's 59 real clips: a ball rolls off a table edge
and lands on the floor. The one-command pipeline tracks it; g comes from the
time of flight, g = 2h/T^2, using the annotated table height h and the frame
rate only (no pixel scale: the edge is the vertex of the vertical parabola).

    .venv/Scripts/python eval/apple_pi_projectile.py

Apple-PI (CC BY 4.0) is used for evaluation only; nothing is tuned on it.
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

from apple_pi import OUT
from apple_pi_auto import AUTO, cached_tracker

from strobophone.bounce import g_from_time_of_flight, measure_restitution
from strobophone.calibration import Scale
from strobophone.drops import parse_apple_pi_annotation
from strobophone.measure import measure_video

DATA = (
    ROOT / "data" / "apple_pi" / "cases" / "UniversalGravitation" / "ProjectileMotion"
)


def cases() -> list[Path]:
    return [
        c
        for c in sorted(DATA.iterdir())
        if json.loads((c / "metadata.json").read_text(encoding="utf-8")).get(
            "is_realworld"
        )
    ]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")
    rows = []
    for case in cases():
        md = json.loads((case / "metadata.json").read_text(encoding="utf-8"))
        g_true, balls = parse_apple_pi_annotation(md["annotation"])
        row = {
            "case": case.name,
            "balls": "+".join(b["name"] for b in balls),
            "g_true": g_true,
        }
        work = AUTO / f"projectile_{case.name}"
        work.mkdir(parents=True, exist_ok=True)
        try:
            m = measure_video(
                case / "rgb" / "video.mp4",
                work,
                Scale(1.0, 0.0),
                tracker=cached_tracker(f"projectile_{case.name}", False),
            )
        except Exception as e:
            rows.append({**row, "status": f"failed: {e}"})
            print(rows[-1], flush=True)
            continue
        h = {b["h"] for b in balls}
        h_m = h.pop() if len(h) == 1 else None
        try:
            g_tof = g_from_time_of_flight(m.y_px, h_m, m.fps) if h_m else float("nan")
        except ValueError:
            g_tof = float("nan")
        try:
            e = measure_restitution(m.y_px).e
        except ValueError:
            e = float("nan")
        rows.append(
            {
                **row,
                "status": "ok",
                "fps": m.fps,
                "concept": m.concept,
                "h_m": h_m,
                "g_tof": round(g_tof, 3),
                "e_auto": round(e, 3),
                "e_ball": balls[0]["e"] if len(balls) == 1 else None,
                "warnings": len(m.result.warnings),
            }
        )
        print(rows[-1], flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / "projectile.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r.get("status") == "ok" and np.isfinite(r["g_tof"])]
    err = [abs(r["g_tof"] - r["g_true"]) / r["g_true"] for r in ok]
    print(
        f"\n{sum(r.get('status') == 'ok' for r in rows)}/{len(rows)} clips tracked; time-of-flight g on {len(ok)}: "
        f"median {np.median([r['g_tof'] for r in ok]):.2f}, median |error| {100 * np.median(err):.1f}%"
    )


if __name__ == "__main__":
    main()
