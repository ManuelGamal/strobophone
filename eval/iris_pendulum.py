"""Pendulum g on IRIS real clips (rasulkhanbayov/IRIS, CC BY-NC 4.0; evaluation
only, nothing redistributed). The clips were shrunk to 720p / first 20 s on
Kaggle (kaggle/iris_prep). g comes from the period, with the annotated rope
length 0.5 m and the frame rate only; no pixel scale.

    .venv/Scripts/kaggle kernels output manuelgamal/strobophone-iris-prep -p data/iris_small
    .venv/Scripts/python eval/iris_pendulum.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

from apple_pi_auto import cached_tracker

from strobophone.measure import horizontal_span, track_video
from strobophone.pendulum import measure_pendulum

DATA = next((p for p in (ROOT / "data" / "iris_small").rglob("parameters.json")), None)
OUT = ROOT / "outputs" / "iris"
CONCEPTS = ("pendulum bob", "ball", "weight", "sphere", "round object")
G_REF = 9.81  # what IRIS lists for its falling-ball setting


def setting(path: Path, settings: dict) -> str | None:
    """The parameters.json setting a clip belongs to: by name, else by its number."""
    p = str(path).lower()
    for k in settings:
        if k.lower() in p:
            return k
    for k in settings:
        n = re.findall(r"\d+", k)
        if n and re.search(rf"(?<!\d){n[0]}(?!\d)", path.parent.name):
            return k
    return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")
    if DATA is None:
        sys.exit("pull the Kaggle output into data/iris_small first")
    root = DATA.parent
    params = json.loads(DATA.read_text(encoding="utf-8"))["pendulum"]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for clip in sorted((root / "Pendulum").rglob("*.mp4")):
        if len(sys.argv) > 1 and not any(k in clip.as_posix() for k in sys.argv[1:]):
            continue  # optional filters, e.g. pendulum_45/01
        key = setting(clip.relative_to(root), params)
        row = {"clip": str(clip.relative_to(root)), "setting": key}
        if key is None:
            rows.append({**row, "status": "unknown setting"})
            print(rows[-1], flush=True)
            continue
        L = params[key]["rope_length"]["mean"]
        name = "iris_" + "_".join(clip.relative_to(root).with_suffix("").parts)
        work = OUT / name
        work.mkdir(parents=True, exist_ok=True)
        try:
            tr = track_video(
                clip, work, cached_tracker(name, False), CONCEPTS, pick=horizontal_span
            )
            x, y = (
                (tr.boxes[:, 0] + tr.boxes[:, 2]) / 2,
                (tr.boxes[:, 1] + tr.boxes[:, 3]) / 2,
            )
            m = measure_pendulum(np.arange(len(x)) / tr.fps, x, y, length_m=L)
        except Exception as e:
            rows.append({**row, "status": f"failed: {e}"})
            print(rows[-1], flush=True)
            continue
        rows.append(
            {
                **row,
                "status": "ok",
                "fps": tr.fps,
                "concept": tr.concept,
                "L_m": L,
                "angle_deg": params[key]["angle"]["mean"],
                "amp0_deg": round(float(np.degrees(m.amplitudes[0])), 1),
                "n_cycles": m.n_cycles,
                "T_mean_s": round(float(np.mean(m.periods)), 4),
                "g": round(m.g, 3),
                "ci_low": round(m.ci[0], 3),
                "ci_high": round(m.ci[1], 3),
                "g_small_angle": round(m.g_small_angle, 3),
            }
        )
        print(rows[-1], flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / "pendulum.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        err = [100 * (r["g"] - G_REF) / G_REF for r in ok]
        print(
            f"\n{len(ok)}/{len(rows)} clips measured; g median {np.median([r['g'] for r in ok]):.3f}, "
            f"median |error| {np.median(np.abs(err)):.2f}%, reference inside CI on "
            f"{sum(r['ci_low'] <= G_REF <= r['ci_high'] for r in ok)}/{len(ok)}"
        )


if __name__ == "__main__":
    main()
