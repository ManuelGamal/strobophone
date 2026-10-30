"""Summaries of the IRIS evals by setting (reads outputs/iris/*.csv).

Pendulum: if the amplitude correction is right, corrected g is the same at 20,
45 and 90 degrees while the small-angle formula drifts. The length that would
give g = 9.81 is reported too: IRIS states "rope length 0.5 m" without saying
whether it is pivot-to-bob-centre.

    .venv/Scripts/python eval/iris_summary.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "outputs" / "iris"
G = 9.81


def rows(name: str) -> list[dict]:
    path = OUT / name
    return list(csv.DictReader(path.open())) if path.exists() else []


def pendulum() -> None:
    by = defaultdict(list)
    for r in rows("pendulum.csv"):
        if r["status"] == "ok":
            by[r["setting"]].append(r)
    if not by:
        return
    print("pendulum   n  amp0(deg)  g corrected        g small-angle      L for g=9.81")
    for key in sorted(by, key=lambda k: float(k.split("_")[-1])):
        rs = by[key]
        g = np.array([float(r["g"]) for r in rs])
        g0 = np.array([float(r["g_small_angle"]) for r in rs])
        amp = np.median([float(r["amp0_deg"]) for r in rs])
        L = float(rs[0]["L_m"]) * G / np.median(g)
        print(
            f"{key:12s}{len(rs):2d}  {amp:6.1f}    {np.median(g):.3f} ± {np.std(g, ddof=1):.3f}    "
            f"{np.median(g0):.3f} ± {np.std(g0, ddof=1):.3f}    {L:.3f} m"
        )
    med = [np.median([float(r["g"]) for r in rs]) for rs in by.values()]
    med0 = [np.median([float(r["g_small_angle"]) for r in rs]) for rs in by.values()]
    print(
        f"spread of setting medians: corrected {100 * (max(med) - min(med)) / np.mean(med):.1f}%, "
        f"small-angle {100 * (max(med0) - min(med0)) / np.mean(med0):.1f}%"
    )


def fall() -> None:
    by = defaultdict(list)
    for r in rows("fall.csv"):
        if r["status"] == "ok":
            by[r["clip"].split("\\")[0].split("/")[0] + "/" + r["setting"]].append(r)
    if not by:
        return
    print(
        "\nfall                    n  g time-of-flight      g ball-size scale   ref in CI"
    )
    for key in sorted(by):
        rs = by[key]
        tof = np.array([float(r["g_tof"]) for r in rs])
        tof = tof[np.isfinite(tof)]
        sc = [r for r in rs if r["g_scale"] not in ("", "nan")]
        gs = np.array([float(r["g_scale"]) for r in sc])
        inside = sum(float(r["ci_low"]) <= G <= float(r["ci_high"]) for r in sc)
        t = (
            f"{np.median(tof):.3f} ± {np.std(tof, ddof=1):.3f} ({len(tof)})"
            if len(tof) > 1
            else "-"
        )
        s = f"{np.median(gs):.3f} ± {np.std(gs, ddof=1):.3f}" if len(gs) > 1 else "-"
        print(
            f"{key:22s}{len(rs):3d}  {t:20s}  {s:18s}  {f'{inside}/{len(sc)}' if sc else '-'}"
        )


if __name__ == "__main__":
    pendulum()
    fall()
