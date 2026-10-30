"""Amplitude test inside each real IRIS pendulum clip, independent of the annotated
release angle and of the rope-length definition: as the swing dies down, g from
each full cycle should stay flat after the exact amplitude correction, while the
small-angle formula drifts. Uses the cached SAM tracks from eval/iris_pendulum.py.

    .venv/Scripts/python eval/iris_pendulum_cycles.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import special, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

from apple_pi_auto import cached_tracker
from iris_pendulum import DATA, OUT

from strobophone.measure import PENDULUM_CONCEPTS, horizontal_span, track_video
from strobophone.pendulum import swings


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    root = DATA.parent
    slopes = defaultdict(list)
    for r in csv.DictReader((OUT / "pendulum.csv").open()):
        if r["status"] != "ok":
            continue
        rel = Path(r["clip"].replace("\\", "/"))
        name = "iris_" + "_".join(rel.with_suffix("").parts)
        tr = track_video(
            root / rel,
            OUT / name,
            cached_tracker(name, False),
            PENDULUM_CONCEPTS,
            pick=horizontal_span,
        )
        x, y = (
            (tr.boxes[:, 0] + tr.boxes[:, 2]) / 2,
            (tr.boxes[:, 1] + tr.boxes[:, 3]) / 2,
        )
        s = swings(np.arange(len(x)) / tr.fps, x, y)
        T, amp = np.array([p for p, _ in s]), np.array([a for _, a in s])
        L = float(r["L_m"])
        g_corr = (
            4
            * np.pi**2
            * L
            / (T * np.pi / (2 * special.ellipk(np.sin(amp / 2) ** 2))) ** 2
        )
        g_small = 4 * np.pi**2 * L / T**2
        deg = np.degrees(amp)
        slopes[r["setting"]].append(
            (
                deg.max(),
                deg.min(),
                10 * stats.theilslopes(g_corr, deg)[0],
                10 * stats.theilslopes(g_small, deg)[0],
            )
        )
    print(
        "change in per-cycle g for every 10 deg of amplitude (m/s^2), as the swing decays"
    )
    print("setting       n  amplitude range   corrected        small-angle")
    for key in sorted(slopes, key=lambda k: float(k.split("_")[-1])):
        a = np.array(slopes[key])
        print(
            f"{key:12s}{len(a):3d}  {np.median(a[:, 0]):5.1f} -> {np.median(a[:, 1]):4.1f} deg   "
            f"{np.median(a[:, 2]):+.3f}           {np.median(a[:, 3]):+.3f}"
        )


if __name__ == "__main__":
    main()
