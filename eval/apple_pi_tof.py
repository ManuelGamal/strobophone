"""g from time of flight (g = 2h/T^2) on Apple-PI drops from rest: release and
impact times come from the SAM track at sub-frame precision; no pixel scale.
Motivated by the pixel-scale g results on this same set (see README notes), so
it is reported as a post-hoc method, not a pre-registered one.

    .venv/Scripts/python eval/apple_pi_tof.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from apple_pi import OUT, kind, matched_balls, real_cases  # noqa: E402
from strobophone.bounce import fall_time_frames  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    for case in real_cases():
        g_true, fps, _, _, balls, objs = matched_balls(case)
        for ball, (_, b) in zip(balls, objs):
            if ball["v_0"] != 0:
                continue
            try:
                t_rel, t_imp = fall_time_frames((b[:, 1] + b[:, 3]) / 2)
            except ValueError as exc:
                rows.append({"case": case.name, "ball": ball["name"], "note": str(exc)})
                continue
            T = (t_imp - t_rel) / fps
            rows.append({"case": case.name, "ball": ball["name"], "h_m": ball["h"], "fall_frames": round(t_imp - t_rel, 2),
                         "g_tof": round(2 * ball["h"] / T**2, 3), "g_true": g_true})
    with open(OUT / "tof.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case", "ball", "h_m", "fall_frames", "g_tof", "g_true", "note"])
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if "g_tof" in r]
    print(f"{len(rows)} drops from rest, {len(ok)} with a measurable bounce")
    print(f"{'ball':<13}{'n':>3}{'g_tof med':>11}{'IQR':>18}{'med|err|%':>11}")
    for name in sorted({kind(r["ball"]) for r in ok}) + ["ALL"]:
        g = np.array([r["g_tof"] for r in ok if name == "ALL" or kind(r["ball"]) == name])
        print(f"{name:<13}{len(g):>3}{np.median(g):>11.3f}{f'[{np.percentile(g, 25):.2f}, {np.percentile(g, 75):.2f}]':>18}"
              f"{np.median(abs(g - 9.81) / 9.81 * 100):>11.1f}")


if __name__ == "__main__":
    main()
