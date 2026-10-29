"""Coefficient of restitution on Apple-PI's real FreeFall clips, against the
instrument-measured e in each case's annotation (two-drop test, +/-0.02).

    .venv/Scripts/python eval/apple_pi_restitution.py

e is a velocity ratio at the impact instant, so it needs neither the pixel scale
nor the frame rate -- the two unknowns that limit g on this dataset. Uses the
SAM boxes cached by eval/apple_pi.py. Evaluation only: nothing is tuned here.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from apple_pi import OUT, kind, matched_balls, real_cases

from strobophone.bounce import measure_restitution


def mask_track(maps: np.ndarray, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """Human-mask centroid y nearest to SAM's centre in each frame (NaN if none)."""
    y = np.full(len(maps), np.nan)
    for i, m in enumerate(maps):
        if np.isnan(cx[i]):
            continue
        lab, k = ndimage.label(m)
        if not k:
            continue
        c = np.array(ndimage.center_of_mass(m, lab, range(1, k + 1)))
        d = np.hypot(c[:, 1] - cx[i], c[:, 0] - cy[i])
        if d.min() < 60:
            y[i] = c[np.argmin(d), 0]
    return y


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    for case in real_cases():
        _, _, frames, _, balls, objs = matched_balls(case)
        maps = np.load(case / "mask" / "maps.npz")["maps"]
        for ball, (oid, b) in zip(balls, objs):
            cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
            row = {
                "case": case.name,
                "ball": ball["name"],
                "e_true": ball["e"],
                "frames": len(frames),
            }
            for label, y in (("sam", cy), ("mask", mask_track(maps, cx, cy))):
                try:
                    r = measure_restitution(y)
                    row[f"e_{label}"] = round(r.e, 3)
                    row[f"e_{label}_ci"] = f"[{r.ci[0]:.2f}, {r.ci[1]:.2f}]"
                    row[f"n_{label}"] = f"{r.n_before}+{r.n_after}"
                except ValueError as exc:
                    row[f"e_{label}"] = None
                    row[f"note_{label}"] = str(exc)
            rows.append(row)
            print({k: v for k, v in row.items() if v is not None}, flush=True)

    keys = [
        "case",
        "ball",
        "frames",
        "e_true",
        "e_sam",
        "e_sam_ci",
        "n_sam",
        "note_sam",
        "e_mask",
        "e_mask_ci",
        "n_mask",
        "note_mask",
    ]
    with open(OUT / "restitution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r.get("e_sam") is not None]
    print(f"\n{len(rows)} balls, {len(ok)} with a measurable bounce")
    print(
        f"{'ball':<14}{'n':>3}{'e_true':>8}{'e_sam med':>11}{'med|err|':>10}{'e_mask med':>12}{'med|err|':>10}"
    )
    for name in sorted({kind(r["ball"]) for r in ok}) + ["ALL"]:
        sel = [r for r in ok if name == "ALL" or kind(r["ball"]) == name]
        es = np.array([r["e_sam"] for r in sel])
        em = np.array([r["e_mask"] for r in sel if r.get("e_mask") is not None])
        et = np.array([r["e_true"] for r in sel])
        etm = np.array([r["e_true"] for r in sel if r.get("e_mask") is not None])
        true = f"{np.median(et):.2f}" if name != "ALL" else "-"
        print(
            f"{name:<14}{len(sel):>3}{true:>8}{np.median(es):>11.3f}{np.median(abs(es - et)):>10.3f}"
            f"{np.median(em):>12.3f}{np.median(abs(em - etm)):>10.3f}"
        )
    print(f"\nwrote {OUT / 'restitution.csv'}")


if __name__ == "__main__":
    main()
