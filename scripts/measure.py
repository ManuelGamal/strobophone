"""Measure g from a raw free-fall clip with no manual preprocessing flags.

    .venv/Scripts/python scripts/measure.py VIDEO --px-per-m 400 [--scale-rel-unc 0.02]
    .venv/Scripts/python scripts/measure.py VIDEO --reference A4

Field order, crop + zoom and the SAM concept phrase are chosen automatically
and printed, so every choice is visible in the output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strobophone.baseline import find_white_sheet, read_video
from strobophone.calibration import Scale, scale_from_reference
from strobophone.measure import measure_video


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--px-per-m", type=float, help="scale in original-video pixels")
    group.add_argument("--reference", choices=["A4", "letter"], help="white sheet visible in the first frame")
    ap.add_argument("--scale-rel-unc", type=float, default=0.0)
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")

    if args.reference:
        _, _, w, h = find_white_sheet(read_video(args.video)[0][0])
        scale = scale_from_reference(w, h, args.reference)
    else:
        scale = Scale(args.px_per_m, args.scale_rel_unc)

    workdir = ROOT / "outputs" / "measure" / args.video.stem
    workdir.mkdir(parents=True, exist_ok=True)
    m = measure_video(args.video, workdir, scale)
    p = m.prep
    print(f"field order: {p.parity or 'progressive'} -> {m.fps:g} fps")
    print(f"crop {p.crop} zoom {p.zoom:g}x" if p.crop else "no crop/zoom needed")
    print("concepts tried:", ", ".join(f"{c!r} ({r})" for c, r in m.attempts))
    r = m.result
    lo, hi = r.g_ci_with_scale
    print(f"object {m.object_id}: {np.count_nonzero(~np.isnan(m.y_px))}/{len(m.y_px)} frames, "
          f"release frame {r.release_frame}, {r.fit.n_points} points, residual {r.fit.residual_rms_px:.2f} px")
    print(f"scale {scale.px_per_m:.1f} px/m (+/-{100 * scale.rel_uncertainty:.1f}%)")
    print(f"g = {r.fit.g:.3f} m/s^2, 95% CI [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
