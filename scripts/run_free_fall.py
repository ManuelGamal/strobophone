"""Measure g from a real free-fall video with SAM 3.1 tracking.

    .venv/Scripts/python scripts/run_free_fall.py VIDEO --concept ball --px-per-m 400         [--deinterlace tff|bff] [--crop W:H:X:Y] [--zoom 4]

--deinterlace splits interlaced frames into fields (yadif, double rate); the
field order matters: the wrong one makes the track zigzag. --crop/--zoom
magnify a small object for SAM (it misses a ~7 px ball); the scale is
multiplied by --zoom automatically.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strobophone import sam
from strobophone.baseline import read_video
from strobophone.calibration import Scale
from strobophone.pipeline import measure_free_fall


def preprocess(video: Path, parity: str | None, crop: str | None, zoom: float) -> Path:
    filters = []
    if parity:
        filters.append(f"yadif=mode=1:parity={parity}")
    if crop:
        filters.append(f"crop={crop}")
    if zoom != 1:
        filters.append(f"scale=iw*{zoom}:ih*{zoom}:flags=lanczos")
    if not filters:
        return video
    out = ROOT / "outputs" / f"{video.stem}_prepped.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-i", str(video),
                    "-vf", ",".join(filters), "-c:v", "libx264", "-crf", "12", str(out)], check=True)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--concept", default="ball")
    ap.add_argument("--px-per-m", type=float, required=True)
    ap.add_argument("--scale-rel-unc", type=float, default=0.0)
    ap.add_argument("--deinterlace", choices=["tff", "bff"])
    ap.add_argument("--crop", help="ffmpeg crop W:H:X:Y, applied after deinterlacing")
    ap.add_argument("--zoom", type=float, default=1.0)
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")

    video = preprocess(args.video, args.deinterlace, args.crop, args.zoom)
    frames, fps = read_video(video)
    tracks = sam.track(video, args.concept, len(frames))
    if not tracks:
        sys.exit(f"SAM 3.1 found nothing for {args.concept!r}")
    for oid, xy in tracks.items():
        print(
            f"object {oid}: {np.count_nonzero(~np.isnan(xy[:, 1]))}/{len(frames)} frames"
        )
    # The falling object is the one that moves furthest vertically.
    oid, xy = max(
        tracks.items(), key=lambda kv: np.nanmax(kv[1][:, 1]) - np.nanmin(kv[1][:, 1])
    )
    y = xy[:, 1]
    print("y track (px):", np.round(y, 1).tolist())

    m = measure_free_fall(y, fps, Scale(args.px_per_m * args.zoom, args.scale_rel_unc))
    lo, hi = m.g_ci_with_scale
    print(
        f"{video.name} @ {fps:g} fps, object {oid}, release frame {m.release_frame}, "
        f"{m.fit.n_points} points, residual {m.fit.residual_rms_px:.2f} px"
    )
    print(f"g = {m.fit.g:.3f} m/s^2, 95% CI [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
