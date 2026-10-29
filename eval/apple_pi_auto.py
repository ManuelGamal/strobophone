"""End-to-end check of the one-command pipeline (strobophone.measure.measure_video)
on Apple-PI's 36 real FreeFall clips: automatic prep (field order, crop + zoom),
fallback concepts, track mapped back to original pixels.

    .venv/Scripts/python eval/apple_pi_auto.py [--refresh]

Per clip it reports what the pipeline chose, the tracking error of the chosen
ball against the human masks (in original pixels, so it tests the zoom and the
mapping back), g with the diameter scale, and the restitution e. Compare with
eval/apple_pi.py, which tracked at native resolution with no prep.
Apple-PI is CC BY 4.0 and used for evaluation only; nothing is tuned on it.
"""

from __future__ import annotations

import argparse
import hashlib
import csv
import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from apple_pi import OUT, mask_centroids, matched_balls, real_cases

from strobophone import sam
from strobophone.bounce import measure_restitution
from strobophone.calibration import Scale
from strobophone.drops import parse_apple_pi_annotation
from strobophone.measure import measure_video

AUTO = OUT / "auto"


def cached_tracker(case: str, refresh: bool):
    def tracker(video, concept, n_frames):
        # keyed on the prepared clip's bytes: a different crop/zoom needs a new SAM run
        digest = hashlib.sha1(Path(video).read_bytes()).hexdigest()[:10]
        cache = AUTO / f"{case}_{concept.replace(' ', '_')}_{digest}.json"
        if cache.exists() and not refresh:
            d = json.loads(cache.read_text())
        else:
            boxes = sam.track_boxes(video, concept, n_frames)
            d = {k: np.where(np.isnan(v), None, v).tolist() for k, v in boxes.items()}
            cache.write_text(json.dumps(d))
        return {k: np.array(v, float) for k, v in d.items()}

    return tracker


def track_error(boxes: np.ndarray, centroids: list[np.ndarray]) -> tuple[float, int]:
    """Mean distance from the box centre to the nearest human-mask centroid."""
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    err = []
    for i in range(min(len(cx), len(centroids))):
        if np.isnan(cx[i]) or not len(centroids[i]):
            continue
        d = np.hypot(centroids[i][:, 0] - cx[i], centroids[i][:, 1] - cy[i]).min()
        if d < 60:
            err.append(d)
    return (float(np.mean(err)) if err else float("nan")), len(err)


def evaluate(case: Path, refresh: bool) -> dict:
    md = json.loads((case / "metadata.json").read_text(encoding="utf-8"))
    g_true, balls = parse_apple_pi_annotation(md["annotation"])
    workdir = AUTO / case.name
    workdir.mkdir(parents=True, exist_ok=True)
    row = {
        "case": case.name,
        "balls": "+".join(b["name"] for b in balls),
        "g_true": g_true,
    }
    try:
        # Placeholder scale: g is recomputed below with the chosen ball's own scale.
        m = measure_video(
            case / "rgb" / "video.mp4",
            workdir,
            Scale(1.0, 0.0),
            tracker=cached_tracker(case.name, refresh),
        )
    except Exception as e:
        return {**row, "status": f"failed: {e}"}
    width = float(
        np.nanmedian(m.boxes[m.result.fit_frames, 2] - m.boxes[m.result.fit_frames, 0])
    )
    # Which annotated ball did the pipeline pick? Match its centre to the balls that
    # eval/apple_pi.py already matched (cached native-resolution SAM run, no API call).
    ball = balls[0] if len(balls) == 1 else None
    _, _, _, _, m_balls, objs = matched_balls(case)
    if objs:
        cx = np.nanmedian((m.boxes[:, 0] + m.boxes[:, 2]) / 2)
        dist = [abs(np.nanmedian((b[:, 0] + b[:, 2]) / 2) - cx) for _, b in objs]
        if min(dist) < 30:
            ball = m_balls[int(np.argmin(dist))]
    scale = width / ball["d"] if ball else float("nan")
    g = m.result.fit.g * 1.0 / scale if ball else float("nan")  # fit used 1 px/m
    err, n_err = track_error(
        m.boxes, mask_centroids(np.load(case / "mask" / "maps.npz")["maps"])
    )
    try:
        e = measure_restitution(m.y_px).e
    except ValueError:
        e = float("nan")
    return {
        **row,
        "status": "ok",
        "parity": m.prep.parity or "progressive",
        "zoom": m.prep.zoom,
        "concept": m.concept,
        "attempts": len(m.attempts),
        "ball": ball["name"] if ball else "?",
        "width_px": round(width, 1),
        "track_err_px": round(err, 2),
        "n_err": n_err,
        "fit_points": m.result.fit.n_points,
        "g_auto_d": round(g, 3),
        "e_true": ball["e"] if ball else None,
        "e_auto": round(e, 3),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    AUTO.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in real_cases():
        r = evaluate(case, args.refresh)
        print(r, flush=True)
        rows.append(r)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / "auto.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"\n{len(ok)}/{len(rows)} clips measured with no manual settings")
    print(
        "zoom:",
        {z: sum(r["zoom"] == z for r in ok) for z in sorted({r["zoom"] for r in ok})},
    )
    errs = [r["track_err_px"] for r in ok if np.isfinite(r["track_err_px"])]
    print(
        f"median track error vs human masks: {np.median(errs):.2f} px (n={len(errs)})"
    )


if __name__ == "__main__":
    main()
