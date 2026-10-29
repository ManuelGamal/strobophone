"""Evaluate StroboPhone's free-fall measurement on Apple-PI's 36 real FreeFall clips.

    .venv/Scripts/python eval/apple_pi.py            # uses cached SAM boxes when present
    .venv/Scripts/python eval/apple_pi.py --refresh  # re-query SAM 3.1

Apple-PI (CC BY 4.0, https://huggingface.co/datasets/yaorunmao/Apple-PI-GT) is an
evaluation set: nothing here is tuned on it. Per ball it reports g with three
scale/track combinations, so tracking error and scale error can be told apart:
  g_sam_d   SAM 3.1 track, scale from the annotated ball diameter (box width)
  g_mask_d  human-mask track, same scale  -> isolates SAM's tracking error
  g_sam_h   SAM 3.1 track, scale from the annotated drop height
plus SAM's centroid error against the human masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strobophone import sam
from strobophone.baseline import read_video
from strobophone.drops import contact_frame, parse_apple_pi_annotation, scale_from_drop
from strobophone.fits import detect_release, fit_free_fall

DATA = ROOT / "data" / "apple_pi" / "cases" / "UniversalGravitation" / "FreeFall"
OUT = ROOT / "outputs" / "apple_pi"
CONCEPTS = ["ball", "sphere", "round object"]
MIN_SPAN_PX = 100  # a falling ball moves; the hand and static clutter do not


def sam_boxes(
    case: Path, video: Path, n: int, refresh: bool
) -> tuple[str, dict[str, np.ndarray]]:
    cache = OUT / f"{case.name}_sam.json"
    if cache.exists() and not refresh:
        d = json.loads(cache.read_text())
        return d["concept"], {k: np.array(v, float) for k, v in d["boxes"].items()}
    for concept in CONCEPTS:
        boxes = sam.track_boxes(video, concept, n)
        if boxes:
            break
    cache.write_text(
        json.dumps(
            {
                "concept": concept,
                "boxes": {
                    k: np.where(np.isnan(v), None, v).tolist() for k, v in boxes.items()
                },
            }
        )
    )
    return concept, {k: np.array(v, float) for k, v in boxes.items()}


def mask_centroids(maps: np.ndarray) -> list[np.ndarray]:
    """Per frame, an (k, 2) array of (x, y) centroids of the human-mask components."""
    out = []
    for m in maps:
        lab, k = ndimage.label(m)
        out.append(
            np.array(ndimage.center_of_mass(m, lab, range(1, k + 1)))[:, ::-1]
            if k
            else np.empty((0, 2))
        )
    return out


def fit_g(y: np.ndarray, release: int, contact: int, fps: float, px_per_m: float):
    idx = np.array([i for i in range(release, contact) if not np.isnan(y[i])])
    if len(idx) < 4:
        return None
    return fit_free_fall((idx - release) / fps, y[idx], px_per_m)


def matched_balls(case: Path, refresh: bool = False):
    """(g_true, fps, frames, concept, balls, objs): SAM objects matched one-to-one to
    the annotated balls, or objs=[] when the counts differ."""
    md = json.loads((case / "metadata.json").read_text(encoding="utf-8"))
    g_true, balls = parse_apple_pi_annotation(md["annotation"])
    video = case / "rgb" / "video.mp4"
    frames, fps = read_video(video)
    concept, boxes = sam_boxes(case, video, len(frames), refresh)
    moving = {
        k: b
        for k, b in boxes.items()
        if np.count_nonzero(~np.isnan(b[:, 3])) >= 4
        and np.nanmax(b[:, 3]) - np.nanmin(b[:, 3]) > MIN_SPAN_PX
    }
    # Match SAM objects to annotated balls: by size when diameters differ, else left-to-right
    # (names say Left/Right, or the listed order -- an assumption for unnamed pairs).
    objs = sorted(
        moving.items(), key=lambda kv: np.nanmedian((kv[1][:, 0] + kv[1][:, 2]) / 2)
    )
    if len(balls) == 2 and len(objs) == 2 and balls[0]["d"] != balls[1]["d"]:
        width = lambda b: np.nanmedian(b[:, 2] - b[:, 0])
        objs = sorted(objs, key=lambda kv: width(kv[1]))
        balls = sorted(balls, key=lambda b: b["d"])
    return g_true, fps, frames, concept, balls, objs if len(objs) == len(balls) else []


def real_cases() -> list[Path]:
    return [
        c
        for c in sorted(DATA.iterdir())
        if json.loads((c / "metadata.json").read_text(encoding="utf-8")).get(
            "is_realworld"
        )
    ]


def evaluate_case(case: Path, refresh: bool) -> list[dict]:
    g_true, fps, frames, concept, balls, objs = matched_balls(case, refresh)
    centroids = mask_centroids(np.load(case / "mask" / "maps.npz")["maps"])
    rows = []
    for ball, obj in zip(balls, objs) if len(objs) == len(balls) else []:
        oid, b = obj
        cx, cy, bottom = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 3]
        tracked = np.nonzero(~np.isnan(cy))[0]
        release = int(tracked[detect_release(cy[tracked])])
        contact = contact_frame(bottom, release)
        in_flight = [i for i in range(release, contact) if not np.isnan(cy[i])]

        # Human-mask track for the same ball: nearest component to SAM's centre each frame.
        mask_y, err = np.full(len(frames), np.nan), []
        for i in range(len(frames)):
            if np.isnan(cx[i]) or not len(centroids[i]):
                continue
            dist = np.hypot(centroids[i][:, 0] - cx[i], centroids[i][:, 1] - cy[i])
            j = int(np.argmin(dist))
            if dist[j] < 60:
                mask_y[i] = centroids[i][j, 1]
                err.append(dist[j])

        width_px = float(np.nanmedian(b[in_flight, 2] - b[in_flight, 0]))
        s_d = width_px / ball["d"]
        try:
            s_h = scale_from_drop(
                bottom[release], np.nanmax(bottom[release:]), ball["h"]
            )
        except ValueError:
            s_h = np.nan
        fits = {
            "g_sam_d": fit_g(cy, release, contact, fps, s_d),
            "g_mask_d": fit_g(mask_y, release, contact, fps, s_d),
            "g_sam_h": fit_g(cy, release, contact, fps, s_h)
            if np.isfinite(s_h)
            else None,
        }
        rows.append(
            {
                "case": case.name,
                "ball": ball["name"],
                "d_m": ball["d"],
                "h_m": ball["h"],
                "v0": ball["v_0"],
                "concept": concept,
                "frames_tracked": len(tracked),
                "fit_points": len(in_flight),
                "track_err_px": round(float(np.mean(err)), 2) if err else None,
                "ball_width_px": round(width_px, 1),
                "g_true": g_true,
                **{k: (round(f.g, 3) if f else None) for k, f in fits.items()},
                "g_sam_d_ci": f"[{fits['g_sam_d'].ci_low:.2f}, {fits['g_sam_d'].ci_high:.2f}]"
                if fits["g_sam_d"]
                else None,
            }
        )
    if not rows:
        rows.append(
            {
                "case": case.name,
                "ball": "+".join(b["name"] for b in balls),
                "concept": concept,
                "frames_tracked": 0,
                "note": f"SAM moving objects did not match the {len(balls)} annotated balls",
            }
        )
    return rows


def kind(name: str) -> str:
    return name.replace("Left", "").replace("Right", "")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    OUT.mkdir(parents=True, exist_ok=True)

    cases = [
        c
        for c in sorted(DATA.iterdir())
        if json.loads((c / "metadata.json").read_text(encoding="utf-8")).get(
            "is_realworld"
        )
    ]
    rows = []
    for c in cases:
        r = evaluate_case(c, args.refresh)
        rows += r
        for x in r:
            print(
                {
                    k: x.get(k)
                    for k in (
                        "case",
                        "ball",
                        "frames_tracked",
                        "fit_points",
                        "track_err_px",
                        "g_sam_d",
                        "g_mask_d",
                        "g_sam_h",
                        "note",
                    )
                    if x.get(k) is not None
                },
                flush=True,
            )

    keys = sorted(
        {k for r in rows for k in r},
        key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99,
    )
    with open(OUT / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r.get("g_sam_d") is not None]
    print(
        f"\n{len(cases)} clips, {len([r for r in rows if r.get('frames_tracked')])} balls tracked, {len(ok)} fitted"
    )
    print(
        f"{'ball':<14}{'n':>3}{'track err px':>14}"
        + "".join(f"{k + ' med|err|%':>20}" for k in ("g_sam_d", "g_mask_d", "g_sam_h"))
    )
    for name in sorted({kind(r["ball"]) for r in ok}) + ["ALL"]:
        sel = [r for r in ok if name == "ALL" or kind(r["ball"]) == name]
        te = [r["track_err_px"] for r in sel if r.get("track_err_px") is not None]
        cols = []
        for k in ("g_sam_d", "g_mask_d", "g_sam_h"):
            e = [
                abs(r[k] - r["g_true"]) / r["g_true"] * 100
                for r in sel
                if r.get(k) is not None
            ]
            cols.append(f"{np.median(e):>14.1f} (n={len(e)})" if e else f"{'-':>20}")
        print(
            f"{name:<14}{len(sel):>3}{np.median(te) if te else float('nan'):>14.2f}"
            + "".join(f"{c:>20}" for c in cols)
        )
    print(f"\nwrote {OUT / 'results.csv'}")


if __name__ == "__main__":
    main()
