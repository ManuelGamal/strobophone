"""'Muse Spark alone' baseline on Apple-PI's real FreeFall clips: the model watches
the video and estimates each ball's restitution and g, with the same facts
StroboPhone gets (fps, ball names, diameters, drop heights) and no measurement
tools. Answers are cached in outputs/apple_pi/spark_<case>.json.

    .venv/Scripts/python eval/spark_baseline.py [--refresh]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from apple_pi import OUT, kind, real_cases

from strobophone.drops import parse_apple_pi_annotation

MODEL = "muse-spark-1.3"
SCHEMA = {
    "type": "object",
    "properties": {
        "balls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "restitution": {"type": "number"},
                    "g_m_per_s2": {"type": "number"},
                    "reasoning": {"type": "string"},
                },
                "required": ["name", "restitution", "g_m_per_s2", "reasoning"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["balls"],
    "additionalProperties": False,
}


def prompt(fps: float, balls: list[dict]) -> str:
    listing = "; ".join(
        f"{b['name']} (diameter {b['d']} m, released {b['h']} m above the floor, "
        f"initial downward speed {b['v_0']} m/s)"
        for b in balls
    )
    return (
        f"This is a {fps:g} fps video of a physics lab drop. Balls: {listing}. "
        "For each ball, estimate from what you see in the video: (1) its coefficient of restitution "
        "for its first bounce on this floor (rebound speed / impact speed), and (2) the gravitational "
        "acceleration in m/s^2 implied by its fall. Give your best numerical estimates."
    )


def ask(client: OpenAI, case: Path, refresh: bool, fps_claim: float | None = None) -> dict:
    """Spark's estimates for one clip. `fps_claim` overrides the frame rate stated in
    the prompt (a counterfactual: the implied g then scales with fps_claim^2)."""
    suffix = f"_fps{fps_claim:g}" if fps_claim else ""
    cache = OUT / f"spark_{case.name}{suffix}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    md = json.loads((case / "metadata.json").read_text(encoding="utf-8"))
    _, balls = parse_apple_pi_annotation(md["annotation"])
    video = case / "rgb" / "video.mp4"
    up = client.files.create(
        file=(video.name, video.read_bytes(), "video/mp4"), purpose="user_data"
    )
    r = client.responses.create(
        model=MODEL,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt(fps_claim or md["gt_fps"], balls)},
                    {"type": "input_video", "file_id": up.id},
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "estimates",
                "schema": SCHEMA,
                "strict": True,
            }
        },
    )
    out = {
        "case": case.name,
        "answer": json.loads(r.output_text),
        "usage": {"input": r.usage.input_tokens, "output": r.usage.output_tokens}
        if r.usage
        else None,
    }
    cache.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    client = OpenAI(
        base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"]
    )

    cases = real_cases()
    with ThreadPoolExecutor(4) as pool:
        answers = list(pool.map(lambda c: ask(client, c, args.refresh), cases))

    measured = {
        (r["case"], r["ball"]): r
        for r in csv.DictReader(open(OUT / "restitution.csv", encoding="utf-8"))
    }
    rows, tokens = [], [0, 0]
    for case, a in zip(cases, answers):
        if a.get("usage"):
            tokens[0] += a["usage"]["input"]
            tokens[1] += a["usage"]["output"]
        md = json.loads((case / "metadata.json").read_text(encoding="utf-8"))
        g_true, balls = parse_apple_pi_annotation(md["annotation"])
        est = {e["name"]: e for e in a["answer"]["balls"]}
        for b in balls:
            e = est.get(b["name"])
            m = measured.get((case.name, b["name"]), {})
            rows.append(
                {
                    "case": case.name,
                    "ball": b["name"],
                    "e_annot": b["e"],
                    "e_strobophone": float(m["e_sam"]) if m.get("e_sam") else None,
                    "e_spark": e["restitution"] if e else None,
                    "g_true": g_true,
                    "g_spark": e["g_m_per_s2"] if e else None,
                }
            )

    with open(OUT / "spark_baseline.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(
        f"{len(cases)} clips, {len(rows)} balls; Spark tokens {tokens[0]:,} in / {tokens[1]:,} out "
        f"(~${tokens[0] * 1.25e-6 + tokens[1] * 4.25e-6:.2f})"
    )
    print(
        f"{'ball':<13}{'n':>3}{'e_annot':>8}{'e_StroboPhone':>15}{'e_Spark':>9}{'Spark g med':>13}{'Spark g IQR':>16}"
    )
    ok = [
        r for r in rows if r["e_spark"] is not None and r["e_strobophone"] is not None
    ]
    for name in sorted({kind(r["ball"]) for r in ok}) + ["ALL"]:
        sel = [r for r in ok if name == "ALL" or kind(r["ball"]) == name]
        gs = np.array([r["g_spark"] for r in sel])
        ann = f"{sel[0]['e_annot']:.2f}" if name != "ALL" else "-"
        print(
            f"{name:<13}{len(sel):>3}{ann:>8}{np.median([r['e_strobophone'] for r in sel]):>15.3f}"
            f"{np.median([r['e_spark'] for r in sel]):>9.3f}{np.median(gs):>13.2f}"
            f"{f'[{np.percentile(gs, 25):.2f}, {np.percentile(gs, 75):.2f}]':>16}"
        )
    # Agreement with StroboPhone's measurement on balls where it matches the annotation (tennis, baseball).
    trusted = [r for r in ok if kind(r["ball"]) in ("TennisBall", "Baseball")]
    if trusted:
        es = np.array([abs(r["e_spark"] - r["e_annot"]) for r in trusted])
        em = np.array([abs(r["e_strobophone"] - r["e_annot"]) for r in trusted])
        print(
            f"\ntennis+baseball (n={len(trusted)}): median |e error| StroboPhone {np.median(em):.3f} vs Spark alone {np.median(es):.3f}"
        )
    print(f"wrote {OUT / 'spark_baseline.csv'}")


if __name__ == "__main__":
    main()
