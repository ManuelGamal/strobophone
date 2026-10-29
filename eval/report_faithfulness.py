"""Does the lab report say only what was measured? On Apple-PI's 36 real clips,
Muse Spark writes the results section twice from the same facts:
  free   - Spark types the numbers itself (baseline, no checker)
  agent  - StroboPhone: placeholders + checker + up to 2 revisions
and the checker counts numbers that no tool output accounts for.

    .venv/Scripts/python eval/report_faithfulness.py [--refresh]

Apple-PI's g values are biased low (30 fps, rounded annotations), which makes
this a stress test: a model that "knows" g = 9.81 is tempted to say so.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

from apple_pi import OUT, real_cases
from apple_pi_auto import AUTO, cached_tracker

from strobophone.calibration import Scale
from strobophone.drops import parse_apple_pi_annotation
from strobophone.measure import measure_video
from strobophone.pipeline import measure_free_fall
from strobophone.report import (
    facts_for,
    untraced_numbers,
    write_free_report,
    write_report,
)

REPORTS = OUT / "report"
TEXTBOOK = {"9.8", "9.81", "9.807", "9.80665", "10"}
BOX_WIDTH_UNC = 0.05  # assumed scale uncertainty of a box-width diameter scale


def measurement(case: Path):
    """The pipeline's measurement with the diameter scale of the ball it tracked."""
    md = json.loads((case / "metadata.json").read_text(encoding="utf-8"))
    _, balls = parse_apple_pi_annotation(md["annotation"])
    tracker = cached_tracker(case.name, refresh=False)
    m = measure_video(
        case / "rgb" / "video.mp4", AUTO / case.name, Scale(1.0, 0.0), tracker=tracker
    )
    width = float(
        np.nanmedian(m.boxes[m.result.fit_frames, 2] - m.boxes[m.result.fit_frames, 0])
    )
    d = balls[0]["d"] if len({b["d"] for b in balls}) == 1 else None
    if d is None:  # two different balls: take the one whose size ratio matches best
        d = min(
            balls,
            key=lambda b: abs(
                width / b["d"] - np.median([width / x["d"] for x in balls])
            ),
        )["d"]
    scale = Scale(width / d, BOX_WIDTH_UNC)
    return dataclasses.replace(m, result=measure_free_fall(m.y_px, m.fps, scale)), scale


def cached(path: Path, refresh: bool, make):
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    out = make()
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
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
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in real_cases():
        m, scale = measurement(case)
        facts = facts_for(m, scale)
        free = cached(
            REPORTS / f"{case.name}_free.json",
            args.refresh,
            lambda: {"text": write_free_report(facts, client)},
        )
        agent = cached(
            REPORTS / f"{case.name}_agent.json",
            args.refresh,
            lambda: dataclasses.asdict(write_report(facts, client)),
        )
        bad_free = untraced_numbers(free["text"], facts)
        bad_agent = (
            untraced_numbers(agent["text"], facts)
            if agent["accepted"]
            else agent["untraced"]
        )
        lo, hi = m.result.g_ci_with_scale
        rows.append(
            {
                "case": case.name,
                "g": round(m.result.fit.g, 3),
                "ref_in_ci": lo <= 9.79 <= hi,
                "free_untraced": " ".join(bad_free),
                "free_n_untraced": len(bad_free),
                "free_textbook": sum(t in TEXTBOOK for t in bad_free),
                "agent_accepted": agent["accepted"],
                "agent_revisions": agent["revisions"],
                "agent_n_untraced": len(bad_agent),
            }
        )
        print(rows[-1], flush=True)
    with (OUT / "report_faithfulness.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    print(
        f"\nfree Spark: {sum(r['free_n_untraced'] > 0 for r in rows)}/{n} reports contain untraced numbers "
        f"({sum(r['free_n_untraced'] for r in rows)} in total, {sum(r['free_textbook'] for r in rows)} textbook g values)"
    )
    print(
        f"StroboPhone agent: {sum(r['agent_accepted'] for r in rows)}/{n} accepted, "
        f"{sum(r['agent_n_untraced'] for r in rows)} untraced numbers in accepted text, "
        f"revisions {sum(r['agent_revisions'] for r in rows)}"
    )


if __name__ == "__main__":
    main()
