"""Counterfactual frame-rate test on a second experiment: Muse Spark watches real
IRIS pendulum clips (60 fps) and is told either the true rate or 30 fps. At
"30 fps" every swing takes twice as long, so the implied g is a quarter. We also
ask how many frames one full swing takes and compare with StroboPhone's tracked
period. IRIS (CC BY-NC 4.0) is used for evaluation only.

    .venv/Scripts/python eval/spark_pendulum_counterfactual.py [--refresh]
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
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

from iris_pendulum import DATA, OUT

MODEL = "muse-spark-1.3"
TRUE_FPS, CLAIM = 60.0, 30.0
CLIPS = [
    f"Pendulum/{s}/{t}.mp4"
    for s in ("pendulum_20", "pendulum_45", "pendulum_90")
    for t in ("01", "02")
]
SCHEMA = {
    "type": "object",
    "properties": {
        "frames_per_full_swing": {"type": "number"},
        "period_s": {"type": "number"},
        "g_m_per_s2": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["frames_per_full_swing", "period_s", "g_m_per_s2", "reasoning"],
    "additionalProperties": False,
}


def prompt(fps: float) -> str:
    return (
        f"This is a {fps:g} fps video of a simple pendulum. The string length from the pivot to the "
        "centre of the bob is 0.5 m. From what you see in the video, estimate (1) how many video frames "
        "one full swing (there and back) takes, (2) the period in seconds, and (3) the gravitational "
        "acceleration in m/s^2 implied by the motion. Give your best numerical estimates."
    )


def ask(client: OpenAI, clip: str, fps: float, refresh: bool) -> dict:
    cache = OUT / "spark" / f"{clip.replace('/', '_')[:-4]}_fps{fps:g}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    video = DATA.parent / clip
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
                    {"type": "input_text", "text": prompt(fps)},
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
    out = {"clip": clip, "fps_claim": fps, "answer": json.loads(r.output_text)}
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
    ours = {
        r["clip"].replace("\\", "/"): r
        for r in csv.DictReader((OUT / "pendulum.csv").open())
        if r["status"] == "ok"
    }
    jobs = [(c, f) for c in CLIPS for f in (TRUE_FPS, CLAIM)]
    with ThreadPoolExecutor(4) as pool:
        answers = list(pool.map(lambda j: ask(client, j[0], j[1], args.refresh), jobs))
    print(
        "clip                         told   Spark: frames/swing  g      StroboPhone: frames/swing  g"
    )
    for (clip, fps), a in zip(jobs, answers):
        o = ours[clip]
        frames = float(o["T_mean_s"]) * TRUE_FPS
        g_ours = (
            float(o["g"]) * (fps / TRUE_FPS) ** 2
        )  # the measurement scales exactly with the stated rate
        s = a["answer"]
        print(
            f"{clip:28s} {fps:4g}   {s['frames_per_full_swing']:8.1f}         {s['g_m_per_s2']:5.2f}"
            f"   {frames:8.1f}                {g_ours:5.2f}"
        )
    for fps in (TRUE_FPS, CLAIM):
        g = [a["answer"]["g_m_per_s2"] for (c, f), a in zip(jobs, answers) if f == fps]
        ratio = [
            a["answer"]["frames_per_full_swing"]
            / (float(ours[c]["T_mean_s"]) * TRUE_FPS)
            for (c, f), a in zip(jobs, answers)
            if f == fps
        ]
        print(
            f"told {fps:g} fps: Spark median g {np.median(g):.2f}, stated frames/swing / tracked {np.median(ratio):.2f}"
        )


if __name__ == "__main__":
    main()
