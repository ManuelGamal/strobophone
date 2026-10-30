"""Counterfactual frame-rate test (after QuantiPhy's counterfactual priors): tell
Muse Spark the Apple-PI clips were filmed at 15 fps instead of 30. Frames are
then 1/15 s apart, so the motion implies g_true * (15/30)^2 = g/4. A measurement
scales with the stated fps; an answer recited from prior knowledge stays near 9.8.
StroboPhone's time-of-flight g is rescaled exactly by (15/30)^2 for comparison.

    .venv/Scripts/python eval/spark_counterfactual.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from apple_pi import OUT, real_cases  # noqa: E402
from spark_baseline import ask  # noqa: E402

CLAIM, TRUE_FPS = 15.0, 30.0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")
    client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])
    cases = real_cases()
    with ThreadPoolExecutor(4) as pool:
        answers = list(pool.map(lambda c: ask(client, c, False, CLAIM), cases))
    factor = (CLAIM / TRUE_FPS) ** 2
    spark = [b["g_m_per_s2"] for a in answers for b in a["answer"]["balls"]]
    tof = [float(r["g_tof"]) * factor for r in csv.DictReader(open(OUT / "tof.csv", encoding="utf-8")) if r.get("g_tof")]
    tokens = [sum(a["usage"][k] for a in answers if a.get("usage")) for k in ("input", "output")]
    print(f"stated fps {CLAIM:g} (true {TRUE_FPS:g}) -> implied g = 9.81 x {factor:g} = {9.81 * factor:.2f} m/s^2")
    print(f"Muse Spark alone : median g {np.median(spark):.2f}  IQR [{np.percentile(spark, 25):.2f}, {np.percentile(spark, 75):.2f}]  (n={len(spark)})")
    print(f"StroboPhone (ToF): median g {np.median(tof):.2f}  IQR [{np.percentile(tof, 25):.2f}, {np.percentile(tof, 75):.2f}]  (n={len(tof)})")
    print(f"Spark answers within 25% of 9.8: {np.mean(np.abs(np.array(spark) - 9.8) < 2.45):.0%}; "
          f"within 25% of {9.81 * factor:.2f}: {np.mean(np.abs(np.array(spark) - 9.81 * factor) < 0.61):.0%}")
    # Did Spark misreport what it saw to keep 9.8? Compare the fall duration it states
    # ("impact at ~8 frames") with the tracked one (single-ball clips in tof.csv).
    frames = {r["case"]: float(r["fall_frames"]) for r in csv.DictReader(open(OUT / "tof.csv", encoding="utf-8")) if r.get("fall_frames")}
    ratio = []
    for case, a in zip(cases, answers):
        for b in a["answer"]["balls"]:
            m = re.search(r"(?:impact|contact|land\w*|hit\w*|fall\w*)[^.]{0,40}?~?\s*(\d+(?:\.\d+)?)\s*frames", b["reasoning"], re.I)
            m = m or re.search(r"(\d+(?:\.\d+)?)\s*frames", b["reasoning"])
            if m and case.name in frames:
                ratio.append(float(m.group(1)) / frames[case.name])
    if ratio:
        print(f"Spark's stated fall duration / tracked: median {np.median(ratio):.2f} "
              f"IQR [{np.percentile(ratio, 25):.2f}, {np.percentile(ratio, 75):.2f}] on {len(ratio)} clips")
    print(f"tokens {tokens[0]:,} in / {tokens[1]:,} out (~${tokens[0] * 1.25e-6 + tokens[1] * 4.25e-6:.2f})")


if __name__ == "__main__":
    main()
