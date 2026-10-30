"""Arabic lab reports: does Muse Spark keep the [[fact]] placeholders when it writes
Modern Standard Arabic, and does the checker still catch typed numbers (including
Arabic-Indic digits)? Same Apple-PI measurements as report_faithfulness.py.

    .venv/Scripts/python eval/report_arabic.py [-n 12] [--refresh]
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

from apple_pi import real_cases
from report_faithfulness import REPORTS, cached, measurement

from strobophone.report import facts_for, localize, untraced_numbers, write_report


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=12)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    client = OpenAI(
        base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"]
    )
    accepted = revisions = 0
    cases = real_cases()[: args.n]
    for case in cases:
        m, scale = measurement(case)
        facts = localize(facts_for(m, scale), "ar")
        r = cached(
            REPORTS / f"{case.name}_agent_ar.json",
            args.refresh,
            lambda: dataclasses.asdict(write_report(facts, client, language="ar")),
        )
        accepted += r["accepted"]
        revisions += r["revisions"]
        leftover = (
            untraced_numbers(r["text"], facts) if r["accepted"] else r["untraced"]
        )
        print(
            case.name,
            "accepted" if r["accepted"] else "BLOCKED",
            r["revisions"],
            leftover,
            flush=True,
        )
    print(f"\nArabic: {accepted}/{len(cases)} accepted, {revisions} revisions in total")
    print(
        "\nsample:\n"
        + next(
            (
                cached(REPORTS / f"{c.name}_agent_ar.json", False, lambda: None) or {}
            ).get("text", "")
            for c in cases
        )
    )


if __name__ == "__main__":
    main()
