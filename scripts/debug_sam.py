"""Dump SAM 3.1's raw stream and parsed records for one video + concept.

.venv/Scripts/python scripts/debug_sam.py VIDEO "noun phrase"
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from meta_sam_parser import parse_responses_stream, video_segmentation_format
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "smoke"


async def main(video: Path, concept: str) -> None:
    load_dotenv(ROOT / ".env")
    api = AsyncOpenAI(
        base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"]
    )
    up = await api.files.create(
        file=(video.name, video.read_bytes(), "video/mp4"), purpose="user_data"
    )
    stream = await api.responses.create(
        model="sam-3.1",
        stream=True,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": concept},
                    {"type": "input_video", "file_id": up.id},
                ],
            }
        ],
    )
    events = [e.model_dump() for e in [e async for e in stream]]
    log = OUT / f"sam_events_{video.stem}.jsonl"
    log.write_text(
        "\n".join(json.dumps(e, default=str) for e in events), encoding="utf-8"
    )
    print("event types:", Counter(e.get("type") for e in events))
    text = "".join(
        e.get("delta", "")
        for e in events
        if e.get("type") == "response.output_text.delta"
    )
    print(f"output text: {len(text)} chars; head: {text[:600]!r}")
    final = [
        e
        for e in events
        if e.get("type")
        in ("response.completed", "response.incomplete", "response.failed")
    ]
    if final:
        resp = final[-1].get("response", {})
        print(
            "final status:",
            resp.get("status"),
            "| usage:",
            resp.get("usage"),
            "| error:",
            resp.get("error"),
            "| incomplete:",
            resp.get("incomplete_details"),
        )

    # Parse the same events the way the smoke test does.
    async def replay():
        done = set()
        for data in events:
            lane = (
                data.get("item_id"),
                data.get("output_index"),
                data.get("content_index"),
            )
            if data.get("type") == "response.output_text.done":
                if lane not in done:
                    done.add(lane)
                    yield data
                continue
            yield data
            part = data.get("part")
            if (
                data.get("type") == "response.content_part.done"
                and lane not in done
                and isinstance(part, dict)
                and part.get("type") == "output_text"
            ):
                done.add(lane)
                yield {
                    "type": "response.output_text.done",
                    "item_id": data.get("item_id"),
                    "output_index": data.get("output_index"),
                    "content_index": data.get("content_index"),
                    "text": part["text"],
                }

    parsed = parse_responses_stream(replay(), video_segmentation_format())
    async with parsed:
        async for _ in parsed:
            pass
        result = await parsed.final_result()
    print("parsed outcome:", result.outcome, "| media:", result.media)
    print("record kinds:", Counter(r.kind for r in result.records))
    print(
        "diagnostics:",
        [(d.severity, d.code, d.message) for d in result.diagnostics][:10],
    )
    print("raw log:", log)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main(Path(sys.argv[1]), sys.argv[2]))
