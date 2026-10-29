"""SAM 3.1 video tracking through the Meta Model API (Files API upload +
streamed Responses call, parsed with Meta's meta_sam_parser)."""

from __future__ import annotations

import asyncio
import os
from contextlib import aclosing
from pathlib import Path

import numpy as np
from meta_sam_parser import parse_responses_stream, video_segmentation_format
from openai import AsyncOpenAI

BASE_URL = "https://api.meta.ai/v1"
MODEL = "sam-3.1"


async def _segment(video: Path, concept: str, api_key: str):
    async with AsyncOpenAI(base_url=BASE_URL, api_key=api_key) as api:
        up = await api.files.create(
            file=(video.name, video.read_bytes(), "video/mp4"), purpose="user_data"
        )
        stream = await api.responses.create(
            model=MODEL,
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

        # Same event adaptation as Meta's cookbook (07_sam/01_api_basics/python/sam_api.py):
        # the API streams content_part.done but not output_text.done, which the parser needs.
        async def events():
            done = set()
            async for event in stream:
                data = event.model_dump()
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

        async with stream, aclosing(events()) as evs:
            parsed = parse_responses_stream(evs, video_segmentation_format())
            async with parsed:
                async for _ in parsed:
                    pass
                return await parsed.final_result()


def boxes_from_records(records, n_frames: int) -> dict[str, np.ndarray]:
    """Per-object (n_frames, 4) arrays of (left, top, right, bottom); NaN where absent."""
    boxes: dict[str, np.ndarray] = {}
    for r in records:
        if r.kind != "box" or r.frame is None:
            continue
        arr = boxes.setdefault(r.object_id, np.full((n_frames, 4), np.nan))
        arr[r.frame.frame_index] = (r.left, r.top, r.right, r.bottom)
    return boxes


def track_boxes(
    video: Path, concept: str, n_frames: int, api_key: str | None = None
) -> dict[str, np.ndarray]:
    """SAM 3.1 boxes per object for `concept`. Empty dict if nothing matched."""
    key = api_key or os.environ["MODEL_API_KEY"]
    result = asyncio.run(_segment(video, concept, key))
    if result.outcome.status != "completed":
        raise RuntimeError(f"SAM 3.1 did not complete: {result.outcome}")
    return boxes_from_records(result.records, n_frames)


def track(
    video: Path, concept: str, n_frames: int, api_key: str | None = None
) -> dict[str, np.ndarray]:
    """Per-object (n_frames, 2) arrays of box centres (x, y) in pixels; NaN where
    the object was not found. Empty dict if SAM matched nothing for `concept`."""
    return {
        oid: np.column_stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2])
        for oid, b in track_boxes(video, concept, n_frames, api_key).items()
    }
