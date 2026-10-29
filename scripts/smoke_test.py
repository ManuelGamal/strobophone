"""Smoke-test every Meta model StroboPhone depends on, from this machine.

    .venv/Scripts/python scripts/smoke_test.py            # all checks
    .venv/Scripts/python scripts/smoke_test.py sam voice  # a subset

Checks: spark (text + structured output), spark_video (Files API upload + video
understanding), sam (SAM 3.1 video tracking -> free-fall fit), voice (Muse Voice
Transcribe with keyword biasing). Test media is synthesized locally, so the
expected answers are known. Raw responses go to outputs/smoke/.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from contextlib import aclosing
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from meta_sam_parser import parse_responses_stream, video_segmentation_format
from openai import AsyncOpenAI, OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strobophone.calibration import scale_from_reference  # noqa: E402
from strobophone.pipeline import measure_free_fall  # noqa: E402
from strobophone.synthetic import FreeFallTruth, make_free_fall_video  # noqa: E402

OUT = ROOT / "outputs" / "smoke"
BASE_URL = "https://api.meta.ai/v1"
SPARK = "muse-spark-1.3"
SAM = "sam-3.1"
VOICE = "muse-voice-transcribe-1.0"

SPOKEN = "String length eighty centimetres. Hanging mass fifty grams. Frequency one hundred hertz."


def key() -> str:
    load_dotenv(ROOT / ".env")
    value = os.environ.get("MODEL_API_KEY")
    if not value:
        sys.exit(
            "MODEL_API_KEY is not set. Put it in .env (gitignored) as MODEL_API_KEY=..."
        )
    return value


def make_speech_wav(path: Path) -> None:
    """Windows SAPI text-to-speech -> 24 kHz mono 16-bit WAV."""
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(24000,"
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,"
        "[System.Speech.AudioFormat.AudioChannel]::Mono);"
        f"$s.SetOutputToWaveFile('{path}', $f); $s.Speak('{SPOKEN}'); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


def save(name: str, data: object) -> None:
    (OUT / name).write_text(
        json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8"
    )


def check_spark(api_key: str) -> str:
    client = OpenAI(base_url=BASE_URL, api_key=api_key)
    schema = {
        "type": "object",
        "properties": {
            "quantity": {"type": "string"},
            "value": {"type": "number"},
            "unit": {"type": "string"},
        },
        "required": ["quantity", "value", "unit"],
        "additionalProperties": False,
    }
    r = client.responses.create(
        model=SPARK,
        input="What is the standard acceleration due to gravity? Answer with the quantity, value and SI unit.",
        text={
            "format": {
                "type": "json_schema",
                "name": "measurement",
                "schema": schema,
                "strict": True,
            }
        },
    )
    save("spark.json", r.model_dump())
    parsed = json.loads(r.output_text)
    return f"structured output ok: {parsed} | usage {r.usage.input_tokens} in / {r.usage.output_tokens} out"


def check_spark_video(api_key: str, video: Path) -> str:
    client = OpenAI(base_url=BASE_URL, api_key=api_key)
    up = client.files.create(
        file=(video.name, video.read_bytes(), "video/mp4"), purpose="user_data"
    )
    schema = {
        "type": "object",
        "properties": {
            "experiment": {"type": "string"},
            "tracked_object": {"type": "string"},
            "scale_reference": {"type": "string"},
            "notes": {"type": "string"},
        },
        "required": ["experiment", "tracked_object", "scale_reference", "notes"],
        "additionalProperties": False,
    }
    r = client.responses.create(
        model=SPARK,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "This is a physics experiment video. Identify the experiment, give a short noun phrase "
                            "for the object to track, and name any object of known size usable as a length scale."
                        ),
                    },
                    {"type": "input_video", "file_id": up.id},
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "plan",
                "schema": schema,
                "strict": True,
            }
        },
    )
    save("spark_video.json", r.model_dump())
    return f"file {up.id} | plan {json.loads(r.output_text)} | usage {r.usage.input_tokens} in / {r.usage.output_tokens} out"


async def sam_track(api_key: str, video: Path, concept: str):
    async with AsyncOpenAI(base_url=BASE_URL, api_key=api_key) as api:
        up = await api.files.create(
            file=(video.name, video.read_bytes(), "video/mp4"), purpose="user_data"
        )
        stream = await api.responses.create(
            model=SAM,
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


def check_sam(api_key: str, video: Path, truth: FreeFallTruth) -> str:
    result = asyncio.run(sam_track(api_key, video, "orange ball"))
    boxes = [r for r in result.records if r.kind == "box" and r.frame is not None]
    save("sam_boxes.json", [
        {"object_id": b.object_id, "frame": b.frame.frame_index, "l": b.left, "t": b.top,
         "r": b.right, "b": b.bottom, "c": b.confidence}
        for b in boxes
    ])
    if result.outcome.status != "completed" or not boxes:
        return f"FAILED: outcome {result.outcome.status}, {len(boxes)} boxes, media {result.media}"
    ids = sorted({b.object_id for b in boxes})
    main = max(ids, key=lambda i: sum(b.object_id == i for b in boxes))
    y = np.full(len(truth.ball_y_px), np.nan)
    for b in boxes:
        if b.object_id == main:
            y[b.frame.frame_index] = (b.top + b.bottom) / 2
    # Scale from the true A4 box here: this check isolates SAM's tracking quality.
    _, _, w, h = truth.a4_box_px
    m = measure_free_fall(y, truth.fps, scale_from_reference(w, h, "A4"))
    tracked = ~np.isnan(y)
    px_err = np.abs(y[tracked] - truth.ball_y_px[tracked]).mean()
    err = 100 * abs(m.fit.g - truth.g) / truth.g
    return (f"{len(ids)} object(s), {tracked.sum()}/{len(y)} frames tracked | mean centre error "
            f"{px_err:.2f} px | g = {m.fit.g:.3f} m/s^2, 95% CI [{m.g_ci_with_scale[0]:.3f}, "
            f"{m.g_ci_with_scale[1]:.3f}] (true {truth.g}, error {err:.2f}%)")


def check_voice(api_key: str, wav: Path) -> str:
    request = {
        "mode": "PUSH_TO_TALK",
        "model": VOICE,
        "audioEncoding": "WAV",
        "keywords": ["centimetres", "grams", "hertz", "Melde"],
    }
    with open(wav, "rb") as audio:
        r = requests.post(
            f"{BASE_URL}/asr/transcribe",
            headers={"Authorization": f"Bearer {api_key}"},
            files={
                "request": (None, json.dumps(request), "application/json"),
                "audio": (wav.name, audio),
            },
            timeout=120,
        )
    if r.status_code != 200:
        return f"FAILED: HTTP {r.status_code}: {r.text[:300]}"
    data = r.json()
    save("voice.json", data)
    return f"transcript: {data.get('transcript')!r} | expected: {SPOKEN!r}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    api_key = key()
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = set(sys.argv[1:]) or {"spark", "spark_video", "sam", "voice"}

    video, wav = OUT / "free_fall_synthetic.mp4", OUT / "spoken_params.wav"
    truth = make_free_fall_video(video)
    if "voice" in wanted:
        make_speech_wav(wav)

    checks = {
        "spark": lambda: check_spark(api_key),
        "spark_video": lambda: check_spark_video(api_key, video),
        "sam": lambda: check_sam(api_key, video, truth),
        "voice": lambda: check_voice(api_key, wav),
    }
    for name, fn in checks.items():
        if name not in wanted:
            continue
        start = time.perf_counter()
        try:
            msg = fn()
        except Exception as exc:  # report every check, even if one fails
            msg = f"FAILED: {type(exc).__name__}: {exc}"
        print(f"[{name}] {time.perf_counter() - start:.1f}s  {msg}", flush=True)


if __name__ == "__main__":
    main()
