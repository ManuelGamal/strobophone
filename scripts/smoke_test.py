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

import imageio_ffmpeg
import numpy as np
import requests
from dotenv import load_dotenv
from meta_sam_parser import parse_responses_stream, video_segmentation_format
from openai import AsyncOpenAI, OpenAI

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "smoke"
BASE_URL = "https://api.meta.ai/v1"
SPARK = "muse-spark-1.3"
SAM = "sam-3.1"
VOICE = "muse-voice-transcribe-1.0"

# Synthetic free fall: known scale and g, so the SAM fit has a ground truth.
FPS = 120
SCALE_PX_PER_M = 300.0
G_TRUE = 9.81
BALL_RADIUS_M = 0.04
REST_FRAMES = 15
FALL_S = 0.5
WIDTH, HEIGHT = 480, 520
Y0_PX = 60.0

SPOKEN = "String length eighty centimetres. Hanging mass fifty grams. Frequency one hundred hertz."


def key() -> str:
    load_dotenv(ROOT / ".env")
    value = os.environ.get("MODEL_API_KEY")
    if not value:
        sys.exit(
            "MODEL_API_KEY is not set. Put it in .env (gitignored) as MODEL_API_KEY=..."
        )
    return value


def make_fall_video(path: Path) -> np.ndarray:
    """Render an orange ball falling past an A4 sheet; return true centre y per frame."""
    n_fall = int(FALL_S * FPS) + 1
    t = np.arange(n_fall) / FPS
    ys = np.concatenate(
        [np.full(REST_FRAMES, Y0_PX), Y0_PX + 0.5 * G_TRUE * t**2 * SCALE_PX_PER_M]
    )
    cx = WIDTH * 0.6
    r = BALL_RADIUS_M * SCALE_PX_PER_M
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    wall = np.empty((HEIGHT, WIDTH, 3), np.float32)
    wall[:] = (205, 200, 190)
    wall += np.random.default_rng(0).normal(0, 4, wall.shape)  # light texture
    a4_w, a4_h = 0.210 * SCALE_PX_PER_M, 0.297 * SCALE_PX_PER_M
    ax, ay = WIDTH * 0.15, HEIGHT * 0.35
    wall[(xx >= ax) & (xx < ax + a4_w) & (yy >= ay) & (yy < ay + a4_h)] = (
        250,
        250,
        248,
    )

    writer = imageio_ffmpeg.write_frames(
        str(path),
        (WIDTH, HEIGHT),
        fps=FPS,
        codec="libx264",
        pix_fmt_out="yuv420p",
        quality=8,
        macro_block_size=8,
    )
    writer.send(None)
    for y in ys:
        frame = wall.copy()
        d = np.sqrt((xx - cx) ** 2 + (yy - y) ** 2)
        alpha = np.clip(r - d + 0.5, 0, 1)[..., None]  # antialiased edge
        frame = frame * (1 - alpha) + np.array((240, 110, 20), np.float32) * alpha
        writer.send(np.clip(frame, 0, 255).astype(np.uint8).tobytes())
    writer.close()
    return ys


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


def check_sam(api_key: str, video: Path, true_ys: np.ndarray) -> str:
    result = asyncio.run(sam_track(api_key, video, "orange ball"))
    boxes = [r for r in result.records if r.kind == "box" and r.frame is not None]
    save(
        "sam_boxes.json",
        [
            {
                "object_id": b.object_id,
                "frame": b.frame.frame_index,
                "l": b.left,
                "t": b.top,
                "r": b.right,
                "b": b.bottom,
                "c": b.confidence,
            }
            for b in boxes
        ],
    )
    if result.outcome.status != "completed" or not boxes:
        return f"FAILED: outcome {result.outcome.status}, {len(boxes)} boxes, media {result.media}"
    ids = sorted({b.object_id for b in boxes})
    main = max(ids, key=lambda i: sum(b.object_id == i for b in boxes))
    track = {
        b.frame.frame_index: (b.top + b.bottom) / 2
        for b in boxes
        if b.object_id == main
    }
    frames = np.array(sorted(f for f in track if f >= REST_FRAMES))
    y = np.array([track[f] for f in frames])
    t = (frames - REST_FRAMES) / FPS
    a2, _, _ = np.polyfit(t, y, 2)
    g = 2 * a2 / SCALE_PX_PER_M
    px_err = np.abs(y - true_ys[frames]).mean()
    return (
        f"{len(ids)} object(s), {len(track)}/{len(true_ys)} frames tracked | mean centre error "
        f"{px_err:.2f} px | fitted g = {g:.3f} m/s^2 (true {G_TRUE}, error {100 * abs(g - G_TRUE) / G_TRUE:.2f}%)"
    )


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
    true_ys = make_fall_video(video)
    if "voice" in wanted:
        make_speech_wav(wav)

    checks = {
        "spark": lambda: check_spark(api_key),
        "spark_video": lambda: check_spark_video(api_key, video),
        "sam": lambda: check_sam(api_key, video, true_ys),
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
