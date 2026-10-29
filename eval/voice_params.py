"""Spoken parameters end to end: Windows text-to-speech dictates experiment
parameters (varied wording, units and voices), Muse Voice Transcribe transcribes,
Muse Spark locates each parameter's phrase, deterministic code parses it.
Every value is compared with what was said.

    .venv/Scripts/python eval/voice_params.py [--refresh]

Synthetic speech is cleaner than a student in a lab, so this is an upper bound
on transcription accuracy; it does test the extraction and the verification.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strobophone.voice import ONES, TENS, extract_parameters, transcribe

OUT = ROOT / "outputs" / "voice"
VOICES = ["Microsoft David Desktop", "Microsoft Zira Desktop"]
ONES_W = {v: k for k, v in ONES.items()}
TENS_W = {v: k for k, v in TENS.items()}


def say_int(n: int) -> str:
    if n < 20:
        return ONES_W[n]
    if n < 100:
        return TENS_W[n // 10 * 10] + ("" if n % 10 == 0 else " " + ONES_W[n % 10])
    if n < 1000:
        rest = n % 100
        return (
            ONES_W[n // 100]
            + " hundred"
            + ("" if rest == 0 else " and " + say_int(rest))
        )
    rest = n % 1000
    return say_int(n // 1000) + " thousand" + ("" if rest == 0 else " " + say_int(rest))


def say(x: float) -> str:
    whole, _, frac = f"{x:g}".partition(".")
    return say_int(int(whole)) + (
        "" if not frac else " point " + " ".join(ONES_W[int(d)] for d in frac)
    )


def utterances(n: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    makers = [
        lambda: (
            "drop_height",
            round(rng.uniform(0.5, 2.0), 2),
            "m",
            [
                "Drop height {} metres.",
                "I dropped it from {} metres.",
                "The release height was {} metres.",
            ],
        ),
        lambda: (
            "drop_height",
            rng.randrange(40, 200, 5),
            "cm",
            ["Drop height {} centimetres.", "It falls {} centimetres to the floor."],
        ),
        lambda: (
            "ball_diameter",
            rng.choice([38, 40, 57, 65, 67]),
            "mm",
            ["The ball diameter is {} millimetres.", "Ball diameter {} millimetres."],
        ),
        lambda: (
            "string_length",
            round(rng.uniform(0.4, 1.6), 1),
            "m",
            ["String length {} metres.", "The string is {} metres long."],
        ),
        lambda: (
            "hanging_mass",
            rng.randrange(10, 300, 10),
            "g",
            ["Hanging mass {} grams.", "I hung {} grams on the pan."],
        ),
        lambda: (
            "drive_frequency",
            rng.randrange(20, 200, 5),
            "Hz",
            ["Frequency {} hertz.", "The vibrator runs at {} hertz."],
        ),
        lambda: (
            "frame_rate",
            rng.choice([30, 60, 120, 240]),
            "fps",
            ["Frame rate {} frames per second.", "Recorded at {} frames per second."],
        ),
    ]
    factor = {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "g": 1e-3, "Hz": 1.0, "fps": 1.0}
    out = []
    for i in range(n):
        picked, used = [], set()
        for make in rng.sample(makers, len(makers)):
            name, value, unit, templates = make()
            if name in used:
                continue
            used.add(name)
            picked.append((name, value, unit, rng.choice(templates)))
            if len(picked) == rng.choice([2, 3]):
                break
        text = " ".join(t.format(say(v)) for _, v, _, t in picked)
        truth = {name: v * factor[u] for name, v, u, _ in picked}
        out.append(
            {"id": f"u{i:02d}", "voice": VOICES[i % 2], "text": text, "truth": truth}
        )
    return out


def speak(text: str, voice: str, path: Path) -> None:
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$s.SelectVoice('{voice}');"
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(24000,"
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,"
        "[System.Speech.AudioFormat.AudioChannel]::Mono);"
        f"$s.SetOutputToWaveFile('{path}', $f); $s.Speak('{text}'); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("-n", type=int, default=24)
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    key = os.environ["MODEL_API_KEY"]
    client = OpenAI(base_url="https://api.meta.ai/v1", api_key=key)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, n_params, n_right, n_wrong, n_missed = [], 0, 0, 0, 0
    for u in utterances(args.n):
        cache = OUT / f"{u['id']}.json"
        if cache.exists() and not args.refresh:
            res = json.loads(cache.read_text(encoding="utf-8"))
        else:
            wav = OUT / f"{u['id']}.wav"
            speak(u["text"], u["voice"], wav)
            transcript = transcribe(
                wav,
                key,
                keywords=["centimetres", "millimetres", "hertz", "frames per second"],
            )
            params, rejected = extract_parameters(transcript, client)
            res = {
                "transcript": transcript,
                "params": {k: v[0] for k, v in params.items()},
                "rejected": rejected,
            }
            cache.write_text(json.dumps(res, indent=1), encoding="utf-8")
        for name, true in u["truth"].items():
            n_params += 1
            got = res["params"].get(name)
            if got is None:
                n_missed += 1
                verdict = "missed"
            elif abs(got - true) <= 1e-6 + 1e-3 * abs(true):
                n_right += 1
                verdict = "right"
            else:
                n_wrong += 1
                verdict = "WRONG"
            rows.append(
                {
                    "id": u["id"],
                    "voice": u["voice"],
                    "parameter": name,
                    "said": u["text"],
                    "true": true,
                    "got": got,
                    "verdict": verdict,
                    "transcript": res["transcript"],
                }
            )
        extra = set(res["params"]) - set(u["truth"])
        for name in extra:
            n_wrong += 1
            rows.append(
                {
                    "id": u["id"],
                    "parameter": name,
                    "true": None,
                    "got": res["params"][name],
                    "verdict": "WRONG (not said)",
                    "said": u["text"],
                    "transcript": res["transcript"],
                }
            )
        print(
            u["id"],
            {r["parameter"]: r["verdict"] for r in rows if r["id"] == u["id"]},
            flush=True,
        )
    with (OUT / "voice_params.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            [
                "id",
                "voice",
                "parameter",
                "true",
                "got",
                "verdict",
                "said",
                "transcript",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(
        f"\n{n_right}/{n_params} parameters right, {n_missed} missed (rejected or not found), {n_wrong} wrong values accepted"
    )


if __name__ == "__main__":
    main()
