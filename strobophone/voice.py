"""Spoken lab parameters: Muse Voice Transcribe -> transcript, Muse Spark finds
which phrase states which parameter (an exact quote), deterministic code turns
the quote into an SI value. Spark never produces the number, and a quote that
is not in the transcript, or has the wrong kind of unit, is rejected."""

from __future__ import annotations

import json
import re

import requests

ASR_URL = "https://api.meta.ai/v1/asr/transcribe"
VOICE_MODEL = "muse-voice-transcribe-1.0"

ONES = {
    w: i
    for i, w in enumerate(
        ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    )
}
TENS = {
    w: 10 * (i + 2)
    for i, w in enumerate(
        ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    )
}
SCALES = {"hundred": 100, "thousand": 1000}

# unit word -> (factor to SI, SI unit)
UNITS = {
    "millimetres": (1e-3, "m"),
    "millimeters": (1e-3, "m"),
    "millimetre": (1e-3, "m"),
    "millimeter": (1e-3, "m"),
    "mm": (1e-3, "m"),
    "centimetres": (1e-2, "m"),
    "centimeters": (1e-2, "m"),
    "centimetre": (1e-2, "m"),
    "centimeter": (1e-2, "m"),
    "cm": (1e-2, "m"),
    "metres": (1.0, "m"),
    "meters": (1.0, "m"),
    "metre": (1.0, "m"),
    "meter": (1.0, "m"),
    "m": (1.0, "m"),
    "grams": (1e-3, "kg"),
    "gram": (1e-3, "kg"),
    "g": (1e-3, "kg"),
    "kilograms": (1.0, "kg"),
    "kilogram": (1.0, "kg"),
    "kg": (1.0, "kg"),
    "hertz": (1.0, "Hz"),
    "hz": (1.0, "Hz"),
    "frames per second": (1.0, "fps"),
    "fps": (1.0, "fps"),
}
# what each parameter must be measured in
PARAMETERS = {
    "drop_height": "m",
    "ball_diameter": "m",
    "string_length": "m",
    "hanging_mass": "kg",
    "drive_frequency": "Hz",
    "frame_rate": "fps",
}


def words_to_number(text: str) -> float:
    """'two hundred and forty' -> 240, 'one point two six' -> 1.26, '1.26' -> 1.26."""
    text = text.strip().lower().replace("-", " ")
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    whole, _, frac = text.partition(" point ")
    total, current = 0, 0
    for w in whole.split():
        if w == "and":
            continue
        if w in ONES:
            current += ONES[w]
        elif w in TENS:
            current += TENS[w]
        elif w in SCALES:
            current = max(current, 1) * SCALES[w]
            if SCALES[w] >= 1000:
                total, current = total + current, 0
        else:
            raise ValueError(f"not a number word: {w!r}")
    value = float(total + current)
    if frac:
        digits = [ONES[w] for w in frac.split()]
        if any(d > 9 for d in digits):
            raise ValueError(f"bad decimal digits: {frac!r}")
        value += float("0." + "".join(str(d) for d in digits))
    return value


def parse_quantity(quote: str) -> tuple[float, str]:
    """'eighty centimetres' -> (0.8, 'm'). Raises ValueError without a known unit."""
    q = quote.strip().lower().rstrip(".")
    for unit in sorted(
        UNITS, key=len, reverse=True
    ):  # longest first: 'frames per second' before 'second'
        m = re.fullmatch(rf"(.+?)\s*{re.escape(unit)}", q)
        if m:
            factor, si = UNITS[unit]
            return words_to_number(m[1]) * factor, si
    raise ValueError(f"no unit in {quote!r}")


def verified_parameters(transcript: str, found: list[dict]) -> tuple[dict, list[dict]]:
    """Accept a {parameter, quote} only if the quote occurs verbatim in the
    transcript and parses to a quantity in the parameter's unit."""
    params, rejected = {}, []
    low = transcript.lower()
    for f in found:
        name, quote = f["parameter"], f["quote"]
        if quote.lower() not in low:
            rejected.append({**f, "reason": "quote not in transcript"})
            continue
        try:
            value, unit = parse_quantity(quote)
        except ValueError as e:
            rejected.append({**f, "reason": f"unparseable: {e}"})
            continue
        if PARAMETERS.get(name) != unit:
            rejected.append({**f, "reason": f"unit {unit} does not suit {name}"})
            continue
        params[name] = (value, unit)
    return params, rejected


def transcribe(wav_path, api_key: str, keywords: list[str] | None = None) -> str:
    request = {
        "mode": "PUSH_TO_TALK",
        "model": VOICE_MODEL,
        "audioEncoding": "WAV",
        "keywords": keywords or [],
    }
    with open(wav_path, "rb") as audio:
        r = requests.post(
            ASR_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            files={
                "request": (None, json.dumps(request), "application/json"),
                "audio": (str(wav_path), audio),
            },
            timeout=120,
        )
    r.raise_for_status()
    return r.json()["transcript"]


SCHEMA = {
    "type": "object",
    "properties": {
        "found": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "parameter": {"type": "string", "enum": list(PARAMETERS)},
                    "quote": {"type": "string"},
                },
                "required": ["parameter", "quote"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["found"],
    "additionalProperties": False,
}

PROMPT = """A student dictated the parameters of a physics experiment. For each parameter they state, return the
parameter name and the exact words from the transcript that give its value and unit (copy them character for
character, number and unit only, e.g. "eighty centimetres"). Skip parameters that are not stated.
Parameters: drop_height, ball_diameter, string_length, hanging_mass, drive_frequency, frame_rate.

Transcript: """


def extract_parameters(
    transcript: str, client, model: str = "muse-spark-1.3"
) -> tuple[dict, list[dict]]:
    """Spark locates the phrases; verified_parameters turns them into SI values."""
    r = client.responses.create(
        model=model,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": PROMPT + transcript}],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "parameters",
                "schema": SCHEMA,
                "strict": True,
            }
        },
    )
    return verified_parameters(transcript, json.loads(r.output_text)["found"])
