# StroboPhone

A phone becomes a physics-lab measurement instrument. Film an experiment and SAM 3.1 tracks the moving objects. Deterministic physics code fits the model and computes the uncertainty. A Muse Spark agent then writes a lab report in which every number traces back to a tool call.

Entry for the Meta Global AI Developer Hackathon 2026 (Indie track, *Push the Frontier*).

- **Live demo:** https://huggingface.co/spaces/ManuelGamal/strobophone. No clip handy? Load one of the simulated samples (a drop at 240 fps, or a 0.60 m pendulum), each with a known answer.
- **3-minute video:** https://youtu.be/G-nkOVWi_rU

## Why

A school physics lab measures g with photogates or ticker timers that many schools do not have. Every student has a phone. StroboPhone turns a phone video into a measurement **with an honest 95% interval**, plus a lab report in English or Arabic, and it warns when the setup, rather than the physics, produced the number.

## The finding that shaped the design

We told Muse Spark that real Apple-π drop clips were filmed at 15 fps instead of 30. Frames are then twice as far apart, so the motion implies g = 9.81 × (15/30)² ≈ 2.45 m/s².

| Told "15 fps" (true 30) | g reported | Expected from the evidence |
|---|---|---|
| Muse Spark alone (54 answers) | median **9.80** (IQR 9.61–9.83); 54/54 within 25% of 9.8 | 2.45 |
| StroboPhone (36 clips) | median **2.40** (IQR 2.25–2.59) | 2.45 |

Spark also **misreported what it saw so the answer matched the textbook value**. On all 33 single-ball clips where the check is possible, the fall duration Spark stated was half the tracked one (median ratio 0.50, IQR 0.48–0.51). It wrote "impact at ~8 frames" when the ball takes ~16. (`eval/spark_counterfactual.py`)

**The same happens with a pendulum.** We gave Spark real IRIS pendulum clips (60 fps) and told it either 60 or 30 fps:
- **Told 30 fps:** it still answered g ≈ 9.6–9.8 (6/6), and halved the frame count it claims for one swing (0.48× the tracked count).
- **Told the true 60 fps:** it said one swing takes 85 frames in 5 of 6 clips. That is exactly the textbook 2π√(L/g) × 60 = 85.1, whatever the amplitude. The real swings take 87–92 frames and get longer at larger amplitude, which StroboPhone measures and corrects for. (`eval/spark_pendulum_counterfactual.py`)

So in StroboPhone, models do what they are good at and code does the arithmetic:

```
phone video ──► clip prep (field order, crop/zoom, streamed: O(1 frame) memory)
            ──► SAM 3.1 video tracking (fallback concept phrases)
            ──► deterministic fits: joint multi-arc parabola / pendulum period
                bootstrap 95% CI, arc-consistency check (Birge ratio)
            ──► Muse Spark writes the report with [[fact]] placeholders
                code fills every number; drafts with untraceable numbers are blocked
  spoken parameters ──► Muse Voice Transcribe ──► Spark finds the phrase ──► code parses + verifies it
```

## What it measures

- **Free fall:** g from a drop, scaled by an A4 sheet in the fall plane or a known px/m. All bounces are fitted jointly with one shared g. Restitution e comes from the bounce, with no scale needed.
- **Pendulum:** g from the period, needing only the string length (no pixel scale). It applies the exact large-amplitude correction T = (2/π)·K(sin²(θ₀/2))·T₀. At 90° the textbook formula is off by more than 10%.
- **Wrong slow-motion metadata:** enter the true capture frame rate (phones often label 240 fps footage as 30 fps). g follows it exactly, as g ∝ fps².

## Evidence

| Check | Result | Script |
|---|---|---|
| Real Apple-π free-fall clips, no manual settings | **36/36** measured | `eval/apple_pi_auto.py` |
| SAM 3.1 tracking vs human masks | median **0.72 px** | same |
| 95% interval coverage (simulated drops, 30–240 fps, 0.5–2 px noise) | **92.8–97.0%** | `eval/ci_coverage.py` |
| Arc-consistency check flags clips whose arcs disagree | 8/36 flagged, interval widened, warning shown | `strobophone/pipeline.py` |
| Restitution e on tennis-ball drops | median error 0.008 | `eval/apple_pi_auto.py` |
| Report agent (English): untraced numbers in accepted reports | **0** in 36 reports | `eval/report_faithfulness.py` |
| Report agent (Arabic) accepted on the first draft | **12/12** | `eval/report_arabic.py` |
| IRIS real 4K pendulum clips: g stays flat as a real swing dies down (per 10° of amplitude) | **+0.004 / −0.022 m/s²** at 20° / 45°, vs −0.15 / −0.26 for the textbook formula | `eval/iris_pendulum_cycles.py` |
| IRIS pendulum, spread of g across release angles 20° / 45° / ~75° | **3.2%** corrected vs 9.7% small-angle; ±0.01–0.03 m/s² between takes (17 clips) | `eval/iris_pendulum.py`, `eval/iris_summary.py` |
| IRIS real drops (20 side-on clips), ball-size scale | median g 9.47; 95% interval contains 9.81 in **19/20** | `eval/iris_fall.py` |

## Limitations (measured, not hidden)

- **Absolute g on Apple-π is not quoted.** The only scale there is the ball's own box width, so values span 4–13 m/s². The arc check exists because one such clip gave a convincing-looking 10.39.
- **Time-of-flight g on Apple-π projectile clips is unreliable** (median error 22.7% on 31 clips at 30 fps). The table-edge release is not detected precisely enough.
- **IRIS pendulum absolute g is 4–5% low** (9.16–9.46 m/s²) with the stated "rope length 0.5 m". That matches a pivot-to-centre length of 0.52 m; IRIS does not say where the length was measured. Above ~60° amplitude some drift remains (−0.10 m/s² per 10°), consistent with air drag on a light tennis ball.
- **IRIS drops of light balls:** the arc-consistency check flags all 20 clips. On a light 22 cm ball, air drag makes the fast drop and the slow rebounds disagree, so the intervals are honest but wide (median ±0.98 m/s²). Time-of-flight g at 60 fps is 9.46 median, but one frame of release timing is worth ~7%. IRIS's Dropping_ball clips are filmed top-down (the ball falls along the camera axis), which side-on tracking cannot measure; they are excluded.
- **The faithfulness eval shows a guarantee, not an improvement.** Given the same finished facts, free-writing Spark also copied numbers correctly. The counterfactual above is where unconstrained Spark fails. The number checker does not catch unsupported qualitative claims.

## Run it

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Python 3.13
echo MODEL_API_KEY=... > .env                                          # Meta API key, never committed
.venv/Scripts/python -m uvicorn app.server:app --port 8765               # web app
.venv/Scripts/python scripts/measure.py clip.mp4 --reference A4          # command line
.venv/Scripts/python -m pytest -q                                        # tests
```

## Meta models used

`sam-3.1` (video tracking), `muse-spark-1.3` (report writing, parameter extraction; structured JSON output), `muse-voice-transcribe-1.0` (spoken parameters).

## Data

- **Apple-π** (yaorunmao/Apple-PI-GT, CC BY 4.0): evaluation only; nothing was tuned on it, except one disclosed change (zoom only objects under 16 px).
- **IRIS** (rasulkhanbayov/IRIS, CC BY-NC 4.0): evaluation only. Clips were shrunk on Kaggle (`kaggle/iris_prep`) and are not redistributed. One change was made after seeing it: the pendulum concept list now tries "ball" before "pendulum bob" (the latter found nothing on real clips). This saves a SAM call and does not change any measurement.

## Built during the hackathon

Everything in this repository was written during the hackathon building period, 27 Oct – 4 Nov 2026. The repository was created on 29 Oct 2026 and the commit history is kept unsquashed.
