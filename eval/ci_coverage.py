"""Are StroboPhone's 95% intervals honest? Simulate free-fall tracks with known g
and Gaussian tracking noise, fit them, and count how often the CI contains the
truth. No API calls; deterministic (seeded).

    .venv/Scripts/python eval/ci_coverage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strobophone.fits import fit_free_fall

G, PX_PER_M, DROP_M = 9.81, 500.0, 1.0
TRIALS = 400


def coverage(fps: float, noise_px: float, seed: int) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    n = int(np.sqrt(2 * DROP_M / G) * fps)
    t = np.arange(n) / fps
    hits, widths = 0, []
    for k in range(TRIALS):
        y = 100 + 0.5 * G * PX_PER_M * t**2 + rng.normal(0, noise_px, n)
        r = fit_free_fall(t, y, PX_PER_M, seed=k)
        hits += r.ci_low <= G <= r.ci_high
        widths.append(r.ci_high - r.ci_low)
    return hits / TRIALS, float(np.median(widths)) / 2, n


def main() -> None:
    print(f"{'fps':>5} {'noise px':>8} {'points':>6} {'coverage':>8} {'+/- g':>8}")
    for fps in (30, 60, 120, 240):
        for noise in (0.5, 1.0, 2.0):
            cov, half, n = coverage(fps, noise, seed=int(fps * 10 + noise * 10))
            print(f"{fps:>5} {noise:>8.1f} {n:>6} {cov:>8.1%} {half:>8.3f}")


if __name__ == "__main__":
    main()
