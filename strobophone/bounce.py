"""Coefficient of restitution from a bouncing-ball track. The result is a ratio
of velocities at the same instant, so neither the pixel scale nor the frame
rate enters it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from strobophone.fits import detect_release

BOOTSTRAP_SAMPLES = 2000


@dataclass(frozen=True)
class RestitutionResult:
    e: float
    ci: tuple[float, float]
    impact_frame: float
    n_before: int
    n_after: int


def find_bounces(y_px: np.ndarray, min_rise_px: float = 3.0) -> list[int]:
    """Frame indices of the lowest point of each bounce (image y grows downward):
    a local maximum of y that the ball clearly rises away from afterwards."""
    y = np.asarray(y_px, float)
    valid = np.nonzero(~np.isnan(y))[0]
    bounces = []
    for k in range(1, len(valid) - 1):
        i, prev, nxt = valid[k], valid[k - 1], valid[k + 1]
        if y[i] >= y[prev] and y[i] > y[nxt]:
            after = y[valid[k + 1 : k + 4]]
            if y[i] - after.min() >= min_rise_px:
                bounces.append(int(i))
    return bounces


def _coeffs(t: np.ndarray, y: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Joint fit y = a t^2 + b t + c with shared a and separate (b, c) before and
    after impact. Returns ((a, b1, c1, b2, c2), design matrix)."""
    before = ~after
    X = np.column_stack([t**2, t * before, before, t * after, after.astype(float)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef, X


def _solve(t: np.ndarray, y: np.ndarray, after: np.ndarray) -> tuple[float, float, np.ndarray]:
    """(e, impact time, fitted values) from the joint fit."""
    (a, b1, c1, b2, c2), X = _coeffs(t, y, after)
    t_c = (c2 - c1) / (b1 - b2)  # where the two parabolas meet
    v_in, v_out = 2 * a * t_c + b1, 2 * a * t_c + b2
    return float(-v_out / v_in), float(t_c), X @ np.array([a, b1, c1, b2, c2])


def _first_bounce(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """(t, y, after-mask, n_before, n_after) for the frames around the first bounce."""
    bounces = find_bounces(y)
    if not bounces:
        raise ValueError("no bounce found in the track")
    hit = bounces[0]
    end = bounces[1] if len(bounces) > 1 else len(y)
    valid = np.nonzero(~np.isnan(y))[0]
    pre_valid = valid[valid <= hit]
    start = int(pre_valid[detect_release(y[pre_valid])])
    pre = valid[(valid >= start) & (valid < hit)]
    post = valid[(valid > hit) & (valid < end)]
    if len(pre) < 3 or len(post) < 3:
        raise ValueError(f"need 3+ frames each side of the bounce, got {len(pre)} and {len(post)}")
    idx = np.concatenate([pre, post])
    after = np.concatenate([np.zeros(len(pre), bool), np.ones(len(post), bool)])
    return idx.astype(float), y[idx], after, len(pre), len(post)


def measure_restitution(y_px: np.ndarray, seed: int = 0) -> RestitutionResult:
    """e = rebound speed / impact speed at the first bounce, with a residual-bootstrap CI."""
    t, yy, after, n_pre, n_post = _first_bounce(np.asarray(y_px, float))
    e, t_c, fitted = _solve(t, yy, after)
    rng = np.random.default_rng(seed)
    resid = yy - fitted
    samples = [_solve(t, fitted + rng.choice(resid, len(resid)), after)[0] for _ in range(BOOTSTRAP_SAMPLES)]
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return RestitutionResult(e, (float(lo), float(hi)), t_c, n_pre, n_post)


def fall_time_frames(y_px: np.ndarray) -> tuple[float, float]:
    """(release, impact) in fractional frames, for a drop from rest: release is the
    vertex of the falling parabola, impact is where it meets the rebound parabola."""
    t, yy, after, _, _ = _first_bounce(np.asarray(y_px, float))
    (a, b1, c1, b2, c2), _ = _coeffs(t, yy, after)
    return float(-b1 / (2 * a)), float((c2 - c1) / (b1 - b2))


def g_from_time_of_flight(y_px: np.ndarray, h_m: float, fps: float) -> float:
    """g = 2h / T^2 for a drop from rest through a known height h. Uses only times,
    so it needs no pixel scale (but does rely on the frame rate)."""
    t_release, t_impact = fall_time_frames(y_px)
    return 2 * h_m / ((t_impact - t_release) / fps) ** 2
