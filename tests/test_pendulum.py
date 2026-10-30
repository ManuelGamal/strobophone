import numpy as np
import pytest
from scipy.integrate import solve_ivp
from strobophone.pendulum import fit_circle, measure_pendulum, swings

FPS = 60.0


def simulate(
    theta0_deg,
    L=0.5,
    g=9.81,
    damping=0.02,
    seconds=20,
    px_per_m=800,
    pivot=(960, 100),
    noise=0.5,
    seed=0,
):
    """Bob centre (x, y) in image pixels for a nonlinear damped pendulum."""

    def rhs(t, s):
        th, om = s
        return [om, -(g / L) * np.sin(th) - damping * om]

    t = np.arange(int(seconds * FPS)) / FPS
    sol = solve_ivp(
        rhs, (0, t[-1]), [np.radians(theta0_deg), 0.0], t_eval=t, rtol=1e-10, atol=1e-12
    )
    th = sol.y[0]
    rng = np.random.default_rng(seed)
    x = pivot[0] + px_per_m * L * np.sin(th) + rng.normal(0, noise, len(t))
    y = pivot[1] + px_per_m * L * np.cos(th) + rng.normal(0, noise, len(t))
    return t, x, y


def test_circle_fit_finds_pivot_and_radius():
    _, x, y = simulate(45)
    cx, cy, r = fit_circle(x, y)
    assert (cx, cy, r) == pytest.approx((960, 100, 400), abs=1.0)


def test_swings_give_period_and_amplitude_per_cycle():
    t, x, y = simulate(20, damping=0.0, noise=0.0)
    s = swings(t, x, y)
    assert len(s) >= 10
    T, amp = s[0]
    assert np.degrees(amp) == pytest.approx(20, abs=0.5)


@pytest.mark.parametrize("theta0", [20, 45, 90])
def test_g_from_pendulum_corrects_large_amplitude(theta0):
    t, x, y = simulate(theta0)
    m = measure_pendulum(t, x, y, length_m=0.5)
    assert m.g == pytest.approx(9.81, rel=0.003)
    assert m.ci[0] < 9.81 < m.ci[1]


def test_small_angle_formula_would_be_badly_wrong_at_90_degrees():
    t, x, y = simulate(90)
    m = measure_pendulum(t, x, y, length_m=0.5)
    assert m.g_small_angle < 0.9 * 9.81  # naive 4 pi^2 L / T^2 underestimates by >10%
