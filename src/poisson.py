"""Dixon-Coles corrected Poisson utilities for O/U 2.5 calibration."""
from __future__ import annotations
import math


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _dc_tau(x: int, y: int, lh: float, la: float, rho: float) -> float:
    """Dixon-Coles correction factor for low-scoring outcomes (0-0, 1-0, 0-1, 1-1)."""
    if x == 0 and y == 0:
        return 1.0 - lh * la * rho
    if x == 1 and y == 0:
        return 1.0 + la * rho
    if x == 0 and y == 1:
        return 1.0 + lh * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def dixon_coles_p_over25(lambda_h: float, lambda_a: float, rho: float = -0.08) -> float:
    """
    P(total goals > 2.5) with Dixon-Coles correction.

    Corrects the standard independent-Poisson model at low scorelines —
    the exact zone (2-3 goals) where O/U 2.5 calibration degrades.
    rho = -0.08 is the published Dixon-Coles correlation estimate.
    """
    lambda_h = max(0.1, min(float(lambda_h), 8.0))
    lambda_a = max(0.1, min(float(lambda_a), 8.0))

    p_under = 0.0
    for h in range(3):
        for a in range(3 - h):   # all (h, a) with h + a <= 2
            tau = _dc_tau(h, a, lambda_h, lambda_a, rho)
            p_under += tau * _poisson_pmf(h, lambda_h) * _poisson_pmf(a, lambda_a)

    return float(max(0.0, min(1.0, 1.0 - p_under)))
