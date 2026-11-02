"""Exact P/L totals and their display, in one place.

P/L values are stored with a few decimals (e.g. -0.95, 1.075). Summing them as floats makes the last
bit depend on summation order, and a total that is exactly on a half (-8.445u) then rounds to -8.44 on
one page and -8.45 on another (issue #19). Every P/L sum that is shown goes through total(), every
P/L shown goes through fmt():

    total(values) -> the exact decimal sum of the stored values (each read as Decimal(str(v)), missing
                     values skipped like pandas .sum()), returned as the float whose shortest repr IS
                     that decimal - so it is a drop-in float and fmt() recovers the exact decimal.
    fmt(x, places=2, sign=True, suffix="u") -> Decimal quantize ROUND_HALF_UP; "—" when x is missing.
"""
from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Iterable

MISSING = "—"


def _dec(v) -> Decimal | None:
    if v is None:
        return None
    if isinstance(v, Decimal):
        return None if v.is_nan() else v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    try:
        return Decimal(str(v)) if isinstance(v, (int, float, str)) else Decimal(repr(f))
    except InvalidOperation:
        return Decimal(repr(f))


def total_dec(values: Iterable) -> Decimal:
    """Exact Decimal sum; missing (None / NaN / non-numeric) values are skipped."""
    s = Decimal(0)
    if values is None:
        return s
    for v in values:
        d = _dec(v)
        if d is not None:
            s += d
    return s


def total(values: Iterable) -> float:
    """Exact decimal sum of the stored P/L values, as a float (see module doc)."""
    return float(total_dec(values))


def running(values) -> list[float]:
    """Exact running totals (missing values add nothing), as floats - for curves and drawdowns."""
    s, out = Decimal(0), []
    for v in values:
        d = _dec(v)
        if d is not None:
            s += d
        out.append(float(s))
    return out


def rnd(x, places: int = 2) -> float:
    """Half-up rounded float for numeric table cells (NaN stays NaN)."""
    d = _dec(x)
    if d is None:
        return float("nan")
    return float(d.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)) + 0.0


def fmt(x, places: int = 2, sign: bool = True, suffix: str = "u") -> str:
    """Round half-up to `places` decimals; '+' on non-negative values when sign; never '-0.00'."""
    d = _dec(x)
    if d is None:
        return MISSING
    q = d.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    if q == 0:
        q = abs(q)
    return f"{q:{'+' if sign else ''}f}{suffix}"
