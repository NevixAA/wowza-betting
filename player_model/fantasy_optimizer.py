"""A genuinely legal FPL squad: 15 players, £100m, max three per club, solved not sorted.

    from player_model.fantasy_optimizer import optimal_squad
    sq = optimal_squad(proj_df)

WHY A SOLVER AND NOT A SORT. best_xi() takes the top N per position by projected points and
tries each formation. Its own docstring is honest that "budget is reported (not
hard-constrained)", and there is no per-club limit anywhere in it, so what it returns cannot be
called a legal FPL team -- on 2026-09-27 it came in at £60.4m with at most three per club purely
by luck, not because anything stopped it.

Greedy sorting cannot fix that. Budget and the three-per-club rule are COUPLING constraints:
whether a £12m forward belongs in the squad depends on what the other fourteen picks cost, and
whether a third Arsenal defender is affordable depends on which Arsenal players are already in.
No per-position ranking can see that, which is exactly the case integer programming exists for.

WHAT IT MAXIMISES. Starting XI points, plus the captain again (a captain is doubled), plus a
small weight on the bench -- a bench is not worthless, but a point sitting on it is worth far
less than a point on the pitch, and weighting it equally would buy expensive reserves.

POINTS ARE UNCONDITIONAL. It optimises `xpts_uncond` (conditional points x P(start)) where
available, because a player who will not start is worth zero however good he is. Feeding it
conditional points is how a squad ends up full of injured stars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: FPL squad composition.
SQUAD_QUOTA = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
#: Legal starting-XI bounds per position (exactly one keeper).
XI_BOUNDS = {"GKP": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
SQUAD_SIZE, XI_SIZE = 15, 11
BUDGET = 100.0
MAX_PER_CLUB = 3
#: A bench point is worth this fraction of a pitch point -- enough to prefer a useful bench,
#: far too little to justify spending real money on one.
BENCH_WEIGHT = 0.10


def _points_column(df: pd.DataFrame, points_col: str | None) -> pd.Series:
    """Unconditional points if we can get them; never silently conditional."""
    if points_col and points_col in df.columns:
        return pd.to_numeric(df[points_col], errors="coerce").fillna(0.0)
    for c in ("xpts_uncond", "xpts_rot"):
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    base = pd.to_numeric(df.get("fantasy_pts", 0), errors="coerce").fillna(0.0)
    p = pd.to_numeric(df.get("p_start", 1.0), errors="coerce").fillna(0.0).clip(0, 1)
    return base * p


def optimal_squad(proj: pd.DataFrame, budget: float = BUDGET,
                  points_col: str | None = None,
                  exclude_unavailable: bool = True) -> dict:
    """Solve for the best legal 15. Returns {} when no feasible squad exists.

    Keys: squad, xi, bench (ordered), captain, vice_captain, formation, total_cost,
    money_remaining, xi_points, captain_points, squad_points, feasible, solver.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds

    df = proj.copy().reset_index(drop=True)
    if df.empty or "position" not in df.columns or "price" not in df.columns:
        return {}

    if exclude_unavailable and "availability" in df.columns:
        av = df["availability"].astype(str).str.lower()
        df = df[~av.isin(("injured", "unavailable", "suspended"))].reset_index(drop=True)
    if df.empty:
        return {}

    pts = _points_column(df, points_col).to_numpy(dtype=float)
    price = pd.to_numeric(df["price"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pos = df["position"].astype(str).to_numpy()
    club = df.get("team", pd.Series("", index=df.index)).astype(str).to_numpy()
    n = len(df)

    # Not enough bodies in a position to field a legal squad -> say so rather than return junk.
    for p, q in SQUAD_QUOTA.items():
        if (pos == p).sum() < q:
            return {"feasible": False, "reason": f"only {(pos == p).sum()} {p} available, need {q}"}

    # Variables: x (in squad) | y (in XI) | c (captain), each length n.
    X, Y, C = slice(0, n), slice(n, 2 * n), slice(2 * n, 3 * n)
    N = 3 * n
    obj = np.zeros(N)
    obj[Y] = -pts                       # XI points
    obj[C] = -pts                       # captain counts a second time
    obj[X] = -pts * BENCH_WEIGHT        # everyone gets the bench weight...
    obj[Y] += pts * BENCH_WEIGHT        # ...removed again for those who start

    cons = []

    def row(**parts):
        r = np.zeros(N)
        for k, v in parts.items():
            r[{"x": X, "y": Y, "c": C}[k]] = v
        return r

    cons.append(LinearConstraint(row(x=1), SQUAD_SIZE, SQUAD_SIZE))
    cons.append(LinearConstraint(row(y=1), XI_SIZE, XI_SIZE))
    cons.append(LinearConstraint(row(c=1), 1, 1))
    cons.append(LinearConstraint(row(x=price), -np.inf, budget))

    # y_i <= x_i and c_i <= y_i, as one sparse block each.
    A = np.zeros((n, N)); A[np.arange(n), n + np.arange(n)] = 1; A[np.arange(n), np.arange(n)] = -1
    cons.append(LinearConstraint(A, -np.inf, 0))
    B = np.zeros((n, N)); B[np.arange(n), 2 * n + np.arange(n)] = 1; B[np.arange(n), n + np.arange(n)] = -1
    cons.append(LinearConstraint(B, -np.inf, 0))

    for p, q in SQUAD_QUOTA.items():
        cons.append(LinearConstraint(row(x=(pos == p).astype(float)), q, q))
    for p, (lo, hi) in XI_BOUNDS.items():
        cons.append(LinearConstraint(row(y=(pos == p).astype(float)), lo, hi))
    for cl in pd.unique(club):
        if cl:
            cons.append(LinearConstraint(row(x=(club == cl).astype(float)), -np.inf, MAX_PER_CLUB))

    res = milp(c=obj, constraints=cons, integrality=np.ones(N),
               bounds=Bounds(np.zeros(N), np.ones(N)))
    if not res.success or res.x is None:
        return {"feasible": False, "reason": res.message}

    sel = np.round(res.x).astype(int)
    in_squad = sel[X].astype(bool)
    in_xi = sel[Y].astype(bool) & in_squad
    cap_i = int(np.argmax(sel[C]))

    squad = df[in_squad].copy(); squad["xpts"] = pts[in_squad]
    xi = df[in_xi].copy(); xi["xpts"] = pts[in_xi]
    bench = df[in_squad & ~in_xi].copy(); bench["xpts"] = pts[in_squad & ~in_xi]
    # Bench ORDER matters in FPL: the keeper is a separate slot, the outfield three are ranked.
    bench_gk = bench[bench["position"] == "GKP"]
    bench_out = bench[bench["position"] != "GKP"].sort_values("xpts", ascending=False)
    bench = pd.concat([bench_out, bench_gk]).reset_index(drop=True)

    xi_sorted = xi.sort_values("xpts", ascending=False).reset_index(drop=True)
    vice = xi_sorted.iloc[1] if len(xi_sorted) > 1 else None
    counts = xi["position"].value_counts()
    return {
        "feasible": True,
        "solver": "scipy.optimize.milp (HiGHS)",
        "squad": squad.reset_index(drop=True),
        "xi": xi.reset_index(drop=True),
        "bench": bench,
        "captain": df.iloc[cap_i],
        "vice_captain": vice,
        "formation": f"{counts.get('DEF', 0)}-{counts.get('MID', 0)}-{counts.get('FWD', 0)}",
        "total_cost": round(float(price[in_squad].sum()), 1),
        "money_remaining": round(budget - float(price[in_squad].sum()), 1),
        "xi_points": round(float(pts[in_xi].sum()), 2),
        "captain_points": round(float(pts[cap_i]), 2),
        "squad_points": round(float(pts[in_squad].sum()), 2),
    }


def verify_legal(sq: dict, budget: float = BUDGET) -> list[str]:
    """Return a list of rule violations. Empty list means the squad is genuinely legal."""
    if not sq or not sq.get("feasible"):
        return ["no feasible squad"]
    bad = []
    squad, xi = sq["squad"], sq["xi"]
    if len(squad) != SQUAD_SIZE:
        bad.append(f"squad has {len(squad)} players, need {SQUAD_SIZE}")
    if len(xi) != XI_SIZE:
        bad.append(f"XI has {len(xi)}, need {XI_SIZE}")
    for p, q in SQUAD_QUOTA.items():
        got = int((squad["position"] == p).sum())
        if got != q:
            bad.append(f"squad has {got} {p}, need {q}")
    for p, (lo, hi) in XI_BOUNDS.items():
        got = int((xi["position"] == p).sum())
        if not (lo <= got <= hi):
            bad.append(f"XI has {got} {p}, legal range {lo}-{hi}")
    cost = float(pd.to_numeric(squad["price"], errors="coerce").sum())
    if cost > budget + 1e-6:
        bad.append(f"cost £{cost:.1f}m exceeds £{budget:.0f}m")
    if "team" in squad.columns:
        over = squad["team"].value_counts()
        for cl, k in over[over > MAX_PER_CLUB].items():
            bad.append(f"{k} players from {cl}, max {MAX_PER_CLUB}")
    if not set(xi["player_name"]).issubset(set(squad["player_name"])):
        bad.append("XI contains a player not in the squad")
    return bad
