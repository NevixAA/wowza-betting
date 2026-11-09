"""Test helper: give leagues their OWN owner-applied tier bars (owner 2026-11-09: no global bars).

Since 2026-11-09 serving tiers a league x market only on its own owner-applied sniper / marksman /
valuable bars; without them every fixture is AVOID (NO_LEAGUE_BAR). Tests that need a tip install
explicit bars here instead of relying on the removed global fallback:

    from tests.own_bars import install
    install(monkeypatch, ["Championship"], markets=("ou25",), bars=(0.12, 0.08, 0.04))

It patches only the bar lookups (the params readers and the applied-entry ledger), never
config.MODELS_DIR, so the model payloads still load from the repository.
"""
from __future__ import annotations

#: the values the removed global fallback used to serve, for tests that assert those historical tiers
OLD_SIDE = (0.10, 0.08, 0.04)
OLD_OU25 = (0.12, 0.08, 0.03)


def install(monkeypatch, leagues, markets=("ou25", "btts", "over15", "over35"), bars=None,
            extra: dict | None = None) -> dict:
    """Owner-applied bars for every league x market given. bars = (sniper, marksman, valuable); by
    default OLD_OU25 for ou25 and OLD_SIDE for the side markets. Returns {"standard", "side"}."""
    import src.betting as B
    import src.gate_resolver as G
    import src.side_thresholds as ST
    import pipeline as P

    std, side, ids = {}, {}, set()
    for m in markets:
        sn, mm, val = bars if bars is not None else (OLD_OU25 if m == "ou25" else OLD_SIDE)
        entry = {"sniper_th": sn, "marksman_th": mm, "valuable_th": val, "approved": False,
                 "status": "UNPROVEN", **(extra or {})}
        for lg in leagues:
            if m == "ou25":
                std[lg] = dict(entry)
            else:
                side.setdefault(m, {})[lg] = dict(entry)
            ids.add(f"{m}|{lg}")
    frozen = frozenset(ids)
    monkeypatch.setattr(ST, "applied_entries", lambda path=None: frozen)
    monkeypatch.setattr(B, "_STANDARD_TH_CACHE", std)
    monkeypatch.setattr(G, "_standard_params", lambda: std)
    monkeypatch.setattr(G, "_side_params", lambda: side)
    monkeypatch.setattr(P, "_load_side_market_thresholds", lambda: side)
    return {"standard": std, "side": side}


class _AnyLeague(dict):
    """A params mapping that answers for EVERY league (tests only)."""

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def get(self, k, default=None):
        return self._fn(str(k))

    def __getitem__(self, k):
        return self._fn(str(k))

    def __contains__(self, k):
        return True

    def __bool__(self):
        return True


class _AnyId(frozenset):
    def __contains__(self, k):
        return True


def install_historical(monkeypatch, markets=("ou25", "btts", "over15", "over35")) -> None:
    """Give EVERY league owner-applied bars equal to what the removed global path served before
    2026-11-09 (read lazily from the historical config values, so a test that patches them still
    moves them). For tests whose expectations predate the cutover; it is not a serving rule."""
    import config
    import src.betting as B
    import src.gate_resolver as G
    import src.side_thresholds as ST
    import pipeline as P

    def ou25(lg):
        sn = config.LEAGUE_SNIPER_THRESHOLDS.get(lg, config.SNIPER_THRESHOLD_OVER)
        mm = min(config.LEAGUE_MARKSMAN_THRESHOLDS.get(lg, config.MARKSMAN_THRESHOLD), sn)
        val = min(config.VALUABLE_THRESHOLD, mm)
        return {"sniper_th": sn, "marksman_th": mm, "valuable_th": val, "status": "UNPROVEN"}

    def side(lg):
        g = config.HISTORICAL_SIDE_GLOBAL_BARS
        return {"sniper_th": g["sniper"], "marksman_th": g["marksman"], "valuable_th": g["valuable"],
                "status": "UNPROVEN"}

    std = _AnyLeague(ou25) if "ou25" in markets else {}
    sidep = {m: _AnyLeague(side) for m in markets if m != "ou25"}
    monkeypatch.setattr(ST, "applied_entries", lambda path=None: _AnyId())
    monkeypatch.setattr(B, "_STANDARD_TH_CACHE", std)
    monkeypatch.setattr(G, "_standard_params", lambda: std)
    monkeypatch.setattr(G, "_side_params", lambda: sidep)
    monkeypatch.setattr(P, "_load_side_market_thresholds", lambda: sidep)
