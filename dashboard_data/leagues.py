"""Leagues — per-league results, tips, rating and averages. ONE source for the app and Telegram.

    from dashboard_data import leagues as L
    d  = L.settled()                      # settled live tips since PERFORMANCE_CUTOFF_DATE
    ov = L.overview(d, today)             # one row per league (the page table AND the Monday picture)

Owner request 2026-10-25: "a page for it in the app? and telegram? 'leagues' with the ability to see
each league results and tip and rating and avg's". pages/17_🏟️_Leagues.py and
telegram_bot/daily_pack.leagues_image() both read THESE functions, so the two cannot disagree
(tests/test_leagues_page.py asserts it on the same input).

SAME ROWS AND BASES AS THE REST OF THE APP - nothing here is a new definition:
  rows      telegram_bot.daily_pack.settled_tips(): live tips from bets_ledger.csv + side_bets_ledger.csv,
            every sent tip (no first-per-fixture filter, exactly as the dashboard's record)
  window    match date >= config.PERFORMANCE_CUTOFF_DATE, settled only (pnl present), flat 1u
  staked    SNIPER + MARKSMAN (daily_pack.STAKED); VALUABLE is shown on its own line as paper
  paper     config.PAPER_LEAGUES rows are KEPT so the league can be shown, labelled PAPER LEAGUE,
            and are excluded from every all-league total
  upcoming  telegram_bot.tips_table.tips_rows() over tip days today..+7 (src/tip_day.py) - the sent
            tips board, so a paper league shows none
  hit       won / settled (a void counts as settled and not won) - the dashboard's (pnl > 0).mean()

MODEL PROBABILITY. Side-market ledger rows record model_prob. The O/U 2.5 ledger does not; there
v9's edge is model_prob - 1/odds (src/betting.py), so model_prob = 1/odds + edge_pct/100 - derived,
and labelled so. Side-market edge is measured against the de-vigged price, so "avg edge" across
markets mixes two bases; the by-market table keeps them apart.

A VIEWER: reads two small committed ledgers (column-projected), two small JSON configs, Pro's small
ratings JSON and one small Pro CSV. Never a parquet, never a model.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.tier_labels import PAPER_TIER_LABEL

import config

V9_DIR = Path(__file__).resolve().parents[1]
POLICY_PATH = V9_DIR / "config" / "bankroll_policy.json"
REGIMES_PATH = V9_DIR / "config" / "league_market_regimes.json"
PRO_WEEKLY = "output/studies/league_results_weekly.csv"

STAKED = ("SNIPER", "MARKSMAN")
#: overview_table column for paper (VALUABLE) P/L - display words from src/tier_labels.py
PAPER_PL_COL = f"{PAPER_TIER_LABEL} (paper) P/L"
TIERS = ("SNIPER", "MARKSMAN", "VALUABLE")
#: Display markets. Sides are never pooled: O/U 2.5 Over and Under are separate rows.
MARKETS = ["O/U 2.5 Over", "O/U 2.5 Under", "BTTS", "Over 1.5", "Over 3.5"]
_PICK_TO_MARKET = {"O/U 2.5 OVER": "O/U 2.5 Over", "O/U 2.5 UNDER": "O/U 2.5 Under", "BTTS YES": "BTTS",
                   "O/U 1.5 OVER": "Over 1.5", "O/U 3.5 OVER": "Over 3.5"}
#: display market -> (policy market, regime side)
_TRACK = {"O/U 2.5 Over": ("OU25", "OVER"), "O/U 2.5 Under": ("OU25", "UNDER"), "BTTS": ("BTTS", "YES"),
          "Over 1.5": ("OVER15", "OVER"), "Over 3.5": ("OVER35", "OVER")}
_POLICY_LABEL = {"OU25": "O/U 2.5", "OU25_OVER": "O/U 2.5 Over", "OU25_UNDER": "O/U 2.5 Under",
                 "BTTS": "BTTS", "OVER15": "Over 1.5", "OVER35": "Over 3.5"}
#: strongest first - the league's headline state is its strongest market's
STATE_ORDER = ["ACTIVE_STRONG", "ACTIVE", "ACTIVE_LOW", "PROBATION", "DEGRADED", "PAPER", "BLOCKED"]
REGIME_ORDER = ["PROTECTED", "REPAIR", "LEARNING"]
PAPER_STATE = "PAPER LEAGUE"
#: best / worst market need this many settled staked tips, so a 1-for-1 market cannot "win"
MIN_MARKET_N = 5


# ── rows ──────────────────────────────────────────────────────────────────────────────────────
def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """settled_tips(include_paper=True) frame -> the league frame: settled, from the cutoff, with
    display market, staked / paper flags, implied and model probability."""
    cols = ["date", "league", "match", "tier", "mkt", "pick", "odds", "result", "pnl", "edge_pct",
            "model_prob", "implied", "staked", "paper", "track"]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=cols)
    cut = str(getattr(config, "PERFORMANCE_CUTOFF_DATE", "1970-01-01"))[:10]
    d = raw[(raw["date"].astype(str) >= cut) & pd.to_numeric(raw["pnl"], errors="coerce").notna()
            & raw["tier"].isin(TIERS)].copy()
    d["pnl"] = pd.to_numeric(d["pnl"], errors="coerce")
    d["odds"] = pd.to_numeric(d["odds"], errors="coerce")
    d["edge_pct"] = pd.to_numeric(d.get("edge_pct"), errors="coerce") if "edge_pct" in d.columns else float("nan")
    d["mkt"] = d["pick"].astype(str).map(_PICK_TO_MARKET).fillna(d["pick"].astype(str))
    d["implied"] = 1.0 / d["odds"]
    mp = pd.to_numeric(d["model_prob"], errors="coerce") if "model_prob" in d.columns else pd.Series(
        float("nan"), index=d.index)
    ou = d["mkt"].str.startswith("O/U 2.5")
    d["model_prob"] = mp.where(~ou, d["implied"] + d["edge_pct"] / 100.0)
    d["staked"] = d["tier"].isin(STAKED)
    d["paper"] = d["league"].map(config.is_paper_league).astype(bool)
    d["track"] = d["league"].map(config.model_type_for_league)
    return d[cols].sort_values("date", kind="mergesort").reset_index(drop=True)


def settled() -> pd.DataFrame:
    """Every settled live tip since the cutoff, paper leagues included (flagged)."""
    from telegram_bot.daily_pack import settled_tips
    return prepare(settled_tips(include_paper=True))


def upcoming(today: str, days: int = 7) -> pd.DataFrame:
    """The sent tips board (tips_rows) for tip days today..today+days. Paper leagues never appear."""
    from telegram_bot.tips_table import tips_rows
    hi = (pd.Timestamp(today) + pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        return tips_rows(today, hi)
    except Exception:                                                  # noqa: BLE001 - a viewer never dies
        return pd.DataFrame()


# ── context: rating, production state, research regime ──────────────────────────────────────────
def ratings() -> dict:
    """Pro's league rating 1-10 (the stake plan's source). {} when missing or stale - shown as '–'."""
    try:
        from src.stake_plan import load_league_ratings
        return load_league_ratings()
    except Exception:                                                  # noqa: BLE001
        return {}


def _json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:                                                  # noqa: BLE001
        return {}


def policy() -> dict:
    return _json(POLICY_PATH)


def regimes() -> dict:
    return _json(REGIMES_PATH)


def production_states(league: str, pol: dict) -> dict:
    """{policy market: state} for the league's track, or {} for a paper league / unknown track."""
    if config.is_paper_league(league) or league in set((pol or {}).get("paper_leagues") or []):
        return {}
    mt = config.model_type_for_league(league)
    out = {}
    for key, v in ((pol or {}).get("tracks") or {}).items():
        if "|" in key and key.split("|", 1)[0] == mt and isinstance(v, dict):
            out[key.split("|", 1)[1]] = str(v.get("state", "?"))
    return out


def production_state(league: str, pol: dict) -> tuple[str, str]:
    """(headline, detail). Headline = PAPER LEAGUE, or the strongest market state of the league's
    track. Detail spells out every market."""
    if config.is_paper_league(league) or league in set((pol or {}).get("paper_leagues") or []):
        return PAPER_STATE, "paper league: collected and settled, never sent, counted or bet"
    s = production_states(league, pol)
    if not s:
        return "–", "no policy track for this league"
    rank = {k: i for i, k in enumerate(STATE_ORDER)}
    head = min(s.values(), key=lambda v: rank.get(v, len(STATE_ORDER)))
    detail = " · ".join(f"{_POLICY_LABEL.get(m, m)} {v}" for m, v in
                        sorted(s.items(), key=lambda kv: rank.get(kv[1], len(STATE_ORDER))))
    return head, detail


def regime_entries(league: str, reg: dict) -> dict:
    """{display market: research regime} from config/league_market_regimes.json."""
    out = {}
    back = {v: k for k, v in _TRACK.items()}
    for e in (reg or {}).get("entries") or []:
        if e.get("league") == league:
            mk = back.get((e.get("market"), e.get("side")))
            if mk:
                out[mk] = str(e.get("research_regime", "?"))
    return out


def research_regime(league: str, reg: dict) -> str:
    """Headline regime: PROTECTED if any market is, else REPAIR if any is, else LEARNING. With the count."""
    r = regime_entries(league, reg)
    if not r:
        return "–"
    for k in REGIME_ORDER:
        n = sum(v == k for v in r.values())
        if n:
            return k if k == "LEARNING" and n == len(r) else f"{k} ({n})"
    return "–"


# ── aggregation ───────────────────────────────────────────────────────────────────────────────
def summarise(g: pd.DataFrame) -> dict:
    """n, W, L, voids, hit, ROI, P/L, avg odds, avg edge, avg model probability, avg implied."""
    n = int(len(g))
    if n == 0:
        return {"tips": 0, "won": 0, "lost": 0, "void": 0, "hit": float("nan"), "roi": float("nan"),
                "pl": 0.0, "avg_odds": float("nan"), "avg_edge": float("nan"),
                "avg_model_prob": float("nan"), "avg_implied": float("nan")}
    pnl = g["pnl"]
    won, lost = int((pnl > 0).sum()), int((pnl < 0).sum())
    pl = float(pnl.sum())
    return {"tips": n, "won": won, "lost": lost, "void": n - won - lost, "hit": won / n, "roi": pl / n,
            "pl": round(pl, 4), "avg_odds": float(g["odds"].mean()), "avg_edge": float(g["edge_pct"].mean()),
            "avg_model_prob": float(g["model_prob"].mean()), "avg_implied": float(g["implied"].mean())}


def _best_worst(st: pd.DataFrame) -> tuple[str, str]:
    if st.empty:
        return "–", "–"
    m = st.groupby("mkt")["pnl"].agg(["size", "sum"])
    m = m[m["size"] >= MIN_MARKET_N] if (m["size"] >= MIN_MARKET_N).any() else m
    if len(m) == 0:
        return "–", "–"
    m = m.sort_values("sum", ascending=False)
    fmt = lambda k: f"{k} ({m.loc[k, 'sum']:+.1f}u)"                    # noqa: E731
    best = fmt(m.index[0])
    worst = fmt(m.index[-1]) if len(m) > 1 else "–"
    return best, worst


def leagues_list(d: pd.DataFrame) -> list[str]:
    """Every league that is live-predicted or has a settled live tip."""
    seen = set(d["league"].dropna().astype(str)) if len(d) else set()
    return sorted(seen | set(getattr(config, "ENABLED_LEAGUES", set())))


def overview(d: pd.DataFrame, today: str, up: pd.DataFrame | None = None, rat: dict | None = None,
             pol: dict | None = None, reg: dict | None = None, only_with_tips: bool = False) -> pd.DataFrame:
    """One row per league. Results columns are STAKED tiers (SNIPER + MARKSMAN); VALUABLE is its
    own pair of columns (paper). Sorted best P/L first, paper leagues last."""
    rat = {} if rat is None else rat
    week = (pd.Timestamp(today) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    up_n = (up["league"].value_counts() if up is not None and len(up) and "league" in up.columns
            else pd.Series(dtype=int))
    rows = []
    for lg in leagues_list(d):
        g = d[d["league"] == lg]
        st = g[g["staked"]]
        if only_with_tips and g.empty:
            continue
        s = summarise(st)
        v = g[g["tier"] == "VALUABLE"]
        w7 = st[st["date"] >= week]
        state, state_detail = production_state(lg, pol or {})
        best, worst = _best_worst(st)
        r = rat.get(lg)
        rows.append({"league": lg,
                     "track": {"standard": "standard", "new_format": "new-format"}.get(
                         config.model_type_for_league(lg), "unknown"),
                     "rating": int(r) if r is not None else None,
                     "state": state, "state_detail": state_detail,
                     "regime": research_regime(lg, reg or {}),
                     **s,
                     "pl_7d": round(float(w7["pnl"].sum()), 4), "tips_7d": int(len(w7)),
                     "valuable_tips": int(len(v)), "valuable_pl": round(float(v["pnl"].sum()), 4),
                     "upcoming": int(up_n.get(lg, 0)),
                     "best_market": best, "worst_market": worst,
                     "paper": bool(config.is_paper_league(lg))})
    ov = pd.DataFrame(rows)
    if ov.empty:
        return ov
    return ov.sort_values(["paper", "pl", "tips"], ascending=[True, False, False],
                          kind="mergesort").reset_index(drop=True)


def wl(won, lost, void) -> str:
    return f"{int(won)}-{int(lost)}" + (f" ({int(void)}v)" if void else "")


def overview_table(ov: pd.DataFrame) -> pd.DataFrame:
    """The display table the page shows AND the Telegram picture draws (rounded once, here)."""
    if ov is None or ov.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "League": ov["league"] + ov["paper"].map({True: " (paper)", False: ""}),
        "Track": ov["track"],
        "Rating": ov["rating"].map(lambda x: "–" if x is None or pd.isna(x) else f"{int(x)}/10"),
        "State": ov["state"],
        "Regime": ov["regime"],
        "Tips": ov["tips"].astype(int),
        "W-L": [wl(w, l, v) for w, l, v in zip(ov["won"], ov["lost"], ov["void"])],
        "Hit %": (ov["hit"] * 100).round(1),
        "ROI %": (ov["roi"] * 100).round(1),
        "P/L (u)": ov["pl"].round(2),
        "Avg odds": ov["avg_odds"].round(2),
        "Avg edge %": ov["avg_edge"].round(1),
        "Model %": (ov["avg_model_prob"] * 100).round(1),
        "Implied %": (ov["avg_implied"] * 100).round(1),
        "P/L 7d (u)": ov["pl_7d"].round(2),
        PAPER_PL_COL: ov["valuable_pl"].round(2),
        "Upcoming": ov["upcoming"].astype(int),
        "Best market": ov["best_market"],
        "Worst market": ov["worst_market"],
    })


def totals(d: pd.DataFrame) -> dict:
    """All leagues, staked, paper leagues EXCLUDED - the same number the dashboard's record shows."""
    return summarise(d[d["staked"] & ~d["paper"]])


def by_market(g: pd.DataFrame, tiers=STAKED, reg: dict | None = None, league: str | None = None) -> pd.DataFrame:
    """One row per display market (sides apart), on the given tiers. Markets with no tips are omitted."""
    x = g[g["tier"].isin(tiers)]
    regs = regime_entries(league, reg) if league and reg else {}
    rows = []
    for mk in MARKETS + sorted(set(x["mkt"]) - set(MARKETS)):
        s = summarise(x[x["mkt"] == mk])
        if s["tips"]:
            rows.append({"market": mk, **s, "regime": regs.get(mk, "–")})
    return pd.DataFrame(rows)


def by_tier(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in TIERS:
        s = summarise(g[g["tier"] == t])
        if s["tips"]:
            rows.append({"tier": t + ("" if t in STAKED else " (paper)"), **s})
    return pd.DataFrame(rows)


def recent(g: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    x = g.sort_values("date", kind="mergesort").tail(n).iloc[::-1]
    return pd.DataFrame({"date": x["date"], "match": x["match"], "market": x["mkt"],
                         "tier": x["tier"].where(x["staked"], x["tier"] + " (paper)"),
                         "odds": x["odds"].round(2), "result": x["result"], "pnl": x["pnl"].round(2)})


def cumulative(g: pd.DataFrame) -> pd.DataFrame:
    """Running staked P/L by match date (one point per day)."""
    st = g[g["staked"]]
    if st.empty:
        return pd.DataFrame(columns=["date", "cum_pl"])
    s = st.groupby("date")["pnl"].sum().cumsum()
    return pd.DataFrame({"date": pd.to_datetime(s.index), "cum_pl": s.values.round(3)})


def last_week(d: pd.DataFrame, today: str) -> pd.DataFrame:
    """Each league's settled tips in the 7 match days before `today` (the Monday picture's window,
    same as daily_pack.week_images): staked W-L / P/L and paper VALUABLE. Only leagues with tips."""
    start = (pd.Timestamp(today) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    w = d[(d["date"] >= start) & (d["date"] < today)]
    rows = []
    for lg, g in w.groupby("league"):
        s = summarise(g[g["staked"]])
        v = g[g["tier"] == "VALUABLE"]
        rows.append({"league": lg, "paper": bool(config.is_paper_league(lg)), **s,
                     "valuable_tips": int(len(v)), "valuable_pl": round(float(v["pnl"].sum()), 4)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["paper", "pl"], ascending=[True, False], kind="mergesort").reset_index(drop=True)


# ── Pro context + freshness ──────────────────────────────────────────────────────────────────
def pro_weekly() -> pd.DataFrame:
    """Pro's weekly league table (src/studies/league_results.py), local checkout or public repo."""
    try:
        from dashboard_data import pro
        return pro.read_csv(pro._f(PRO_WEEKLY))
    except Exception:                                                  # noqa: BLE001
        return pd.DataFrame()


def pro_context(weekly: pd.DataFrame, league: str) -> pd.DataFrame:
    """Latest week's rows for the league (season + last 30 days windows)."""
    if weekly is None or weekly.empty or "league" not in weekly.columns:
        return pd.DataFrame()
    x = weekly[(weekly["week"] == weekly["week"].max()) & (weekly["league"] == league)]
    return x.reset_index(drop=True)


def freshness_stamps(weekly: pd.DataFrame | None = None) -> dict:
    """Recorded timestamps of every input (never file mtime). None = unknown (treated as old)."""
    out = {}
    newest = None
    for f in ("bets_ledger.csv", "side_bets_ledger.csv"):
        p = config.OUTPUT_DIR / f
        if p.exists():
            try:
                t = pd.to_datetime(pd.read_csv(p, usecols=["generated_at"])["generated_at"],
                                   errors="coerce", utc=True).max()
                newest = t if newest is None or (pd.notna(t) and t > newest) else newest
            except Exception:                                          # noqa: BLE001
                pass
    out["ledgers (newest tip recorded)"] = newest if newest is not None and pd.notna(newest) else None
    reg = regimes()
    out["research regimes (generated for)"] = reg.get("generated_for")
    out["bankroll policy (generated)"] = policy().get("generated_at")
    if weekly is not None and len(weekly) and "updated_at" in weekly.columns:
        out["Pro league table (updated)"] = str(weekly["updated_at"].dropna().max()).replace(" UTC", "Z")
    return out
