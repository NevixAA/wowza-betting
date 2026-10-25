"""Top picks — the 5 tips a person should take from a long list, in three lists (owner 2026-10-18).

    from src.top_picks import freeze_day, todays_lists, scoreboard

WHY. Some days carry 60+ tips. People take a few at random, and two people picking 5 from the same
list get 0/5 and 5/5. Validated in Pro (wowzaV9-Pro src/studies/tip_ranker.py, TIP_RANKER.md, 36
days with >= 10 tips, walk-forward): a random 5 hit 47%; the lists below hit 55-63%.

THE THREE LISTS (top 5 of the day's tips each; a tip may appear in more than one):
  BEST_BETS  highest expected profit among tips with P(win) >= 45%      (backtest 55% hit, +9% ROI)
  SAFEST     the 5 SHORTEST PRICES - chosen over the ML version because it
             won the backtest (63% hit, no 0/5 day vs 57%), owner 2026-10-18 (backtest 63%, +9%)
  VALUE      highest expected profit                                     (backtest 55%, +12%)
P(win) is learned each morning from every tip settled before that day: logistic regression on
logit(1/odds), logit(model probability), market and model track. The raw model probability is not
used directly - it is overconfident by ~13.6 points. Expected profit = P(win) x odds - 1.

TRACKABLE. freeze_day() writes the day's lists ONCE to output/top_picks_log.csv (first sight; a day
already logged is never rewritten), and scoreboard() grades the frozen picks against the ledgers'
results. What users were shown is exactly what is scored - and is the data the ranker learns from.
No tip, tier, stake or bet changes here; this only orders and records.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

LISTS = ("BEST_BETS", "SAFEST", "VALUE")
LIST_LABEL = {"BEST_BETS": "Best bets", "SAFEST": "Safest", "VALUE": "Value"}
TIERS = ("SNIPER", "MARKSMAN", "VALUABLE")
TOP_K = 5
HIT_FLOOR = 0.45
MIN_TRAIN = 300
LOG = config.OUTPUT_DIR / "top_picks_log.csv"
MARKET_LABEL = {"OU25_OVER": "Over 2.5", "OU25_UNDER": "Under 2.5", "btts": "BTTS Yes",
                "over15": "Over 1.5", "over35": "Over 3.5"}
KEY = ["day", "home_team", "away_team", "market"]


def _lg(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def load_tips() -> pd.DataFrame:
    """Every live tiered tip from both ledgers, one per fixture x market (first tip stands)."""
    parts = []
    p = config.OUTPUT_DIR / "bets_ledger.csv"
    if p.exists():
        m = pd.read_csv(p, low_memory=False)
        if "source" in m.columns:
            m = m[m["source"].fillna("live").astype(str).eq("live")]
        m["market"] = "OU25_" + m["side"].astype(str).str.upper()
        parts.append(m)
    p = config.OUTPUT_DIR / "side_bets_ledger.csv"
    if p.exists():
        s = pd.read_csv(p, low_memory=False)
        s["market"] = s["market"].astype(str).str.lower()
        parts.append(s)
    if not parts:
        return pd.DataFrame()
    d = config.drop_paper_leagues(pd.concat(parts, ignore_index=True))
    d = d[d["signal_tier"].isin(TIERS) & d["market"].isin(MARKET_LABEL)].copy()
    d["odds"] = pd.to_numeric(d["odds"], errors="coerce")
    d = d[d["odds"] > 1.01]
    d["implied"] = 1.0 / d["odds"]
    e = pd.to_numeric(d["edge_pct"], errors="coerce")
    e = e / 100.0 if e.abs().median() > 1.0 else e                  # stored as % or as a fraction
    mp = pd.to_numeric(d["model_prob"], errors="coerce") if "model_prob" in d.columns else pd.Series(np.nan, index=d.index)
    d["model_p"] = mp.where(mp.notna(), d["implied"] + e).clip(0.02, 0.98)
    d["model_type"] = d["model_type"].fillna("unknown").astype(str)
    # day = TIP DAY, the album that carries the tip (src/tip_day.py, 2026-10-25); match_day kept so
    # lists frozen before the change (keyed by match date) still grade.
    from src.tip_day import add_tip_day
    d["match_day"] = d["match_date"].astype(str).str[:10]
    d["day"] = add_tip_day(d, date_col="match_day")["tip_day"].values
    d["result"] = d["result"].fillna("").astype(str).str.upper()
    d = d.sort_values("generated_at", kind="mergesort").drop_duplicates(KEY, keep="first")
    return d.reset_index(drop=True)


def _design(d: pd.DataFrame, markets: list[str], tracks: list[str]) -> np.ndarray:
    cols = [_lg(d["implied"]), _lg(d["model_p"])]
    cols += [(d["market"] == m).astype(float).to_numpy() for m in markets[1:]]
    cols += [(d["model_type"] == t).astype(float).to_numpy() for t in tracks[1:]]
    return np.column_stack(cols)


def p_win(tips: pd.DataFrame, day: str, history: pd.DataFrame | None = None) -> pd.Series:
    """P(win) for `tips`, from a model fitted on tips settled BEFORE `day`. NaN if too little history."""
    from sklearn.linear_model import LogisticRegression
    h = history if history is not None else load_tips()
    h = h[(h["day"] < day) & h["result"].isin(["WIN", "LOSS"])]
    if len(h) < MIN_TRAIN or tips.empty:
        return pd.Series(np.nan, index=tips.index)
    markets, tracks = sorted(h["market"].unique()), sorted(h["model_type"].unique())
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(_design(h, markets, tracks), (h["result"] == "WIN").astype(int))
    return pd.Series(lr.predict_proba(_design(tips, markets, tracks))[:, 1], index=tips.index)


def todays_lists(day: str, tips: pd.DataFrame | None = None) -> pd.DataFrame:
    """The three lists for `day` (not frozen). Columns: list, rank, day, league, home_team, away_team,
    market, pick, odds, p_win, ev, signal_tier, model_type."""
    allt = tips if tips is not None else load_tips()
    if allt is None or allt.empty:
        return pd.DataFrame()
    g = allt[allt["day"] == day].copy()
    if g.empty:
        return pd.DataFrame()
    g["p_win"] = p_win(g, day, allt)
    g["ev"] = g["p_win"] * g["odds"] - 1.0
    picks = {"SAFEST": g.nsmallest(TOP_K, "odds")}
    if g["p_win"].notna().any():
        picks["VALUE"] = g.nlargest(TOP_K, "ev")
        picks["BEST_BETS"] = g[g["p_win"] >= HIT_FLOOR].nlargest(TOP_K, "ev")
    out = []
    for name in LISTS:
        p = picks.get(name)
        if p is None or p.empty:
            continue
        p = p.assign(list=name, rank=range(1, len(p) + 1))
        out.append(p)
    if not out:
        return pd.DataFrame()
    d = pd.concat(out, ignore_index=True)
    d["pick"] = d["market"].map(MARKET_LABEL)
    cols = ["list", "rank", "day", "league", "home_team", "away_team", "market", "pick", "odds", "p_win", "ev",
            "signal_tier", "model_type"]
    d = d[cols]
    d[["p_win", "ev"]] = d[["p_win", "ev"]].round(4)
    return d


def frozen(day: str | None = None) -> pd.DataFrame:
    if not LOG.exists():
        return pd.DataFrame()
    d = pd.read_csv(LOG)
    return d if day is None else d[d["day"].astype(str) == day]


def freeze_day(day: str) -> pd.DataFrame:
    """Log the day's lists once (first sight) and return what is frozen for the day. Never raises."""
    try:
        have = frozen(day)
        if not have.empty:
            return have
        d = todays_lists(day)
        if d.empty:
            return d
        d.insert(0, "frozen_at", pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"))
        d.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
        try:                                     # SHADOW contract check (src/contracts); observe only
            from src.contracts.shadow import observe
            observe("top_picks_log.csv (appended rows)", d, "TOP_PICKS_ROW")
        except Exception:                                             # noqa: BLE001
            pass
        log.info("top picks frozen for %s: %s", day, d.groupby("list").size().to_dict())
        return d
    except Exception as e:                                           # noqa: BLE001 - never break the album
        log.warning("top picks freeze skipped: %s", e)
        return pd.DataFrame()


def graded(days: int | None = None) -> pd.DataFrame:
    """Frozen picks joined to their results (WIN / LOSS / pending)."""
    f = frozen()
    if f.empty:
        return f
    t = load_tips()
    f["day"] = f["day"].astype(str)
    g = f.merge(t[KEY + ["result"]], on=KEY, how="left")
    # lists frozen before 2026-10-25 used the match date as the day: grade those by match date
    alt = t[["match_day", "home_team", "away_team", "market", "result"]].rename(
        columns={"match_day": "day", "result": "_r2"}).drop_duplicates(KEY)
    g = g.merge(alt, on=KEY, how="left")
    g["result"] = g["result"].where(g["result"].notna() & (g["result"] != ""), g["_r2"])
    g = g.drop(columns="_r2")
    g["result"] = g["result"].fillna("").replace({"": "PENDING"})
    if days:
        cut = (pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        g = g[g["day"] >= cut]
    return g


def scoreboard(days: int | None = None) -> pd.DataFrame:
    """Per list: settled picks, hit rate, flat-1u ROI, days, share of 0/5 days."""
    g = graded(days)
    if g.empty:
        return pd.DataFrame(columns=["list", "days", "picks", "wins", "hit_rate", "roi", "zero_days"])
    s = g[g["result"].isin(["WIN", "LOSS"])].copy()
    s["win"] = (s["result"] == "WIN").astype(int)
    s["ret"] = np.where(s["win"] == 1, s["odds"] - 1.0, -1.0)
    rows = []
    for name in LISTS:
        x = s[s["list"] == name]
        if x.empty:
            continue
        per_day = x.groupby("day")["win"].agg(["sum", "size"])
        rows.append({"list": name, "days": int(len(per_day)), "picks": int(len(x)), "wins": int(x["win"].sum()),
                     "hit_rate": round(float(x["win"].mean()), 3), "roi": round(float(x["ret"].mean()), 3),
                     "zero_days": round(float((per_day["sum"] == 0).mean()), 3)})
    return pd.DataFrame(rows)


def day_line(day: str) -> str:
    """'Best bets 3/5 · Safest 4/5 · Value 2/5' for a graded day ('' if nothing frozen)."""
    g = graded()
    if g.empty:
        return ""
    g = g[g["day"] == day]
    parts = []
    for name in LISTS:
        x = g[(g["list"] == name) & g["result"].isin(["WIN", "LOSS"])]
        if len(x):
            parts.append(f"{LIST_LABEL[name]} {int((x['result'] == 'WIN').sum())}/{len(x)}")
    return " · ".join(parts)
