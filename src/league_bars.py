"""Per-league tip bars (SNIPER / MARKSMAN / VALUABLE) from the clean Phase 2 retrain - PROPOSALS ONLY.

    python scripts/league_bars.py --predictions <run>/predictions.parquet --track standard ...

OWNER DECISIONS (2026-11-08)
  * EVERY league gets its OWN bars, for every market and every tier, VALUABLE included. There is no
    global bar anywhere in this module ("it can't be global... every league has different numbers").
  * "Reasonable bars": a league's bar comes from its OWN walk-forward bets. A league with few bets is
    pulled toward the pooled (all leagues, same track + market) estimate by an empirical-Bayes weight
    LEARNED from the walk-forward data (DerSimonian-Laird between-league variance) - no typed
    shrinkage constant. The result is still one number per league.
  * STRICT pass rule: PROVEN only when the HOLDOUT has >= MIN_HOLDOUT_BETS bets AND the ROI is above 0
    with Holm-adjusted one-sided p < HOLM_ALPHA_ONE_SIDED (= a 95% two-sided interval above zero after
    the correction) across EVERY league x market tested. Otherwise UNPROVEN, with the reason.
  * NO_PRICE_DATA only for a league x market with no real two-sided bet-time price AT ANY TIME. No
    number is invented for it. ("no way with 100% real data we don't have true bars": a league priced
    only from 2026-04-01 gets IN_PERIOD_SPLIT bars, below.)

WHAT A BAR IS. The minimum edge (p_model - fair_prob) at which a fixture is tipped at that tier. The
fair probability is the POWER DE-VIG of a same-snapshot (or same-book) two-sided pair, captured before
kickoff; a fixture without such a pair is never bet. The closing price is never used.

THE THREE BARS, each with one meaning and no tuning constant (shrunk ROI curve r(t), bets n(t)):
  VALUABLE  the lowest bar whose shrunk ROI is above zero              (break-even bar)
  MARKSMAN  the bar that maximises shrunk total profit r(t) * n(t)     (most units won)
  SNIPER    the bar that maximises shrunk ROI per bet r(t)             (best return per bet)
  forced into VALUABLE <= MARKSMAN <= SNIPER. Only bars with >= MIN_BETS_FIT selection-period bets of
  the league itself are eligible (bar_basis OWN_BETS); a priced league that never reaches that is read
  off its shrunk curve where the POOLED bets do (bar_basis SHRUNK_TO_POOL, its weight reported).
  Reused from scripts/league_thresholds.py: coarse grid, bars chosen on one window, scored on another,
  block bootstrap by match date.

NOTHING IS SCORED ON THE ROWS THAT CHOSE IT. Two kinds of evidence, labelled on every row:
  WALK_FORWARD_HOLDOUT  bars chosen on stage == "walk_forward" rows only; the holdout is scored once,
                        after every bar is fixed.
  IN_PERIOD_SPLIT       (owner 2026-11-08) the league has no priced walk-forward row but has priced
                        holdout rows (e.g. Over 1.5 / Over 3.5, two-sided only from 2026-04-01). The
                        league's priced holdout matchdays are split in time order at a month boundary
                        so the CHECK part holds about 40% of them; bars are chosen on the earlier part
                        (shrunk toward every league's priced rows in that same window) and scored once
                        on the later part. Same strict rule; split dates are recorded per league.
                        Weaker evidence than the full walk-forward + holdout (a shorter, single window).

This module writes nothing outside output/threshold_proposals/ (src.threshold_proposals._safe_write).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import team_names as TN
from src.market_edge import power_devig

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("_league_thresholds", ROOT / "scripts" / "league_thresholds.py")
_LT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_LT)

#: reused from scripts/league_thresholds.py (coarse on purpose; a bar that only wins at a finer
#: resolution is fitted to the sample)
GRID = _LT.GRID
MIN_BETS_FIT = _LT.MIN_BETS_FIT
N_BOOT = _LT.N_BOOT
block_ci = _LT._block_ci

#: owner's strict rule (2026-11-08)
MIN_HOLDOUT_BETS = 150
HOLM_ALPHA_ONE_SIDED = 0.025

STATUS_PROVEN = "PROVEN"
STATUS_UNPROVEN = "UNPROVEN"
STATUS_NO_PRICE = "NO_PRICE_DATA"
STATUS_THIN = "INSUFFICIENT_BETS"

#: markets that bet one side only (the production side generator bets Yes / Over); O/U 2.5 bets both
ONE_SIDED_BET = {"btts": ("btts_yes", "btts_no"), "over15": ("over15", "under15"),
                 "over35": ("over35", "under35"),
                 # P7 item 2 (2026-11-09): first-half goals O/U 0.5 / 1.5. Two-sided pairs ARE captured
                 # (standard_sidemarket_odds_history since 2026-08-21, 17 leagues; new_format from the
                 # P7 item 3 parser on). The bars had them as NO_PRICE_DATA only because no loader read them.
                 "ht_over05": ("ht_over05", "ht_under05"), "ht_over15": ("ht_over15", "ht_under15")}
TWO_SIDED_BET = {"over25": ("over25", "under25")}
SNAPSHOT_KEYS = {**ONE_SIDED_BET, **TWO_SIDED_BET}


# ── prices ──────────────────────────────────────────────────────────────────────────────────────

def snapshot_pairs(raw: pd.DataFrame, source: str, market: str, which: str = "first") -> pd.DataFrame:
    """One bet-time two-sided pair per (match_date, league, match) from a snapshot capture file.

    Rules: only rows price_provenance marks market_mapping_valid; both sides from the SAME snapshot;
    captured BEFORE kickoff - snapshot_ts < kickoff_utc when the kickoff is recorded, otherwise
    snapshot_ts <= match_date 00:00 UTC (a backfill stamp) or on an earlier date; any other row is
    unprovable and dropped. Of the provable pre-kickoff pairs captured from match_date - 1 day on, the
    EARLIEST is the bet-time price (the latest would be the closing price)."""
    from src.integrity.price_provenance import label_market_mapping
    yes_k, no_k = SNAPSHOT_KEYS[market]
    d = raw[raw["market"].isin([yes_k, no_k])].copy()
    if d.empty:
        return pd.DataFrame(columns=["match_date", "league", "match", "yes", "no", "snapshot_ts", "price_origin"])
    d = label_market_mapping(d, source)
    d = d[d["market_mapping_valid"]]
    d["ts"] = pd.to_datetime(d["snapshot_ts"], errors="coerce", utc=True, format="mixed")
    d["ko"] = pd.to_datetime(d["kickoff_utc"], errors="coerce", utc=True, format="mixed") \
        if "kickoff_utc" in d.columns else pd.NaT
    d["day0"] = pd.to_datetime(d["match_date"], errors="coerce").dt.tz_localize("UTC")
    d = d[d["ts"].notna() & d["day0"].notna()]
    pre = np.where(d["ko"].notna(), d["ts"] < d["ko"], d["ts"] <= d["day0"])
    d = d[pre & (d["ts"] >= d["day0"] - pd.Timedelta(days=1))]
    d["odds"] = pd.to_numeric(d["odds"], errors="coerce")
    w = d.pivot_table(index=["match_date", "league", "match", "ts", "price_origin"], columns="market",
                      values="odds", aggfunc="first").reset_index()
    if yes_k not in w or no_k not in w:
        return pd.DataFrame(columns=["match_date", "league", "match", "yes", "no", "snapshot_ts", "price_origin"])
    w = w[w[yes_k].notna() & w[no_k].notna()].sort_values("ts")
    if which not in ("first", "last"):
        raise ValueError(f"which={which!r}: 'first' (bet-time price) or 'last' (closing, CLV only)")
    g = w.groupby(["match_date", "league", "match"], as_index=False)
    w = g.first() if which == "first" else g.last()     # last = the latest pre-kickoff pair (CLV only)
    return w.rename(columns={yes_k: "yes", no_k: "no", "ts": "snapshot_ts"})[
        ["match_date", "league", "match", "yes", "no", "snapshot_ts", "price_origin"]]


def split_match(s: str) -> tuple[str | None, str | None]:
    parts = str(s).split(" vs ")
    return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (None, None)


def fd_ou25_pairs(fd_raw_dir) -> pd.DataFrame:
    """Football-data PRE-MATCH (non-closing) O/U 2.5 pairs, one SAME-BOOK pair per match: Bet365
    (B365>2.5 / B365<2.5), else Pinnacle (P>2.5 / P<2.5). Average / maximum columns are never used
    (not one book's price, not executable). Closing (C) columns are never read."""
    from src.standard_repair import FD_CODES
    inv = {v: k for k, v in FD_CODES.items()}
    frames = []
    for p in sorted(Path(fd_raw_dir).glob("*_*.csv")):
        lg = inv.get(p.stem.rsplit("_", 1)[0])
        if not lg:
            continue
        try:
            raw = pd.read_csv(p, encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
        except UnicodeDecodeError:
            raw = pd.read_csv(p, encoding="cp1252", encoding_errors="replace", on_bad_lines="skip",
                              low_memory=False)
        if not {"Date", "HomeTeam", "AwayTeam"} <= set(raw.columns):
            continue
        out = pd.DataFrame({"date": pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce").dt.normalize(),
                            "league": lg, "home": raw["HomeTeam"], "away": raw["AwayTeam"]})
        out["yes"], out["no"], out["book"] = np.nan, np.nan, ""
        for book, o, u in (("B365", "B365>2.5", "B365<2.5"), ("Pinnacle", "P>2.5", "P<2.5")):
            if o not in raw or u not in raw:
                continue
            oo, uu = pd.to_numeric(raw[o], errors="coerce"), pd.to_numeric(raw[u], errors="coerce")
            take = out["yes"].isna() & (oo > 1) & (uu > 1)
            out.loc[take, "yes"], out.loc[take, "no"], out.loc[take, "book"] = oo[take], uu[take], book
        frames.append(out[out["date"].notna() & out["yes"].notna()])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["date", "league", "home", "away", "yes", "no", "book"])


def attach_prices(pred: pd.DataFrame, prices: pd.DataFrame, aliases: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Join a pair to each prediction row. prices: date, league, home, away, yes, no.
    Clubs are matched ONLY by src.team_names.resolve_identity against the league's canonical pool;
    the date must be equal, or within one day when that league pair of clubs is unique in the window
    (a provider/UK date difference). Unmatched rows stay unpriced and are counted, never guessed.
    Returns (pred with yes/no columns, {league: {"rows": n, "priced": k}})."""
    aliases = aliases or {}
    p = pred.copy()
    parts = p["row_id"].str.split("|", expand=True)
    p["home"], p["away"] = parts[2], parts[3]
    p["d"] = pd.to_datetime(p["date"]).dt.normalize()
    p["yes"], p["no"] = np.nan, np.nan
    stats = {}
    for lg, g in p.groupby("league"):
        pool = sorted(set(g["home"]) | set(g["away"]))
        pr = prices[prices["league"] == lg]
        idx = {}
        if not pr.empty:
            amap = aliases.get(lg)
            cache = {}

            def res(n):
                if n not in cache:
                    cache[n] = TN.resolve_identity(n, pool, amap) if isinstance(n, str) else None
                return cache[n]
            for r in pr.itertuples(index=False):
                h, a = res(r.home), res(r.away)
                if h and a and h != a:
                    idx.setdefault((h, a), []).append((pd.Timestamp(r.date).normalize(), r.yes, r.no))
        n_priced = 0
        for i, r in zip(g.index, g.itertuples(index=False)):
            cands = idx.get((r.home, r.away), [])
            exact = [c for c in cands if c[0] == r.d]
            near = [c for c in cands if abs((c[0] - r.d).days) <= 1]
            hit = exact[0] if len(exact) == 1 else (near[0] if not exact and len(near) == 1 else None)
            if hit is not None:
                p.at[i, "yes"], p.at[i, "no"] = hit[1], hit[2]
                n_priced += 1
        stats[lg] = {"rows": int(len(g)), "priced": n_priced}
    return p.drop(columns=["d"]), stats


# ── edges and P/L ───────────────────────────────────────────────────────────────────────────────

def bets_frame(p: pd.DataFrame, market: str) -> pd.DataFrame:
    """Edge and 1-unit P/L of the side production would take. Rows without a pair or whose pair
    cannot be de-vigged are dropped (no bet)."""
    d = p[p["yes"].notna() & p["no"].notna()].copy()
    fair = np.array([power_devig(float(a), float(b)) for a, b in zip(d["yes"], d["no"])], dtype=object)
    ok = np.array([f is not None for f in fair], dtype=bool)
    d = d[ok].copy()
    d["fair_yes"] = np.array([float(f) for f in fair[ok]])
    e_yes = d["p"].to_numpy(float) - d["fair_yes"].to_numpy(float)
    y = d["y"].to_numpy(float)
    if market in TWO_SIDED_BET:
        take_yes = e_yes >= 0
        d["side"] = np.where(take_yes, "yes", "no")
        d["edge"] = np.abs(e_yes)
        odds = np.where(take_yes, d["yes"].to_numpy(float), d["no"].to_numpy(float))
        won = np.where(take_yes, y == 1, y == 0)
    else:
        d["side"] = "yes"
        d["edge"] = e_yes
        odds = d["yes"].to_numpy(float)
        won = y == 1
    d["odds_taken"] = odds
    d["pnl"] = np.where(won, odds - 1.0, -1.0)
    return d


# ── bar selection ───────────────────────────────────────────────────────────────────────────────

def roi_curve(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    e, pnl = d["edge"].to_numpy(float), d["pnl"].to_numpy(float)
    for t in GRID:
        m = e >= t
        n = int(m.sum())
        rows.append({"t": float(t), "n": n, "roi": float(pnl[m].mean()) if n else np.nan,
                     "var": float(pnl[m].var(ddof=1)) if n > 1 else np.nan})
    return pd.DataFrame(rows)


def between_league_variance(curves: dict, t: float) -> float:
    """DerSimonian-Laird tau^2 of the leagues' ROI at bar t (method of moments, floored at 0)."""
    r, v = [], []
    for c in curves.values():
        row = c[c["t"] == t].iloc[0]
        if row["n"] >= 2 and np.isfinite(row["roi"]) and np.isfinite(row["var"]) and row["var"] > 0:
            r.append(row["roi"])
            v.append(row["var"] / row["n"])
    k = len(r)
    if k < 2:
        return 0.0
    r, v = np.array(r), np.array(v)
    w = 1.0 / v
    mu = (w * r).sum() / w.sum()
    q = (w * (r - mu) ** 2).sum()
    c = w.sum() - (w ** 2).sum() / w.sum()
    return float(max(0.0, (q - (k - 1)) / c)) if c > 0 else 0.0


def choose_bars(curve: pd.DataFrame, pooled: pd.DataFrame, tau2: dict) -> dict:
    """Shrink the league's ROI curve toward the pooled one, then read the three bars off it."""
    c = curve.merge(pooled[["t", "roi", "n"]].rename(columns={"roi": "roi_pool", "n": "n_pool"}), on="t")
    se2 = c["var"] / c["n"]
    t2 = c["t"].map(tau2)
    # fewer than 2 own bets at a bar = no own variance there: the league's weight at that bar is 0
    w = np.where((c["n"] >= 2) & np.isfinite(se2) & (se2 > 0), t2 / (t2 + se2), 0.0)
    c["w"] = w
    own = c["roi"].where(np.isfinite(c["roi"]), c["roi_pool"])
    c["roi_shrunk"] = w * own + (1 - w) * c["roi_pool"]
    basis = "OWN_BETS"
    elig = c[(c["n"] >= MIN_BETS_FIT) & np.isfinite(c["roi_shrunk"])]
    if elig.empty:
        # owner 2026-11-08 ("no way with 100% real data we don't have true bars"): a priced league with
        # fewer than MIN_BETS_FIT own bets at every bar still gets bars - read off its shrunk curve where
        # the POOLED real bets reach MIN_BETS_FIT; its own weight (reported) is then small or 0.
        basis = "SHRUNK_TO_POOL"
        elig = c[(c["n_pool"] >= MIN_BETS_FIT) & np.isfinite(c["roi_shrunk"])]
    if elig.empty:
        return {"status": STATUS_THIN, "reason": f"fewer than {MIN_BETS_FIT} bets at every bar, own and pooled"}
    best = elig["roi_shrunk"].max()
    sniper = float(elig.loc[elig["roi_shrunk"] == best, "t"].max())
    positive = elig[elig["roi_shrunk"] > 0]
    if positive.empty:
        valuable = marksman = sniper
        note = "no bar has a positive shrunk selection-period ROI: all three bars set to the least-negative bar"
    else:
        valuable = float(positive["t"].min())
        prof = positive["roi_shrunk"] * np.maximum(positive["n"], 1)
        marksman = float(positive.loc[prof.idxmax(), "t"])
        note = ""
    marksman = max(marksman, valuable)
    sniper = max(sniper, marksman)
    at = c.set_index("t")
    own_roi = at.at[valuable, "roi"]
    return {"status": None, "valuable_th": valuable, "marksman_th": marksman, "sniper_th": sniper,
            "bar_basis": basis, "shrink_weight_at_valuable": round(float(at.at[valuable, "w"]), 4),
            "wf_bets": int(at.at[valuable, "n"]),
            "wf_roi": round(float(own_roi), 4) if np.isfinite(own_roi) else None,
            "wf_roi_shrunk": round(float(at.at[valuable, "roi_shrunk"]), 4), "note": note}


# ── holdout ─────────────────────────────────────────────────────────────────────────────────────

def one_sided_p(pnl: np.ndarray, days: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> float:
    """p = (k + 1) / (B + 1), k = bootstrap means (resampled by match date) <= 0."""
    if len(pnl) == 0:
        return float("nan")
    uniq = np.unique(days)
    if len(uniq) < 3:
        return float("nan")
    rng = np.random.default_rng(seed)
    by = {d: pnl[days == d] for d in uniq}
    k = 0
    for _ in range(n_boot):
        s = np.concatenate([by[d] for d in rng.choice(uniq, size=len(uniq), replace=True)])
        k += int(s.mean() <= 0)
    return (k + 1) / (n_boot + 1)


def holm(p: np.ndarray, extra: int = 0) -> np.ndarray:
    """Holm step-down adjusted p-values. `extra` = hypotheses searched or tested ELSEWHERE (earlier runs,
    the configurations of this one) whose p-values are not in `p`: they enlarge the family and are never
    credited as rejected (each counts as if its p were 1, the conservative placement)."""
    p = np.asarray(p, float)
    if extra < 0:
        raise ValueError("extra hypotheses must be >= 0")
    out = np.full(len(p), np.nan)
    fin = np.where(np.isfinite(p))[0]
    m = len(fin) + int(extra)
    if not m:
        return out
    order = fin[np.argsort(p[fin])]
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        out[i] = run
    return out


def score_holdout(d: pd.DataFrame, bars: dict) -> dict:
    h = d[d["edge"] >= bars["valuable_th"]]
    pnl, days = h["pnl"].to_numpy(float), pd.to_datetime(h["date"]).dt.strftime("%Y-%m-%d").to_numpy()
    lo, hi = block_ci(pnl, days)
    tiers = {}
    for name, lo_t, hi_t in (("VALUABLE", bars["valuable_th"], bars["marksman_th"]),
                             ("MARKSMAN", bars["marksman_th"], bars["sniper_th"]),
                             ("SNIPER", bars["sniper_th"], np.inf)):
        m = (h["edge"] >= lo_t) & (h["edge"] < hi_t)
        tiers[name] = {"bets": int(m.sum()), "roi": round(float(h.loc[m, "pnl"].mean()), 4) if m.any() else None}
    return {"ho_bets": int(len(h)), "ho_roi": round(float(pnl.mean()), 4) if len(h) else None,
            "ho_ci95": [None if not np.isfinite(lo) else round(lo, 4), None if not np.isfinite(hi) else round(hi, 4)],
            "ho_matchdays": int(len(np.unique(days))), "p_one_sided": one_sided_p(pnl, days), "ho_tiers": tiers}


# ── one track ────────────────────────────────────────────────────────────────────────────

#: owner 2026-11-08: when a league x market has no bet-time price before the holdout, the holdout
#: period itself is split in time order and the CHECK part holds about this share of the matchdays
IN_PERIOD_CHECK_SHARE = 0.40
EVIDENCE_WF = "WALK_FORWARD_HOLDOUT"
EVIDENCE_SPLIT = "IN_PERIOD_SPLIT"

_EMPTY_SCORE = {"ho_bets": 0, "ho_roi": None, "ho_ci95": [None, None], "ho_matchdays": 0,
                "p_one_sided": float("nan"), "ho_tiers": {}}


def in_period_split(dates: pd.Series) -> pd.Timestamp | None:
    """The first CHECK date: the month boundary whose later part holds the share of matchdays nearest
    IN_PERIOD_CHECK_SHARE (a matchday quantile when the period spans one month). None under 2 matchdays."""
    days = pd.DatetimeIndex(np.sort(pd.to_datetime(dates).dt.normalize().unique()))
    if len(days) < 2:
        return None
    bounds = list(pd.date_range(days[0] + pd.offsets.MonthBegin(1), days[-1], freq="MS"))
    if bounds:
        share = {b: float((days >= b).mean()) for b in bounds}
        return min(bounds, key=lambda b: (abs(share[b] - IN_PERIOD_CHECK_SHARE), b))
    k = int(round(len(days) * (1 - IN_PERIOD_CHECK_SHARE)))
    return days[min(max(k, 1), len(days) - 1)]


def _bars_for(sel: pd.DataFrame, leagues) -> tuple[dict, pd.DataFrame | None, dict]:
    curves = {lg: roi_curve(sel[sel["league"] == lg]) for lg in leagues if (sel["league"] == lg).any()}
    pooled = roi_curve(sel) if len(sel) else None
    tau2 = {float(t): between_league_variance(curves, float(t)) for t in GRID} if curves else {}
    return curves, pooled, tau2


def _period(d: pd.DataFrame) -> tuple[str | None, str | None]:
    if not len(d):
        return None, None
    x = pd.to_datetime(d["date"])
    return str(x.min().date()), str(x.max().date())


def run_market(pred: pd.DataFrame, market: str, track: str) -> list[dict]:
    """Bars and out-of-sample scores for every league of one track x market. `pred` must already carry
    yes/no prices (attach_prices).

    WALK_FORWARD_HOLDOUT: bars on the walk-forward rows, scored on the holdout.
    IN_PERIOD_SPLIT (owner 2026-11-08): a league with NO priced walk-forward row but priced holdout rows
    gets its OWN split of its priced holdout matchdays (in_period_split); bars are picked on the earlier
    part - shrunk toward every league's priced rows in that same earlier window - and scored once on the
    later part. NO_PRICE_DATA only when the league has no real two-sided bet-time price at any time."""
    out = []
    p = pred[pred["market"] == market]
    wf_all = bets_frame(p[p["stage"] == "walk_forward"], market)
    ho_all = bets_frame(p[p["stage"] == "holdout"], market)
    ho_date = pd.to_datetime(ho_all["date"])
    leagues = sorted(p["league"].unique())
    curves, pooled, tau2 = _bars_for(wf_all, leagues)
    by_split: dict = {}
    for lg in leagues:
        base = {"track": track, "market": market, "league": lg,
                "wf_rows": int(((p["league"] == lg) & (p["stage"] == "walk_forward")).sum()),
                "wf_priced": int((wf_all["league"] == lg).sum()),
                "ho_rows": int(((p["league"] == lg) & (p["stage"] == "holdout")).sum()),
                "ho_priced": int((ho_all["league"] == lg).sum())}
        if lg in curves:
            bars = choose_bars(curves[lg], pooled, tau2)
            row = base | {"evidence": EVIDENCE_WF} | bars
            if bars["status"] != STATUS_THIN:
                lh = ho_all[ho_all["league"] == lg]
                row |= score_holdout(lh, bars) if len(lh) else dict(_EMPTY_SCORE)
            out.append(row)
            continue
        if not base["ho_priced"]:
            out.append(base | {"status": STATUS_NO_PRICE, "evidence": None,
                               "reason": "no real two-sided bet-time price at any time"})
            continue
        lh = ho_all[ho_all["league"] == lg]
        split = in_period_split(lh["date"])
        if split is None:
            out.append(base | {"status": STATUS_THIN, "evidence": EVIDENCE_SPLIT,
                               "reason": "real prices on fewer than 2 matchdays: no earlier part to pick bars on"})
            continue
        if split not in by_split:
            by_split[split] = _bars_for(ho_all[ho_date < split], leagues)
        s_curves, s_pooled, s_tau2 = by_split[split]
        sel, chk = lh[pd.to_datetime(lh["date"]) < split], lh[pd.to_datetime(lh["date"]) >= split]
        (sf, st), (cf, ct) = _period(sel), _period(chk)
        days = pd.to_datetime(lh["date"]).dt.normalize()
        split_info = {"select_from": sf, "select_to": st, "check_from": cf, "check_to": ct,
                      "check_share_of_matchdays": round(float(days[days >= split].nunique() / days.nunique()), 3)}
        bars = choose_bars(s_curves[lg], s_pooled, s_tau2)
        row = base | {"evidence": EVIDENCE_SPLIT, "split": split_info} | bars
        if bars["status"] != STATUS_THIN:
            row |= score_holdout(chk, bars) if len(chk) else dict(_EMPTY_SCORE)
        out.append(row)
    return out


def finalise(rows: list[dict]) -> list[dict]:
    """Holm across EVERY tested league x market x track, then the strict pass rule."""
    tested = [i for i, r in enumerate(rows) if r.get("status") is None and "ho_bets" in r]
    ph = holm(np.array([rows[i]["p_one_sided"] for i in tested], float))
    for i, adj in zip(tested, ph):
        r = rows[i]
        r["p_holm"] = None if not np.isfinite(adj) else round(float(adj), 4)
        r["family_size"] = int(np.isfinite(ph).sum())
        why = []
        if r["ho_bets"] < MIN_HOLDOUT_BETS:
            why.append(f"{r['ho_bets']} out-of-sample bets < {MIN_HOLDOUT_BETS}")
        if r["ho_roi"] is None or r["ho_roi"] <= 0:
            why.append("out-of-sample ROI not above 0")
        if r["p_holm"] is None or r["p_holm"] >= HOLM_ALPHA_ONE_SIDED:
            why.append(f"Holm-adjusted p {r['p_holm']} not < {HOLM_ALPHA_ONE_SIDED}")
        r["status"] = STATUS_UNPROVEN if why else STATUS_PROVEN
        r["reason"] = "; ".join(why)
        r["p_one_sided"] = None if not np.isfinite(r["p_one_sided"]) else round(float(r["p_one_sided"]), 4)
    return rows


# ── candidates ──────────────────────────────────────────────────────────────────────────────────

def candidate_params(rows: list[dict]) -> dict:
    """{market: {league: params}} in the production params shape (+ valuable_th and the evidence).
    `approved` is True ONLY for PROVEN. Leagues without bars (NO_PRICE_DATA / thin) are listed with
    null bars and drop=True - no number is invented for them."""
    out: dict = {}
    for r in rows:
        mk = "ou25" if r["market"] == "over25" else r["market"]
        has = r.get("valuable_th") is not None
        out.setdefault(mk, {})[r["league"]] = {
            "sniper_th": r.get("sniper_th") if has else None,
            "marksman_th": r.get("marksman_th") if has else None,
            "valuable_th": r.get("valuable_th") if has else None,
            "approved": r["status"] == STATUS_PROVEN, "drop": not has,
            "status": r["status"], "reason": r.get("reason", ""),
            "bets_insample": r.get("wf_bets"), "roi_insample": r.get("wf_roi"),
            "bets_oos": r.get("ho_bets"), "roi_oos": r.get("ho_roi"),
            "oos_roi_ci95": r.get("ho_ci95"), "oos_matchdays": r.get("ho_matchdays"),
            "p_holm": r.get("p_holm"), "shrink_weight": r.get("shrink_weight_at_valuable"),
            "evidence": r.get("evidence"), "split": r.get("split"), "bar_basis": r.get("bar_basis"),
            "source": "league_bars/1 (Phase 2 clean retrain, champion_recipe)"}
    return out


def write_outputs(rows: list[dict], track: str, out_dir: Path, meta: dict) -> list[Path]:
    """Candidates + evidence under output/threshold_proposals/ only (src.threshold_proposals guards)."""
    from src import threshold_proposals as TP
    out_dir = Path(out_dir)
    written = []
    params = candidate_params(rows)
    if track == "standard":
        if "ou25" in params:
            written.append(TP.write_candidate("standard", params["ou25"], out_dir / track))
        side = {m: v for m, v in params.items() if m != "ou25"}
        if side:
            written.append(TP.write_candidate("side_markets", side, out_dir / track))
    else:
        for m, v in params.items():
            written.append(TP.write_candidate(f"{track}_{m}", v, out_dir / track))
    written.append(TP._safe_write(out_dir / track / "league_bars_evidence.json",
                                  json.dumps({"meta": meta, "rows": rows}, indent=1, default=str)))
    return written
