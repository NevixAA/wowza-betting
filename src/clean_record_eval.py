"""Honest post-repair performance record (owner deep-repair brief 2026-11-22, P1 item 8).

    python -m src.clean_record_eval                 # print the summary (JSON) to stdout
    python -m src.clean_record_eval --write         # write output/clean_record_evaluation.json (committed)

WHAT IT IS. One reproducible evaluation of every record the system keeps, in SEGMENTS THAT ARE NEVER POOLED:

  pre_repair_historical        settled live tips generated BEFORE the Clean Record start (unrepaired system,
                               older models), corrections sidecars applied (src/trustworthy_performance view A);
                               plus the prediction-ledger learning rows predicted before the start.
  post_repair_forward          the Clean Record (generated AND played on or after the start,
                               config/trustworthy_performance.json clean_record_start_utc):
                                 top3_staked / top3_unstaked   output/top3_league_ledger.csv, split on the FROZEN
                                                               stake_reason (dashboard_data.clean_record.top3_staked)
                                 tips                          bets / side / HT ledgers (view C), staked and not
                                                               staked counted apart inside every cell
                                 props_tips                    player_ledger tips + props_top3_ledger (no stake)
                                 prediction_ledger_learning    every fixture x market prediction (owner "predict
                                                               every match"): learning rows, NOT bets - no P/L
  research_only                weekly league research panel, sharp tracker (src/trustworthy_performance view B)
  shadow_experiments           shadow logs: inventory; v91_shadow scored (model vs de-vigged market)
  real_price_backtests         backtest outputs: inventory only (older models; a backtest proves nothing,
                               owner rule "live proof only")
  synthetic_price_diagnostics  quarantined (src/integrity/synthetic_roi.py): listed, no figure carried

CELL = league x market x model version (the market carries the side where a market has two). Every cell
reports: settled bets, independent matchdays, real-odds coverage, flat-1u P/L and ROI with a matchday block-
bootstrap CI (config/trustworthy_performance.json "bootstrap"), staked n / ROI where a stake was recorded, max
drawdown (flat 1u, match-date order), log loss and Brier of the model, the same of the de-vigged bookmaker
probability on the same rows (and the model-minus-market difference), calibration (mean p vs hit rate), CLV
eligibility and coverage (CERTIFIED closes only; stored ledger CLV is shown apart, labelled uncertified),
the registered forward-test status (config/track_registry.json + registry/preregistered_hypotheses.json) and
an evidence verdict. "NO PROVEN EDGE YET" is the expected headline until the owner's minimums are met.

NO PRICES. The output carries counts, units, percentages and probability scores only; never an odds value,
never a mean price or a mean market probability.

OWNER RULES KEPT. performance_claims_enabled is read and echoed, never changed; no threshold, bar or stake is
tuned from these numbers; the verdict never says PROVEN for a historical, research, shadow or backtest segment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_FILE = "clean_record_evaluation.json"
SCHEMA = "CRE-1"
SETTLED = ("WIN", "LOSS", "PUSH", "HALF_WIN", "HALF_LOSS")
EPS = 1e-6
NO_PROVEN_EDGE = "NO PROVEN EDGE YET"

#: verdicts
NOT_ENOUGH, NEGATIVE, NO_EDGE, POS_UNCONFIRMED, CANDIDATE = (
    "NOT_ENOUGH_DATA", "NEGATIVE_EDGE_CI_BELOW_ZERO", "NO_PROVEN_EDGE", "POSITIVE_ROI_UNCONFIRMED",
    "EDGE_CANDIDATE_FOR_OWNER_REVIEW")

#: identity fields of a pre-start live row: never change once written (immutability digest). No price column.
IDENTITY = {"bets_ledger.csv": ["generated_at", "match_date", "league", "home_team", "away_team", "side",
                                "model_type", "signal_tier", "model_version"],
            "side_bets_ledger.csv": ["generated_at", "match_date", "league", "home_team", "away_team", "market",
                                     "model_type", "signal_tier", "model_version"],
            "ht_ledger.csv": ["generated_at", "match_date", "league", "home_team", "away_team", "market", "side",
                              "model_type", "model_version"],
            "top3_league_ledger.csv": ["pick_id", "issued_at", "match_date", "league", "home_team", "away_team",
                                       "market", "side", "stake_reason", "prediction_engine_version"],
            "player_ledger.csv": ["signal_date", "match_date", "league", "player_name", "market", "tier"]}
#: the generation-time / match-date columns of each live source (the Clean Record split)
GEN_DATE = {"bets_ledger.csv": ("generated_at", "match_date"), "side_bets_ledger.csv": ("generated_at", "match_date"),
            "ht_ledger.csv": ("generated_at", "match_date"), "top3_league_ledger.csv": ("issued_at", "match_date"),
            "player_ledger.csv": ("signal_date", "match_date"), "props_top3_ledger.csv": ("issued_at", "match_date")}

#: owner's historical figures (brief 2026-11-22), verified on the current ledgers by `historical_figures`
OWNER_FIGURES = (("Std O/U 2.5", "bets_ledger.csv", "standard", None, 164, -16.0),
                 ("NF O/U 2.5", "bets_ledger.csv", "new_format", None, 501, -8.8),
                 ("NF BTTS", "side_bets_ledger.csv", "new_format", "btts", 155, 16.1),
                 ("NF O1.5", "side_bets_ledger.csv", "new_format", "over15", 115, 8.7))
FIGURE_DEFINITION = ("live tips generated before the Clean Record start, signal_tier SNIPER or MARKSMAN (the retired "
                     "tier basis of 'staked' before 2026-11-04), match_date on or after config.PERFORMANCE_CUTOFF_DATE, "
                     "settled WIN/LOSS/PUSH/HALF (VOID out), flat 1u")

#: shadow logs (inventory; v91 scored)
SHADOW_FILES = (("edge_engine_shadow.csv", "prediction_ts", "market_edge_v2 selection shadow"),
                ("side_edge_devig_shadow.csv", "asof", "side-market de-vig edge shadow"),
                ("quality_tier_shadow.csv", "observed_at", "quality tier shadow"),
                ("drift_shadow.csv", "observed_at", "drift tier shadow"),
                ("v91_shadow.csv", "logged_at", "v9.1 / Pro HGB vs v9 O/U 2.5 shadow"),
                ("props_r2_shadow.csv", "logged_at", "props R2 shadow"),
                ("edge_engine_shadow_pre_reseed_2026-11-20.csv", "prediction_ts", "edge shadow, archived pre re-seed"))
BACKTEST_FILES = ("backtest_results_standard.csv", "backtest_results_newformat.csv", "backtest_results_btts.csv",
                  "backtest_results_over15.csv", "backtest_results_over35.csv", "backtest_all_leagues.csv",
                  "edge_engine_backtest.csv")


# ── small helpers ─────────────────────────────────────────────────────────────────────────────────
def _r(x, nd=4):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else round(x, nd)


def _read(p: Path, **kw) -> pd.DataFrame | None:
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(p, low_memory=False, **kw)
    except Exception:                                                   # noqa: BLE001 - unreadable = absent
        return None


def _num(d: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(d[c], errors="coerce") if c in d.columns else pd.Series(np.nan, index=d.index)


def _str(d: pd.DataFrame, c: str, default: str = "") -> pd.Series:
    return d[c].fillna(default).astype(str).str.strip() if c in d.columns else pd.Series(default, index=d.index)


def tp_config() -> dict:
    from src import trustworthy_performance as TP
    return TP.load_config()


def start_ts(cfg: dict) -> pd.Timestamp:
    from src import trustworthy_performance as TP
    return TP.clean_record_start(cfg)


def record_of(gen: pd.Series, day: pd.Series, start: pd.Timestamp) -> pd.Series:
    """CLEAN / BEFORE / UNPLACED with the rule of dashboard_data.clean_record.assign, for an explicit start:
    CLEAN needs the generation time >= start AND the match day >= the start day; BEFORE = generated before the start
    (or no generation time and a match before the start day); anything else UNPLACED."""
    g = pd.to_datetime(gen, errors="coerce", utc=True, format="mixed")
    dd = pd.to_datetime(day, errors="coerce", utc=True, format="mixed").dt.normalize()
    s_day = start.normalize()
    out = pd.Series("UNPLACED", index=gen.index, dtype=object)
    out[g.notna() & (g < start)] = "BEFORE"
    out[g.isna() & dd.notna() & (dd < s_day)] = "BEFORE"
    out[g.notna() & (g >= start) & dd.notna() & (dd >= s_day)] = "CLEAN"
    if start != s_day:
        raw = gen.astype("string").str.strip()
        out[((raw.str.len() <= 10).fillna(False) & (g.dt.normalize() == s_day).fillna(False)).to_numpy()] = "UNPLACED"
    return out


# ── statistics ────────────────────────────────────────────────────────────────────────────────────
def _block_ci(num, den, days, key, cfg):
    from src import trustworthy_performance as TP
    bs = cfg["bootstrap"]
    lo, hi = TP.block_ci(np.asarray(num, float), np.asarray(den, float), np.asarray(days), key,
                         int(bs["draws"]), float(bs["ci"]))
    return [lo, hi]


def _ll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _drawdown(pnl: np.ndarray) -> float:
    if not len(pnl):
        return 0.0
    c = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[0.0], c]))[1:]
    return float(np.max(peak - c))


def prob_scores(p: pd.Series, y: pd.Series, pm: pd.Series | None = None) -> dict:
    """Log loss / Brier / calibration of the model on rows with a known outcome; the de-vigged market on the
    SAME rows (only rows where both exist). Scores only: no mean market probability is reported."""
    m = p.notna() & y.notna()
    out = {"n_prob": int(m.sum()), "log_loss_model": None, "brier_model": None, "mean_p_model": None,
           "hit_rate": None, "calibration_gap": None, "n_vs_market": 0, "log_loss_model_same_rows": None,
           "log_loss_market": None, "brier_model_same_rows": None, "brier_market": None,
           "d_log_loss_model_minus_market": None, "d_brier_model_minus_market": None}
    if not m.any():
        return out
    pp, yy = p[m].to_numpy(float), y[m].to_numpy(float)
    out.update(log_loss_model=_r(_ll(pp, yy)), brier_model=_r(np.mean((pp - yy) ** 2)), mean_p_model=_r(pp.mean()),
               hit_rate=_r(yy.mean()), calibration_gap=_r(yy.mean() - pp.mean()))
    if pm is not None:
        mm = m & pm.notna()
        out["n_vs_market"] = int(mm.sum())
        if mm.any():
            a, b, t = p[mm].to_numpy(float), pm[mm].to_numpy(float), y[mm].to_numpy(float)
            llm, llk = _ll(a, t), _ll(b, t)
            bm, bk = float(np.mean((a - t) ** 2)), float(np.mean((b - t) ** 2))
            out.update(log_loss_model_same_rows=_r(llm), log_loss_market=_r(llk), brier_model_same_rows=_r(bm),
                       brier_market=_r(bk), d_log_loss_model_minus_market=_r(llm - llk),
                       d_brier_model_minus_market=_r(bm - bk))
    return out


def calibration_bins(p: pd.Series, y: pd.Series, edges=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001)) -> list:
    m = p.notna() & y.notna()
    if not m.any():
        return []
    b = pd.cut(p[m], list(edges), right=False)
    g = pd.DataFrame({"p": p[m], "y": y[m], "b": b}).groupby("b", observed=True)
    return [{"bin": f"{iv.left:.1f}-{min(iv.right, 1.0):.1f}", "n": int(len(x)), "mean_p": _r(x["p"].mean()),
             "observed": _r(x["y"].mean())} for iv, x in g]


def cell_metrics(d: pd.DataFrame, key: str, cfg: dict, *, bets: bool = True, forward: bool = False) -> dict:
    """Metrics of one cell. `d` = normalised rows (see normalise()). `bets` False = learning rows (no P/L)."""
    from src import trustworthy_performance as TP
    mins = TP.minimums(cfg)
    st = d[d["settled"]]
    days = st["match_date"].astype(str).str[:10].to_numpy()
    n, nd = int(len(st)), int(len(set(days)))
    out = {"rows": int(len(d)), "settled": n, "matchdays": nd,
           "first_match_date": (min(days) if n else None), "last_match_date": (max(days) if n else None)}
    if bets:
        pr = st[st["priced"]]
        npr = int(len(pr))
        pnl = pr["pnl"].to_numpy(float)
        pdays = pr["match_date"].astype(str).str[:10].to_numpy()
        order = np.lexsort((pr["gen"].astype(str).to_numpy(), pdays)) if npr else np.array([], int)
        out.update({"real_odds_coverage": _r(npr / n) if n else None, "priced_settled": npr,
                    "pnl_units": _r(pnl.sum()) if npr else None, "roi": _r(pnl.sum() / npr) if npr else None,
                    "roi_ci": _block_ci(pnl, np.ones(npr), pdays, key + "|roi", cfg) if npr else [None, None],
                    "max_drawdown_units": _r(_drawdown(pnl[order])) if npr else None})
        sk = pr[pr["staked"].fillna(False).astype(bool)]
        ns = int(len(sk))
        su = sk["stake_units"].to_numpy(float)
        sp = sk["pnl"].to_numpy(float)
        has_units = ns and np.isfinite(su).all() and su.sum() > 0
        out["staked"] = {"n": ns, "pnl_units_flat": _r(sp.sum()) if ns else None,
                         "roi_flat": _r(sp.sum() / ns) if ns else None,
                         "roi_at_recorded_stake": _r((sp * su).sum() / su.sum()) if has_units else None}
        out["not_staked"] = {"n": npr - ns, "roi_flat": _r((pnl.sum() - sp.sum()) / (npr - ns)) if npr - ns else None}
    w = st[st["won"].notna()]
    out.update(prob_scores(w["p_model"], w["won"], w["p_market"]))
    out["calibration"] = calibration_bins(w["p_model"], w["won"]) if out["n_prob"] >= mins["settled"] else []
    # CLV: certified closes only count; stored (uncertified) ledger CLV is reported apart
    c = st["clv_cert"]
    cm = c.notna()
    nc, ndc = int(cm.sum()), int(len(set(days[cm.to_numpy()]))) if n else 0
    out["clv"] = {"eligible": int(st["clv_eligible"].sum()), "certified_n": nc,
                  "coverage": _r(nc / n) if n else None,
                  "mean_pct": _r(c[cm].mean()) if nc else None,
                  "mean_ci": _block_ci(c[cm].to_numpy(float), np.ones(nc), days[cm.to_numpy()], key + "|clv", cfg)
                  if nc else [None, None],
                  "status": TP.OK if nc >= mins["clv"] and ndc >= mins["matchdays"] else TP.NOT_ENOUGH,
                  "stored_uncertified_n": int(st["clv_stored"].notna().sum()),
                  "stored_uncertified_mean_pct": _r(st["clv_stored"].mean()) if st["clv_stored"].notna().any() else None}
    out["verdict"] = verdict(out, mins, bets=bets, forward=forward)
    return out


def verdict(m: dict, mins: dict, *, bets: bool, forward: bool) -> str:
    """Evidence verdict. Never PROVEN: at most a candidate for the owner's review, and only in the forward record
    with enough settled bets AND matchdays, a ROI CI above 0 AND certified CLV evidence above 0 at its minimum."""
    if not bets:
        return NOT_ENOUGH if m["settled"] < mins["settled"] or m["matchdays"] < mins["matchdays"] else NO_EDGE
    n = m["priced_settled"]
    if n < mins["settled"] or m["matchdays"] < mins["matchdays"]:
        return NOT_ENOUGH
    lo, hi = (m.get("roi_ci") or [None, None])
    if hi is not None and hi < 0:
        return NEGATIVE
    if lo is None or lo <= 0:
        return NO_EDGE
    clv = m["clv"]
    clv_ok = clv["status"] == "OK" and clv["mean_ci"][0] is not None and clv["mean_ci"][0] > 0
    return CANDIDATE if (forward and clv_ok) else POS_UNCONFIRMED


# ── normalised rows per source ─────────────────────────────────────────────────────────────────────
COLS = ["source", "league", "market", "model_version", "model_type", "match_date", "gen", "settled", "won", "pnl",
        "priced", "staked", "stake_units", "p_model", "p_market", "clv_cert", "clv_stored", "clv_eligible"]


def _frame(d: pd.DataFrame, **cols) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    for c in COLS:
        v = cols.get(c, np.nan)
        out[c] = v if not callable(v) else v(d)
    return out


def _won(res: pd.Series) -> pd.Series:
    r = res.fillna("").astype(str).str.strip().str.upper()
    return pd.Series(np.where(r.eq("WIN"), 1.0, np.where(r.eq("LOSS"), 0.0, np.nan)), index=res.index)


def _settled(res: pd.Series, pnl: pd.Series) -> pd.Series:
    return res.fillna("").astype(str).str.strip().str.upper().isin(SETTLED) & pnl.notna()


FAIR_FILES = ("edge_engine_shadow.csv", "edge_engine_shadow_pre_reseed_2026-11-20.csv")
_MKT = {"btts": "BTTS", "over15": "OU15", "over35": "OU35", "ou25": "OU25", "ou15": "OU15", "ou35": "OU35"}


def market_fair(out_dir: Path) -> pd.DataFrame | None:
    """De-vigged first-sight market probability per (fixture key, model_type, market, side) from
    output/edge_engine_shadow.csv (power de-vig, two-sided books). Both sides: p and 1 - p."""
    # the live shadow plus its archive from before the 2026-11-20 re-seed (bytes kept; earlier quotes live there)
    parts = [x for x in (_read(out_dir / f) for f in FAIR_FILES) if x is not None and len(x)]
    e = pd.concat(parts, ignore_index=True) if parts else None
    if e is None or e.empty or "p_market_fair" not in e.columns:
        return None
    e = e[pd.to_numeric(e["p_market_fair"], errors="coerce").between(0, 1, inclusive="neither")].copy()
    e["p_market_fair"] = pd.to_numeric(e["p_market_fair"], errors="coerce")
    e["p_model_raw"] = pd.to_numeric(e.get("p_model_raw"), errors="coerce")
    e["ts"] = pd.to_datetime(e["prediction_ts"], errors="coerce", utc=True, format="mixed")
    opp = {"OVER": "UNDER", "UNDER": "OVER", "YES": "NO", "NO": "YES"}
    a = e.assign(side=e["side"].astype(str).str.upper())
    b = a.assign(side=a["side"].map(opp), p_market_fair=1 - a["p_market_fair"], p_model_raw=1 - a["p_model_raw"])
    f = pd.concat([a, b[b["side"].notna()]], ignore_index=True)
    f["mkey"] = (f["match_date"].astype(str).str[:10] + "|" + f["home_team"].astype(str) + "|" + f["away_team"].astype(str)
                 + "|" + f["model_type"].astype(str) + "|" + f["market"].astype(str).str.upper() + "|" + f["side"])
    ko = pd.to_datetime(f["kickoff_utc"], errors="coerce", utc=True, format="mixed").dt.strftime("%Y-%m-%dT%H:%M")
    f["fkey"] = (ko.fillna("") + "|" + f["home_team"].astype(str) + "|" + f["away_team"].astype(str) + "|"
                 + f["model_type"].astype(str) + "|" + f["market"].astype(str).str.upper() + "|" + f["side"])
    return f[f["ts"].notna()][["mkey", "fkey", "ts", "p_market_fair", "p_model_raw"]].sort_values("ts")


def _asof(rows: pd.DataFrame, keycol: str, tscol: str, fair: pd.DataFrame | None, fair_key: str) -> pd.DataFrame:
    """The latest shadow row of the same key observed at or before the row's time (point in time)."""
    out = pd.DataFrame({"p_market": np.nan, "p_model_shadow": np.nan}, index=rows.index)
    if fair is None or rows.empty:
        return out
    # a ledger time is recorded to the minute while the shadow row of the same predict run carries seconds:
    # the row's own minute counts (+59 s), never a later run
    left = pd.DataFrame({"k": rows[keycol].astype(str), "ts": pd.to_datetime(rows[tscol], errors="coerce", utc=True,
                                                                              format="mixed") + pd.Timedelta(seconds=59),
                         "i": rows.index})
    left = left[left["ts"].notna()].sort_values("ts")
    right = fair.rename(columns={fair_key: "k"})[["k", "ts", "p_market_fair", "p_model_raw"]]
    if left.empty:
        return out
    m = pd.merge_asof(left, right, on="ts", by="k", direction="backward")
    out.loc[m["i"].to_numpy(), "p_market"] = m["p_market_fair"].to_numpy()
    out.loc[m["i"].to_numpy(), "p_model_shadow"] = m["p_model_raw"].to_numpy()
    return out


def team_rows(name: str, d: pd.DataFrame, fair) -> pd.DataFrame:
    """bets / side / HT ledger rows (already corrected and stake-labelled by src/trustworthy_performance)."""
    from src import staked as S
    res, pnl = _str(d, "result"), _num(d, "pnl")
    if name == "bets_ledger.csv":
        mkt, side, code = "OU25_" + _str(d, "side").str.upper(), _str(d, "side").str.upper(), pd.Series("OU25", index=d.index)
        price = _num(d, "odds")
        pm = _num(d, "cell_p_used")
    elif name == "side_bets_ledger.csv":
        code = _str(d, "market").str.lower().map(_MKT).fillna(_str(d, "market").str.upper())
        side = np.where(code.eq("BTTS"), "YES", "OVER")
        side = pd.Series(side, index=d.index)
        mkt = _str(d, "market").str.upper() + "_" + side                # BTTS_YES / OVER15_OVER, as TOP-3 names it
        price = _num(d, "odds")
        pm = _num(d, "model_prob")
    else:
        code = _str(d, "market").str.upper()
        side = _str(d, "side").str.upper()
        mkt = code + "_" + side
        price = _num(d, "entry_odds").fillna(_num(d, "odds")) if "entry_odds" in d.columns else _num(d, "odds")
        pm = _num(d, "model_prob")
    tmp = d.assign(_k=(_str(d, "match_date").str[:10] + "|" + _str(d, "home_team") + "|" + _str(d, "away_team") + "|"
                       + _str(d, "model_type") + "|" + code + "|" + side))
    j = _asof(tmp, "_k", "generated_at", fair, "mkey") if name != "ht_ledger.csv" else None
    pmk = j["p_market"] if j is not None else pd.Series(np.nan, index=d.index)
    if j is not None:
        pm = pm.fillna(j["p_model_shadow"])
    stk = S.is_staked(d["stake_status"]) if "stake_status" in d.columns else pd.Series(False, index=d.index)
    clv = _num(d, "clv_pct")
    return _frame(d, source=name, league=_str(d, "league"), market=mkt, model_version=_str(d, "model_version", "unrecorded")
                  .replace("", "unrecorded"), model_type=_str(d, "model_type"), match_date=_str(d, "match_date").str[:10],
                  gen=_str(d, "generated_at"), settled=_settled(res, pnl), won=_won(res), pnl=pnl,
                  priced=(price > 1).fillna(False), staked=pd.Series(stk, index=d.index).astype(bool),
                  stake_units=_num(d, "stake_units"), p_model=pm, p_market=pmk,
                  clv_cert=pd.Series(np.nan, index=d.index),            # ledger rows carry no certified close
                  clv_stored=clv, clv_eligible=(price > 1).fillna(False))


def top3_rows(d: pd.DataFrame) -> pd.DataFrame:
    from dashboard_data import clean_record as CR
    res, pnl = _str(d, "result"), _num(d, "flat_pnl")
    settled = _settled(res, pnl) & _str(d, "settle_status").str.upper().eq("SETTLED")
    return _frame(d, source="top3_league_ledger.csv", league=_str(d, "league"),
                  market=_str(d, "market").str.upper() + "_" + _str(d, "side").str.upper(),
                  model_version=_str(d, "prediction_engine_version", "unrecorded").replace("", "unrecorded"),
                  model_type=_str(d, "model_type"), match_date=_str(d, "match_date").str[:10], gen=_str(d, "issued_at"),
                  settled=settled, won=_won(res), pnl=pnl, priced=(_num(d, "odds") > 1).fillna(False),
                  staked=CR.top3_staked(d).astype(bool), stake_units=pd.Series(np.nan, index=d.index),
                  p_model=_num(d, "p_model_raw"), p_market=_num(d, "p_market_fair"),
                  clv_cert=pd.Series(np.nan, index=d.index), clv_stored=pd.Series(np.nan, index=d.index),
                  clv_eligible=(_num(d, "odds") > 1).fillna(False))


def props_rows(d: pd.DataFrame, source: str) -> pd.DataFrame:
    if source == "player_ledger.csv":
        res, pnl, price, p, gen = _str(d, "result"), _num(d, "pnl"), _num(d, "market_odds"), _num(d, "model_prob"), \
            _str(d, "signal_date")
        mv = pd.Series("unrecorded", index=d.index)
    else:
        res, pnl, price, p, gen = _str(d, "result"), _num(d, "flat_pnl"), _num(d, "odds"), _num(d, "model_prob"), \
            _str(d, "issued_at")
        mv = pd.Series("unrecorded", index=d.index)
    return _frame(d, source=source, league=_str(d, "league"), market=_str(d, "market").str.upper(), model_version=mv,
                  model_type=pd.Series("props", index=d.index), match_date=_str(d, "match_date").str[:10], gen=gen,
                  settled=_settled(res, pnl), won=_won(res), pnl=pnl, priced=(price > 1).fillna(False),
                  staked=pd.Series(False, index=d.index), stake_units=pd.Series(np.nan, index=d.index),
                  p_model=p, p_market=pd.Series(np.nan, index=d.index), clv_cert=pd.Series(np.nan, index=d.index),
                  clv_stored=pd.Series(np.nan, index=d.index), clv_eligible=(price > 1).fillna(False))


def prediction_ledger(out_dir: Path) -> pd.DataFrame:
    """Every SETTLED prediction-ledger row (output/prediction_ledger_settled), typed."""
    root = out_dir / "prediction_ledger_settled"
    parts = []
    for f in sorted(root.rglob("*.csv")) if root.exists() else []:
        x = _read(f, dtype=str)
        if x is not None and len(x):
            parts.append(x)
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    return d[d["outcome_status"].astype(str).str.upper().eq("SETTLED")].reset_index(drop=True)


def learning_rows(d: pd.DataFrame, fair) -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame(columns=COLS + ["snapshot", "model_id", "record"])
    code = _str(d, "market").str.lower().map(_MKT).fillna("")
    sel = _str(d, "selection").str.upper()
    ko = pd.to_datetime(_str(d, "kickoff_utc"), errors="coerce", utc=True, format="mixed").dt.strftime("%Y-%m-%dT%H:%M")
    tmp = d.assign(_k=ko.fillna("") + "|" + _str(d, "home_team") + "|" + _str(d, "away_team") + "|" + _str(d, "track")
                   + "|" + code + "|" + sel)
    team = code.ne("") & _str(d, "model_id").isin(["v9_standard", "v9_new_format"])
    j = _asof(tmp[team], "_k", "predicted_at", fair, "fkey") if team.any() else None
    pm = pd.Series(np.nan, index=d.index)
    if j is not None:
        pm.loc[j.index] = j["p_market"]
    y = _num(d, "y")
    f = _frame(d, source="prediction_ledger", league=_str(d, "league"),
               market=_str(d, "market").str.upper() + "_" + sel + _str(d, "line").map(lambda s: f"_{s}" if s else ""),
               model_version=_str(d, "model_version", "unrecorded"), model_type=_str(d, "track"),
               match_date=_str(d, "kickoff_utc").str[:10], gen=_str(d, "predicted_at"),
               settled=y.isin([0, 1]), won=y.where(y.isin([0, 1])), pnl=pd.Series(np.nan, index=d.index),
               priced=pd.Series(False, index=d.index), staked=pd.Series(False, index=d.index),
               stake_units=pd.Series(np.nan, index=d.index), p_model=_num(d, "probability"), p_market=pm,
               clv_cert=pd.Series(np.nan, index=d.index), clv_stored=pd.Series(np.nan, index=d.index),
               clv_eligible=pd.Series(False, index=d.index))
    f["snapshot"], f["model_id"] = _str(d, "snapshot"), _str(d, "model_id")
    return f


def research_rows(name: str, d: pd.DataFrame) -> pd.DataFrame:
    if name == "league_weekly_research.csv":
        res, pnl = _str(d, "result"), _num(d, "flat_pnl")
        cert = _num(d, "clv_pct").where(_str(d, "close_status").str.upper().eq("OK"))
        priced = _str(d, "price_quality_state").isin(["TWO_SIDED_MULTI_BOOK", "TWO_SIDED_SINGLE_BOOK"]) & \
            (_num(d, "odds") > 1).fillna(False)
        return _frame(d, source=name, league=_str(d, "league"),
                      market=_str(d, "market").str.upper() + "_" + _str(d, "side").str.upper(),
                      model_version=_str(d, "prediction_engine_version", "unrecorded"), model_type=_str(d, "model_type"),
                      match_date=_str(d, "match_date").str[:10], gen=_str(d, "first_seen_at"),
                      settled=_settled(res, pnl), won=_won(res), pnl=pnl, priced=priced,
                      staked=pd.Series(False, index=d.index), stake_units=pd.Series(np.nan, index=d.index),
                      p_model=_num(d, "p_model_raw"), p_market=_num(d, "p_market_fair"), clv_cert=cert,
                      clv_stored=pd.Series(np.nan, index=d.index), clv_eligible=priced)
    res, pnl = _str(d, "result"), _num(d, "pnl")
    return _frame(d, source=name, league=_str(d, "league"),
                  # the label carries the side and, for 1X2, the team name: the cell is the market x side
                  market=_str(d, "market_label").str.replace(r"\s*\(.*\)\s*$", "", regex=True).str.upper(),
                  model_version=pd.Series("sharp_tracker", index=d.index), model_type=pd.Series("", index=d.index),
                  match_date=_str(d, "match_date").str[:10], gen=_str(d, "signal_date"), settled=_settled(res, pnl),
                  won=_won(res), pnl=pnl, priced=pnl.notna(), staked=pd.Series(False, index=d.index),
                  stake_units=pd.Series(np.nan, index=d.index), p_model=pd.Series(np.nan, index=d.index),
                  p_market=pd.Series(np.nan, index=d.index), clv_cert=pd.Series(np.nan, index=d.index),
                  clv_stored=pd.Series(np.nan, index=d.index), clv_eligible=pd.Series(False, index=d.index))


# ── forward-test registration ──────────────────────────────────────────────────────────────────────
def registrations() -> tuple[dict, list]:
    tr = json.loads((ROOT / "config" / "track_registry.json").read_text(encoding="utf-8"))
    reg = {}
    for e in tr.get("entries", []):
        reg[(e["league"], e["market"], e.get("side"), e["model_type"])] = {
            "production_state": e.get("production_state"), "real_money": bool(e.get("real_money")),
            "research_regime": e.get("research_regime")}
    hyp = json.loads((ROOT / "registry" / "preregistered_hypotheses.json").read_text(encoding="utf-8"))
    return reg, [{"id": h["id"], "status": h.get("status"), "title": h.get("title"), "n_min": h.get("n_min"),
                  "confirmation_starts": h.get("confirmation_starts")} for h in hyp.get("hypotheses", [])]


_REG_MKT = {"OU25_OVER": ("OU25", "OVER"), "OU25_UNDER": ("OU25", "UNDER"), "BTTS": ("BTTS", "YES"),
            "BTTS_YES": ("BTTS", "YES"), "OU15": ("OVER15", "OVER"), "OVER15_OVER": ("OVER15", "OVER"),
            "OU35": ("OVER35", "OVER"), "OVER35_OVER": ("OVER35", "OVER")}


def forward_status(league: str, market: str, model_type: str, reg: dict) -> dict:
    mk = _REG_MKT.get(market)
    e = reg.get((league, mk[0], mk[1], model_type)) if mk else None
    hyps = []
    if mk and mk[0] == "BTTS":
        hyps = ["H-BTTS-01", "H-BTTS-03", "H-BTTS-04"] if league == "Argentina Primera Division" else ["H-BTTS-02"]
    if mk and mk[0] == "OU25" and mk[1] == "OVER":
        hyps = ["H-MODEL-01"]
    return {"track_registry": e or "NOT_IN_TRACK_REGISTRY", "preregistered_hypotheses": hyps}


# ── cells ──────────────────────────────────────────────────────────────────────────────────────────
def cells(rows: pd.DataFrame, seg_key: str, cfg: dict, *, bets: bool = True, forward: bool = False,
          reg: dict | None = None, by=("league", "market", "model_version")) -> dict:
    """{'total': metrics over the segment source, 'cells': [one per league x market x model version]}."""
    if rows is None or rows.empty:
        return {"total": None, "cells": []}
    out = {"total": cell_metrics(rows, seg_key, cfg, bets=bets, forward=forward), "cells": []}
    for k, g in rows.groupby(list(by), dropna=False, sort=True):
        k = k if isinstance(k, tuple) else (k,)
        if not g["settled"].any():
            continue
        c = {col: (None if pd.isna(v) else str(v)) for col, v in zip(by, k)}
        c.update(cell_metrics(g, f"{seg_key}|{'|'.join(map(str, k))}", cfg, bets=bets, forward=forward))
        if reg is not None:
            mt = g["model_type"].dropna().astype(str)
            c["forward_test"] = forward_status(c.get("league") or "", c.get("market") or "",
                                               mt.iloc[0] if len(mt) else "", reg)
        out["cells"].append(c)
    return out


# ── immutability fingerprints ──────────────────────────────────────────────────────────────────────
def pre_start_fingerprint(out_dir: Path, cfg: dict) -> dict:
    """Per live source: how many rows were generated before the start and a sha256 over their identity fields
    (sorted, order-free). Settlement fills result / P&L / closing columns and leaves this unchanged; a rewritten
    identity field, a removed pre-start row or a back-dated new one changes it. `clean_rows_generated_before_start`
    must always be 0."""
    start = start_ts(cfg)
    fp = {}
    for name, cols in IDENTITY.items():
        d = _read(Path(out_dir) / name, dtype=str, keep_default_na=False)
        if d is None:
            fp[name] = {"present": False}
            continue
        gc, dc = GEN_DATE[name]
        g = pd.to_datetime(d[gc], errors="coerce", utc=True, format="mixed")
        rec = record_of(d[gc], d[dc], start)
        pre = d[(g < start).fillna(False)]
        ident = pre.reindex(columns=cols).fillna("").astype(str)
        lines = sorted("\x1f".join(r) for r in ident.itertuples(index=False, name=None))
        h = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
        fp[name] = {"present": True, "pre_start_rows": int(len(pre)), "identity_fields": cols, "identity_sha256": h,
                    "clean_rows_generated_before_start": int(((rec == "CLEAN") & (g < start).fillna(False)).sum())}
    return fp


# ── historical figures ─────────────────────────────────────────────────────────────────────────────
def historical_figures(out_dir: Path, cfg: dict) -> list:
    """The owner's four historical figures recomputed on the CURRENT committed ledgers (raw ledger rows and with
    the view-A corrections applied). Reproduced exactly at commit 0dc7e3326 (results 2026-11-21 04:14 UTC); later
    settlements move them (see reports/P1_CLEAN_RECORD_EVALUATION.md)."""
    import config
    from src import trustworthy_performance as TP
    start, cut = start_ts(cfg), pd.Timestamp(config.PERFORMANCE_CUTOFF_DATE)
    views, _ = TP.frames(out_dir, cfg)
    corrected = {n: x for n, _k, x, _r, _e in views[TP.A]}
    out = []
    for label, f, mt, mk, n_owner, roi_owner in OWNER_FIGURES:
        raw = _read(Path(out_dir) / f)
        res = {"figure": label, "owner_n": n_owner, "owner_roi_pct": roi_owner, "definition": FIGURE_DEFINITION}
        for kind, d in (("raw_ledger", raw), ("corrections_applied", corrected.get(f))):
            if d is None:
                res[kind] = None
                continue
            x = d[d["source"].astype(str).eq("live")] if "source" in d.columns else d
            g = pd.to_datetime(x["generated_at"], errors="coerce", utc=True, format="mixed")
            x = x[(g < start).fillna(False) & x["model_type"].astype(str).eq(mt)
                  & x["signal_tier"].astype(str).isin(["SNIPER", "MARKSMAN"])
                  & (pd.to_datetime(x["match_date"], errors="coerce") >= cut).fillna(False)]
            if mk:
                x = x[x["market"].astype(str).str.lower().eq(mk)]
            p = pd.to_numeric(x["pnl"], errors="coerce")
            x = x[x["result"].astype(str).str.upper().isin(SETTLED) & p.notna()]
            p = pd.to_numeric(x["pnl"], errors="coerce")
            n = int(len(x))
            res[kind] = {"n": n, "pnl_units": _r(p.sum(), 2), "roi_pct": _r(100 * p.sum() / n, 2) if n else None,
                         "roi_ci_pct": [None if v is None else _r(100 * v, 1) for v in
                                        _block_ci(p.to_numpy(float), np.ones(n), x["match_date"].astype(str).str[:10]
                                                  .to_numpy(), f"hist|{label}|{kind}", cfg)] if n else [None, None]}
        c = res["raw_ledger"] or {}
        res["matches_owner_figure_now"] = bool(c.get("n") == n_owner and c.get("roi_pct") is not None
                                               and round(c["roi_pct"], 1) == roi_owner)
        out.append(res)
    return out


# ── inventories ────────────────────────────────────────────────────────────────────────────────────
def shadow_inventory(out_dir: Path, cfg: dict) -> dict:
    inv = []
    for f, tcol, what in SHADOW_FILES:
        d = _read(out_dir / f, usecols=lambda c: c in (tcol, "result", "settle_status"))
        if d is None:
            inv.append({"file": f, "what": what, "present": False})
            continue
        t = pd.to_datetime(d.get(tcol), errors="coerce", utc=True, format="mixed")
        inv.append({"file": f, "what": what, "present": True, "rows": int(len(d)),
                    "first": None if t.isna().all() else t.min().strftime("%Y-%m-%dT%H:%MZ"),
                    "last": None if t.isna().all() else t.max().strftime("%Y-%m-%dT%H:%MZ"),
                    "settled_rows": int(d["result"].astype(str).str.upper().isin(SETTLED).sum()) if "result" in d else None})
    out = {"inventory": inv, "v91_shadow_scored": None}
    v = _read(out_dir / "v91_shadow.csv")
    if v is not None and len(v):
        v = v[v["settle_status"].astype(str).str.upper().eq("SETTLED")]
        y = v["outcome"].astype(str).str.upper().map({"OVER": 1.0, "UNDER": 0.0})
        pm = _num(v, "p_market")
        sc = {}
        for col in ("p_v9", "p_v91", "p_pro_hgb", "p_goal_distribution"):
            if col in v.columns:
                s = prob_scores(_num(v, col), y, pm)
                sc[col] = {k: s[k] for k in ("n_prob", "log_loss_model", "brier_model", "n_vs_market", "log_loss_market",
                                             "brier_market", "d_log_loss_model_minus_market",
                                             "d_brier_model_minus_market")}
        out["v91_shadow_scored"] = {"target": "O/U 2.5 OVER (outcome)", "settled": int(len(v)),
                                    "matchdays": int(v["match_date"].astype(str).str[:10].nunique()), "scores": sc,
                                    "note": "probability scores of each shadow model vs the de-vigged market on the same "
                                            "rows; a lower log loss than the market is the only edge a shadow can show"}
    return out


def backtest_inventory(out_dir: Path) -> list:
    inv = []
    for f in BACKTEST_FILES:
        p = out_dir / f
        if not p.exists():
            inv.append({"file": f, "present": False})
            continue
        head = pd.read_csv(p, nrows=0).columns.tolist()
        with open(p, "rb") as fh:
            n = sum(1 for _ in fh) - 1
        inv.append({"file": f, "present": True, "rows": int(n),
                    "real_price_columns": [c for c in head if "odds" in c.lower()][:12],
                    "status": "OLDER_MODELS_BACKTEST_NOT_PROOF"})
    bo = out_dir / "bars_opt"
    if bo.exists():
        inv.append({"file": "bars_opt/", "present": True, "rows": None,
                    "files": len([x for x in bo.rglob("*") if x.is_file()]),
                    "status": "REAL_PRICE_REPLAY_FOR_BARS_NOT_PROOF",
                    "note": "per-cell walk-forward replays at real prices that set the approved bars; read on the "
                            "Admin Board 'Bars' tab"})
    return inv


def synthetic_inventory() -> list:
    from src.integrity import synthetic_roi as SR
    out = []
    for path, why in SR.QUARANTINED.items():
        p = ROOT / path
        marker = None
        if p.exists():
            try:
                marker = bool(SR.load_for_audit(p).get(SR.MARKER_KEY))
            except Exception:                                           # noqa: BLE001
                marker = None
        out.append({"file": path, "present": p.exists(), "quarantine_marker_in_file": marker, "why": why,
                    "figures_carried": False, "refused_purposes": list(SR.REFUSED_PURPOSES)})
    return out


# ── build ──────────────────────────────────────────────────────────────────────────────────────────
def build(out_dir=None, cfg: dict | None = None) -> dict:
    from src import trustworthy_performance as TP
    from dashboard_data import clean_record as CR
    if out_dir is None:
        import config
        out_dir = config.OUTPUT_DIR
    out_dir = Path(out_dir)
    cfg = cfg or tp_config()
    start = start_ts(cfg)
    reg, hyps = registrations()
    fair = market_fair(out_dir)
    views, corrections = TP.frames(out_dir, cfg)

    def live(view):
        parts = [team_rows(n, x, fair) for n, _k, x, _r, _e in views[view] if len(x)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)

    seg = {}
    # pre-repair
    a = live(TP.A)
    pl = prediction_ledger(out_dir)
    lr = learning_rows(pl, fair)
    if len(lr):
        lr["record"] = record_of(lr["gen"], lr["match_date"], start).to_numpy()
    first = lr[lr["snapshot"].eq("first_sight")] if len(lr) else lr
    seg["pre_repair_historical"] = {
        "label": "Before the repairs: settled live tips generated before the Clean Record start (unrepaired system, "
                 "older models), corrections sidecars applied. Never part of a Clean Record figure.",
        "tips": {src: cells(a[a["source"] == src], f"A|{src}", cfg, reg=reg) for src in sorted(a["source"].unique())},
        "prediction_ledger_learning": cells(first[first["record"] == "BEFORE"] if len(first) else first,
                                            "A|pl", cfg, bets=False),
        "owner_historical_figures": historical_figures(out_dir, cfg),
        "corrections_applied": [{k: c.get(k) for k in ("sidecar", "applied_to", "present", "effect", "rows_changed")}
                                for c in corrections]}
    # forward
    c = live(TP.C)
    t3 = _read(out_dir / "top3_league_ledger.csv")
    t3c = CR.top3_split(t3)[CR.CLEAN] if t3 is not None else pd.DataFrame()
    t3r = top3_rows(t3c) if len(t3c) else pd.DataFrame(columns=COLS)
    pled = _read(out_dir / "player_ledger.csv")
    pt = CR.props_split(pled)[CR.CLEAN] if pled is not None else pd.DataFrame()
    p3 = _read(out_dir / "props_top3_ledger.csv")
    p3c = record_of(p3["issued_at"], p3["match_date"], start) if p3 is not None and len(p3) else None
    props = pd.concat([x for x in (props_rows(pt, "player_ledger.csv") if len(pt) else None,
                                   props_rows(p3[p3c == "CLEAN"], "props_top3_ledger.csv")
                                   if p3c is not None and (p3c == "CLEAN").any() else None) if x is not None]
                      or [pd.DataFrame(columns=COLS)], ignore_index=True)
    fwd_first = first[first["record"] == "CLEAN"] if len(first) else first
    fwd_final = lr[(lr["snapshot"] == "final") & (lr["record"] == "CLEAN")] if len(lr) else lr
    seg["post_repair_forward"] = {
        "label": "Clean Record: generated AND played on or after the start, on the repaired data and clean models.",
        "top3_staked": cells(t3r[t3r["staked"]], "C|top3|staked", cfg, forward=True, reg=reg),
        "top3_unstaked": cells(t3r[~t3r["staked"].astype(bool)], "C|top3|unstaked", cfg, forward=True, reg=reg),
        "tips": {src: cells(c[c["source"] == src], f"C|{src}", cfg, forward=True, reg=reg)
                 for src in sorted(c["source"].unique())},
        "props_tips": {src: cells(props[props["source"] == src], f"C|{src}", cfg, forward=True)
                       for src in sorted(props["source"].unique())},
        "prediction_ledger_learning": {
            "first_sight": cells(fwd_first, "C|pl|first", cfg, bets=False),
            "final": cells(fwd_final, "C|pl|final", cfg, bets=False),
            "note": "learning rows, not bets: every fixture x market the models scored (owner 'predict every match'). "
                    "No P/L. The de-vigged market comparison exists for team markets with an edge-shadow quote."},
        "unplaced_rows": {"top3": int(len(CR.top3_split(t3)[CR.UNPLACED])) if t3 is not None else 0,
                          "props": int(len(CR.props_split(pled)[CR.UNPLACED])) if pled is not None else 0,
                          "prediction_ledger": int((lr["record"] == "UNPLACED").sum()) if len(lr) else 0}}
    # research
    rs = []
    for n, _k, x, _r, _e in views[TP.B]:
        if n in ("league_weekly_research.csv", "sharp_ledger.csv") and len(x):
            rs.append(research_rows(n, x))
    rsd = pd.concat(rs, ignore_index=True) if rs else pd.DataFrame(columns=COLS)
    seg["research_only"] = {"label": "Research only: never staked, never part of the Clean Record.",
                            **{src: cells(rsd[rsd["source"] == src], f"B|{src}", cfg)
                               for src in sorted(rsd["source"].unique())}}
    seg["shadow_experiments"] = {"label": "Shadow experiments: logged beside production, never sent or staked.",
                                 **shadow_inventory(out_dir, cfg)}
    seg["real_price_backtests"] = {"label": "Backtests at real bookmaker prices on OLDER models. A backtest proves "
                                            "nothing (owner rule: live proof only); inventory only.",
                                   "inventory": backtest_inventory(out_dir)}
    seg["synthetic_price_diagnostics"] = {"label": "QUARANTINED: ROI against synthetic prices. Never a performance "
                                                   "figure; listed for audit only.", "inventory": synthetic_inventory()}

    fwd_cells = [x for k in ("top3_staked", "top3_unstaked") for x in seg["post_repair_forward"][k]["cells"]] + \
        [x for v in seg["post_repair_forward"]["tips"].values() for x in v["cells"]] + \
        [x for v in seg["post_repair_forward"]["props_tips"].values() for x in v["cells"]]
    cand = [x for x in fwd_cells if x["verdict"] == CANDIDATE]
    from collections import Counter
    by_verdict = dict(Counter(x["verdict"] for x in fwd_cells))

    def n_settled(block: dict) -> int:                                  # an empty block has no total: no rows
        return block["total"]["settled"] if block["total"] else 0
    headline = {
        "verdict": NO_PROVEN_EDGE if not cand else f"{len(cand)} CELL(S) FOR OWNER REVIEW (not proven)",
        "forward_cells": len(fwd_cells), "forward_cells_by_verdict": by_verdict,
        "forward_settled": {"top3_staked": n_settled(seg["post_repair_forward"]["top3_staked"]),
                            "top3_unstaked": n_settled(seg["post_repair_forward"]["top3_unstaked"]),
                            **{k: n_settled(v) for k, v in seg["post_repair_forward"]["tips"].items()},
                            **{k: n_settled(v) for k, v in seg["post_repair_forward"]["props_tips"].items()}},
        "minimums": TP.minimums(cfg)}
    return {"schema": SCHEMA, "about": __doc__.split("\n\n")[0].strip(),
            "clean_record_start_utc": cfg.get("clean_record_start_utc"),
            "performance_claims_enabled": bool(cfg.get("performance_claims_enabled")),
            "claim_status": "EVALUATION ONLY - NOT A PERFORMANCE CLAIM (performance_claims_enabled is false)"
            if not cfg.get("performance_claims_enabled") else "owner-enabled",
            "as_of_last_settled_match": _as_of(seg), "inputs": _inputs(out_dir),
            "method": METHOD, "verdicts": VERDICTS, "headline": headline, "registered_hypotheses": hyps,
            "segments": seg, "immutability": pre_start_fingerprint(out_dir, cfg)}


METHOD = {
    "cell": "league x market x model version; market carries the side where the market has two",
    "settled": "WIN / LOSS / PUSH / HALF_WIN / HALF_LOSS with a recorded P/L; VOID is not a settled bet",
    "matchdays": "distinct match dates among settled rows (the bootstrap block)",
    "real_odds_coverage": "share of settled rows with a recorded entry price > 1 (P/L and ROI use those rows only)",
    "roi": "flat 1u P/L / priced settled rows; CI = matchday block bootstrap, config/trustworthy_performance.json",
    "staked": "src/staked.py (a recorded policy stake > 0); TOP-3 by the frozen stake_reason; both apart from "
              "not staked; ROI at the recorded stake where stake units were recorded",
    "max_drawdown": "largest peak-to-trough fall of cumulative flat-1u P/L in match-date order",
    "probability_scores": "log loss and Brier of the model probability of the tipped side, WIN/LOSS rows only; the "
                          "de-vigged bookmaker probability (power method, output/edge_engine_shadow.csv and its "
                          "pre-re-seed archive, the latest quote at or before the tip time; TOP-3 and research carry their own first-sight fair "
                          "probability) scored on the same rows; d_* = model minus market (negative = model better)",
    "calibration": "mean model probability vs hit rate (calibration_gap = hit rate - mean p); 10 bins once the cell "
                   "has the minimum settled rows",
    "clv": "certified close only (strictly pre-kickoff, within 30 minutes; the research panel's close_status OK). "
           "Team ledger rows carry no certified close: their stored CLV is shown apart as uncertified",
    "forward_test": "config/track_registry.json entry of the league x market x side x model type and the "
                    "preregistered hypotheses that cover it (registry/preregistered_hypotheses.json)",
    "never": "no raw price, mean price or mean market probability is written; segments are never pooled",
}
VERDICTS = {
    NOT_ENOUGH: "below the owner's minimums (config/trustworthy_performance.json): numbers shown, not readable yet",
    NEGATIVE: "enough data and the ROI CI is entirely below 0",
    NO_EDGE: "enough data, the ROI CI includes 0",
    POS_UNCONFIRMED: "ROI CI above 0 but no certified CLV evidence at its minimum, or not a forward record: a "
                     "hypothesis, never a proven edge (backtests and history prove nothing)",
    CANDIDATE: "forward record only: ROI CI above 0 AND certified CLV CI above 0 at its minimum; goes to the owner "
               "for review. Not PROVEN, no stake or bar change follows from it",
}


def _as_of(seg: dict) -> str | None:
    days = []

    def walk(o):
        if isinstance(o, dict):
            if o.get("last_match_date"):
                days.append(o["last_match_date"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(seg)
    return max(days) if days else None


def _inputs(out_dir: Path) -> dict:
    """sha256 (first 16) of every input ledger as committed bytes, so a reader can tell which data it describes."""
    files = ["bets_ledger.csv", "side_bets_ledger.csv", "ht_ledger.csv", "top3_league_ledger.csv",
             "props_top3_ledger.csv", "player_ledger.csv", "league_weekly_research.csv", "sharp_ledger.csv",
             "edge_engine_shadow.csv", "edge_engine_shadow_pre_reseed_2026-11-20.csv", "v91_shadow.csv",
             "stake_audit.csv", "stake_plan.csv"]
    out = {}
    for f in files:
        p = out_dir / f
        out[f] = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16] if p.exists() else None
    return out


# ── cells table (one CSV row per cell; the JSON keeps the totals) ─────────────────────────────────────
CELL_KEYS = ("segment", "part", "source", "league", "market", "model_version")
CELLS_FILE = "clean_record_evaluation_cells.csv"


def _flat(c: dict) -> dict:
    out = {}
    for k, v in c.items():
        if k == "calibration":
            continue
        if k == "forward_test":
            tr = v.get("track_registry")
            out["ft_track_state"] = tr.get("production_state") if isinstance(tr, dict) else tr
            out["ft_real_money"] = tr.get("real_money") if isinstance(tr, dict) else None
            out["ft_hypotheses"] = ";".join(v.get("preregistered_hypotheses") or [])
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, list) and len(v2) == 2:
                    out[f"{k}_{k2}_lo"], out[f"{k}_{k2}_hi"] = v2
                else:
                    out[f"{k}_{k2}"] = v2
        elif isinstance(v, list) and len(v) == 2:
            out[f"{k}_lo"], out[f"{k}_hi"] = v
        else:
            out[k] = v
    return out


def split_cells(res: dict) -> tuple[dict, pd.DataFrame]:
    """Move every `cells` list out of the summary into one table; the summary keeps each block's total and its
    cell count. Returns new objects (the input is not changed)."""
    rows = []

    def walk(o, path):
        if isinstance(o, dict):
            if isinstance(o.get("cells"), list):
                seg, part, src = (path + [None, None, None])[:3]
                for c in o["cells"]:
                    rows.append({"segment": seg, "part": part, "source": src, **_flat(c)})
                return {**{k: v for k, v in o.items() if k != "cells"}, "n_cells": len(o["cells"])}
            return {k: walk(v, path + [k]) for k, v in o.items()}
        return o
    out = dict(res)
    out["segments"] = walk(res["segments"], [])
    df = pd.DataFrame(rows)
    if len(df):
        lead = [c for c in CELL_KEYS if c in df.columns]
        df = df[lead + [c for c in df.columns if c not in lead]].sort_values(
            lead, na_position="first", kind="mergesort").reset_index(drop=True)
    return out, df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=None, help="ledger directory (default config.OUTPUT_DIR)")
    ap.add_argument("--write", action="store_true", help=f"write output/{OUT_FILE}")
    ap.add_argument("--out", default=None, help="write to this path instead")
    a = ap.parse_args(argv)
    import config
    out_dir = Path(a.out_dir) if a.out_dir else Path(config.OUTPUT_DIR)
    res, table = split_cells(build(out_dir))
    res["cells_file"] = f"output/{CELLS_FILE}"
    text = json.dumps(res, indent=1, default=str, ensure_ascii=False) + "\n"
    if a.write or a.out:
        target = Path(a.out) if a.out else out_dir / OUT_FILE
        target.write_text(text, encoding="utf-8", newline="\n")
        table.to_csv(target.with_suffix(".cells.csv") if a.out else out_dir / CELLS_FILE, index=False,
                     lineterminator="\n")
        print(f"wrote {target} (+ {len(table)} cells): {res['headline']['verdict']}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
