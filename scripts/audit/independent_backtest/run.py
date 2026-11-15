"""AU5 phase-10 driver: replay -> settle -> reconcile -> ROI -> placebos. Read-only.

Writes:
  reports/10_backtest_mismatches.csv        (required columns, one row per mismatching bet)
  reports/10_reconciliation_full.csv        (every production and independent bet, both sides' fields)
  reports/10_independent_backtest_numbers.json
  reports/10_regime_timeline.csv
Usage (from the worktree root):
  python -m scripts.audit.independent_backtest.extract      # once (HEAVY.lock)
  python -m scripts.audit.independent_backtest.run
"""
from __future__ import annotations

import io
import json
import math
import sys

import numpy as np
import pandas as pd

from .common import CACHE, REF, REPO, heavy_lock, show
from .engine import replay
from .regimes import CODE_REGIMES, RegimeTimeline
from .results import ResultIndex, load_results, outcome, pnl
from .selection import decide_ou25, decide_side, snapshot_time

RNG_SEED = 20261101
N_BOOT = 2000
N_PLACEBO = 1000
LIVE_EVIDENCE_FROM = "2026-09-01"
TIER_STAKE = {"SNIPER": 1.0, "MARKSMAN": 0.75, "VALUABLE": 0.5}
OUT = REPO / "reports"


def _ts(x):
    t = pd.Timestamp(str(x))
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except (TypeError, ValueError):
        return math.nan


# ── production ledgers ──────────────────────────────────────────────────────────────────────
def load_production():
    cur_ou = pd.read_csv(io.BytesIO(show(REF, "output/bets_ledger.csv")), dtype=str)
    cur_ou = cur_ou[cur_ou["source"].fillna("live") == "live"].copy()
    cur_ou["market"] = "ou25"
    cur_sd = pd.read_csv(io.BytesIO(show(REF, "output/side_bets_ledger.csv")), dtype=str)
    cur_sd = cur_sd[cur_sd["source"].fillna("live") == "live"].copy()
    cur_sd["side"] = cur_sd.get("side", pd.Series(index=cur_sd.index, dtype=str))
    cur_sd["side"] = cur_sd["side"].fillna(cur_sd["market"].map({"btts": "YES", "over15": "OVER", "over35": "OVER"}))
    fs_ou = pd.read_parquet(CACHE / "ledger_first_seen_ou25.parquet")
    fs_ou["market"] = "ou25"
    fs_sd = pd.read_parquet(CACHE / "ledger_first_seen_side.parquet")
    fs_sd["side"] = fs_sd.get("side", pd.Series(index=fs_sd.index, dtype=str))
    fs_sd["side"] = fs_sd["side"].fillna(fs_sd["market"].map({"btts": "YES", "over15": "OVER", "over35": "OVER"}))
    cur = pd.concat([cur_ou, cur_sd], ignore_index=True)
    fs = pd.concat([fs_ou, fs_sd], ignore_index=True)
    for d in (cur, fs):
        d["match_date"] = d["match_date"].astype(str).str[:10]
        d["key"] = list(zip(d["match_date"], d["home_team"].astype(str), d["away_team"].astype(str),
                            d["market"], d["side"].astype(str).str.strip()))
    return cur, fs


def load_stakes():
    sa = pd.read_csv(io.BytesIO(show(REF, "output/stake_audit.csv")), dtype=str)
    sa["generated_at"] = pd.to_datetime(sa["generated_at"], errors="coerce")
    sa = sa.sort_values("generated_at", kind="mergesort")
    parts = sa["fixture"].str.split("|", expand=True)
    sa["match_date"], sa["home_team"], sa["away_team"] = parts[0], parts[1], parts[2]
    sa["market"] = sa["market"].str.lower().replace({"ou25": "ou25", "btts": "btts", "over15": "over15",
                                                      "over35": "over35"})
    sa["key"] = list(zip(sa["match_date"], sa["home_team"], sa["away_team"], sa["market"], sa["side"].str.upper()))
    sa = sa.drop_duplicates("key", keep="first")
    return dict(zip(sa["key"], pd.to_numeric(sa["stake_amount"], errors="coerce")))


# ── settlement ───────────────────────────────────────────────────────────────────────────────
class Settler:
    def __init__(self):
        self.idx = ResultIndex(load_results())
        self.cache: dict = {}

    def fixture(self, league, d, h, a):
        k = (league, d, h, a)
        if k not in self.cache:
            self.cache[k] = self.idx.find(league, d, h, a)
        return self.cache[k]

    def settle(self, league, d, h, a, market, side, odds, stake=1.0):
        fx = self.fixture(league, d, h, a)
        if fx["status"] not in ("MATCHED", "SWAPPED"):
            return {"res_status": fx["status"], "result": None, "pnl": None, "why": fx.get("why", "")}
        if fx["status"] == "SWAPPED" and market not in ("ou25", "over15", "over35", "btts"):
            return {"res_status": "SWAPPED_SIDE_MARKET", "result": None, "pnl": None, "why": "orientation"}
        try:
            res = outcome(market, side, fx["hg"], fx["ag"])
        except ValueError as e:
            return {"res_status": "INVALID_SIDE", "result": None, "pnl": None, "why": str(e)}
        o = _num(odds)
        return {"res_status": fx["status"], "result": res, "pnl": pnl(res, o, stake) if o > 1 else None,
                "why": "", "score": f"{fx['hg']}-{fx['ag']}", "res_src": fx.get("src")}


# ── statistics ───────────────────────────────────────────────────────────────────────────────
def roi(df, stake_col=None):
    s = df[df["ind_result"].isin(["WIN", "LOSS"])]
    if s.empty:
        return {"n": 0}
    stake = s[stake_col].astype(float) if stake_col else pd.Series(1.0, index=s.index)
    ret = np.where(s["ind_result"] == "WIN", s["odds"].astype(float) - 1.0, -1.0) * stake
    st = stake.sum()
    return {"n": int(len(s)), "staked": round(float(st), 4), "profit": round(float(ret.sum()), 4),
            "roi_pct": round(100 * float(ret.sum()) / st, 2) if st > 0 else None,
            "hit_pct": round(100 * float((s["ind_result"] == "WIN").mean()), 2),
            "mean_odds": round(float(s["odds"].astype(float).mean()), 3)}


def block_boot(df, rng, n=N_BOOT):
    s = df[df["ind_result"].isin(["WIN", "LOSS"])].copy()
    if len(s) < 10:
        return None
    s["ret"] = np.where(s["ind_result"] == "WIN", s["odds"].astype(float) - 1.0, -1.0)
    g = s.groupby("fx_date")["ret"].agg(["sum", "count"])
    sums, cnts = g["sum"].to_numpy(), g["count"].to_numpy()
    k = len(g)
    idx = rng.integers(0, k, size=(n, k))
    rois = sums[idx].sum(1) / cnts[idx].sum(1) * 100
    return [round(float(np.percentile(rois, 2.5)), 2), round(float(np.percentile(rois, 97.5)), 2)]


def placebos(book, universe, settler, rng, label):
    """Random side / random equal-size fixture subset / anti-strategy / outcome permutation."""
    out = {}
    s = book[book["ind_result"].isin(["WIN", "LOSS"])]
    if len(s) < 20:
        return {"note": "too few settled bets"}
    actual = roi(s)["roi_pct"]
    # 1) random side on the SAME fixtures at the SAME first-sight board (O/U only: both prices real)
    if (s["market"] == "ou25").all():
        rs = []
        o_sel, o_opp = s["odds"].astype(float).to_numpy(), s["odds_opposite"].map(_num).to_numpy()
        win_sel = (s["ind_result"] == "WIN").to_numpy()
        ok = np.isfinite(o_opp) & (o_opp > 1)
        for _ in range(N_PLACEBO):
            flip = rng.random(len(s)) < 0.5
            flip &= ok
            r = np.where(flip, np.where(~win_sel, o_opp - 1, -1.0), np.where(win_sel, o_sel - 1, -1.0))
            rs.append(r.mean() * 100)
        rs = np.array(rs)
        out["random_side"] = {"mean_roi": round(float(rs.mean()), 2), "p95": round(float(np.percentile(rs, 95)), 2),
                              "p_value_actual_ge": round(float((rs >= actual).mean()), 4)}
        anti = np.where(~win_sel & ok, o_opp - 1, -1.0)[ok]
        out["anti_strategy_opposite_side"] = {"n": int(ok.sum()), "roi_pct": round(float(anti.mean() * 100), 2)}
    # 2) random fixture subset of equal size from the priced universe in the same window, random side
    lo, hi = s["fx_date"].min(), s["fx_date"].max()
    mk = s["market"].iloc[0]
    u = universe[(universe["market"] == mk) & (universe["fx_date"] >= lo) & (universe["fx_date"] <= hi)]
    if mk == "ou25":
        u = u[u["model_type"].isin(s["model_type"].unique())]
    u = u.copy()
    if len(u) >= len(s):
        rets = {}
        for r in u.itertuples():
            fx = settler.fixture(r.league, r.fx_date, r.home_team, r.away_team)
            if fx["status"] not in ("MATCHED", "SWAPPED"):
                continue
            if mk == "ou25":
                for side in ("OVER", "UNDER"):
                    o = _num(getattr(r, f"odds_{side}"))
                    if o > 1:
                        res = outcome("ou25", side, fx["hg"], fx["ag"])
                        rets.setdefault(r.Index, []).append(o - 1 if res == "WIN" else -1.0)
            else:
                side = "YES" if mk == "btts" else "OVER"
                o = _num(getattr(r, f"odds_{side}", math.nan))
                if o > 1:
                    res = outcome(mk, side, fx["hg"], fx["ag"])
                    rets.setdefault(r.Index, []).append(o - 1 if res == "WIN" else -1.0)
        keys = list(rets)
        if len(keys) >= len(s):
            rs = []
            for _ in range(N_PLACEBO):
                pick = rng.choice(len(keys), size=len(s), replace=False)
                rs.append(np.mean([rets[keys[i]][rng.integers(0, len(rets[keys[i]]))] for i in pick]) * 100)
            rs = np.array(rs)
            out["random_fixture_subset_equal_n"] = {
                "frame_n": len(keys), "draw_n": int(len(s)), "mean_roi": round(float(rs.mean()), 2),
                "p2.5": round(float(np.percentile(rs, 2.5)), 2), "p97.5": round(float(np.percentile(rs, 97.5)), 2),
                "p_value_actual_ge": round(float((rs >= actual).mean()), 4)}
    # 3) negative control: outcomes permuted across the book's own bets (kills any information)
    win = (s["ind_result"] == "WIN").to_numpy()
    o = s["odds"].astype(float).to_numpy()
    rs = []
    for _ in range(N_PLACEBO):
        w = rng.permutation(win)
        rs.append(np.where(w, o - 1, -1.0).mean() * 100)
    rs = np.array(rs)
    out["outcome_permutation_control"] = {"mean_roi": round(float(rs.mean()), 2),
                                          "p_value_actual_ge": round(float((rs >= actual).mean()), 4)}
    out["actual_roi_pct"] = actual
    return out


# ── reconciliation ──────────────────────────────────────────────────────────────────────────
def board_decision_for(snaps_fx: pd.DataFrame, timeline, market: str, when):
    """Independent decision on the committed board nearest to `when` (for explaining a mismatch)."""
    if snaps_fx.empty:
        return None, None
    tt = snaps_fx["_t"]
    i = (tt - when).abs().idxmin()
    r = snaps_fx.loc[i].to_dict()
    reg = timeline.at(r["_t"])
    d = decide_ou25(r, reg, r["_t"]) if market == "ou25" else decide_side(r, market, reg, r["_t"])
    return d, r


def main() -> int:
    rng = np.random.default_rng(RNG_SEED)
    snaps = pd.read_parquet(CACHE / "pred_snapshots.parquet")
    timeline = RegimeTimeline()
    pd.DataFrame(timeline.summary()).to_csv(OUT / "10_regime_timeline.csv", index=False)
    with heavy_lock("AU5 independent backtest replay"):
        fs, universe, dec, boards = replay(snaps, timeline)
        snaps["fx_date"] = snaps["date"].astype(str).str[:10]
        snaps["_t"] = [snapshot_time(r) for r in snaps[["generated_at", "snap_commit_ts"]].to_dict("records")]
        snaps_by_fx = {k: g for k, g in snaps.groupby(["fx_date", "home_team", "away_team"])}
        settler = Settler()
        first_board_t = snaps["_t"].min()

        # settle the independent books (flat stake at first-sight odds)
        res = [settler.settle(r.league, r.fx_date, r.home_team, r.away_team, r.market, r.side, r.odds)
               for r in fs.itertuples()]
        fs["ind_res_status"] = [x["res_status"] for x in res]
        fs["ind_result"] = [x["result"] for x in res]
        fs["ind_pnl"] = [x["pnl"] for x in res]
        fs["score"] = [x.get("score") for x in res]
        fs["key"] = list(zip(fs["fx_date"], fs["home_team"], fs["away_team"], fs["market"], fs["side"]))

        cur, pfs = load_production()
        stakes = load_stakes()
        cur_by_key = {k: g.iloc[0] for k, g in cur.groupby("key", sort=False)}
        dup_keys = {k: len(g) for k, g in cur.groupby("key", sort=False) if len(g) > 1}
        pfs_by_key = {r["key"]: r for r in pfs.to_dict("records")}
        ind_by_key = {r["key"]: r for r in fs.to_dict("records")}
        side_ledger_start = _ts(pfs[pfs["market"] != "ou25"]["first_commit_ts"].min())

        rows = []
        all_prod_keys = set(cur_by_key) | set(pfs_by_key)
        for key in sorted(all_prod_keys, key=lambda k: (k[0], k[1], k[3])):
            d, h, a, mkt, side = key
            c = cur_by_key.get(key)
            f = pfs_by_key.get(key)
            src = c if c is not None else pd.Series(f)
            league = str(src.get("league"))
            reasons = []
            gen = _ts(f["generated_at"]) if f is not None and str(f.get("generated_at")) not in ("", "nan", "None") \
                else (_ts(src.get("generated_at")) if str(src.get("generated_at")) not in ("", "nan", "None") else None)
            if c is None:
                reasons.append("row_removed_from_current_ledger")
            if f is None:
                reasons.append("row_absent_from_git_history_first_seen")
            if key in dup_keys:
                reasons.append(f"duplicate_ledger_rows_x{dup_keys[key]}")
            if side not in ("OVER", "UNDER", "YES", "NO"):
                reasons.append(f"invalid_side_{side}")
            I = ind_by_key.get(key)
            prod_first_odds = _num(f.get("odds")) if f is not None else _num(src.get("odds"))
            prod_cur_odds = _num(c.get("odds")) if c is not None else math.nan
            prod_first_tier = (f.get("signal_tier") if f is not None else src.get("signal_tier"))
            if c is not None and f is not None and abs(prod_cur_odds - prod_first_odds) > 1e-6:
                reasons.append("ledger_odds_overwritten_after_first_log")
            if c is not None and f is not None and str(c.get("signal_tier")) != str(f.get("signal_tier")):
                reasons.append("ledger_tier_overwritten_after_first_log")
            pre_history = gen is not None and gen < first_board_t - pd.Timedelta(minutes=30)
            if pre_history:
                reasons.append("NOT_IDENTIFIABLE_logged_before_first_committed_board")
            if I is None and not pre_history:
                g = snaps_by_fx.get((d, h, a))
                if g is None:
                    reasons.append("fixture_never_on_a_committed_board")
                else:
                    dd, br = board_decision_for(g, timeline, mkt, gen if gen is not None else g["_t"].min())
                    if dd is not None and dd["selected"] and dd["side"] != side:
                        reasons.append(f"independent_selects_other_side_{dd['side']}")
                    elif dd is not None:
                        reasons.append(f"independent_rejects_{dd['reason'] or 'unselected'}")
                        prod_board = str(br.get("bet")) if mkt == "ou25" else ""
                        if mkt == "ou25" and prod_board in ("OVER", "UNDER"):
                            reasons.append(f"production_board_said_{prod_board}_{br.get('signal_tier')}")
            elif I is not None and gen is not None:
                lag = (I["t_first"] - gen).total_seconds() / 60
                if lag < -20:
                    reasons.append("independent_first_sight_earlier_than_production_log")
                elif lag > 20:
                    reasons.append("production_logged_from_uncommitted_board")
                if not math.isnan(prod_first_odds) and abs(float(I["odds"]) - prod_first_odds) > 0.0051 \
                        and abs(lag) <= 20:
                    reasons.append("first_sight_odds_differ_same_run")
                if str(I["tier"]) != str(prod_first_tier) and abs(lag) <= 20:
                    era = "pre_2026-10-26_drift_tier_era" if gen < _ts(CODE_REGIMES["drift_tier_removed"]) else "post_drift_removal"
                    legacy = " (legacy VALUE label)" if str(prod_first_tier) == "VALUE" else ""
                    reasons.append(f"tier_differs_{prod_first_tier}_vs_{I['tier']}_{era}{legacy}")
            # settlement comparison (independent settles at PRODUCTION's own settlement odds here,
            # to isolate settlement from selection/price)
            p_res = str(src.get("result")) if str(src.get("result")) not in ("", "nan", "None") else ""
            p_pnl = _num(src.get("pnl"))
            st = settler.settle(league, d, h, a, mkt, side, prod_cur_odds if c is not None else prod_first_odds)
            i_res = st["result"] or ""
            if p_res in ("WIN", "LOSS") and i_res and p_res != i_res:
                reasons.append(f"RESULT_DISAGREES_score_{st.get('score')}_src_{st.get('res_src')}")
            elif p_res in ("WIN", "LOSS") and not i_res:
                reasons.append(f"independent_result_unavailable_{st['res_status']}")
            elif not p_res and i_res:
                reasons.append("production_unsettled_but_result_known")
            elif p_res == "VOID" and i_res:
                reasons.append(f"production_VOID_but_played_{st.get('score')}")
            if p_res in ("WIN", "LOSS") and i_res == p_res and st["pnl"] is not None and not math.isnan(p_pnl) \
                    and abs(st["pnl"] - p_pnl) > 0.011:
                reasons.append("pnl_arithmetic_differs")
            if p_res in ("WIN", "LOSS") and math.isnan(p_pnl):
                reasons.append("production_settled_without_pnl")
            ind_pnl_first = I["ind_pnl"] if I is not None else None
            rows.append({
                "bet_id": f"P|{d}|{h}|{a}|{mkt}|{side}", "fixture_id": f"{d}|{h}|{a}", "market": mkt, "side": side,
                "league": league, "model_type": src.get("model_type"),
                "production_result": p_res, "independent_result": i_res,
                "production_pnl": None if math.isnan(p_pnl) else round(p_pnl, 4),
                "independent_pnl": ind_pnl_first if ind_pnl_first is not None else st["pnl"],
                "independent_pnl_at_production_odds": st["pnl"],
                "production_first_logged_at": gen.isoformat() if gen is not None else "",
                "production_first_odds": prod_first_odds, "production_current_odds": prod_cur_odds,
                "production_first_tier": prod_first_tier,
                "production_current_tier": c.get("signal_tier") if c is not None else "",
                "independent_first_sight": I["t_first"].isoformat() if I is not None else "",
                "independent_odds": I["odds"] if I is not None else math.nan,
                "independent_edge": I["edge"] if I is not None else math.nan,
                "independent_tier": I["tier"] if I is not None else "",
                "independent_policy_drop": I["policy_drop"] if I is not None else "",
                "result_status": st["res_status"],
                "mismatch_reason": ";".join(reasons), "origin": "production"})
        # independent-only bets inside each ledger's operating window
        for key, I in ind_by_key.items():
            if key in all_prod_keys:
                continue
            start = first_board_t if I["market"] == "ou25" else side_ledger_start
            if I["t_first"] < start:
                continue
            d, h, a, mkt, side = key
            reasons = ["independent_only"]
            other = [k for k in all_prod_keys if k[:4] == key[:4] and k[4] != side]
            if other:
                reasons.append("production_has_opposite_side_first(flip_guard/one_side_policy)")
            if mkt == "ou25":
                pb = str(I.get("prod_board_bet"))
                reasons.append(f"production_board_decision_{pb}_{I.get('prod_board_tier')}")
            if I["t_first"] < _ts(CODE_REGIMES["both_losing_guard_removed"]) and mkt == "ou25"                     and str(I.get("prod_board_bet")) == "AVOID":
                reasons.append("NOT_IDENTIFIABLE_both_losing_guard_v7_params_uncommitted")
            if I["policy_drop"]:
                reasons.append(f"independent_policy_drop_{I['policy_drop']}")
            if I["t_first"] >= _ts("2026-10-31T00:00:00Z"):
                reasons.append("recent_board_ledger_commit_may_lag")
            rows.append({
                "bet_id": f"I|{d}|{h}|{a}|{mkt}|{side}", "fixture_id": f"{d}|{h}|{a}", "market": mkt, "side": side,
                "league": I["league"], "model_type": I["model_type"],
                "production_result": "", "independent_result": I["ind_result"] or "",
                "production_pnl": None, "independent_pnl": I["ind_pnl"], "independent_pnl_at_production_odds": None,
                "production_first_logged_at": "", "production_first_odds": math.nan,
                "production_current_odds": math.nan, "production_first_tier": "", "production_current_tier": "",
                "independent_first_sight": I["t_first"].isoformat(), "independent_odds": I["odds"],
                "independent_edge": I["edge"], "independent_tier": I["tier"],
                "independent_policy_drop": I["policy_drop"], "result_status": I["ind_res_status"],
                "mismatch_reason": ";".join(reasons), "origin": "independent"})
        rec = pd.DataFrame(rows)
        rec.to_csv(OUT / "10_reconciliation_full.csv", index=False)
        req = ["bet_id", "fixture_id", "market", "side", "production_result", "independent_result",
               "production_pnl", "independent_pnl", "mismatch_reason"]
        rec[rec["mismatch_reason"] != ""][req].to_csv(OUT / "10_backtest_mismatches.csv", index=False)

        # ── numbers ──
        nums: dict = {"ref": REF, "n_boards": int(snaps["snap_sha"].nunique()),
                      "board_rows": int(len(snaps)), "first_board_utc": first_board_t.isoformat(),
                      "last_board_utc": snaps["_t"].max().isoformat(),
                      "side_ledger_first_commit": side_ledger_start.isoformat(),
                      "decisions": dec.to_dict("records")}
        # reason tallies
        tally = {}
        for r in rec["mismatch_reason"]:
            for x in str(r).split(";"):
                if x:
                    base = x.split("_score_")[0]
                    for pre in ("independent_rejects_", "tier_differs_", "production_board_said_",
                                "production_board_decision_", "independent_result_unavailable_",
                                "production_VOID_but_played_", "duplicate_ledger_rows_"):
                        if base.startswith(pre) and pre in ("production_VOID_but_played_", "duplicate_ledger_rows_"):
                            base = pre.rstrip("_")
                    tally[base] = tally.get(base, 0) + 1
        nums["mismatch_reason_counts"] = dict(sorted(tally.items(), key=lambda kv: -kv[1]))
        prod = rec[rec["origin"] == "production"]
        nums["reconciliation"] = {
            "production_keys": int(len(prod)), "independent_only": int((rec["origin"] == "independent").sum()),
            "production_exact_match_no_reason": int((prod["mismatch_reason"] == "").sum()),
            "by_market": {m: {"production": int((prod["market"] == m).sum()),
                              "matched_clean": int(((prod["market"] == m) & (prod["mismatch_reason"] == "")).sum()),
                              "independent_also_selected": int(((prod["market"] == m) & (prod["independent_tier"] != "")).sum()),
                              "independent_only": int(((rec["origin"] == "independent") & (rec["market"] == m)).sum())}
                          for m in ["ou25", "btts", "over15", "over35"]},
            "result_comparisons": {
                "both_settled": int(((prod["production_result"].isin(["WIN", "LOSS"])) &
                                     (prod["independent_result"].isin(["WIN", "LOSS"]))).sum()),
                "agree": int(((prod["production_result"].isin(["WIN", "LOSS"])) &
                              (prod["production_result"] == prod["independent_result"])).sum()),
                "disagree": int(prod["mismatch_reason"].str.contains("RESULT_DISAGREES").sum())}}
        # ROI: independent policy book (one position per fixture-direction), flat & documented-tier stake
        book = fs[(fs["policy_drop"] == "")].copy()
        book["tier_stake"] = book["tier"].map(TIER_STAKE).fillna(0.5)
        book["actual_stake"] = book["key"].map(stakes)
        roi_tab = {}
        for (mkt, mt), g in book.groupby(["market", "model_type"]):
            for era, ge in (("all", g), ("from_2026-09-01", g[g["fx_date"] >= LIVE_EVIDENCE_FROM])):
                for side, gs in [("ALL", ge)] + [(s, ge[ge["side"] == s]) for s in sorted(ge["side"].unique())]:
                    k = f"{mkt}|{mt}|{era}|{side}"
                    roi_tab[k] = {"flat": roi(gs), "flat_ci95_day_block": block_boot(gs, rng),
                                  "documented_tier_stake": roi(gs, "tier_stake")}
                    sa = gs[gs["actual_stake"].fillna(0) > 0]
                    roi_tab[k]["actual_stake_audit"] = roi(sa, "actual_stake") if len(sa) else \
                        {"n": 0, "note": "NOT IDENTIFIABLE - no stake_audit row (stake log starts 2026-10-14)"}
        # standard by serving model version
        std = book[(book["market"] == "ou25") & (book["model_type"] == "standard")]
        std_rep = std[std["t_first"] >= _ts(CODE_REGIMES["standard_repaired_serving"])]
        std_leg = std[(std["t_first"] < _ts(CODE_REGIMES["standard_repaired_serving"])) &
                      (std["fx_date"] >= LIVE_EVIDENCE_FROM)]
        roi_tab["ou25|standard|repaired_served_from_2026-10-28"] = {"flat": roi(std_rep)}
        roi_tab["ou25|standard|legacy_2026-09-01_to_10-28"] = {"flat": roi(std_leg),
                                                              "flat_ci95_day_block": block_boot(std_leg, rng)}
        nums["independent_roi"] = roi_tab
        # odds bands for ou25 from 2026-09-01
        ob = book[(book["market"] == "ou25") & (book["fx_date"] >= LIVE_EVIDENCE_FROM)].copy()
        ob["band"] = pd.cut(ob["odds"].astype(float), [1, 2, 2.5, 3, 100], labels=["<2", "2-2.5", "2.5-3", ">3"])
        nums["ou25_from_09_01_by_model_side_band"] = {
            f"{mt}|{s}|{b}": roi(g) for (mt, s, b), g in ob.groupby(["model_type", "side", "band"], observed=True)}
        # production ledger ROI (current file, settled live rows) for side-by-side
        pr = {}
        cur, _ = load_production()
        cur["ind_result"] = cur["result"]
        cur["odds"] = pd.to_numeric(cur["odds"], errors="coerce")
        cur["fx_date"] = cur["match_date"]
        for (mkt, mt), g in cur[cur["side"].isin(["OVER", "UNDER", "YES"])].groupby(["market", "model_type"]):
            for era, ge in (("all", g), ("from_2026-09-01", g[g["fx_date"] >= LIVE_EVIDENCE_FROM])):
                for side, gs in [("ALL", ge)] + [(s, ge[ge["side"] == s]) for s in sorted(ge["side"].unique())]:
                    r = roi(gs)
                    pn = pd.to_numeric(gs["pnl"], errors="coerce")
                    r["ledger_pnl_sum"] = round(float(pn[gs["result"].isin(["WIN", "LOSS"])].sum()), 4)
                    pr[f"{mkt}|{mt}|{era}|{side}"] = r
        nums["production_ledger_roi"] = pr
        # placebos & negative controls on the main live-evidence books
        pl = {}
        for (mkt, mt), g in book[book["fx_date"] >= LIVE_EVIDENCE_FROM].groupby(["market", "model_type"]):
            if g["ind_result"].isin(["WIN", "LOSS"]).sum() >= 20:
                pl[f"{mkt}|{mt}|from_2026-09-01"] = placebos(g, universe, settler, rng, f"{mkt}|{mt}")
        nums["placebos"] = pl
        nums["independent_settlement_status"] = fs["ind_res_status"].value_counts().to_dict()
        nums["first_sight_books"] = fs.groupby(["market", "policy_drop"]).size().rename("n").reset_index() \
            .to_dict("records")
    (OUT / "10_independent_backtest_numbers.json").write_text(json.dumps(nums, indent=1, default=str), encoding="utf-8")
    fs.drop(columns=["key"]).to_parquet(CACHE / "independent_firstsight.parquet", index=False)
    print(json.dumps(nums["reconciliation"], indent=1))
    print(json.dumps(nums["mismatch_reason_counts"], indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
