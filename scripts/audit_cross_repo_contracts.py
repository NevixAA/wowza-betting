"""Phase 2 Part 10 evidence: cross-repository contracts (read-only; light; column-projected reads).

    python scripts/audit_cross_repo_contracts.py [--dev C:/Users/nevo/dev] [--out reports/...json]

Reads committed output files of the five Wowza repos from their local checkouts (each expected at
its data-truth-100 commit; the SHAs are recorded in the output) and RE-DERIVES every figure:
league canonicality, market/side vocabularies, decimal odds validity, P&L reproducibility from
(result, odds), model-version stamping, signal time vs match time, settlement vocabularies and
PAPER/ACTIVE agreement. It never writes to any ledger.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GIT = r"C:\Program Files\Git\cmd\git.exe"
ROOT = Path(__file__).resolve().parents[1]


def sha(repo: Path) -> str:
    try:
        return subprocess.run([GIT, "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=20).stdout.strip() or "UNIDENTIFIABLE"
    except Exception as e:                                            # noqa: BLE001
        return f"UNIDENTIFIABLE ({type(e).__name__})"


def registry(v9: Path):
    r = json.loads((v9 / "config/leagues.json").read_text(encoding="utf-8"))
    names = {e["name"] for e in r["leagues"]}
    aliases = {a: e["name"] for e in r["leagues"] for a in (e.get("aliases") or [])}
    paper = {e["name"] for e in r["leagues"] if e.get("paper")}
    return names, aliases, paper, r.get("registry_version")


def league_check(series: pd.Series, names, aliases) -> dict:
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    canon = s.isin(names)
    alias = ~canon & s.isin(set(aliases))
    unknown = ~canon & ~alias
    return {"rows": int(len(s)), "distinct": int(s.nunique()), "canonical_rows": int(canon.sum()),
            "alias_rows": int(alias.sum()), "unknown_rows": int(unknown.sum()),
            "alias_values": sorted(s[alias].unique().tolist())[:15],
            "unknown_values": sorted(s[unknown].unique().tolist())[:25]}


def flat_pnl(result: pd.Series, odds: pd.Series) -> pd.Series:
    r = result.astype(str).str.upper()
    o = pd.to_numeric(odds, errors="coerce")
    return pd.Series(np.select([r.eq("WIN"), r.eq("LOSS"), r.eq("VOID")], [o - 1.0, -1.0, 0.0],
                               default=np.nan), index=result.index)


def pnl_check(d: pd.DataFrame, odds_col: str, pnl_col: str = "pnl", result_col: str = "result") -> dict:
    res = d[result_col].astype(str).str.upper()
    settled = d[res.isin(["WIN", "LOSS", "VOID"])]
    exp = flat_pnl(settled[result_col], settled[odds_col])
    got = pd.to_numeric(settled[pnl_col], errors="coerce")
    both = exp.notna() & got.notna()
    diff = (exp - got).abs()
    bad = both & (diff > 0.011)
    return {"settled_rows": int(len(settled)), "pnl_missing_on_settled": int(got.isna().sum()),
            "odds_missing_on_settled": int(exp.isna().sum()),
            "pnl_mismatch_vs_flat_1u": int(bad.sum()),
            "pnl_sum_recorded": round(float(got[both].sum()), 2),
            "pnl_sum_recomputed": round(float(exp[both].sum()), 2),
            "mismatch_examples": settled[bad].head(5)[[c for c in (result_col, odds_col, pnl_col, "league",
                                                                   "match_date") if c in settled]]
                                  .astype(str).to_dict("records")}


def odds_check(s: pd.Series) -> dict:
    o = pd.to_numeric(s, errors="coerce")
    present = s.notna() & (s.astype(str).str.strip() != "")
    return {"present": int(present.sum()), "unparseable": int((present & o.isna()).sum()),
            "le_1": int((o <= 1.0).sum()), "gt_50": int((o > 50).sum()),
            "min": None if o.dropna().empty else round(float(o.min()), 3),
            "max": None if o.dropna().empty else round(float(o.max()), 3)}


def vc(s: pd.Series, n=20) -> dict:
    return {str(k): int(v) for k, v in s.astype(object).fillna("<NA>").astype(str).value_counts().head(n).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default=r"C:\Users\nevo\dev")
    ap.add_argument("--dash", default=r"C:\Users\nevo\dev\wtp2\crossrepo_dash")
    ap.add_argument("--out", default=str(ROOT / "reports" / "crossrepo_contracts_evidence.json"))
    a = ap.parse_args()
    dev = Path(a.dev)
    v9, pro, v11, ex, dash = ROOT, dev / "wowzaV9-Pro", dev / "wowza_v11", dev / "wowza-exec", Path(a.dash)
    E: dict = {"inspected": {k: {"path": str(p), "exists": p.exists(), "head": sha(p) if p.exists() else
                                 "UNIDENTIFIABLE (no checkout)"}
                             for k, p in (("v9", v9), ("pro", pro), ("v11", v11), ("exec", ex), ("dashboard", dash))}}
    names, aliases, paper, rv = registry(v9)
    E["registry"] = {"version": rv, "n_leagues": len(names), "n_aliases": len(aliases), "paper": sorted(paper)}

    O = v9 / "output"
    bl = pd.read_csv(O / "bets_ledger.csv", low_memory=False)
    sb = pd.read_csv(O / "side_bets_ledger.csv", low_memory=False)
    ht = pd.read_csv(O / "ht_ledger.csv", low_memory=False)
    pl = pd.read_csv(O / "player_ledger.csv", usecols=["league", "market", "result", "pnl", "market_odds",
                                                        "match_date", "signal_date"], low_memory=False)
    sp = pd.read_csv(O / "stake_plan.csv", low_memory=False)
    p1 = pd.read_csv(pro / "output/paper_1x2.csv", low_memory=False)
    bbs = pd.read_csv(pro / "output/bet_builder_settled.csv", usecols=["league", "combo_result", "fixture_key"],
                      low_memory=False)
    v11s = pd.read_csv(v11 / "output/v11_shadow_log.csv", low_memory=False)
    v11g = pd.read_csv(v11 / "output/v11_graded.csv", low_memory=False)
    v11sc = pd.read_csv(v11 / "output/v11_scoreboard.csv", low_memory=False)
    te = json.loads((pro / "output/bankroll/track_evidence.json").read_text(encoding="utf-8"))
    lr = json.loads((pro / "output/studies/league_ratings.json").read_text(encoding="utf-8"))
    preg = json.loads((pro / "registry/wowza_league_registry.json").read_text(encoding="utf-8"))

    # 1. canonical leagues
    E["leagues"] = {
        "v9/bets_ledger": league_check(bl["league"], names, aliases),
        "v9/side_bets_ledger": league_check(sb["league"], names, aliases),
        "v9/ht_ledger": league_check(ht["league"], names, aliases),
        "v9/player_ledger": league_check(pl["league"], names, aliases),
        "v9/stake_plan": league_check(sp["league"], names, aliases),
        "pro/paper_1x2": league_check(p1["league"], names, aliases),
        "pro/bet_builder_settled": league_check(bbs["league"], names, aliases),
        "pro/league_ratings.json keys": league_check(pd.Series(list((lr.get("ratings") or {}).keys())), names, aliases),
        "pro/wowza_league_registry.json": league_check(pd.Series([x["league_name"] for x in preg["leagues"]]),
                                                        names, aliases),
        "v11/shadow_log": league_check(v11s["league"], names, aliases),
        "v11/graded": league_check(v11g["league"], names, aliases),
    }

    # 2. fixture identity columns
    E["fixture_identity"] = {
        "v9/bets_ledger": [c for c in bl.columns if "fixture" in c],
        "v9/side_bets_ledger": [c for c in sb.columns if "fixture" in c],
        "v9/ht_ledger": [c for c in ht.columns if "fixture" in c],
        "v9/stake_plan": [c for c in sp.columns if "fixture" in c],
        "pro/paper_1x2": [c for c in p1.columns if "fixture" in c],
        "v11/shadow_log": [c for c in v11s.columns if "fixture" in c],
        "v11/graded": [c for c in v11g.columns if "fixture" in c],
        "pro/paper_1x2 fixture_key_status": vc(p1.get("fixture_key_status", pd.Series(dtype=object))),
        "pro/bet_builder_settled fixture_key null": int(bbs["fixture_key"].isna().sum()),
        "pro/bet_builder_settled rows": int(len(bbs)),
        "v11/shadow_log fixture_id null": int(v11s["fixture_id"].isna().sum()),
        "v11/shadow_log rows": int(len(v11s)),
    }
    # v9 ledger duplicates on the name-based key exec uses (date|league|home|away) x market
    for nm, d, mk in (("bets_ledger", bl, None), ("side_bets_ledger", sb, "market")):
        k = (d["match_date"].astype(str).str[:10] + "|" + d["league"].astype(str) + "|" +
             d["home_team"].astype(str) + "|" + d["away_team"].astype(str))
        km = k + "|" + (d[mk].astype(str) if mk else "ou25")
        E["fixture_identity"][f"v9/{nm} rows sharing name-key x market"] = int(km.duplicated(keep=False).sum())
        E["fixture_identity"][f"v9/{nm} same name-key x market with BOTH sides"] = int(
            d.assign(_k=km).groupby("_k")["side"].nunique().gt(1).sum()) if "side" in d else None

    # 3. market keys and sides
    E["markets_sides"] = {
        "v9/bets_ledger side": vc(bl["side"]),
        "v9/side_bets_ledger market": vc(sb["market"]),
        "v9/side_bets_ledger side": vc(sb["side"]),
        "v9/side_bets market x side": vc(sb["market"].astype(str) + "|" + sb["side"].astype(str)),
        "v9/ht_ledger market x side": vc(ht["market"].astype(str) + "|" + ht["side"].astype(str)),
        "v9/stake_plan market x side": vc(sp["market"].astype(str) + "|" + sp["side"].astype(str)),
        "v9/stake_plan track": vc(sp["track"]),
        "pro/track_evidence tracks": sorted((te.get("tracks") or {}).keys()),
        "pro/paper_1x2 notified_pick": vc(p1["notified_pick"]),
        "v11/shadow_log live_side": vc(v11s["live_side"]),
        "v11/shadow_log v11_side": vc(v11s["v11_side"]),
    }

    # 4. decimal odds
    E["odds"] = {
        "v9/bets_ledger odds": odds_check(bl["odds"]),
        "v9/bets_ledger closing_odds": odds_check(bl["closing_odds"]),
        "v9/side_bets_ledger odds": odds_check(sb["odds"]),
        "v9/ht_ledger entry_odds": odds_check(ht["entry_odds"]),
        "v9/player_ledger market_odds": odds_check(pl["market_odds"]),
        "v9/stake_plan odds": odds_check(sp["odds"]),
        "pro/paper_1x2 notified_odds": odds_check(p1["notified_odds"]),
        "v11/shadow_log odds_over25": odds_check(v11s["odds_over25"]),
        "v11/shadow_log odds_under25": odds_check(v11s["odds_under25"]),
    }

    # 5. P&L units: recorded pnl vs flat 1u recompute from (result, odds)
    E["pnl"] = {
        "v9/bets_ledger": pnl_check(bl, "odds"),
        "v9/side_bets_ledger": pnl_check(sb, "odds"),
        "v9/ht_ledger": pnl_check(ht, "entry_odds"),
        "v9/player_ledger": pnl_check(pl, "market_odds"),
        "v9/bets_ledger result vocab": vc(bl["result"]),
        "v9/side_bets_ledger result vocab": vc(sb["result"]),
        "v9/ht_ledger result vocab": vc(ht["result"]),
        "v9/player_ledger result vocab": vc(pl["result"]),
        "pro/paper_1x2 result vocab": vc(p1["result"]),
        "pro/bet_builder combo_result vocab": vc(bbs["combo_result"]),
        "v11/graded over25_result vocab": vc(v11g["over25_result"]),
        "v9/stake_plan stake_amount": {"rows": int(len(sp)),
                                       "gt0": int((pd.to_numeric(sp["stake_amount"], errors="coerce") > 0).sum()),
                                       "bankroll_basis": vc(sp["bankroll_basis"], 5)},
    }

    # 6. model versions
    def mv(d, col="model_version"):
        if col not in d.columns:
            return {"column": "ABSENT"}
        g = pd.to_datetime(d["generated_at"], errors="coerce", format="mixed", utc=True).dt.tz_convert(None)
        out = {"rows": int(len(d)), "null": int(d[col].isna().sum()), "distinct": int(d[col].nunique())}
        since = g >= pd.Timestamp("2026-10-25")
        ga = d["generated_at"].astype(str)
        tzd = ga.str.contains(r"(?:Z|[+-]\d\d:?\d\d)$", regex=True)
        out["generated_at_formats"] = {"with_tz_offset_or_Z": int(tzd.sum()),
                                       "naive": int((~tzd & d["generated_at"].notna()).sum()),
                                       "unparseable": int((g.isna() & d["generated_at"].notna()).sum())}
        out["rows_generated_since_2026-10-25"] = int(since.sum())
        out["null_since_2026-10-25"] = int((since & d[col].isna()).sum())
        return out
    E["model_versions"] = {"v9/bets_ledger": mv(bl), "v9/side_bets_ledger": mv(sb), "v9/ht_ledger": mv(ht),
                           "v9/stake_plan engine_version": vc(sp.get("engine_version", pd.Series(dtype=object)), 5),
                           "v9/stake_plan policy_version": vc(sp["policy_version"], 5),
                           "pro/paper_1x2 calc_version": vc(p1["calc_version"], 5),
                           "v11/shadow_log v9_model_sha null": int(v11s["v9_model_sha"].isna().sum()),
                           "v11/graded calculation_version": vc(v11g["calculation_version"], 5)}

    # 7. signal time vs match time
    def after_match_day(d, gcol, dcol):
        g = pd.to_datetime(d[gcol], errors="coerce", format="mixed", utc=True)
        m = pd.to_datetime(d[dcol].astype(str).str[:10], errors="coerce", utc=True)
        ok = g.notna() & m.notna()
        late = ok & (g.dt.normalize() > m)
        return {"rows": int(len(d)), "comparable": int(ok.sum()),
                "generated_after_match_day": int(late.sum()),
                "generated_after_match_day_settled": int((late & d["result"].astype(str).str.upper()
                                                          .isin(["WIN", "LOSS"])).sum()),
                "examples": d[late].head(5)[[gcol, dcol, "league", "home_team", "away_team"]].astype(str)
                .to_dict("records")}
    E["timestamps"] = {"v9/bets_ledger": after_match_day(bl, "generated_at", "match_date"),
                       "v9/side_bets_ledger": after_match_day(sb, "generated_at", "match_date"),
                       "v9/ht_ledger": after_match_day(ht, "generated_at", "match_date")}
    lg = pd.to_datetime(p1["logged_at"], errors="coerce", utc=True)
    ko = pd.to_datetime(p1["kickoff_utc"], errors="coerce", utc=True)
    na = pd.to_datetime(p1["notified_at"], errors="coerce", utc=True)
    E["timestamps"]["pro/paper_1x2"] = {"rows": int(len(p1)), "logged_at>=kickoff": int((lg >= ko).sum()),
                                        "notified_at>=kickoff": int((na >= ko).sum()),
                                        "notified": int(na.notna().sum())}
    st = pd.to_datetime(v11s["snapshot_ts"], errors="coerce", utc=True)
    k11 = pd.to_datetime(v11s["kickoff_ts"], errors="coerce", utc=True)
    E["timestamps"]["v11/shadow_log"] = {"rows": int(len(v11s)), "kickoff_ts_null": int(k11.isna().sum()),
                                         "snapshot>=kickoff": int((st >= k11).sum()),
                                         "is_post_kickoff": vc(v11s["is_post_kickoff"], 5)}

    # 8. PAPER vs ACTIVE
    bp = json.loads((v9 / "config/bankroll_policy.json").read_text(encoding="utf-8"))
    tstates = {k: v.get("state") for k, v in bp["tracks"].items()}
    spd = sp.copy()
    spd["gen"] = pd.to_datetime(spd["generated_at"], errors="coerce")
    spd["stake"] = pd.to_numeric(spd["stake_amount"], errors="coerce")
    spd["policy_state_now"] = spd["track"].map(tstates)
    E["paper_active"] = {
        "v9 bankroll_policy policy_version": bp.get("policy_version"),
        "v9 track states": tstates,
        "stake_plan deployment_state x policy_state_now": vc(spd["deployment_state"].astype(str) + " | now="
                                                             + spd["policy_state_now"].astype(str), 30),
        "stake_plan rows staked >0 on a track that is PAPER/BLOCKED now": int(
            ((spd["stake"] > 0) & spd["policy_state_now"].isin(["PAPER", "BLOCKED", "BLOCKED_FROM_REAL"])).sum()),
        "stake_plan NF OU25_UNDER rows by policy_version": vc(
            spd.loc[spd["track"] == "new_format|OU25_UNDER", "policy_version"].astype(str) + " | "
            + spd.loc[spd["track"] == "new_format|OU25_UNDER", "deployment_state"].astype(str) + " | stake>0="
            + (spd.loc[spd["track"] == "new_format|OU25_UNDER", "stake"] > 0).astype(str)),
        "stake_plan latest generated_at": str(spd["gen"].max()),
        "exec execution_policy": json.loads((ex / "registry/execution_policy.json").read_text(encoding="utf-8"))
                                     .get("deployment_mode"),
        "pro league_ratings generated_at": lr.get("generated_at"),
    }

    # 9. dashboard 'staked' basis vs v9 stake plan (does the dashboard call a PAPER-track tip staked?)
    sys.path.insert(0, str(v9))
    import config as v9config                                          # noqa: E402
    from dashboard_data import today_view as TV                        # noqa: E402
    from dashboard_data import v9 as DV9                               # noqa: E402
    from src.bankroll_policy import track_key                          # noqa: E402
    ss = TV.staked_settled(O)
    E["dashboard_staked"] = {"rows": int(len(ss)), "pnl": round(float(ss["_pnl"].sum()), 2)}
    # independent recompute of the same definition from the raw ledgers
    cut = pd.Timestamp(v9config.PERFORMANCE_CUTOFF_DATE)
    parts = []
    for nm, d, mk in (("bets", bl, None), ("side", sb, "market"), ("ht", ht, None)):
        x = d.copy()
        x["_mkt"] = x[mk] if mk else ("OU25" if nm == "bets" else x["market"])
        x["_odds"] = pd.to_numeric(x["odds"] if "odds" in x else x["entry_odds"], errors="coerce")
        x["_pnl_rec"] = pd.to_numeric(x["pnl"], errors="coerce")
        x["_pnl_flat"] = flat_pnl(x["result"], x["_odds"])
        x["_dt"] = pd.to_datetime(x["match_date"].astype(str).str[:10], errors="coerce")
        x["_src"] = nm
        keep = (x.get("source", pd.Series("live", index=x.index)).fillna("live").astype(str) == "live") \
            & x["signal_tier"].astype(str).str.upper().isin(["SNIPER", "MARKSMAN"]) if "signal_tier" in x else \
            pd.Series(False, index=x.index)
        keep &= x["_dt"] >= cut
        keep &= ~x["league"].isin(paper)
        keep &= x["_pnl_rec"].notna()
        parts.append(x[keep])
    ind = pd.concat(parts, ignore_index=True)
    ind["track"] = [track_key(mt, str(m).upper(), s) if isinstance(mt, str) else None
                    for mt, m, s in zip(ind.get("model_type"), ind["_mkt"], ind["side"])]
    ind["track_state_now"] = ind["track"].map(tstates)
    E["dashboard_staked"]["independent_rows"] = int(len(ind))
    E["dashboard_staked"]["independent_pnl_recorded"] = round(float(ind["_pnl_rec"].sum()), 2)
    E["dashboard_staked"]["independent_pnl_flat_from_result_odds"] = round(float(ind["_pnl_flat"].sum()), 2)
    E["dashboard_staked"]["rows_where_recorded_pnl_ne_flat"] = int(((ind["_pnl_rec"] - ind["_pnl_flat"]).abs()
                                                                   > 0.011).sum())
    E["dashboard_staked"]["by_source"] = {k: {"n": int(len(g)), "pnl": round(float(g["_pnl_rec"].sum()), 2)}
                                          for k, g in ind.groupby("_src")}
    E["dashboard_staked"]["by_track_state_now"] = {
        str(k): {"n": int(len(g)), "pnl": round(float(g["_pnl_rec"].sum()), 2)}
        for k, g in ind.groupby(ind["track_state_now"].fillna("NO_TRACK"))}
    E["dashboard_staked"]["by_track"] = {
        str(k): {"n": int(len(g)), "pnl": round(float(g["_pnl_rec"].sum()), 2)}
        for k, g in ind.groupby(ind["track"].fillna("NO_TRACK"))}
    # rows the dashboard calls staked vs v9's own stake plan (stake_amount>0) since the plan exists
    spk = spd.assign(_k=spd["fixture"].astype(str) + "|" + spd["market"].astype(str).str.upper() + "|"
                     + spd["side"].astype(str).str.upper())
    plan_start = pd.to_datetime(spd["date"], errors="coerce").min()
    ind2 = ind[ind["_dt"] >= plan_start].copy()
    mk_up = ind2["_mkt"].astype(str).str.upper().replace({"OU25": "OU25"})
    ind2["_k"] = (ind2["_dt"].dt.strftime("%Y-%m-%d") + "|" + ind2["home_team"].astype(str) + "|"
                  + ind2["away_team"].astype(str) + "|" + mk_up + "|" + ind2["side"].astype(str).str.upper())
    planned = spk.groupby("_k")["stake"].max()
    ind2["plan_stake"] = ind2["_k"].map(planned)
    E["dashboard_staked"]["since_stake_plan_start"] = {
        "plan_start": str(plan_start.date()) if pd.notna(plan_start) else None,
        "dashboard_staked_rows": int(len(ind2)),
        "matched_in_stake_plan": int(ind2["plan_stake"].notna().sum()),
        "plan_stake_gt0": int((ind2["plan_stake"] > 0).sum()),
        "plan_stake_eq0": int((ind2["plan_stake"] == 0).sum()),
        "not_in_plan": int(ind2["plan_stake"].isna().sum()),
        "pnl_dashboard_staked": round(float(ind2["_pnl_rec"].sum()), 2),
        "pnl_plan_stake_gt0_only": round(float(ind2.loc[ind2["plan_stake"] > 0, "_pnl_rec"].sum()), 2),
        "pnl_plan_stake_eq0": round(float(ind2.loc[ind2["plan_stake"] == 0, "_pnl_rec"].sum()), 2),
    }
    perf = DV9.performance()
    E["dashboard_v9_performance"] = {t: {"n": v["n"], "pnl": v["pnl"], "staked_only": v["staked_only"]}
                                     for t, v in (perf.get("by_track") or {}).items()}

    # 10. Pro tip scoreboard (displayed on Pro page) recomputed from paper_1x2.csv
    sbj = json.loads((pro / "output/tip_scoreboard.json").read_text(encoding="utf-8"))
    s = p1[p1["notified_at"].notna()].copy()
    s["res"] = pd.to_numeric(s["result"], errors="coerce")
    s["pick_i"] = s["notified_pick"].map({"HOME": 0, "DRAW": 1, "AWAY": 2})
    s["o"] = pd.to_numeric(s["notified_odds"], errors="coerce")
    s = s[s["res"].notna() & s["pick_i"].notna() & (s["o"] > 1)]
    s = s[~s["league"].isin(paper)]
    won = s["res"] == s["pick_i"]
    u = np.where(won, s["o"] - 1, -1.0)
    E["pro_tip_scoreboard_1x2"] = {"displayed": {k: sbj["one_x_two"]["headline"].get(k) for k in
                                                 ("settled", "won", "units", "roi")},
                                   "displayed_generated_at": sbj.get("generated_at"),
                                   "recomputed_now": {"settled": int(len(s)), "won": int(won.sum()),
                                                      "units": round(float(u.sum()), 2),
                                                      "roi": round(float(u.mean()), 4) if len(s) else None}}
    # same, restricted to matches on or before the scoreboard's generation day
    cutoff = pd.Timestamp(sbj["generated_at"]).tz_convert(None).normalize()
    s2 = s[pd.to_datetime(s["match_date"]) <= cutoff]
    w2 = s2["res"] == s2["pick_i"]
    u2 = np.where(w2, s2["o"] - 1, -1.0)
    E["pro_tip_scoreboard_1x2"]["recomputed_as_of_generated_at"] = {
        "settled": int(len(s2)), "won": int(w2.sum()), "units": round(float(u2.sum()), 2)}

    # 11. Pro bankroll track evidence (Bankroll page) recomputed from v9 ledgers at its cutoff
    ev_cut = pd.Timestamp(te["evidence_cutoff"])
    chk = {}
    for tk, tv in (te.get("tracks") or {}).items():
        disp = (tv.get("horizons") or {}).get("FULL_PROSPECTIVE_RECORD") or {}
        sel = ind[(ind["track"] == tk) & (ind["_dt"] <= ev_cut)]
        chk[tk] = {"displayed_n": disp.get("n"), "displayed_units": disp.get("units"),
                   "recomputed_n": int(len(sel)), "recomputed_units_recorded_pnl": round(float(sel["_pnl_rec"].sum()), 2),
                   "recomputed_units_flat": round(float(sel["_pnl_flat"].sum()), 2)}
    E["pro_track_evidence_vs_v9"] = {"evidence_cutoff": te["evidence_cutoff"], "generated_at": te["generated_at"],
                                     "tracks": chk}

    # 12. v11 scoreboard 'v9' line vs v9's own staked ledger
    E["v11_scoreboard"] = v11sc.astype(str).to_dict("records")
    g = v11g.copy()
    E["v11_graded_v9_live"] = {"rows": int(len(g)), "live_tier": vc(g["live_tier"], 6),
                               "live_pnl_sum": round(float(pd.to_numeric(g["live_pnl"], errors="coerce").sum()), 2),
                               "live_pnl_n": int(pd.to_numeric(g["live_pnl"], errors="coerce").notna().sum())}
    ou = ind[ind["_src"] == "bets"]
    E["v11_graded_v9_live"]["v9_bets_ledger_staked_ou25_since_cutoff"] = {
        "n": int(len(ou)), "pnl": round(float(ou["_pnl_rec"].sum()), 2)}
    allou = bl[(bl["result"].isin(["WIN", "LOSS"])) & (pd.to_datetime(bl["match_date"].astype(str).str[:10],
                                                                       errors="coerce") >= cut)
               & ~bl["league"].isin(paper)]
    E["v11_graded_v9_live"]["v9_bets_ledger_all_tiers_since_cutoff"] = {
        "n": int(len(allou)), "pnl": round(float(pd.to_numeric(allou["pnl"], errors="coerce").sum()), 2)}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(E, indent=1, default=str), encoding="utf-8")
    print(json.dumps(E, indent=1, default=str)[:60000])


if __name__ == "__main__":
    main()
