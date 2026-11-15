"""
ht_history_repair_audit.py - Phase 2 Part 3 (P0 half-time history repair) audit. READ-ONLY.

    python scripts/audit/ht_history_repair_audit.py

Reads only committed files (no API call, no write outside reports/ht_repair/):
  output/fd_history.parquet                    the history (football-data.co.uk), never modified
  output/collect/fd_history_ht_fill.parquet    the HT sidecar written by scripts/collect_ht_scores.py
  output/collect/af_ht_scores.parquet          the provider fixtures the sidecar was matched against
  output/collect/ht_seasons.parquet            fetch journal (last collection per league-season)
  output/collect/ht_fill_audit.json            the collector's own counts (compared, not trusted)
  output/af_history.parquet                    API-Football FT score by fixture_id (FT cross-check)

Writes reports/ht_repair/:
  coverage_league_season_month.csv   one row per new-format league x history season x month
  row_status.csv.gz                  one row per new-format history row: HT source / missing reason
  ft_mismatch.csv                    fill rows whose provider FT differs from the history FT
  checks.json                        every verification with its count and pass/fail

Per-row missing reasons are re-derived by an independent re-implementation of the collector's
matching rule (league + date or adjacent date + clubs resolved by src.team_names.resolve_result),
and the re-derived fill set is compared key-for-key with the committed sidecar.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import league_registry as LR  # noqa: E402
from src.season_resolver import is_known, resolve_api_season  # noqa: E402
from src.team_names import resolve_result  # noqa: E402

OUT = ROOT / "reports" / "ht_repair"
KEY = ["league", "date", "home_team", "away_team"]


def nf_leagues() -> dict:
    return {e["name"]: int(e["api_football_id"]) for e in LR.leagues()
            if e.get("model_type") == "new_format" and e.get("api_football_id") is not None}


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    nf = nf_leagues()
    h = pd.read_parquet(ROOT / "output/fd_history.parquet",
                        columns=["league", "date", "season", "home_team", "away_team", "home_goals",
                                 "away_goals", "ht_home_goals", "ht_away_goals"])
    h["canon"] = h["league"].map(lambda x: LR.canonical_name(x) if isinstance(x, str) else None)
    h = h[h["canon"].isin(nf)].copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    fill = pd.read_parquet(ROOT / "output/collect/fd_history_ht_fill.parquet")
    fill["date"] = pd.to_datetime(fill["date"])
    sc_all = pd.read_parquet(ROOT / "output/collect/af_ht_scores.parquet")
    jr = pd.read_parquet(ROOT / "output/collect/ht_seasons.parquet")
    audit = json.loads((ROOT / "output/collect/ht_fill_audit.json").read_text(encoding="utf-8"))
    afh = pd.read_parquet(ROOT / "output/af_history.parquet",
                          columns=["league", "date", "fixture_id", "FTHG", "FTAG"])
    checks: dict = {}

    def chk(name, value, ok, detail=""):
        checks[name] = {"value": value, "pass": bool(ok), "detail": detail}

    # ---- per-row base flags -------------------------------------------------------------------
    h["has_ft"] = h["home_goals"].notna() & h["away_goals"].notna()
    h["ht_hist"] = h["ht_home_goals"].notna() & h["ht_away_goals"].notna()
    h["ht_half_present"] = h["ht_home_goals"].notna() ^ h["ht_away_goals"].notna()
    h["api_season"] = [resolve_api_season(c, d) if pd.notna(d) else "UNKNOWN"
                       for c, d in zip(h["canon"], h["date"])]
    fk = fill[KEY + ["HTHG", "HTAG", "fixture_id", "match_rule", "af_date"]].drop_duplicates(KEY)
    chk("fill_key_unique", int(fill.duplicated(KEY).sum()), not fill.duplicated(KEY).any(),
        "two sidecar rows for one history key")
    h = h.merge(fk, on=KEY, how="left", validate="many_to_one")
    h["ht_fill"] = h["HTHG"].notna() & h["HTAG"].notna() & ~h["ht_hist"]
    h["fill_on_row_with_hist_ht"] = h["HTHG"].notna() & h["ht_hist"]

    # ---- independent re-derivation of the matcher (reasons per row) ---------------------------
    sc_ok = sc_all.dropna(subset=["ht_home", "ht_away"]).copy()
    sc_all["_d"] = pd.to_datetime(sc_all["date"], errors="coerce").dt.normalize()
    sc_ok["_d"] = pd.to_datetime(sc_ok["date"], errors="coerce").dt.normalize()
    day_ok = {k: g for k, g in sc_ok.groupby(["league", "_d"])}
    day_all = {k: g for k, g in sc_all.groupby(["league", "_d"])}
    fetched = {(str(r.league), str(r.season)) for r in jr.itertuples(index=False)}
    last_fetch = jr.assign(t=pd.to_datetime(jr["fetched_at"], utc=True)) \
        .groupby(["league", "season"])["t"].max()

    lack = h[~h["ht_hist"] & h["date"].notna()].copy()
    rep_mask = lack.duplicated(subset=["canon", "date", "home_team", "away_team"])
    reason: dict = {}
    unresolved: Counter = Counter()
    picks = []
    for idx, r in lack.iterrows():
        if rep_mask.loc[idx]:
            continue
        if not is_known(r["api_season"]):
            reason[idx] = "api_season_unknown"
            continue
        d = pd.Timestamp(r["date"]).normalize()
        found, rule = None, None
        for days, nm in (((0,), "same_day"), ((-1, 1), "adjacent_day")):
            parts = [day_ok[(r["canon"], d + pd.Timedelta(days=k))] for k in days
                     if (r["canon"], d + pd.Timedelta(days=k)) in day_ok]
            if not parts:
                continue
            pool = pd.concat(parts)
            hh = resolve_result(str(r["home_team"]), [str(c) for c in pool["home_name"].unique()])
            aa = resolve_result(str(r["away_team"]), [str(c) for c in pool["away_name"].unique()])
            if hh is None or aa is None:
                continue
            f = pool[(pool["home_name"] == hh) & (pool["away_name"] == aa)]
            if len(f):
                found, rule = f, nm
                break
        if found is None:
            # why: provider has nothing that day / club not resolvable / provider HT null
            if (str(r["canon"]), str(r["api_season"])) not in fetched:
                reason[idx] = "league_season_not_fetched"
                continue
            parts = [day_all[(r["canon"], d + pd.Timedelta(days=k))] for k in (0, -1, 1)
                     if (r["canon"], d + pd.Timedelta(days=k)) in day_all]
            if not parts:
                reason[idx] = "no_provider_fixture_on_date_pm1"
                continue
            pool = pd.concat(parts)
            hh = resolve_result(str(r["home_team"]), [str(c) for c in pool["home_name"].unique()])
            aa = resolve_result(str(r["away_team"]), [str(c) for c in pool["away_name"].unique()])
            if hh is not None and aa is not None and \
                    len(pool[(pool["home_name"] == hh) & (pool["away_name"] == aa)]):
                reason[idx] = "provider_ht_null"
            elif hh is None or aa is None:
                reason[idx] = "club_not_resolved"
                for nm_, ok_ in ((r["home_team"], hh), (r["away_team"], aa)):
                    if ok_ is None:
                        unresolved[(r["canon"], str(nm_))] += 1
            else:
                reason[idx] = "clubs_resolved_no_fixture_pairing"
            continue
        if found["fixture_id"].nunique() > 1:
            reason[idx] = "refused_ambiguous"
            continue
        f0 = found.iloc[0]
        picks.append({"_idx": idx, "league": r["league"], "date": r["date"], "home_team": r["home_team"],
                      "away_team": r["away_team"], "fixture_id": int(f0["fixture_id"]),
                      "HTHG_re": int(f0["ht_home"]), "HTAG_re": int(f0["ht_away"]), "rule_re": rule,
                      "canon": r["canon"]})
    pk = pd.DataFrame(picks)
    nd = pk.groupby("fixture_id")["date"].transform("nunique")
    twice = pk["fixture_id"].duplicated(keep=False) & (nd > 1)
    for i in pk.loc[twice, "_idx"]:
        reason[i] = "refused_fixture_claimed_by_two_dates"
    pk = pk[~twice]
    # re-derived set vs committed sidecar
    re_keys = set(map(tuple, pk[KEY].astype(str).values))
    fl_keys = set(map(tuple, fill[KEY].astype(str).values))
    chk("rederived_fill_equals_sidecar_keys",
        {"rederived": len(re_keys), "sidecar": len(fl_keys), "only_rederived": len(re_keys - fl_keys),
         "only_sidecar": len(fl_keys - re_keys)}, re_keys == fl_keys,
        "independent re-implementation of the matching rule reproduces the committed sidecar")
    cmp = pk.merge(fill, on=KEY, how="inner")
    diff = cmp[(cmp["fixture_id_x"] != cmp["fixture_id_y"]) | (cmp["HTHG_re"] != cmp["HTHG"])
               | (cmp["HTAG_re"] != cmp["HTAG"]) | (cmp["rule_re"] != cmp["match_rule"])]
    chk("rederived_fill_equals_sidecar_values", int(len(diff)), len(diff) == 0,
        "same fixture_id, HT score and match rule per key")

    # repeat rows (identical key repeated in the history) take the representative's outcome
    h["missing_reason"] = None
    h.loc[list(reason), "missing_reason"] = pd.Series(reason)
    first_of = lack.loc[~rep_mask].reset_index().set_index(["canon", "date", "home_team", "away_team"])["index"]
    for idx in lack.index[rep_mask]:
        r = lack.loc[idx]
        rep_idx = first_of.get((r["canon"], r["date"], r["home_team"], r["away_team"]))
        h.loc[idx, "missing_reason"] = h.loc[rep_idx, "missing_reason"] if rep_idx is not None else None
    h.loc[h["ht_hist"] | h["ht_fill"], "missing_reason"] = None
    h.loc[~h["ht_hist"] & ~h["ht_fill"] & h["date"].isna(), "missing_reason"] = "no_date"
    # a lacking row that the re-derivation filled but the sidecar did not would show here
    left = h[~h["ht_hist"] & ~h["ht_fill"] & h["missing_reason"].isna()]
    chk("every_missing_row_has_a_reason", int(len(left)), len(left) == 0)

    # ---- fill integrity -----------------------------------------------------------------------
    dup = fill[fill["fixture_id"].duplicated(keep=False)]
    dup_dates = dup.groupby("fixture_id")["date"].nunique()
    chk("fixture_id_claimed_by_two_dates", int((dup_dates > 1).sum()), (dup_dates > 1).sum() == 0,
        "a fixture id written to history rows of different dates = two matches for one fixture")
    chk("fixture_id_shared_same_date_rows", {"fixture_ids": int(dup["fixture_id"].nunique()),
                                             "rows": int(len(dup))}, True,
        "history repeats one match under two club spellings; each copy carries the same score "
        "(informational; the loader's one_club_one_match merges them)")
    s_by_id = sc_all.drop_duplicates("fixture_id").set_index("fixture_id")
    chk("provider_fixture_id_unique", int(sc_all["fixture_id"].duplicated().sum()),
        not sc_all["fixture_id"].duplicated().any())
    j = fill.join(s_by_id[["league", "league_id", "ht_home", "ht_away", "date", "status",
                           "home_name", "away_name"]], on="fixture_id", rsuffix="_p")
    chk("fill_fixture_in_provider_store", int(j["league_p"].isna().sum()), j["league_p"].notna().all())
    j["canon"] = j["league"].map(LR.canonical_name)
    xl = j[(j["canon"] != j["league_p"]) | (j["canon"].map(nf) != j["league_id"])]
    chk("cross_league_joins", int(len(xl)), len(xl) == 0,
        "fill league (canonical) == provider league name and registry api_football_id == provider league_id")
    hv = j[(j["HTHG"] != j["ht_home"]) | (j["HTAG"] != j["ht_away"])]
    chk("fill_ht_equals_provider_halftime", int(len(hv)), len(hv) == 0,
        "every HT value is the provider's score.halftime for that fixture id (none derived from FT)")
    chk("fill_ht_nulls", int(fill[["HTHG", "HTAG"]].isna().sum().sum()), fill[["HTHG", "HTAG"]].notna().all().all())
    nullp = sc_all[sc_all["ht_home"].isna() | sc_all["ht_away"].isna()]
    leaked = fill[fill["fixture_id"].isin(nullp["fixture_id"])]
    chk("provider_null_ht_never_written", {"provider_null_ht_fixtures": int(len(nullp)),
                                           "in_fill": int(len(leaked))}, len(leaked) == 0,
        "a provider null half-time stays missing (never 0)")
    neg = fill[(fill["HTHG"] < 0) | (fill["HTAG"] < 0)]
    chk("fill_ht_negative", int(len(neg)), len(neg) == 0)
    chk("fill_rows_on_history_rows_that_already_had_ht", int(h["fill_on_row_with_hist_ht"].sum()), True,
        "informational: the loader joins nan_only, so a stored history HT is never overwritten")
    gap = (pd.to_datetime(j["date_p"]) - j["date"]).dt.days.abs()
    chk("fill_date_gap_max_days", int(gap.max()), gap.max() <= 1, "same day or adjacent day only")
    chk("fill_match_rule_counts", fill["match_rule"].value_counts().to_dict(), True)

    # HT <= FT against the history's own FT
    hf = h[h["ht_fill"]]
    no_ft = hf[~hf["has_ft"]]
    bad = hf[hf["has_ft"] & ((hf["HTHG"] > hf["home_goals"]) | (hf["HTAG"] > hf["away_goals"]))]
    chk("fill_ht_le_history_ft", {"violations": int(len(bad)), "filled_rows_without_history_ft": int(len(no_ft))},
        len(bad) == 0)
    # power of that check: the same HT scores dealt to RANDOM matches of the same league
    # (seeded shuffle) - how often would a wrong match break HT <= FT?
    rng = np.random.default_rng(20261103)
    viol = []
    for _ in range(20):
        sh = hf.groupby("canon", group_keys=False)[["HTHG", "HTAG"]].transform(
            lambda x: x.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).to_numpy())
        viol.append(float(((sh["HTHG"] > hf["home_goals"]) | (sh["HTAG"] > hf["away_goals"])).mean()))
    chk("placebo_wrong_match_ht_gt_ft_rate", {"mean_share": round(float(np.mean(viol)), 4),
                                              "min_share": round(float(np.min(viol)), 4), "draws": 20},
        float(np.min(viol)) > 0.05,
        "share of rows that would break HT<=FT if each HT were attached to a random match of the "
        "same league; a near-zero real rate against this is evidence the matches are right")
    hh = h[h["ht_hist"] & h["has_ft"]]
    badh = hh[(hh["ht_home_goals"] > hh["home_goals"]) | (hh["ht_away_goals"] > hh["away_goals"])]
    chk("history_ht_le_history_ft", int(len(badh)), len(badh) == 0, "the history's own HT")
    chk("history_rows_with_half_an_ht_score", int(h["ht_half_present"].sum()), True,
        "informational: one HT side present, the other missing")

    # FT cross-check: provider FT (af_history by fixture_id) vs our history FT
    a = afh.dropna(subset=["fixture_id"]).copy()
    a["fixture_id"] = a["fixture_id"].astype("int64")
    a_dup = a["fixture_id"].duplicated(keep=False)
    a_conf = a[a_dup].groupby("fixture_id")[["FTHG", "FTAG"]].nunique().gt(1).any(axis=1)
    a1 = a.drop_duplicates("fixture_id").set_index("fixture_id")
    m = hf.join(a1[["FTHG", "FTAG", "league"]].rename(columns={"league": "af_league"}), on="fixture_id")
    m["af_ft_conflict"] = m["fixture_id"].isin(a_conf[a_conf].index)
    m = m.join(s_by_id[["status", "home_name", "away_name", "kickoff"]], on="fixture_id")
    have = m["FTHG"].notna() & m["FTAG"].notna() & m["has_ft"]
    mm = m[have & ((m["FTHG"] != m["home_goals"]) | (m["FTAG"] != m["away_goals"]))].copy()
    mm["after_extra_time_status"] = mm["status"].isin(["AET", "PEN"])
    mm.sort_values(["canon", "date"])[["canon", "league", "date", "home_team", "away_team", "home_goals",
                                       "away_goals", "fixture_id", "home_name", "away_name", "kickoff",
                                       "status", "FTHG", "FTAG", "HTHG", "HTAG", "match_rule",
                                       "after_extra_time_status", "af_ft_conflict"]] \
        .to_csv(OUT / "ft_mismatch.csv", index=False)
    ht_gt_af = m[have & ((m["HTHG"] > m["FTHG"]) | (m["HTAG"] > m["FTAG"]))]
    chk("ft_crosscheck_coverage", {"filled_rows": int(len(hf)), "provider_ft_available": int(have.sum()),
                                   "unidentifiable_no_provider_ft": int((~have).sum())}, True,
        "provider FT taken from output/af_history.parquet by fixture_id; af_ht_scores stores no FT")
    chk("ft_mismatch_fill_vs_history", {"rows": int(len(mm)),
                                         "of_which_AET_PEN": int(mm["after_extra_time_status"].sum()),
                                         "regular_time_status": int((~mm["after_extra_time_status"]).sum())},
        int((~mm["after_extra_time_status"]).sum()) == 0,
        "possible wrong match: listed in ft_mismatch.csv")
    chk("fill_ht_le_provider_ft", int(len(ht_gt_af)), len(ht_gt_af) == 0)

    # collector audit json vs recount
    recount = Counter(pk["canon"])
    diffs = {lg: (v.get("filled"), recount.get(lg, 0)) for lg, v in audit["per_league"].items()
             if v.get("filled") != recount.get(lg, 0)}
    chk("audit_json_filled_matches_recount", diffs or 0, not diffs)
    chk("audit_json_total_rows", {"audit": audit.get("filled_rows_total"), "sidecar_rows": int(len(fill))},
        audit.get("filled_rows_total") == len(fill))
    chk("sidecar_written_in_one_run", fill["matched_at"].nunique(), True,
        f"matched_at values: {sorted(fill['matched_at'].unique())[:3]}")

    # ---- coverage table -----------------------------------------------------------------------
    h["month"] = h["date"].dt.strftime("%Y-%m")
    h["completed"] = h["date"].notna() & (h["date"] <= pd.Timestamp.now().normalize())
    lf = {(lg, s): t for (lg, s), t in last_fetch.items()}
    h["last_collection"] = [lf.get((c, s)) for c, s in zip(h["canon"], h["api_season"])]
    h["source_provider"] = np.where(h["ht_hist"], "football-data.co.uk",
                                    np.where(h["ht_fill"], "api-football", ""))
    rows = []
    for (lg, se, mo), g in h.groupby(["canon", "season", "month"], dropna=False):
        valid = g["ht_hist"] | g["ht_fill"]
        reasons = g.loc[~valid, "missing_reason"].value_counts().to_dict()
        lc = g["last_collection"].dropna()
        ftm = mm[(mm["canon"] == lg) & (mm["date"].dt.strftime("%Y-%m") == mo)]
        rows.append({"league": lg, "season": se, "month": mo, "completed_fixtures": int(g["completed"].sum()),
                     "with_ft": int(g["has_ft"].sum()), "ht_from_history": int(g["ht_hist"].sum()),
                     "ht_from_fill": int(g["ht_fill"].sum()), "ht_valid_total": int(valid.sum()),
                     "coverage_pct_of_ft": round(100 * valid[g["has_ft"]].mean(), 1) if g["has_ft"].any() else None,
                     "coverage_pct_before_fill": round(100 * g.loc[g["has_ft"], "ht_hist"].mean(), 1) if g["has_ft"].any() else None,
                     "source_provider": "+".join(sorted(set(g["source_provider"]) - {""})) or "none",
                     "fill_rows_with_fixture_id": int(g["fixture_id"].notna().sum() - g["fill_on_row_with_hist_ht"].sum()),
                     "missing": int((~valid).sum()),
                     "missing_reasons": "; ".join(f"{k}={v}" for k, v in sorted(reasons.items())),
                     "last_collection_utc": lc.max().strftime("%Y-%m-%d %H:%M") if len(lc) else "",
                     "ft_mismatch_rows": int(len(ftm))})
    cov = pd.DataFrame(rows)
    cov.to_csv(OUT / "coverage_league_season_month.csv", index=False)

    rs = h[["canon", "league", "season", "api_season", "date", "home_team", "away_team", "home_goals",
            "away_goals", "ht_home_goals", "ht_away_goals", "HTHG", "HTAG", "fixture_id", "match_rule",
            "source_provider", "missing_reason"]].rename(columns={"HTHG": "fill_HTHG", "HTAG": "fill_HTAG"})
    rs.to_csv(OUT / "row_status.csv.gz", index=False, compression="gzip")

    def summ(g):
        valid = g["ht_hist"] | g["ht_fill"]
        ft = g["has_ft"]
        return pd.Series({"rows": len(g), "completed": int(g["completed"].sum()), "with_ft": int(ft.sum()),
                          "ht_from_history": int(g["ht_hist"].sum()), "ht_from_fill": int(g["ht_fill"].sum()),
                          "before_pct": round(100 * g.loc[ft, "ht_hist"].mean(), 1),
                          "after_pct": round(100 * valid[ft].mean(), 1),
                          "missing": int((~valid).sum())})
    by_lg = h.groupby("canon").apply(summ)
    by_ls = h.groupby(["canon", "season"]).apply(summ)
    reasons_lg = h[~(h["ht_hist"] | h["ht_fill"])].groupby(["canon", "missing_reason"]).size().unstack(fill_value=0)
    by_lg.to_csv(OUT / "coverage_by_league.csv")
    by_ls.to_csv(OUT / "coverage_by_league_season.csv")
    reasons_lg.to_csv(OUT / "missing_reasons_by_league.csv")
    pd.DataFrame([{"league": k[0], "club_spelling": k[1], "rows": v} for k, v in unresolved.most_common()]) \
        .to_csv(OUT / "unresolved_club_spellings.csv", index=False)
    tw = h[h["missing_reason"] == "refused_fixture_claimed_by_two_dates"]
    tw[["canon", "date", "home_team", "away_team", "home_goals", "away_goals"]].sort_values(
        ["canon", "home_team", "away_team", "date"]).to_csv(OUT / "refused_claimed_by_two_dates.csv", index=False)
    tot = summ(h)
    res = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "totals": {k: (int(v) if isinstance(v, (int, np.integer)) else float(v)) for k, v in tot.items()},
           "missing_reasons_total": h.loc[~(h["ht_hist"] | h["ht_fill"]), "missing_reason"].value_counts().to_dict(),
           "ft_mismatch_rows": int(len(mm)), "checks": checks,
           "loader_level": loader_level()}
    (OUT / "checks.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))
    print(by_lg.to_string())
    print(reasons_lg.to_string())
    return res


def _labels(d: pd.DataFrame) -> pd.DataFrame:
    """The loader's derived label columns (data_loader CI path) so build_features can run."""
    from src import data_loader as DL
    d = d.copy()
    d["total_goals"] = d["home_goals"] + d["away_goals"]
    ftn = d["total_goals"].notna()
    for k, th in (("over25", 2.5), ("over15", 1.5), ("over35", 3.5)):
        d[k] = (d["total_goals"] > th).astype(float).where(ftn)
    d["btts"] = ((d["home_goals"] > 0) & (d["away_goals"] > 0)).astype(float).where(ftn)
    return DL._ht_labels(d)


def loader_level() -> dict:
    """The HT step as load_all_matches runs it (CI path order: one_club_one_match, then
    _enrich_ht_scores), on the new-format rows of fd_history: with and without the identity-alias
    placement, then has_ht_history in training (build_features) and serving (build_upcoming_features).
    Scope: fd_history rows only - the full loader also adds API-Football-only leagues and the
    backtest archive, which are not new-format HT rows."""
    from src import data_loader as DL
    from src.feature_engineering import build_features, build_upcoming_features
    nf = nf_leagues()
    h = pd.read_parquet(ROOT / "output/fd_history.parquet")
    h = h[h["league"].isin(nf)].copy()
    h["date"] = pd.to_datetime(h["date"])
    h["row_source"] = "fd_cache"
    out = DL.one_club_one_match(h)
    saved = {k: dict(v) for k, v in DL.IDENTITY_ALIASES.items()}
    DL.IDENTITY_ALIASES.clear()
    e0 = DL._enrich_ht_scores(out.copy(), root=ROOT)
    DL.IDENTITY_ALIASES.update(saved)
    e1 = DL._enrich_ht_scores(out.copy(), root=ROOT)
    res = {"history_rows": int(len(h)), "after_one_club_one_match": int(len(out)),
           "identity_aliases_applied": int(sum(len(v) for v in saved.values())),
           "filled_without_alias_placement": e0.attrs["ht_coverage"]["filled_rows"],
           "filled_with_alias_placement": e1.attrs["ht_coverage"]["filled_rows"]}
    has = lambda d: d["ht_home_goals"].notna() & d["ht_away_goals"].notna()  # noqa: E731
    per = {}
    for lg, g in out.groupby("league"):
        i = g.index
        per[lg] = {"rows": int(len(g)), "before_fill_pct": round(100 * has(out.loc[i]).mean(), 1),
                   "after_fill_old_pct": round(100 * has(e0.loc[i]).mean(), 1),
                   "after_fill_fixed_pct": round(100 * has(e1.loc[i]).mean(), 1),
                   "missing_after_fixed": int((~has(e1.loc[i])).sum())}
    res["per_league"] = per
    res["totals"] = {"before_fill": int(has(out).sum()), "after_fill_old": int(has(e0).sum()),
                     "after_fill_fixed": int(has(e1).sum()), "rows": int(len(out))}
    res["ht_gt_ft_rows"] = int(((e1["ht_home_goals"] > e1["home_goals"]) |
                                (e1["ht_away_goals"] > e1["away_goals"])).sum())
    res["half_filled_rows"] = int((e1["ht_home_goals"].notna() ^ e1["ht_away_goals"].notna()).sum())
    lab = _labels(e1)
    t = lab[lab["ht_total_goals"].notna()]
    res["ht_over05_label_mismatch"] = int(((t["ht_total_goals"] >= 1).astype(float) != t["ht_over05"]).sum())
    res["ht_label_where_ht_missing"] = int(lab.loc[lab["ht_total_goals"].isna(), "ht_over05"].notna().sum())
    f1 = build_features(lab)
    f0 = build_features(_labels(out))
    fe = {}
    for nm, f in (("without_fill", f0), ("with_fill_fixed", f1)):
        fe[nm] = {c: round(float(f[c].mean()), 4) for c in ("home_has_ht_history", "away_has_ht_history")}
        fe[nm]["flag_vs_ht_scored_last5_disagree"] = int(sum(
            (f[f"{s}_ht_scored_last5"].notna().astype(float) != f[f"{s}_has_ht_history"]).sum()
            for s in ("home", "away")))
    res["train_has_ht_history_share"] = fe
    # serving parity on real data: each league's latest match served from the history before it
    last = lab.sort_values("date").groupby("league").tail(1)
    hist = lab.drop(index=last.index)
    up = last[["date", "league", "home_team", "away_team", "odds_over25", "odds_under25"]].reset_index(drop=True)
    sv = build_upcoming_features(up, hist)
    k = ["date", "league", "home_team", "away_team"]
    cols = ["home_has_ht_history", "away_has_ht_history", "home_ht_scored_last5", "away_ht_scored_last5"]
    cmp = sv[k + cols].merge(f1[k + cols], on=k, suffixes=("_serve", "_train"))
    dis = 0
    for c in cols:
        a, b = cmp[f"{c}_serve"], cmp[f"{c}_train"]
        dis += int((~(np.isclose(a.astype(float), b.astype(float), equal_nan=True))).sum())
    res["serve_vs_train_parity"] = {"fixtures_compared": int(len(cmp)), "cells_disagreeing": dis,
                                    "serve_has_ht_history_share": round(float(cmp["home_has_ht_history_serve"].mean()), 3)}
    return res


if __name__ == "__main__":
    main()
