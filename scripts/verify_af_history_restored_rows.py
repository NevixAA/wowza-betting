"""
verify_af_history_restored_rows.py - verify the af_history provider rows restored from git (PR #35)
against INDEPENDENT sources before any loader may read them (owner 2026-11-16: "yes, make the models
use them"). Reads committed files only; no API call.
=====================================================================================================
INPUT   output/corrections/af_history_restored_provider_rows_2026-11.parquet   (28,412 provider rows)
        output/corrections/af_history_restored_writeback_cells_2026-11.parquet (loader copies, unproven)
SOURCES (independent of af_history.parquet):
  * API-Football fixture answers already collected beside the history:
      - output/collect/af_history_fullstats_fill.parquet  /fixtures/statistics by a PROVEN fixture id,
        keyed on af_history's own (league, date, home, away): corners, fouls, cards, shots, SoT, xG,
        inside-box, and the fixture id
      - output/collect/af_ht_scores.parquet               /fixtures score.fulltime by fixture id
      - output/strength/competition_fixtures.parquet      /fixtures FT by fixture id
  * football-data (output/fd_history.parquet): score, corners, fouls, shots, SoT - a DIFFERENT provider,
    clubs resolved onto its spellings with src.team_names.resolve (league-scoped; never guessed).
RULES
  * ROW identity: the FT score must equal API-Football's FT for the fixture (by fixture id: fullstats
    key, else provider names + date). No API-Football FT -> football-data's score must equal. A score
    that DISAGREES with any second source, or no second source at all -> the whole row is EXCLUDED.
  * A row whose key is a provider row in the current af_history, or that the owner's duplicate rule
    removed (output/corrections/af_history_removed_duplicates.csv), is EXCLUDED (duplicate).
  * CELL: an impossible value (negative, non-integer count, SoT > shots, beyond a hard ceiling) is
    excluded. A statistic is kept ONLY when API-Football's fixture-id-proven answer holds the same value
    (owner 2026-11-16: cells with no independent second source are excluded, reason "no second
    source"). football-data agreement is reported only: for these leagues its statistic columns share
    the API-Football lineage, so it is not an independent source.
  * xG / inside-box / shots / SoT cells of the loader-copy file (unproven) are usable only when equal
    to an API-Football provider value - and then that provider value already fills the cell, so they
    add nothing; they are verified and reported, never served.
  * Nothing is patched: an excluded cell is missing in the verified file, its reason listed.
OUTPUT  output/corrections/af_history_restored_provider_rows_2026-11_verified.parquet (served source:
        provenance + restored_from_commit per row), ..._excluded.csv (row/cell, reason), ..._verification.json

    python scripts/verify_af_history_restored_rows.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CORR = ROOT / "output" / "corrections"
RESTORED = CORR / "af_history_restored_provider_rows_2026-11.parquet"
WB_CELLS = CORR / "af_history_restored_writeback_cells_2026-11.parquet"
VERIFIED = CORR / "af_history_restored_provider_rows_2026-11_verified.parquet"
EXCLUDED = CORR / "af_history_restored_provider_rows_2026-11_excluded.csv"
REPORT = CORR / "af_history_restored_provider_rows_2026-11_verification.json"
FULLSTATS = ROOT / "output" / "collect" / "af_history_fullstats_fill.parquet"
HT_SCORES = ROOT / "output" / "collect" / "af_ht_scores.parquet"
COMP_FIX = ROOT / "output" / "strength" / "competition_fixtures.parquet"
FD = ROOT / "output" / "fd_history.parquet"
AF = ROOT / "output" / "af_history.parquet"
AF_DUP_REMOVED = CORR / "af_history_removed_duplicates.csv"
KEY = ["league", "date", "home_team", "away_team"]
PROVENANCE = "af_history provider row (scripts/backfill_af_history.py, API-Football) restored from git"

#: restored column -> (API-Football fullstats column, football-data column)
STATS = {"HS": ("home_shots", "home_shots"), "AS": ("away_shots", "away_shots"),
         "HST": ("home_sot", "home_sot"), "AST": ("away_sot", "away_sot"),
         "HC": ("home_corners", "home_corners"), "AC": ("away_corners", "away_corners"),
         "HF": ("home_fouls", "home_fouls"), "AF": ("away_fouls", "away_fouls"),
         "HY": ("home_yellow", None), "AY": ("away_yellow", None),
         "HR": ("home_red", None), "AR": ("away_red", None),
         "HXG": ("home_xg", None), "AXG": ("away_xg", None),
         "HIB": ("home_insidebox", None), "AIB": ("away_insidebox", None)}
#: hard ceilings: a value above is not a football match statistic
CEIL = {"FTHG": 15, "FTAG": 15, "HS": 60, "AS": 60, "HST": 40, "AST": 40, "HC": 30, "AC": 30,
        "HF": 50, "AF": 50, "HY": 12, "AY": 12, "HR": 5, "AR": 5, "HXG": 10, "AXG": 10, "HIB": 50, "AIB": 50}
COUNT_COLS = [c for c in CEIL if c not in ("HXG", "AXG")]


def _k(df, cols=KEY):
    d = pd.to_datetime(df[cols[1]], errors="coerce").dt.strftime("%Y-%m-%d")
    return pd.Series(list(zip(df[cols[0]].astype(str), d, df[cols[2]].astype(str), df[cols[3]].astype(str))),
                     index=df.index, dtype=object)


def _canon(s: pd.Series) -> pd.Series:
    from src import league_registry as LR
    m = {x: (LR.canonical_name(x) or x) for x in s.dropna().astype(str).unique()}
    return s.astype(str).map(m)


def _eq(a, b) -> pd.Series:
    a, b = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    return (a - b).abs() < 1e-9


def build() -> dict:
    r = pd.read_parquet(RESTORED).reset_index(drop=True)
    r["_k"] = _k(r)
    r["_lg"] = _canon(r["league"])
    r["_d"] = pd.to_datetime(r["date"]).dt.strftime("%Y-%m-%d")
    excl_rows: dict[int, str] = {}
    excl_cells: list[dict] = []

    # ── duplicates against the current af_history / the owner's duplicate rule ───────────────────
    af = pd.read_parquet(AF)
    af_prov = af[pd.to_numeric(af["FTHG"], errors="coerce").notna()]
    prov_keys = set(_k(af_prov))
    wb_keys = set(_k(af[pd.to_numeric(af["FTHG"], errors="coerce").isna()]))
    removed = set()
    if AF_DUP_REMOVED.exists():
        rd = pd.read_csv(AF_DUP_REMOVED, low_memory=False)
        if set(KEY) <= set(rd.columns):
            removed = set(_k(rd))
    # same league + date + one club as a CURRENT provider row = the same fixture spelled differently
    side = set(zip(_canon(af_prov["league"]), pd.to_datetime(af_prov["date"]).dt.strftime("%Y-%m-%d"),
                   af_prov["home_team"].astype(str))) | set(
        zip(_canon(af_prov["league"]), pd.to_datetime(af_prov["date"]).dt.strftime("%Y-%m-%d"),
            af_prov["away_team"].astype(str)))
    dup_counts = {"key_is_current_provider_row": 0, "key_removed_by_duplicate_rule": 0,
                  "club_already_has_provider_row_that_day": 0, "key_not_a_current_written_back_row": 0,
                  "duplicate_key_within_restored": int(r["_k"].duplicated().sum())}
    for i, row in r.iterrows():
        k = row["_k"]
        if k in prov_keys:
            excl_rows[i] = "duplicate: key is a provider row in the current af_history"
            dup_counts["key_is_current_provider_row"] += 1
        elif k in removed:
            excl_rows[i] = "duplicate: key removed by the owner's af_history duplicate rule"
            dup_counts["key_removed_by_duplicate_rule"] += 1
        elif (row["_lg"], row["_d"], str(row["home_team"])) in side or \
                (row["_lg"], row["_d"], str(row["away_team"])) in side:
            excl_rows[i] = "duplicate: a club of this row already has a provider row that day"
            dup_counts["club_already_has_provider_row_that_day"] += 1
        elif k not in wb_keys:
            excl_rows[i] = "not a current written-back row"
            dup_counts["key_not_a_current_written_back_row"] += 1

    # ── API-Football: fullstats (af_history key) -> fixture id -> FT score ─────────────────────────
    fs = pd.read_parquet(FULLSTATS)
    fs["_k"] = _k(fs)
    fs = fs.drop_duplicates("_k", keep="last").set_index("_k")
    fsr = fs.reindex(r["_k"].to_numpy())
    fsr.index = r.index
    ht = pd.read_parquet(HT_SCORES)
    cf = pd.read_parquet(COMP_FIX)
    ft_by_id = {}
    for src_df, a, b in ((cf, "ft_home", "ft_away"), (ht, "ft_home", "ft_away")):
        s = src_df.dropna(subset=["fixture_id", a, b])
        ft_by_id.update(dict(zip(s["fixture_id"].astype(int), zip(s[a].astype(float), s[b].astype(float)))))
    by_name = {}
    for src_df, lgc, hn, an in ((cf, "league", "home_name", "away_name"), (ht, "league", "home_name", "away_name")):
        s = src_df.dropna(subset=["ft_home", "ft_away"]).copy()
        lgv = _canon(s[lgc]).to_numpy()
        dv = pd.to_datetime(s["date"], errors="coerce").dt.strftime("%Y-%m-%d").to_numpy()
        for lg_, d_, h_, a_, fh, fa in zip(lgv, dv, s[hn].astype(str), s[an].astype(str),
                                         s["ft_home"].astype(float), s["ft_away"].astype(float)):
            by_name.setdefault((lg_, d_, h_, a_), set()).add((fh, fa))
    af_ft = pd.Series([None] * len(r), index=r.index, dtype=object)
    af_ft_how = pd.Series([None] * len(r), index=r.index, dtype=object)
    for i in r.index:
        fid = fsr.at[i, "fixture_id"] if "fixture_id" in fsr.columns else np.nan
        if pd.notna(fid) and int(fid) in ft_by_id:
            af_ft[i], af_ft_how[i] = ft_by_id[int(fid)], "fixture_id"
            continue
        hit = by_name.get((r.at[i, "_lg"], r.at[i, "_d"], str(r.at[i, "home_team"]), str(r.at[i, "away_team"])))
        if hit and len(hit) == 1:
            af_ft[i], af_ft_how[i] = next(iter(hit)), "provider_names"

    # ── football-data, clubs resolved onto its spellings ─────────────────────────────────────────
    from src.data_loader import _resolve_onto_history
    fd = pd.read_parquet(FD)
    fd["_lg"] = _canon(fd["league"])
    fd["_d"] = pd.to_datetime(fd["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    fdk = fd.assign(league=fd["_lg"])
    rr = r[["_lg", "_d", "home_team", "away_team"]].rename(columns={"_lg": "league", "_d": "dstr"})
    rr["ridx"] = r.index
    res = _resolve_onto_history(rr, fdk, "home_team", "away_team", "verify_restored_vs_fd")
    fdcols = ["home_goals", "away_goals"] + [v[1] for v in STATS.values() if v[1]]
    fkeys = list(zip(fdk["league"], fdk["_d"], fdk["home_team"].astype(str), fdk["away_team"].astype(str)))
    kc = pd.Series(fkeys).value_counts()
    vals = fdk[fdcols].to_numpy(dtype=float)
    fd_map = {k: vals[j] for j, k in enumerate(fkeys) if kc[k] == 1}      # an ambiguous key is no source
    fdr = pd.DataFrame(np.nan, index=r.index, columns=fdcols)
    for lg_, d_, h_, a_, i_ in zip(res["league"], res["dstr"], res["_h"], res["_a"], res["ridx"]):
        hit = fd_map.get((lg_, d_, h_, a_))
        if hit is not None:
            fdr.loc[i_] = hit

    # ── row identity: the score ──────────────────────────────────────────────────────────────────
    score = {"af_ft_match": 0, "af_ft_mismatch": 0, "fd_match_only": 0, "fd_mismatch_only": 0,
             "no_second_source": 0, "fd_also_checked": 0, "fd_disagrees_where_af_agrees": 0}
    score_mm = []
    for i in r.index:
        if i in excl_rows:
            continue
        h, a = float(r.at[i, "FTHG"]), float(r.at[i, "FTAG"])
        fdh, fda = fdr.at[i, "home_goals"], fdr.at[i, "away_goals"]
        has_fd = pd.notna(fdh) and pd.notna(fda)
        if af_ft[i] is not None:
            if af_ft[i] == (h, a):
                score["af_ft_match"] += 1
                if has_fd:
                    score["fd_also_checked"] += 1
                    if (fdh, fda) != (h, a):
                        score["fd_disagrees_where_af_agrees"] += 1
            else:
                score["af_ft_mismatch"] += 1
                excl_rows[i] = f"score {h:.0f}-{a:.0f} != API-Football FT {af_ft[i][0]:.0f}-{af_ft[i][1]:.0f}"
                score_mm.append({**{k: r.at[i, k] for k in KEY}, "restored": f"{h:.0f}-{a:.0f}",
                                 "api_football": f"{af_ft[i][0]:.0f}-{af_ft[i][1]:.0f}", "via": af_ft_how[i]})
        elif has_fd:
            if (fdh, fda) == (h, a):
                score["fd_match_only"] += 1
            else:
                score["fd_mismatch_only"] += 1
                excl_rows[i] = f"score {h:.0f}-{a:.0f} != football-data {fdh:.0f}-{fda:.0f} (no API-Football FT)"
                score_mm.append({**{k: r.at[i, k] for k in KEY}, "restored": f"{h:.0f}-{a:.0f}",
                                 "football_data": f"{fdh:.0f}-{fda:.0f}", "via": "football-data"})
        else:
            score["no_second_source"] += 1
            excl_rows[i] = "no second source for the row's score (API-Football FT or football-data)"

    # ── cells ─────────────────────────────────────────────────────────────────────────────────────
    v = r.copy()
    per_col = {}
    alive = pd.Series(~r.index.isin(list(excl_rows)), index=r.index)
    for c in ["FTHG", "FTAG"] + list(STATS):
        if c not in r.columns:
            continue
        x = pd.to_numeric(r[c], errors="coerce")
        present = x.notna() & alive                      # cells of an excluded row are not re-listed
        imp = present & ((x < 0) | (x > CEIL[c]) | ((x % 1 != 0) if c in COUNT_COLS else False))
        if c in ("HST", "AST"):
            sh = pd.to_numeric(r["HS" if c == "HST" else "AS"], errors="coerce")
            imp |= present & sh.notna() & (x > sh)
        stat = {"present": int(present.sum()), "impossible": int(imp.sum())}
        bad = imp.copy()
        if c in STATS:
            afc, fdc = STATS[c]
            a = pd.to_numeric(fsr[afc], errors="coerce") if afc in fsr.columns else pd.Series(np.nan, index=r.index)
            both = present & a.notna()
            eq = _eq(x, a)
            stat.update(af_checked=int(both.sum()), af_match=int((both & eq).sum()),
                        af_mismatch=int((both & ~eq).sum()))
            bad |= both & ~eq
            f = fdr[fdc] if fdc else pd.Series(np.nan, index=r.index)
            fboth = present & f.notna()
            feq = _eq(x, f)
            stat.update(fd_checked=int(fboth.sum()), fd_match=int((fboth & feq).sum()),
                        fd_mismatch=int((fboth & ~feq).sum()))
            # owner 2026-11-16 ("Exclude them"): a statistic is used only when an INDEPENDENT second
            # source confirms it. For these leagues football-data's statistic columns share the
            # API-Football lineage, so API-Football's fixture-id-proven answer is the only one.
            no_second = present & a.isna()
            bad |= no_second
            stat["no_second_source"] = int(no_second.sum())
            stat["af_match_pct"] = round(100 * stat["af_match"] / stat["af_checked"], 2) if stat["af_checked"] else None
            stat["fd_match_pct"] = round(100 * stat["fd_match"] / stat["fd_checked"], 2) if stat["fd_checked"] else None
            for i in r.index[(both & ~eq).to_numpy()]:
                excl_cells.append({**{k: r.at[i, k] for k in KEY}, "column": c, "value": x[i],
                                   "reason": f"!= API-Football {a[i]:g}"})
            for i in r.index[no_second.to_numpy()]:
                excl_cells.append({**{k: r.at[i, k] for k in KEY}, "column": c, "value": x[i],
                                   "reason": "no second source"})
        for i in r.index[imp.to_numpy()]:
            excl_cells.append({**{k: r.at[i, k] for k in KEY}, "column": c, "value": x[i], "reason": "impossible value"})
        v.loc[bad, c] = np.nan
        stat["excluded_cells"] = int(bad.sum())
        per_col[c] = stat

    rows_out = [{**{k: r.at[i, k] for k in KEY}, "column": "*row*", "value": None, "reason": why}
                for i, why in sorted(excl_rows.items())]
    v = v.drop(index=list(excl_rows)).drop(columns=["_k", "_lg", "_d"])
    v["provenance"] = PROVENANCE
    v["verified_by"] = [("API-Football FT (" + af_ft_how[i] + ")") if af_ft[i] is not None else "football-data score"
                        for i in v.index]
    keep_cols = KEY + ["season", "FTHG", "FTAG", "FTR"] + list(STATS) + \
        ["provenance", "verified_by", "restored_from_commit", "restored_from_commit_date"]
    v = v[[c for c in keep_cols if c in v.columns]].reset_index(drop=True)

    # ── the loader-copy cells (unproven) ──────────────────────────────────────────────────────────
    wb = pd.read_parquet(WB_CELLS)
    wb["_k"] = _k(wb)
    wbm = {"HS": "home_shots", "AS": "away_shots", "HST": "home_sot", "AST": "away_sot",
           "HXG": "home_xg", "AXG": "away_xg", "HIB": "home_insidebox", "AIB": "away_insidebox"}
    wbs = {}
    for c, g in wb.groupby("column"):
        a = pd.to_numeric(fs.reindex(g["_k"].to_numpy())[wbm[c]], errors="coerce").to_numpy() \
            if c in wbm and wbm[c] in fs.columns else np.full(len(g), np.nan)
        val = g["value"].to_numpy(dtype=float)
        chk = ~np.isnan(a)
        wbs[c] = {"cells": int(len(g)), "provider_checked": int(chk.sum()),
                  "equal_to_provider": int((chk & (np.abs(val - a) < 1e-9)).sum()),
                  "differ_from_provider": int((chk & ~(np.abs(val - a) < 1e-9)).sum()),
                  "no_provider_value": int((~chk).sum())}

    excl = pd.DataFrame(rows_out + excl_cells, columns=KEY + ["column", "value", "reason"])
    reasons = excl.loc[excl["column"] == "*row*", "reason"].str.replace(r"\d+-\d+", "N-N", regex=True)
    report = {
        "restored_rows": int(len(r)), "verified_rows": int(len(v)), "excluded_rows": len(excl_rows),
        "excluded_rows_by_reason": {k: int(n) for k, n in reasons.value_counts().items()},
        "duplicates": dup_counts, "score_identity": score, "score_mismatches_listed": len(score_mm),
        "columns": per_col, "excluded_cells": int((excl["column"] != "*row*").sum()),
        "served_cells_after_verification": {c: int(v[c].notna().sum()) for c in STATS if c in v.columns},
        "writeback_cells_unproven": wbs,
        "writeback_cells_note": "served: none. A loader copy equal to an API-Football value adds nothing (the "
                                "provider value already fills that cell); one that differs or has no provider "
                                "value stays excluded.",
        "sources": {"api_football_statistics": FULLSTATS.relative_to(ROOT).as_posix(),
                    "api_football_ft": [HT_SCORES.relative_to(ROOT).as_posix(), COMP_FIX.relative_to(ROOT).as_posix()],
                    "football_data": FD.relative_to(ROOT).as_posix()},
        "api_calls": 0,
    }
    return {"verified": v, "excluded": excl, "report": report, "score_mismatches": score_mm}


def main() -> int:
    out = build()
    out["verified"].to_parquet(VERIFIED, index=False)
    out["excluded"].to_csv(EXCLUDED, index=False)
    rep = dict(out["report"], score_mismatches=out["score_mismatches"])
    REPORT.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: out["report"][k] for k in ("restored_rows", "verified_rows", "excluded_rows",
                                                     "excluded_rows_by_reason", "score_identity",
                                                     "excluded_cells")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
