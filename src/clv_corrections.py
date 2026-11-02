"""Display-side corrections read from output/corrections/ (round-2 data-truth findings, 2026-11-02).

    from src.clv_corrections import (sharp_record, exclude_contaminated_clv, ht_priced_mask,
                                     top_picks_late_mask)

ONE PLACE for every consumer (Telegram, dashboard, research views) to apply the sidecar corrections.
The committed ledgers are never rewritten: each correction is a sidecar file beside the original, with
an audit note in output/corrections/README_round2_display_2026-11-02.md, and the original values stay
in the ledger (and in a `*_stored` column of every frame this module returns).

V1  SHARP RECORD. sharp_ledger.csv books P/L at `opening_odds`, a price that was already gone when the
    signal fired (opening > current on 100% of rows): +22.22u as booked vs -159.14u at the signal-time
    price on 2,010 settled rows. The restated record is written by the settlement work to
    output/corrections/sharp_ledger_restated.csv. sharp_record() reads it; when it is ABSENT the record
    is UNDER RESTATEMENT and no P/L figure is returned at all - never the inflated booked figure.
    Sidecar contract (tolerant): the ledger key columns that are present among SHARP_KEY (at least
    match_date, home_team, away_team, side) plus one restated P/L column among SHARP_PNL_COLUMNS, and
    optionally a restated result column (SHARP_RESULT_COLUMNS). A settled ledger row the sidecar does
    not cover is EXCLUDED and counted (`unrestated`), never shown at the booked price.

V2  FIRST-HALF CLOSES. Archive rows 2026-07-13..08-23 stored first-half prices as full-match closes;
    170 bets_ledger rows (all new_format, |clv_pct| > 50) carry them. They are listed in
    output/corrections/clv_first_half_contaminated.csv (the archive rows that supplied them in
    clv_first_half_contaminated_archive_rows.csv - quarantined, not deleted). exclude_contaminated_clv()
    blanks clv_pct / closing_odds on the listed rows for every CLV consumer; P/L is untouched.
    SIDE BETS (2026-11-02): 6 side_bets_ledger rows (BTTS YES, |clv_pct| > 50, closes 6.0-8.0 = a
    first-half BTTS price, signalled 2026-08-22) are listed in side_clv_first_half_contaminated.csv,
    with the 7 archive snapshots that supplied them in side_clv_first_half_contaminated_archive_rows.csv
    (same evidence check). A frame with a `market` column is matched against that sidecar, keyed with
    the market. Audit note: output/corrections/README_side_clv_2026-11-02.md.

HT  UNPRICED HALF-TIME BETS. 47 of 61 settled ht_ledger rows have no market price (entry_odds blank)
    and settled at the model's own fair odds. Listed in output/corrections/ht_ledger_unpriced.csv;
    ht_priced_mask() is THE rule every HT P/L display uses.

TP  TOP PICKS FROZEN AFTER KICKOFF. 25 of 219 frozen picks (2026-10-20..25) were frozen after their
    match had kicked off. Listed in output/corrections/top_picks_frozen_after_kickoff.csv and excluded
    from grading (src/top_picks.graded); src/top_picks.freeze_day now refuses such picks.

Rebuild the sidecars (read-only on the ledgers):  python -m src.clv_corrections build
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

UNDER_RESTATEMENT = "UNDER RESTATEMENT"

SHARP_KEY = ["signal_date", "match_date", "league", "home_team", "away_team", "market_label", "side"]
SHARP_PNL_COLUMNS = ("pnl_restated", "restated_pnl", "pnl_signal_price", "pnl_at_signal", "pnl_current_odds")
SHARP_RESULT_COLUMNS = ("result_restated", "restated_result", "result")

CLV_KEY = ["match_date", "league", "home_team", "away_team", "side"]
#: detection rule for the 2026-07/08 first-half contamination (no stored |CLV| between 40.9% and 50%:
#: the clean and contaminated populations do not overlap)
CLV_CONTAMINATION_ABS = 50.0
SIDE_CLV_KEY = ["match_date", "league", "home_team", "away_team", "market", "side"]
SIDE_CLV_SIDECAR = "side_clv_first_half_contaminated.csv"
SIDE_CLV_ARCHIVE_SIDECAR = "side_clv_first_half_contaminated_archive_rows.csv"
#: side market -> (archive market key, archive files by model_type)
_SIDE_ARCHIVE = {("btts", "YES"): "btts_yes", ("btts", "NO"): "btts_no", ("over15", "OVER"): "over15",
                 ("over35", "OVER"): "over35"}
_ARCHIVE_FILES = {"new_format": "newformat_odds_history.csv",
                  "standard": "standard_sidemarket_odds_history.csv"}
HT_KEY = ["source", "generated_at", "match_date", "league", "home_team", "away_team", "market", "side"]
TP_KEY = ["frozen_at", "list", "rank", "day", "home_team", "away_team", "market"]


def corrections_dir(out_dir=None) -> Path:
    return Path(out_dir if out_dir is not None else config.OUTPUT_DIR) / "corrections"


def sidecar(name: str, out_dir=None) -> Path:
    return corrections_dir(out_dir) / name


def _read(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception as e:                                              # noqa: BLE001
        log.warning("corrections sidecar %s unreadable (%s)", p.name, e)
        return None


def _keyframe(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    k = pd.DataFrame(index=d.index)
    for c in cols:
        v = d[c] if c in d.columns else pd.Series("", index=d.index)
        v = v.astype(str).str.strip().replace({"nan": "", "None": "", "<NA>": ""})
        if c.endswith("date") or c == "day":
            v = v.str[:10]
        if c == "side":
            v = v.str.upper()
        k[c] = v
    return k


def _member(d: pd.DataFrame, listed: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Row mask: d's key tuple appears in `listed` (string-normalised)."""
    if listed is None or listed.empty or d.empty:
        return pd.Series(False, index=d.index)
    a = _keyframe(d, cols).apply(tuple, axis=1)
    b = set(_keyframe(listed, cols).apply(tuple, axis=1))
    return a.isin(b)


# ── V1 sharp record ──────────────────────────────────────────────────────────────────────────────
def sharp_record(ledger: pd.DataFrame, out_dir=None) -> tuple[pd.DataFrame | None, dict]:
    """The sharp ledger with `pnl` / `result` RESTATED from the sidecar, or (None, status) while the
    record is under restatement. status: {"state": "RESTATED" | UNDER_RESTATEMENT, "unrestated": n,
    "source": sidecar path}. The booked values are kept as pnl_stored / result_stored."""
    p = sidecar("sharp_ledger_restated.csv", out_dir)
    st = {"state": UNDER_RESTATEMENT, "unrestated": 0, "source": str(p)}
    r = _read(p)
    if r is None or ledger is None:
        return None, st
    pcol = next((c for c in SHARP_PNL_COLUMNS if c in r.columns), None)
    keys = [c for c in SHARP_KEY if c in r.columns and c in ledger.columns]
    if pcol is None or not {"match_date", "home_team", "away_team", "side"} <= set(keys):
        log.warning("sharp_ledger_restated.csv lacks a restated P/L column or the ledger key - "
                    "sharp record stays UNDER RESTATEMENT")
        return None, st
    rcol = next((c for c in SHARP_RESULT_COLUMNS if c in r.columns), None)
    rk = _keyframe(r, keys)
    rk["_pnl_r"] = pd.to_numeric(r[pcol], errors="coerce")
    rk["_res_r"] = r[rcol].astype(str).str.upper().replace({"NAN": ""}) if rcol else ""
    rk = rk.drop_duplicates(keys, keep="last")
    out = ledger.copy()
    out["pnl_stored"] = out.get("pnl")
    out["result_stored"] = out.get("result")
    lk = _keyframe(out, keys)
    m = lk.merge(rk, on=keys, how="left")
    m.index = out.index
    covered = m["_pnl_r"].notna()
    settled_stored = out["result"].astype(str).str.upper().isin(["WIN", "LOSS"])
    unrestated = settled_stored & ~covered
    out["pnl"] = m["_pnl_r"].where(covered, np.nan)
    if rcol:
        res = m["_res_r"].where(covered & (m["_res_r"] != ""), out["result"].where(covered, np.nan))
        out["result"] = res
    else:
        out["result"] = out["result"].where(covered, np.nan)
    out["restatement"] = np.where(covered, "RESTATED", np.where(unrestated, "UNRESTATED", ""))
    st.update(state="RESTATED", unrestated=int(unrestated.sum()))
    if st["unrestated"]:
        log.warning("sharp record: %d settled row(s) not in the restated sidecar - excluded", st["unrestated"])
    return out, st


# ── V2 first-half CLV contamination ──────────────────────────────────────────────────────────────
def contaminated_rule(ledger: pd.DataFrame) -> pd.Series:
    """The detection rule that built the sidecar: |clv_pct| > 50 on a new-format row."""
    clv = pd.to_numeric(ledger.get("clv_pct"), errors="coerce")
    nf = ledger.get("model_type", pd.Series("", index=ledger.index)).astype(str).eq("new_format")
    return (clv.abs() > CLV_CONTAMINATION_ABS) & nf


def side_contaminated_rule(ledger: pd.DataFrame) -> pd.Series:
    """side_bets_ledger detection rule: |clv_pct| > 50 AND the close is a first-half BTTS price
    (BTTS YES closing above price_provenance.BTTS_YES_FULLMATCH_MAX = 3.20 - no full-match BTTS
    price sits there; the parser fix 8d9962c68 landed 2026-08-23). 6 rows at data-truth-100, all
    signalled 2026-08-22, closes 6.0-8.0; the next largest stored |clv_pct| is 47.4 (2026-11-02)."""
    from src.integrity.price_provenance import BTTS_YES_FULLMATCH_MAX
    clv = pd.to_numeric(ledger.get("clv_pct"), errors="coerce")
    close = pd.to_numeric(ledger.get("closing_odds"), errors="coerce")
    mk = ledger.get("market", pd.Series("", index=ledger.index)).astype(str).str.lower()
    side = ledger.get("side", pd.Series("", index=ledger.index)).astype(str).str.upper()
    return (clv.abs() > CLV_CONTAMINATION_ABS) & mk.eq("btts") & side.eq("YES") & (close > BTTS_YES_FULLMATCH_MAX)


def contaminated_mask(ledger: pd.DataFrame, out_dir=None) -> pd.Series:
    """Ledger rows whose closing price is a quarantined first-half price. The sidecar is the
    record; when it is missing the detection rule is applied (fail-safe) and a warning logged.
    A frame with a `market` column is a side_bets_ledger frame: its own sidecar, keyed WITH the
    market (an O1.5 OVER row must never match a bets_ledger O2.5 OVER entry)."""
    if ledger is None or ledger.empty:
        return pd.Series(False, index=getattr(ledger, "index", None))
    if "market" in ledger.columns:
        # row-wise, so a frame mixing both ledgers (market blank on bets_ledger rows) works too
        mk = ledger["market"].astype(str).str.strip().str.lower()
        is_side = ledger["market"].notna() & ~mk.isin(["", "nan", "none", "<na>"])
        listed = _read(sidecar(SIDE_CLV_SIDECAR, out_dir))
        if listed is None:
            log.warning("%s missing - applying the side detection rule directly", SIDE_CLV_SIDECAR)
            side = side_contaminated_rule(ledger)
        else:
            side = _member(ledger, listed, SIDE_CLV_KEY)
        if is_side.all():
            return side & is_side
        rest = ledger[~is_side].drop(columns=["market"])
        return (side & is_side) | contaminated_mask(rest, out_dir).reindex(ledger.index, fill_value=False)
    listed = _read(sidecar("clv_first_half_contaminated.csv", out_dir))
    if listed is None:
        log.warning("clv_first_half_contaminated.csv missing - applying the detection rule directly")
        return contaminated_rule(ledger)
    return _member(ledger, listed, CLV_KEY)


def exclude_contaminated_clv(ledger: pd.DataFrame, out_dir=None) -> pd.DataFrame:
    """Copy of a bets_ledger frame with clv_pct / closing_odds blanked on quarantined rows
    (originals kept as clv_pct_stored / closing_odds_stored, flag in clv_quarantined)."""
    out = ledger.copy()
    m = contaminated_mask(out, out_dir).reindex(out.index, fill_value=False).astype(bool)
    out["clv_quarantined"] = m
    for c in ("clv_pct", "closing_odds"):
        if c in out.columns:
            out[f"{c}_stored"] = out[c]
            out[c] = out[c].where(~m, np.nan)
    return out


# ── HT unpriced ──────────────────────────────────────────────────────────────────────────────────
def ht_priced_mask(ht: pd.DataFrame, out_dir=None) -> pd.Series:
    """True where an ht_ledger row has a real market price: `odds` (tip time) or `entry_odds`
    (filled at settlement), and is not listed as unpriced in the sidecar."""
    px = pd.to_numeric(ht["odds"], errors="coerce") if "odds" in ht.columns else pd.Series(np.nan, index=ht.index)
    if "entry_odds" in ht.columns:
        px = px.fillna(pd.to_numeric(ht["entry_odds"], errors="coerce"))
    listed = _read(sidecar("ht_ledger_unpriced.csv", out_dir))
    return px.notna() & ~_member(ht, listed, HT_KEY)


# ── Top picks frozen after kickoff ───────────────────────────────────────────────────────────────
def top_picks_late_mask(frozen: pd.DataFrame, out_dir=None) -> pd.Series:
    listed = _read(sidecar("top_picks_frozen_after_kickoff.csv", out_dir))
    return _member(frozen, listed, TP_KEY)


# ── sidecar builder (read-only on the ledgers) ───────────────────────────────────────────────────
def _archive_rows(bad: pd.DataFrame, out: Path, archive: str = "newformat_odds_history.csv",
                  market_of=None) -> pd.DataFrame:
    """The `archive` snapshots that supplied each quarantined close: same match date, same canonical
    league, the row's market key (default: the side's O/U 2.5 key), resolver-matched clubs, price ==
    the stored close."""
    from src.league_registry import canonical_name
    from src.team_names import resolve_result
    market_of = market_of or (lambda r: "under25" if str(r["side"]).upper() == "UNDER" else "over25")
    p = out / archive
    if not p.exists() or bad.empty:
        return pd.DataFrame()
    a = pd.read_csv(p, usecols=["snapshot_ts", "match_date", "league", "match", "market", "odds"], low_memory=False)
    a["_row"] = a.index + 2                                            # 1-based file line (header = 1)
    a = a[a["market"].isin({market_of(r) for _, r in bad.iterrows()})]
    a["_dk"] = a["match_date"].astype(str).str[:10]
    a["_lk"] = a["league"].map(lambda x: canonical_name(x))
    hv = a["match"].astype(str).str.split(" vs ", n=1, expand=True)
    a["_h"], a["_a"] = hv[0].str.strip(), hv[1].str.strip()
    rows = []
    for _, r in bad.iterrows():
        mk = market_of(r)
        same = a[(a["_dk"] == str(r["match_date"])[:10]) & (a["_lk"] == canonical_name(r["league"]))
                 & (a["market"] == mk) & a["_a"].notna()]
        if same.empty:
            continue
        rh = resolve_result(str(r["home_team"]), sorted(set(same["_h"])))
        ra = resolve_result(str(r["away_team"]), sorted(set(same["_a"])))
        if not rh or not ra:
            continue
        hit = same[(same["_h"] == rh) & (same["_a"] == ra)
                   & (pd.to_numeric(same["odds"], errors="coerce").round(3) == round(float(r["closing_odds"]), 3))]
        for _, h in hit.iterrows():
            rows.append({"archive_file": f"output/{archive}", "archive_line": int(h["_row"]),
                         "snapshot_ts": h["snapshot_ts"], "match_date": h["match_date"], "league": h["league"],
                         "match": h["match"], "market": h["market"], "odds": h["odds"],
                         "ledger_home_team": r["home_team"], "ledger_away_team": r["away_team"],
                         "ledger_side": r["side"], "status": "QUARANTINED"})
    return pd.DataFrame(rows)


def build_side(out_dir=None) -> dict:
    """Write the side_bets_ledger first-half CLV sidecars (read-only on the ledger): the rows the
    side rule flags, and the archive snapshots that supplied each close - the same evidence check as
    the 170 bets_ledger rows (same date, canonical league, market key, resolver-matched clubs, price
    equal to the stored close), in the archive of the row's model type."""
    out = Path(out_dir if out_dir is not None else config.OUTPUT_DIR)
    cdir = corrections_dir(out)
    cdir.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(out / "side_bets_ledger.csv", low_memory=False)
    bad = s[side_contaminated_rule(s)]
    cols = [c for c in ["source", "generated_at", "signal_date", *SIDE_CLV_KEY, "model_type", "odds",
                        "closing_odds", "clv_pct", "result", "pnl", "mv_source"] if c in s.columns]
    arch = []
    for mt, f in _ARCHIVE_FILES.items():
        sub = bad[bad.get("model_type", pd.Series("", index=bad.index)).astype(str) == mt]
        a = _archive_rows(sub, out, f, lambda r: _SIDE_ARCHIVE.get(
            (str(r["market"]).lower(), str(r["side"]).upper()), ""))
        if len(a):
            arch.append(a)
    arch = pd.concat(arch, ignore_index=True) if arch else pd.DataFrame(
        columns=["archive_file", "archive_line", "snapshot_ts", "match_date", "league", "match", "market",
                 "odds", "ledger_home_team", "ledger_away_team", "ledger_side", "status"])
    covered = set(zip(arch.get("ledger_home_team", []), arch.get("ledger_away_team", [])))
    flagged = bad[cols].assign(
        archive_evidence=[(h, a) in covered for h, a in zip(bad["home_team"], bad["away_team"])],
        reason="first-half BTTS price recorded as the full-match close (|clv_pct| > 50, BTTS YES close > 3.20)",
        action="excluded from every CLV consumer; ledger value kept")
    flagged.to_csv(cdir / SIDE_CLV_SIDECAR, index=False)
    arch.to_csv(cdir / SIDE_CLV_ARCHIVE_SIDECAR, index=False)
    clv = pd.to_numeric(s["clv_pct"], errors="coerce")
    return {"rows": int(len(bad)), "with_archive_evidence": int(flagged["archive_evidence"].sum()),
            "archive_rows": int(len(arch)), "mean_clv_stored": round(float(clv.mean()), 2),
            "mean_clv_clean": round(float(clv[~side_contaminated_rule(s)].mean()), 2),
            "n_clv_stored": int(clv.notna().sum()),
            "n_clv_clean": int(clv[~side_contaminated_rule(s)].notna().sum())}


def build(out_dir=None) -> dict:
    """Write the V2 / HT / top-picks sidecars from the committed files. Never touches a ledger."""
    out = Path(out_dir if out_dir is not None else config.OUTPUT_DIR)
    cdir = corrections_dir(out)
    cdir.mkdir(parents=True, exist_ok=True)
    summary = {}

    b = pd.read_csv(out / "bets_ledger.csv", low_memory=False)
    bad = b[contaminated_rule(b)]
    cols = [c for c in ["source", "generated_at", *CLV_KEY, "model_type", "odds", "closing_odds", "clv_pct",
                        "result", "pnl"] if c in b.columns]
    bad[cols].assign(reason="first-half price recorded as the full-match close (|clv_pct| > 50, new_format)",
                     action="excluded from every CLV consumer; ledger value kept") \
        .to_csv(cdir / "clv_first_half_contaminated.csv", index=False)
    arch = _archive_rows(bad, out)
    arch.to_csv(cdir / "clv_first_half_contaminated_archive_rows.csv", index=False)
    nf = b[b.get("model_type").astype(str) == "new_format"]
    clv = pd.to_numeric(nf["clv_pct"], errors="coerce")
    summary["clv"] = {"rows": int(len(bad)), "archive_rows": int(len(arch)),
                      "nf_mean_stored": round(float(clv.mean()), 2),
                      "nf_mean_clean": round(float(clv[~contaminated_rule(nf)].mean()), 2)}

    h = pd.read_csv(out / "ht_ledger.csv", low_memory=False)
    settled = h["result"].astype(str).str.upper().isin(["WIN", "LOSS"])
    px = pd.to_numeric(h.get("odds"), errors="coerce") if "odds" in h.columns else pd.Series(np.nan, index=h.index)
    px = px.fillna(pd.to_numeric(h.get("entry_odds"), errors="coerce"))
    unp = h[settled & px.isna()]
    hcols = [c for c in [*HT_KEY, "model_prob", "fair_odds", "entry_odds", "result", "pnl"] if c in h.columns]
    unp[hcols].assign(reason="settled at the model's own fair odds - no market price recorded",
                      action="excluded from every HT P/L display; ledger value kept") \
        .to_csv(cdir / "ht_ledger_unpriced.csv", index=False)
    summary["ht"] = {"settled": int(settled.sum()), "unpriced": int(len(unp)),
                     "pnl_all": round(float(pd.to_numeric(h.loc[settled, "pnl"]).sum()), 2),
                     "pnl_unpriced": round(float(pd.to_numeric(unp["pnl"]).sum()), 2),
                     "pnl_priced": round(float(pd.to_numeric(h.loc[settled & px.notna(), "pnl"]).sum()), 2)}

    from src import top_picks
    late = top_picks.frozen_after_kickoff(top_picks.frozen(), out_dir=out)
    late.to_csv(cdir / "top_picks_frozen_after_kickoff.csv", index=False)
    summary["top_picks"] = {"frozen": int(len(top_picks.frozen())), "late": int(len(late))}
    return summary


if __name__ == "__main__":                                              # pragma: no cover
    import json
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        print(json.dumps(build(), indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "build-side":
        print(json.dumps(build_side(), indent=2))
    else:
        print(__doc__)
