"""Props CLV integrity audit (Phase 2 part 7) -> reports/p1_props_clv_data/.

    python scripts/props_clv_integrity.py            # read the committed files, write the status tables

An INTEGRITY audit of output/clv_records.csv, not a backtest. Each record gets one status from
player_model.clv_integrity.classify_stored (same rules the writer now applies to new records):
is the stored close a quote of the same player / market / line / side / fixture as the entry, quoted
before kickoff, and is it same-book or best-of-books? It reports counts per status, market and month.
It computes no CLV mean, no ROI and no edge.

Inputs (read-only):
  output/clv_records.csv                                   the records
  output/player_prop_odds_history.csv + _archive/<date>    every prop quote (src.rolling_store.read_all)
  output/player_ledger.csv                                 provider player ids (namesake check)
  reports/au4_data/prop_kickoff_index.csv                  recorded kickoffs from git history (AU4)
  output/player_tips.csv                                   recorded kickoffs on the current board
  output/prop_quotes/*.csv                                 per-book quotes (same-book proof, from 2026-10-14)
Outputs: clv_record_status.csv (one row per record), status_by_market_month.csv, summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from player_model import clv_integrity as CI  # noqa: E402
from player_model.odds_fetcher import _norm  # noqa: E402

OUT = ROOT / "output"
DEST = ROOT / "reports" / "p1_props_clv_data"
KICKOFF_INDEX = ROOT / "reports" / "au4_data" / "prop_kickoff_index.csv"


def kickoffs(index: pd.DataFrame | None, tips: pd.DataFrame | None) -> tuple[dict, set]:
    """({(date, match): kickoff}, {(date, match) recorded with two kickoffs}). The AU4 index keeps a
    fixture's LAST value but counts every value it saw (n_distinct_kickoff): more than one = refused."""
    parts, bad = [], set()
    if index is not None and not index.empty:
        k = index[["date", "match", "kickoff_utc"]].copy()
        if "n_distinct_kickoff" in index.columns:
            multi = pd.to_numeric(index["n_distinct_kickoff"], errors="coerce") > 1
            bad |= set(zip(index.loc[multi, "date"].astype(str).str[:10], index.loc[multi, "match"].astype(str)))
        parts.append(k)
    if tips is not None and not tips.empty and {"date", "match", "kickoff_utc"} <= set(tips.columns):
        parts.append(tips[["date", "match", "kickoff_utc"]])
    if not parts:
        return {}, bad
    k = pd.concat(parts, ignore_index=True).dropna()
    k["date"] = k["date"].astype(str).str[:10]
    k["match"] = k["match"].astype(str)
    k["_t"] = pd.to_datetime(k["kickoff_utc"], errors="coerce", utc=True)
    k = k.dropna(subset=["_t"])
    n = k.groupby(["date", "match"])["_t"].nunique()
    bad |= set(n[n > 1].index)
    first = k.groupby(["date", "match"])["_t"].first()
    return {key: t for key, t in first.items() if key not in bad}, bad


def classify(clv: pd.DataFrame, hist: pd.DataFrame, ko_map: dict, ko_bad: set,
             ledger: pd.DataFrame | None, quotes: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per record: status, reason, books, kickoff basis and the record's labels. Same book is
    decided on the per-book `quotes` (output/prop_quotes): the committed history kept one row per price."""
    idx = CI.index_history(CI.prepare_history(hist))
    qidx = CI.index_history(CI.prepare_history(quotes if quotes is not None else pd.DataFrame(), per_book=True))
    ids = CI.ledger_ids(ledger)
    # the file's last write: a record without a close whose match was not played by then is not a gap
    as_of = str(pd.to_datetime(clv["ts_bet"], errors="coerce", utc=True, format="mixed").max())[:10]
    rows = []
    for rec in clv.to_dict("records"):
        parsed = CI.parse_bet_id(rec.get("bet_id"))
        d, player, market = parsed if parsed else ("", str(rec.get("player", "")), str(rec.get("market", "")))
        match = str(rec.get("match", "") or "")
        pids = ids.get((_norm(player), market, d), set())
        m = CI.identity_rows(idx, d, player, market, match, next(iter(pids)) if len(pids) == 1 else None)
        own = rec.get("kickoff_utc") if isinstance(rec.get("kickoff_utc"), str) else ""
        ko = CI.utc(own) if own else ko_map.get((d, match), pd.NaT)
        q = CI.identity_rows(qidx, d, player, market, match)
        res = CI.classify_stored(rec, m, ko, kickoff_ambiguous=(d, match) in ko_bad, namesake=len(pids) > 1,
                                 quotes=q)
        if res["status"] == CI.NO_CLOSE and res["reason"] == "no close recorded" and d >= as_of:
            res["reason"] = f"no close recorded; match on or after the file's last entry ({as_of})"
        notes = str(rec.get("notes", "") or "")
        rows.append({
            "bet_id": rec.get("bet_id"), "match_date": d, "month": d[:7], "market": market,
            "tier": notes.split(";")[0] if notes else "",
            # production (main) wrote every committed close as the LAST history row of any book; the
            # best-of-books close and its notes label exist on data-truth-100 only
            "close_basis": "best_vs_best" if "clv_basis=best_vs_best" in notes else "last_row_any_book",
            "status": res["status"], "reason": res["reason"],
            "entry_books": ";".join(sorted(res["entry_books"])), "close_books": ";".join(sorted(res["close_books"])),
            "kickoff_basis": "record" if own else ("index" if (d, match) in ko_map else "none"),
            "player_id": next(iter(pids)) if len(pids) == 1 else "",
            "n_provider_ids": len(pids), "n_identity_rows": int(len(m)),
            "has_clv_value": str(rec.get("clv_pct", "")).strip() not in ("", "nan"),
        })
    return pd.DataFrame(rows)


def tables(st: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """(status x market x month counts, summary dict). Counts only."""
    by = (st.groupby(["month", "market", "status"]).size().rename("n").reset_index()
          .sort_values(["month", "market", "status"]))
    summ = {
        "records": int(len(st)),
        "by_status": {s: int((st["status"] == s).sum()) for s in CI.STATUSES},
        "by_status_market": {s: st[st["status"] == s]["market"].value_counts().to_dict() for s in CI.STATUSES},
        "by_status_month": {s: st[st["status"] == s]["month"].value_counts().sort_index().to_dict()
                            for s in CI.STATUSES},
        "by_reason": st.groupby(["status", "reason"]).size().rename("n").reset_index().to_dict("records"),
        "by_close_basis": st.groupby(["close_basis", "status"]).size().rename("n").reset_index().to_dict("records"),
        "kickoff_basis": st["kickoff_basis"].value_counts().to_dict(),
        "player_id_attached": int((st["player_id"] != "").sum()),
        "namesake_records": int((st["n_provider_ids"] > 1).sum()),
        "clv_value_on_non_valid": int((st["has_clv_value"] & ~st["status"].isin(CI.VALID)).sum()),
    }
    return by, summ


def main() -> None:
    from src.rolling_store import read_all
    clv = pd.read_csv(OUT / "clv_records.csv", dtype=str, keep_default_na=False)
    hist = read_all(OUT / "player_prop_odds_history.csv")
    ledger = pd.read_csv(OUT / "player_ledger.csv", dtype=str,
                         usecols=lambda c: c in ("player_name", "market", "match_date", "player_id"))
    index = pd.read_csv(KICKOFF_INDEX, dtype=str) if KICKOFF_INDEX.exists() else None
    qdir = OUT / "prop_quotes"
    quotes = (pd.concat([pd.read_csv(p, dtype=str) for p in sorted(qdir.glob("*.csv"))], ignore_index=True)
              if qdir.exists() else pd.DataFrame())
    tips = pd.read_csv(OUT / "player_tips.csv", dtype=str, usecols=lambda c: c in ("date", "match", "kickoff_utc"))
    ko_map, ko_bad = kickoffs(index, tips)
    st = classify(clv, hist, ko_map, ko_bad, ledger, quotes)
    by, summ = tables(st)
    summ["inputs"] = {"clv_records": int(len(clv)), "history_rows": int(len(hist)),
                      "prop_quote_rows": int(len(quotes)),
                      "prop_quote_match_dates": ([str(quotes["match_date"].min()), str(quotes["match_date"].max())]
                                                 if len(quotes) else []),
                      "history_match_dates": [str(hist["match_date"].min()), str(hist["match_date"].max())],
                      "kickoff_fixtures": len(ko_map), "kickoff_refused_two_values": len(ko_bad)}
    DEST.mkdir(parents=True, exist_ok=True)
    st.to_csv(DEST / "clv_record_status.csv", index=False, lineterminator="\n")
    by.to_csv(DEST / "status_by_market_month.csv", index=False, lineterminator="\n")
    (DEST / "summary.json").write_text(json.dumps(summ, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: summ[k] for k in ("records", "by_status", "kickoff_basis", "clv_value_on_non_valid")},
                     indent=1))


if __name__ == "__main__":
    main()
