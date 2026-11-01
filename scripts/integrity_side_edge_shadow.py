"""SHADOW: side-market edge on two-sided de-vigged prices vs production's `(1/odds)/1.08`.

    python scripts/integrity_side_edge_shadow.py [--no-append]

Reads only committed files: output/predictions.csv (current board), output/side_bets_ledger.csv
(production side tips, first sight), output/book_odds_snapshots.csv + book_odds_snapshots_archive/
(per-book change-log quotes, carried forward). Writes:
    output/side_edge_devig_shadow.csv   append-only, first sight per fixture/market (board rows)
    output/side_edge_devig_health.json  current board + ledger summary (RETROSPECTIVE for ledger)
Production selection is not touched. See src/integrity/side_edge_devig.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def load_quotes():
    from src.integrity.side_edge_devig import load_quotes as _lq
    return _lq(ROOT / "output")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-append", action="store_true")
    ap.add_argument("--out-json", default="output/side_edge_devig_health.json")
    ap.add_argument("--out-csv", default="output/side_edge_devig_shadow.csv")
    a = ap.parse_args(argv)
    import pandas as pd
    import config
    from src.integrity import side_edge_devig as SE

    params = SE.load_league_params(config.MODELS_DIR)
    quotes = load_quotes()
    ko_map = (quotes.dropna(subset=["kickoff_utc"]).drop_duplicates("match", keep="last")
                    .set_index("match")["kickoff_utc"].astype(str).to_dict())

    board = pd.read_csv(ROOT / "output" / "predictions.csv", low_memory=False)
    ev_board = SE.evaluate_rows(SE.board_rows(board), quotes, params)

    led = pd.read_csv(ROOT / "output" / "side_bets_ledger.csv", low_memory=False)
    lr = SE.ledger_rows(led, ko_map)
    q0 = pd.to_datetime(quotes["snapshot_ts"], errors="coerce", utc=True).min()
    lr = lr[lr["asof"] >= q0] if len(lr) and pd.notna(q0) else lr
    ev_led = SE.evaluate_rows(lr, quotes, params) if len(lr) else pd.DataFrame()
    open_mask = None
    if len(ev_led):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        md = lr["match_date"].astype(str).str[:10].to_numpy()
        ev_led["match_date"] = md
        open_mask = ev_led["match_date"] >= today

    doc = {"artifact": "side_edge_devig_health", "shadow_version": SE.SHADOW_VERSION,
           "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "production_rule": "two-sided de-vig since 2026-11-01 (config.SIDE_EDGE_FAIR_PRICE); "
                              "edge_prod below = the old (1/odds)/1.08, kept for comparison",
           "shadow_rule": "edge = p_model - median_books(power_devig(same-book two-sided pair)), "
                          "pre-kickoff, carried forward to the decision time; same thresholds",
           "quotes_from": str(q0), "quote_rows": int(len(quotes)),
           "board": {"generated_at": str(board["generated_at"].iloc[0]) if len(board) else None,
                     **SE.summarise(ev_board)},
           "ledger_retrospective": {"label": "RETROSPECTIVE", "tips_since_quotes_exist": int(len(lr)),
                                    **SE.summarise(ev_led)},
           "ledger_open_tips": SE.summarise(ev_led[open_mask]) if open_mask is not None else {}}
    if len(ev_led):
        only = ev_led[ev_led["classification"] == "ONLY_BECAUSE_OF_1_08"]
        doc["ledger_retrospective"]["examples_only_because_of_1_08"] = [
            {"fixture": f"{r.home_team} v {r.away_team}", "league": r.league, "market": r.market,
             "tier_prod": r.tier_prod, "edge_prod": round(r.edge_prod, 4),
             "edge_devig": round(r.edge_devig, 4), "odds": r.odds,
             "p_fair_devig": round(r.p_fair_devig, 4), "books": r.books}
            for r in only.tail(8).itertuples()]
    p = ROOT / a.out_json
    p.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    if not a.no_append:
        n = SE.append_first_sight(ev_board, ROOT / a.out_csv)
        print(f"appended {n} first-sight board rows to {a.out_csv}")
    print(json.dumps({k: doc[k] for k in ("board", "ledger_retrospective", "ledger_open_tips")},
                     indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
