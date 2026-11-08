"""Per-league tip bars from the clean Phase 2 retrain - PROPOSALS ONLY (src/league_bars.py explains).

    python scripts/league_bars.py --track standard \
        --predictions <run>/predictions.parquet --fd-raw <Pro>/experiments/ou_info_inputs_2026_10/fd_raw
    python scripts/league_bars.py --track new_format --predictions <run>/predictions.parquet
    python scripts/league_bars.py --report        # rebuild reports/P3_LEAGUE_BARS.md from the evidence

Writes ONLY output/threshold_proposals/league_bars/<track>/ (candidates in the production params shape
+ league_bars_evidence.json) and, with --report, reports/P3_LEAGUE_BARS.md. Never models/, never
config/threshold_approvals.json: the owner approves entries through src/threshold_proposals.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import league_bars as LB  # noqa: E402

OUT_DIR = ROOT / "output" / "threshold_proposals" / "league_bars"
REPORT = ROOT / "reports" / "P3_LEAGUE_BARS.md"
SNAPSHOT_FILES = {"standard": [("standard_sidemarket_odds_history", "output/standard_sidemarket_odds_history.csv")],
                  "new_format": [("newformat_odds_history", "output/newformat_odds_history.csv")]}


def load_aliases(path: str | None) -> dict:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    from src.retrain_harness import protocol as P
    from src import data_loader as DL
    cfg = json.loads((ROOT / "config" / "phase2_retrain.json").read_text(encoding="utf-8"))
    P.load_committed_frame(cfg)
    return {lg: dict(m) for lg, m in DL.IDENTITY_ALIASES.items()}


def snapshot_prices(track: str, market: str) -> tuple[pd.DataFrame, str]:
    frames, how = [], []
    for source, rel in SNAPSHOT_FILES[track]:
        raw = pd.read_csv(ROOT / rel, low_memory=False)
        pr = LB.snapshot_pairs(raw, source, market)
        hv = pr["match"].map(LB.split_match)
        pr["home"], pr["away"] = hv.str[0], hv.str[1]
        pr["date"] = pd.to_datetime(pr["match_date"], errors="coerce")
        frames.append(pr)
        how.append(f"{rel}: earliest provable pre-kickoff same-snapshot pair from match_date-1d "
                   f"({pr['price_origin'].value_counts().to_dict()})")
    return pd.concat(frames, ignore_index=True), "; ".join(how)


def run_track(a) -> int:
    pred = pd.read_parquet(a.predictions)
    pred = pred[(pred["track"] == a.track) & (pred["candidate"] == a.candidate) & pred["deployable"]]
    aliases = load_aliases(a.aliases)
    rows, match_rates, price_notes = [], {}, {}
    for market in sorted(pred["market"].unique()):
        pm = pred[pred["market"] == market]
        if market in LB.TWO_SIDED_BET and a.track == "standard":
            prices = LB.fd_ou25_pairs(a.fd_raw)
            note = f"football-data PRE-MATCH same-book pair ({prices['book'].value_counts().to_dict()})"
        elif market in LB.SNAPSHOT_KEYS:
            prices, note = snapshot_prices(a.track, market)
        else:
            prices, note = pd.DataFrame(columns=["date", "league", "home", "away", "yes", "no"]), \
                "no price exists for this market"
        priced, stats = LB.attach_prices(pm, prices, aliases)
        match_rates[market] = stats
        price_notes[market] = note
        rows += LB.run_market(priced, market, a.track)
    meta = {"track": a.track, "candidate": a.candidate, "predictions": str(a.predictions),
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "price_sources": price_notes, "price_match": match_rates,
            "rules": {"grid": [float(x) for x in LB.GRID], "min_bets_fit": LB.MIN_BETS_FIT,
                      "min_holdout_bets": LB.MIN_HOLDOUT_BETS, "holm_alpha_one_sided": LB.HOLM_ALPHA_ONE_SIDED,
                      "n_boot": LB.N_BOOT}}
    LB.finalise(rows)                                   # per-track family; --report re-runs it across tracks
    for p in LB.write_outputs(rows, a.track, OUT_DIR, meta):
        print("wrote", p.relative_to(ROOT))
    return 0


def _fmt(x, pct=False):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    return f"{100 * x:+.1f}%" if pct else f"{x:g}"


def build_report() -> str:
    """Families are corrected together: Holm is recomputed over every track's tested rows."""
    ev = {}
    for f in sorted(OUT_DIR.glob("*/league_bars_evidence.json")):
        ev[f.parent.name] = json.loads(f.read_text(encoding="utf-8"))
    allrows = [r for t in ev.values() for r in t["rows"]]
    for r in allrows:                                   # re-finalise across tracks (one family)
        if r["status"] in (LB.STATUS_PROVEN, LB.STATUS_UNPROVEN):
            r["status"] = None
            r["p_one_sided"] = float("nan") if r["p_one_sided"] is None else r["p_one_sided"]
    LB.finalise(allrows)
    for track, t in ev.items():                         # candidates follow the cross-track family
        LB.write_outputs([r for r in allrows if r["track"] == track], track, OUT_DIR, t["meta"])
    L = ["# P3 - Per-league tip bars from the clean retrain (PROPOSALS ONLY)", "",
         "Owner decisions 2026-11-08: every league has its OWN bars for every market and tier (no global bar); "
         "small leagues are shrunk toward the pooled estimate by an empirical-Bayes weight learned from the "
         "selection data; PROVEN only under the strict rule (>= 150 out-of-sample bets and ROI above 0 with "
         f"Holm-adjusted one-sided p < {LB.HOLM_ALPHA_ONE_SIDED} across every league x market tested). Owner, "
         "same day: \"no way with 100% real data we don't have true bars\" - a league x market priced only from "
         "2026-04-01 gets IN_PERIOD_SPLIT bars; NO_PRICE_DATA only where no real two-sided bet-time price exists "
         "at any time (no number is invented there).", "",
         "Nothing here serves. The candidates are in `output/threshold_proposals/league_bars/<track>/`; production "
         "changes only through `config/threshold_approvals.json` after the owner approves an entry. Serving today "
         "uses a league's own bars only when `approved` is true and has no per-league VALUABLE bar: using these "
         "bars for every league (and VALUABLE per league) needs a serving change the owner decides at go-live.", "",
         "Context: on the Standard holdout the de-vigged CLOSING O/U 2.5 price beat every model "
         "(champion_recipe log loss minus market_only = +0.0146; market_only is a non-deployable benchmark). "
         "Bars cannot add information the model does not have; expect few or no PROVEN cells.", "",
         "## How the bars are built", "",
         "* Model: `champion_recipe` (the design that won the retrain comparison), deployable rows.",
         "* **WALK_FORWARD_HOLDOUT** evidence: bars chosen on walk-forward rows (< 2026-04-01) only; the holdout "
         "(2026-04-01..2026-11-08) is scored once.",
         "* **IN_PERIOD_SPLIT** evidence (no priced walk-forward row): the holdout period is split in time order at "
         f"the month boundary that leaves about {LB.IN_PERIOD_CHECK_SHARE:.0%} of the LEAGUE's priced matchdays in the "
         "CHECK part (its own split; shrinkage pools every league's priced rows in the same earlier window); bars are "
         "chosen on the earlier part, scored once on the later part, same strict rule. One short window: weaker "
         "evidence than WALK_FORWARD_HOLDOUT. Split dates per league are in the tables.",
         "* Edge = p_model - power de-vigged fair probability of a two-sided pair captured before kickoff. "
         "Never the closing price; a fixture without a pair is not bet.",
         "* VALUABLE = lowest bar with shrunk walk-forward ROI > 0; MARKSMAN = bar maximising shrunk total profit; "
         f"SNIPER = bar maximising shrunk ROI per bet; only bars with >= {LB.MIN_BETS_FIT} own selection bets "
         "(basis OWN_BETS), else where the pooled bets reach it (basis SHRUNK_TO_POOL: mostly the pooled curve; "
         "the league's own weight is the 'Shrink w' column).",
         "* Out-of-sample bets = every tip at or above VALUABLE, 1 unit flat; 95% CI by block bootstrap over match "
         "dates.", ""]
    for track, t in ev.items():
        m = t["meta"]
        L += [f"## {track}", "", f"Predictions: `{m['predictions']}`; built {m['created_at']}.", "",
              "Price sources:", ""]
        L += [f"* **{k}**: {v}" for k, v in m["price_sources"].items()]
        L += [""]
        rows = [r for r in allrows if r["track"] == track]
        fr = pd.DataFrame(rows)
        fr["cell"] = fr["status"] + fr["evidence"].map(lambda e: f" ({e})" if isinstance(e, str) else "")
        cnt = fr.groupby(["market", "cell"]).size().unstack(fill_value=0)
        L += ["Status count per market:", "", "| Market | " + " | ".join(cnt.columns) + " |",
              "|---|" + "---|" * len(cnt.columns)]
        L += [f"| {mk} | " + " | ".join(str(int(v)) for v in cnt.loc[mk]) + " |" for mk in cnt.index]
        L += [""]
        for market in sorted({r["market"] for r in rows}):
            L += [f"### {track} / {market}", ""]
            L += ["| League | Status | Evidence | Split: select / check (check share) | Basis | VALUABLE | MARKSMAN | "
                  "SNIPER | Shrink w | Selection bets | Selection ROI | OOS bets | OOS ROI | 95% CI | p Holm | "
                  "Price match (rows priced / rows) | Reason |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            st = m["price_match"].get(market, {})
            for r in [x for x in rows if x["market"] == market]:
                pm = st.get(r["league"]) or {"priced": "-", "rows": "-"}
                ci = r.get("ho_ci95") or [None, None]
                sp = r.get("split")
                spl = (f"{sp['select_from']}..{sp['select_to']} / {sp['check_from']}..{sp['check_to']} "
                       f"({sp['check_share_of_matchdays']:.0%})") if sp else "-"
                L.append(f"| {r['league']} | {r['status']} | {r.get('evidence') or '-'} | {spl} | {r.get('bar_basis') or '-'} | "
                         f"{_fmt(r.get('valuable_th'))} | {_fmt(r.get('marksman_th'))} | "
                         f"{_fmt(r.get('sniper_th'))} | {_fmt(r.get('shrink_weight_at_valuable'))} | {_fmt(r.get('wf_bets'))} | "
                         f"{_fmt(r.get('wf_roi'), True)} | {_fmt(r.get('ho_bets'))} | {_fmt(r.get('ho_roi'), True)} | "
                         f"[{_fmt(ci[0], True)}, {_fmt(ci[1], True)}] | {_fmt(r.get('p_holm'))} | "
                         f"{pm['priced']}/{pm['rows']} | {r.get('reason') or ''} |")
            L.append("")
    for track in ("standard", "new_format"):
        if track not in ev:
            L += [f"## {track}", "", f"NOT BUILT: no `{track}` retrain predictions are available (the job did not "
                  "finish with an artifact). Run `scripts/league_bars.py --track " + track + "` on its "
                  "predictions.parquet, then `--report`.", ""]
    L += ["## What the owner decides", "",
          "1. Approve or reject each PROVEN entry (only those can be real money under the strict rule).",
          "2. Whether UNPROVEN leagues tip on their own bars as paper, and the serving change for per-league VALUABLE.",
          "3. NO_PRICE_DATA cells (no real two-sided bet-time price at any time, e.g. half-time markets): no bars "
          "exist; they need price capture first.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--track", choices=["standard", "new_format"])
    ap.add_argument("--predictions")
    ap.add_argument("--candidate", default="champion_recipe")
    ap.add_argument("--fd-raw")
    ap.add_argument("--aliases", help="cached identity alias map (JSON); default: run the committed loader")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    if a.track:
        if not a.predictions:
            ap.error("--predictions is required with --track")
        if a.track == "standard" and not a.fd_raw:
            ap.error("--fd-raw is required for the standard track (O/U 2.5 pre-match prices)")
        run_track(a)
    if a.report:
        REPORT.write_text(build_report(), encoding="utf-8")
        print("wrote", REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
