"""Re-run inputs readiness (owner 2026-11-11): per family x league, the study rows that ALREADY hold a bet-time
price from proven own captures (protocol v4/v5 own-backfill strip) or the certified football-data join (Standard
O/U 2.5), the rows still without one, and how many of those the OddsAPI tier BT can reach (league has an OddsAPI
sport key, market in BT's request, match date inside BT's history window). Counts only - no price is read out.

    python scripts/rerun_inputs_readiness.py --aliases <loader identity map json> [--bt-coverage 0.43] [--write]

--bt-coverage is an ASSUMPTION (the share of reachable events BT returns a bet-time pair for), stated in the
output; the expected BT addition is reachable rows x that share, per league.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

#: BT (src/oddsapi_history.py on prep/oddsapi-history, 0f4d57036): event tier, markets btts + alternate_totals,
#: "team" leagues, 2023-05-03 .. HISTORY_END. HT markets are not requested.
BT_MARKETS = {"over25", "btts", "over15", "over35"}
BT_START = pd.Timestamp("2023-05-03")
UNIVERSES = {"standard": ROOT / "output" / "bars_opt" / "universe_standard.parquet",
             "new_format": ROOT / "output" / "bars_opt" / "universe_new_format.parquet"}


def readiness(aliases: dict, fd_raw: Path, bt_share: float) -> list[dict]:
    from src import fd_prematch as F
    from src import league_registry as LR
    from src.bars_opt import oddsapi_prices as P
    out = []
    for track, path in UNIVERSES.items():
        u = pd.read_parquet(path)
        u["date"] = pd.to_datetime(u["date"])
        own, _ = P.strip_unproven_own(u, track, aliases)
        if track == "standard":
            o25 = u[u["market"] == "over25"]
            cert, _ = F.certified_ou25(o25, aliases, fd_raw)
            own.loc[o25.index, ["yes", "no"]] = cert[["yes", "no"]].to_numpy()
        priced = own["yes"].notna() & own["no"].notna()
        for (market, league), g in own.groupby(["market", "league"]):
            p = priced.loc[g.index]
            gap = g[~p]
            reach = gap[(gap["date"] >= BT_START) & bool(LR.odds_api_sport_key(league))] \
                if market in BT_MARKETS else gap.iloc[0:0]
            out.append({"track": track, "market": market, "league": league, "rows": int(len(g)),
                        "bet_time_priced": int(p.sum()),
                        "source": ("certified_fd_join" if (track, market) == ("standard", "over25")
                                   else "proven_own_captures"),
                        "unpriced": int(len(gap)), "bt_reachable": int(len(reach)),
                        "bt_expected_add": int(math.floor(len(reach) * bt_share))})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aliases", required=True)
    ap.add_argument("--fd-raw", default=str(ROOT / "output" / "collect" / "fd_raw"))
    ap.add_argument("--bt-coverage", type=float, default=0.43)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    rows = readiness(json.loads(Path(a.aliases).read_text(encoding="utf-8")), Path(a.fd_raw), a.bt_coverage)
    df = pd.DataFrame(rows)
    fam = df.groupby(["track", "market"])[["rows", "bet_time_priced", "unpriced", "bt_reachable",
                                           "bt_expected_add"]].sum().reset_index()
    print(fam.to_string(index=False))
    doc = {"about": __doc__.strip().splitlines()[0], "bt_coverage_assumed": a.bt_coverage,
           "bt_coverage_basis": "BT test run: about 43% event coverage at T-153h / T-125h (coordinator 2026-11-11)",
           "families": fam.to_dict("records"), "leagues": rows}
    if a.write:
        (ROOT / "output" / "bars_opt" / "rerun_inputs_readiness.json").write_text(
            json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
