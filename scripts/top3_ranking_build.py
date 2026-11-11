"""RESEARCH (no serving change): selection-window walk-forward bets of the FROZEN study_58f10476 choices.

For each family with a frozen choice, the chosen configuration (MODEL leagues) and its league-wide twin
(LEAGUE_WIDE leagues) are refitted on each inner training block of the SELECTION window (rows < 2026-09-01)
exactly as src.bars_opt.study.inner_validate does, and their bets on the next block are kept for the leagues
the frozen choice has ON in that mode. Every bet decision is out-of-sample inside SELECT; only the set of ON
leagues comes from the whole SELECT window (descriptive replay, labelled as such).

    python scripts/top3_ranking_build.py   -> output/research/top3_ranking/sel_bets.parquet, sel_rows.parquet (own prices only)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bars_opt import study as S  # noqa: E402

STUDY = ROOT / "output" / "bars_opt" / "study_58f10476"
OUT = ROOT / "output" / "research" / "top3_ranking"


def main() -> None:
    choices = {(c["track"], c["mside"]): c for c in json.loads((STUDY / "choices.json").read_text(encoding="utf-8"))}
    parts, rows = [], []
    for f, track, mside in S.load_families(ROOT / "output" / "bars_opt"):
        ch = choices[(track, mside)]
        if not ch.get("chosen"):
            continue
        on = {lg: b["mode"] for lg, b in ch["bars"].items() if "off" not in b}
        if not on:
            continue
        sel = S.select_rows(f, S.SELECT_END)
        starts = S.inner_blocks(sel["date"])
        chosen = S.parse_cfg(ch["chosen"])
        for k in range(len(starts) - 1):
            t = sel[sel["date"] < starts[k]]
            e = sel[(sel["date"] >= starts[k]) & (sel["date"] < starts[k + 1])]
            if t.empty or e.empty:
                continue
            for mode, cfg in ((S.MODE_MODEL, chosen), (S.MODE_LW, S.lw_of(chosen))):
                mine = [lg for lg, m in on.items() if m == mode]
                if not mine:
                    continue
                bars, scorer = S.fit_cfg(t, cfg)
                if mode == S.MODE_MODEL:
                    # every priced row of the ON leagues up to the end of this block, scored by THIS fold's
                    # source (training rows in-sample for fitted sources; the block's rows out-of-sample):
                    # the calibration rows of rule R2 (a week W reads only rows dated before W)
                    rr = scorer(sel[(sel["date"] < starts[k + 1]) & sel["league"].isin(mine)])
                    rows.append(rr[["track", "mside", "league", "date", "fair", "odds", "p_used", "won"]]
                                .assign(fold=k, fold_start=starts[k], fold_end=starts[k + 1]))
                b = S.apply_bars(scorer(e[e["league"].isin(mine)]), {lg: bars[lg] for lg in mine if lg in bars}, mode)
                if len(b):
                    parts.append(b.assign(fold=k, fold_start=starts[k]))
        print(track, mside, "done", flush=True)
    bets = pd.concat(parts, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    bets.to_parquet(OUT / "sel_bets.parquet", index=False)
    pd.concat(rows, ignore_index=True).to_parquet(OUT / "sel_rows.parquet", index=False)
    print(len(bets), bets["date"].min(), bets["date"].max())


if __name__ == "__main__":
    main()
