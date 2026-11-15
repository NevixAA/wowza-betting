"""Certify the football-data PRE-MATCH join (src/fd_prematch.py) and write the proven club-name crosswalk.

    python scripts/fd_prematch_join.py --aliases <loader identity map json> [--fd-raw output/collect/fd_raw] [--write]

Served fixtures = the Standard study universe's fixtures (served names); the score check uses the committed bank
output/fd_history.parquet (names mapped by the identity map). Prints the join rate per league. --write stores
config/fd_raw_crosswalk.json (football-data spelling -> served club, each with its joined-row evidence) and
output/bars_opt/fd_prematch_join_report.json (counts only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import fd_prematch as F  # noqa: E402


def served_fixtures(aliases: dict) -> pd.DataFrame:
    u = pd.read_parquet(ROOT / "output" / "bars_opt" / "universe_standard.parquet", columns=["row_id", "league", "date"])
    parts = u["row_id"].astype(str).str.split("|", expand=True)
    sv = pd.DataFrame({"league": u["league"].to_numpy(), "date": pd.to_datetime(u["date"]).dt.normalize().to_numpy(),
                       "home_team": parts[2].to_numpy(), "away_team": parts[3].to_numpy()}).drop_duplicates()
    b = pd.read_parquet(ROOT / "output" / "fd_history.parquet",
                        columns=["league", "date", "home_team", "away_team", "home_goals", "away_goals"])
    for c in ("home_team", "away_team"):
        b[c] = [aliases.get(lg, {}).get(n, n) for lg, n in zip(b["league"], b[c])]
    b["date"] = pd.to_datetime(b["date"]).dt.normalize()
    b = b.dropna(subset=["home_goals", "away_goals"]).drop_duplicates(["league", "date", "home_team", "away_team"])
    return sv.merge(b, on=["league", "date", "home_team", "away_team"], how="left").reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aliases", required=True)
    ap.add_argument("--fd-raw", default=str(ROOT / "output" / "collect" / "fd_raw"))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    aliases = json.loads(Path(a.aliases).read_text(encoding="utf-8"))
    fd = F.load_fd_raw(a.fd_raw)
    sv = served_fixtures(aliases)
    j, rep = F.join(fd, sv, aliases, {})
    cw = F.used_names(j)
    j2, rep2 = F.join(fd, sv, aliases, {lg: {k: v["served"] for k, v in m.items()} for lg, m in cw.items()})
    tot_s = sum(r["served"] for r in rep2)
    tot_j = sum(r["joined"] for r in rep2)
    doc = {"rule": F.__doc__.split("THE JOIN.")[1].split("THE CROSSWALK")[0].strip(),
           "served_fixtures": tot_s, "joined": tot_j, "join_rate": round(tot_j / tot_s, 4) if tot_s else None,
           "score_checked": int(sv["home_goals"].notna().sum()), "leagues": rep2,
           "pm_book": j2.loc[j2["fd_file"].astype(str) != "", "pm_source"].value_counts().to_dict()}
    print(pd.DataFrame(rep2).fillna(0).to_string(index=False))
    print(json.dumps({k: v for k, v in doc.items() if k not in ("leagues", "rule")}))
    if a.write:
        (ROOT / "config" / "fd_raw_crosswalk.json").write_text(json.dumps({
            "about": "football-data (fd_raw) club spelling -> served club, per league: every pair the CERTIFIED join "
                     "(src/fd_prematch.py) used, each fd spelling to exactly ONE served club on >= "
                     f"{F.MIN_DATES} joined rows (name-resolved, same date or unique +-1 day, one-to-one, score "
                     "checked where the bank has it). Built by scripts/fd_prematch_join.py --write; never typed.",
            "map": {lg: {k: v["served"] for k, v in m.items()} for lg, m in sorted(cw.items())},
            "evidence": cw}, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        (ROOT / "output" / "bars_opt" / "fd_prematch_join_report.json").write_text(
            json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
