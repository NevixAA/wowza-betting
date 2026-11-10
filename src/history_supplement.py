"""history_supplement.py - provider fixtures the served history lacks, added BESIDE the banks (owner 2026-11-10).

THE GAP. The served match rows come from football-data (fd_history + the CI download), the backtest
archive and, for K-League 1 / Saudi Pro League, API-Football results fetched with status FT only.
football-data does not publish most play-offs (Austrian Europa Conference play-off, Belgian / Dutch /
Scottish / EFL / Serie B / Ligue 2 / La Liga 2 / National League play-offs, MLS play-offs), nor the
J.League 2026 special season's games decided on penalties; the FT-only fetch leaves out every game
that went to extra time. Earlier seasons' play-offs reached training only because an old API-Football
import wrote provider rows into fd_history (e.g. Austria 2024/25 'Lask Linz v Rapid Vienna', season
'2024'); the 2025/26 ones were never written.

THE PATTERN (the same as that import, kept beside the banks): provider rows - API-Football fixture id,
the provider's date, both clubs, score.fulltime (the 90-minute score; owner rule: settlement and
training on 90 minutes) and score.halftime where stored - are joined to the history BEFORE the loader's
identity pipeline (src.data_loader.one_club_one_match), with row_source 'provider_supplement'. Clubs are
written in the served history's own spelling of that provider club (scripts/build_history_supplement.py
maps them by the provider team id through the fixtures the served rows already claim; a club the served
history never had keeps the provider's name). Statistics are not copied: the loader's af_history join
fills shots / corners / xG from the provider's own rows where they exist; anything else stays missing.

output/corrections/history_provider_supplement.csv is built by scripts/build_history_supplement.py
(stored provider data, 0 calls); output/fd_history.parquet is never rewritten.
Rollback: WOWZA_HISTORY_SUPPLEMENT=0.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENT = ROOT / "output" / "corrections" / "history_provider_supplement.csv"
ROW_SOURCE = "provider_supplement"
#: the loader's match columns a supplement row carries (everything else stays missing)
MATCH_COLS = ["date", "league", "season", "home_team", "away_team", "home_goals", "away_goals",
              "ht_home_goals", "ht_away_goals"]
EVIDENCE_COLS = ["provider_fixture_id", "provider_date", "provider_home", "provider_away", "provider_home_id",
                 "provider_away_id", "provider_status", "provider_season", "home_name_rule", "away_name_rule",
                 "gap"]


def enabled() -> bool:
    return os.getenv("WOWZA_HISTORY_SUPPLEMENT", "1").strip() != "0"


def read_supplement(path: Path | None = None) -> pd.DataFrame:
    """The supplement rows in the loader's columns (+ provider_fixture_id). Empty when disabled, absent
    or unreadable (logged)."""
    p = Path(path or SUPPLEMENT)
    if not enabled() or not p.exists():
        return pd.DataFrame(columns=MATCH_COLS + ["provider_fixture_id"])
    try:
        d = pd.read_csv(p)
    except (OSError, ValueError, pd.errors.EmptyDataError) as e:
        log.error(f"[history-supplement] {p.name} unreadable ({e}) - NOT applied")
        return pd.DataFrame(columns=MATCH_COLS + ["provider_fixture_id"])
    miss = set(MATCH_COLS + ["provider_fixture_id"]) - set(d.columns)
    if miss:
        log.error(f"[history-supplement] {p.name} lacks {sorted(miss)} - NOT applied")
        return pd.DataFrame(columns=MATCH_COLS + ["provider_fixture_id"])
    d = d[MATCH_COLS + ["provider_fixture_id"]].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in ("home_goals", "away_goals", "ht_home_goals", "ht_away_goals"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["season"] = d["season"].astype(str)
    return d.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"]).reset_index(drop=True)


def add_supplement(df: pd.DataFrame, supplement: pd.DataFrame | None = None, tag: str = "history",
                   source_col: str = "row_source") -> pd.DataFrame:
    """`df` plus the supplement rows (row_source 'provider_supplement') whose (league, date, home, away)
    key `df` does not already hold. Counts per league in out.attrs['history_supplement']."""
    sup = read_supplement() if supplement is None else supplement
    if sup is None or sup.empty or df is None:
        return df
    have = set()
    if not df.empty and {"league", "date", "home_team", "away_team"} <= set(df.columns):
        have = set(zip(df["league"].astype(str), pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d"),
                       df["home_team"].astype(str), df["away_team"].astype(str)))
    keys = list(zip(sup["league"].astype(str), sup["date"].dt.strftime("%Y-%m-%d"),
                    sup["home_team"].astype(str), sup["away_team"].astype(str)))
    new = sup[[k not in have for k in keys]].drop(columns=["provider_fixture_id"])
    if new.empty:
        return df
    if source_col:
        new = new.assign(**{source_col: ROW_SOURCE})
    out = pd.concat([df, new], ignore_index=True)
    per = {k: int(v) for k, v in new["league"].value_counts().items()}
    out.attrs["history_supplement"] = {"rows": int(len(new)), "per_league": per}
    log.info(f"[{tag}] history supplement: {len(new)} provider fixture(s) the banks lack added "
             f"({len(sup) - len(new)} already held): {per}")
    return out
