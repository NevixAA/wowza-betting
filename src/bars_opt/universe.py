"""Priced prediction rows for the bars optimization study (reports/P4_BARS_OPTIMIZATION.md).

One row per (track, market, prediction row) with:
  yes / no        the BET-TIME two-sided pair (src.league_bars rules: football-data pre-match same-book pair
                  for Standard O/U 2.5; the EARLIEST provable pre-kickoff same-snapshot pair otherwise)
  yes_c / no_c    the CLOSING pair of the same source, used ONLY to measure CLV after the fact:
                  football-data same-book closing columns (B365C, else PC) / the LATEST provable
                  pre-kickoff same-snapshot pair. Never an input to a probability, an edge or a bar.
Rows without a bet-time pair are kept with NaN prices (counted, never bet). Clubs are matched only by
src.team_names.resolve_identity (src.league_bars.attach_prices).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import league_bars as LB

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_FILES = {"standard": ("standard_sidemarket_odds_history", "output/standard_sidemarket_odds_history.csv"),
                  "new_format": ("newformat_odds_history", "output/newformat_odds_history.csv")}
#: ht_over05 / ht_over15 (P7 item 2, 2026-11-09): the captured first-half two-sided pairs, joined so the
#: HT markets are no longer NO_PRICE_DATA. Snapshot files only (football-data has no HT price).
PRICED_MARKETS = ("over25", "btts", "over15", "over35", "ht_over05", "ht_over15")


def fd_ou25_close_pairs(fd_raw_dir) -> pd.DataFrame:
    """Football-data CLOSING O/U 2.5 same-book pair (B365C, else PC) - for CLV only."""
    from src.standard_repair import FD_CODES
    inv = {v: k for k, v in FD_CODES.items()}
    frames = []
    for p in sorted(Path(fd_raw_dir).glob("*_*.csv")):
        lg = inv.get(p.stem.rsplit("_", 1)[0])
        if not lg:
            continue
        try:
            raw = pd.read_csv(p, encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
        except UnicodeDecodeError:
            raw = pd.read_csv(p, encoding="cp1252", encoding_errors="replace", on_bad_lines="skip",
                              low_memory=False)
        if not {"Date", "HomeTeam", "AwayTeam"} <= set(raw.columns):
            continue
        out = pd.DataFrame({"date": pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce").dt.normalize(),
                            "league": lg, "home": raw["HomeTeam"], "away": raw["AwayTeam"]})
        out["yes"], out["no"] = np.nan, np.nan
        for o, u in (("B365C>2.5", "B365C<2.5"), ("PC>2.5", "PC<2.5")):
            if o not in raw or u not in raw:
                continue
            oo, uu = pd.to_numeric(raw[o], errors="coerce"), pd.to_numeric(raw[u], errors="coerce")
            take = out["yes"].isna() & (oo > 1) & (uu > 1)
            out.loc[take, "yes"], out.loc[take, "no"] = oo[take], uu[take]
        frames.append(out[out["date"].notna() & out["yes"].notna()])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["date", "league", "home", "away", "yes", "no"])


def _snapshot(track: str, market: str, which: str) -> pd.DataFrame:
    source, rel = SNAPSHOT_FILES[track]
    raw = pd.read_csv(ROOT / rel, low_memory=False)
    pr = LB.snapshot_pairs(raw, source, market, which=which)
    hv = pr["match"].map(LB.split_match)
    pr["home"], pr["away"] = hv.str[0], hv.str[1]
    pr["date"] = pd.to_datetime(pr["match_date"], errors="coerce")
    return pr


def build(pred: pd.DataFrame, track: str, aliases: dict, fd_raw: str | None) -> tuple[pd.DataFrame, dict]:
    """pred: one track's champion_recipe deployable rows. Returns (priced rows, match stats)."""
    frames, stats = [], {}
    for market in PRICED_MARKETS:
        pm = pred[pred["market"] == market]
        if pm.empty:
            continue
        if market == "over25" and track == "standard":
            if not fd_raw:
                raise ValueError("standard O/U 2.5 needs --fd-raw (football-data pre-match prices)")
            entry, close = LB.fd_ou25_pairs(fd_raw), fd_ou25_close_pairs(fd_raw)
        else:
            entry, close = _snapshot(track, market, "first"), _snapshot(track, market, "last")
        pe, st = LB.attach_prices(pm, entry, aliases)
        pc, _ = LB.attach_prices(pm, close, aliases)
        pe["yes_c"], pe["no_c"] = pc["yes"].to_numpy(), pc["no"].to_numpy()
        stats[market] = st
        frames.append(pe)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out, stats
