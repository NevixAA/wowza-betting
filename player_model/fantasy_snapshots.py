"""Point-in-time FPL snapshot archive: one row per (target gameweek, player), taken BEFORE that deadline.

    from player_model.fantasy_snapshots import update_archive
    update_archive(bootstrap, fetched_at=<unix ts of the fetch>)     # called by fantasy_v2.run()

    python scripts/fantasy_backfill_snapshots.py    # one-off: rebuild from git history

WHY THIS FILE EXISTS. Every question Fantasy V2 asks — "does a residual on top of FPL's ep_next
beat ep_next?", "how often does a 0.9 start probability actually start?" — needs, for every past
gameweek, what was KNOWABLE before its deadline and what HAPPENED after it. `fpl_bootstrap.json`
is overwritten every run, so that history only existed as 40 git versions of a 1.7 MB blob, and
a CI checkout is depth-1, so the pipeline could never see it. This table is the durable form.

KEY = (target_gw, element). A snapshot taken while gameweek g is `is_next` describes the state a
manager faced before g's deadline. Each run REPLACES that gameweek's rows, so after the deadline
the archive holds the last pre-deadline view — exactly the "last snapshot you could have acted
on" rule fantasy_settle uses. A snapshot taken at or after the deadline is refused for g.

ACTUALS NEED NO EXTRA CALLS. Cumulative season counters (total_points, minutes, starts, goals...)
in row (g+1) minus row (g) are what happened in gameweek g, bonus included, because row (g+1) is
taken before the NEXT deadline. The row for the in-progress gameweek is provisional by
construction and is flagged as such downstream.

GROWTH. ~670 players x 38 gameweeks ~ 25k rows a season (~5 MB of CSV), not 670 x 365 days.

FRESHNESS comes from the recorded fetch timestamp (fpl_cache_meta.json), never from file mtime.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

ARCHIVE = Path(__file__).resolve().parents[1] / "output" / "fantasy_fpl_snapshots.csv"

POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

#: bootstrap element field -> archive column. Cumulative season counters are what make
#: per-gameweek actuals reconstructable; the rest is the pre-deadline information set.
FIELDS = {
    "id": "element", "web_name": "web_name", "team": "team_id", "element_type": "element_type",
    "now_cost": "now_cost", "status": "status",
    "chance_of_playing_next_round": "chance_next", "news": "news",
    "ep_next": "ep_next", "ep_this": "ep_this", "form": "form", "points_per_game": "ppg",
    "selected_by_percent": "selected_pct", "transfers_in_event": "transfers_in_event",
    "transfers_out_event": "transfers_out_event", "cost_change_event": "cost_change_event",
    "penalties_order": "pen_order", "corners_and_indirect_freekicks_order": "corner_order",
    "direct_freekicks_order": "dfk_order",
    # cumulative season counters
    "total_points": "total_points", "minutes": "minutes", "starts": "starts",
    "goals_scored": "goals", "assists": "assists", "clean_sheets": "clean_sheets",
    "goals_conceded": "goals_conceded", "bonus": "bonus", "bps": "bps", "saves": "saves",
    "yellow_cards": "yellow_cards", "red_cards": "red_cards",
    "expected_goals": "xg", "expected_assists": "xa", "expected_goals_conceded": "xgc",
    "threat": "threat", "creativity": "creativity", "influence": "influence",
    "defensive_contribution": "def_contrib",
}
CUMULATIVE = ["total_points", "minutes", "starts", "goals", "assists", "clean_sheets",
              "goals_conceded", "bonus", "bps", "saves", "yellow_cards", "red_cards",
              "xg", "xa", "xgc", "threat", "creativity", "influence", "def_contrib"]
COLUMNS = ["target_gw", "deadline", "fetched_at", "player_name", "team", "position"] + \
          list(FIELDS.values())


def target_event(bootstrap: dict) -> dict | None:
    """The gameweek a snapshot taken now is the pre-deadline view of (FPL's `is_next`)."""
    ev = (bootstrap or {}).get("events") or []
    nxt = next((e for e in ev if e.get("is_next")), None)
    return nxt


def rows_from_bootstrap(bootstrap: dict, fetched_at: float) -> pd.DataFrame:
    """Compact per-player rows for the gameweek this snapshot precedes. Empty if none/late."""
    ev = target_event(bootstrap)
    if not ev or not bootstrap.get("elements"):
        return pd.DataFrame(columns=COLUMNS)
    deadline = pd.Timestamp(ev["deadline_time"]).tz_convert("UTC") \
        if pd.Timestamp(ev["deadline_time"]).tzinfo else pd.Timestamp(ev["deadline_time"], tz="UTC")
    fetched = pd.Timestamp(float(fetched_at), unit="s", tz="UTC")
    if fetched >= deadline:
        # FPL flips is_next a little after the deadline; a fetch in that gap is not pre-deadline.
        return pd.DataFrame(columns=COLUMNS)
    teams = {t["id"]: t.get("name", "") for t in bootstrap.get("teams", [])}
    recs = []
    for e in bootstrap["elements"]:
        r = {dst: e.get(src) for src, dst in FIELDS.items()}
        r["news"] = " ".join(str(r.get("news") or "").split())[:160]
        r["player_name"] = f"{e.get('first_name', '')} {e.get('second_name', '')}".strip()
        r["team"] = teams.get(e.get("team"), "")
        r["position"] = POS.get(e.get("element_type"), "")
        recs.append(r)
    d = pd.DataFrame(recs)
    d["target_gw"] = int(ev["id"])
    d["deadline"] = deadline.strftime("%Y-%m-%dT%H:%M:%SZ")
    d["fetched_at"] = fetched.strftime("%Y-%m-%dT%H:%M:%SZ")
    for c in ("ep_next", "ep_this", "form", "ppg", "selected_pct", "xg", "xa", "xgc",
              "threat", "creativity", "influence"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d[COLUMNS]


def merge_rows(archive: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Replace the target gameweek's rows with the newer snapshot; keep every other gameweek."""
    if new is None or new.empty:
        return archive
    if archive is None or archive.empty:
        return new.reset_index(drop=True)
    gw = int(new["target_gw"].iloc[0])
    old = archive[archive["target_gw"].astype(int) == gw]
    if not old.empty and str(old["fetched_at"].max()) > str(new["fetched_at"].iloc[0]):
        return archive          # never overwrite a newer view with an older one
    keep = archive[archive["target_gw"].astype(int) != gw]
    out = pd.concat([keep, new], ignore_index=True)
    return out.sort_values(["target_gw", "element"]).reset_index(drop=True)


def load(path: Path | None = None) -> pd.DataFrame:
    p = path or ARCHIVE
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(p)


def update_archive(bootstrap: dict, fetched_at: float | None, path: Path | None = None) -> int:
    """Fold one bootstrap snapshot into the archive. Returns rows written for its gameweek.

    An unknown fetch time is REFUSED, not assumed to be now: an undatable snapshot cannot be
    placed on the right side of a deadline, and placing it wrongly is look-ahead leakage.
    """
    if fetched_at is None:
        log.warning("[fantasy_snapshots] no recorded fetch time — snapshot not archived")
        return 0
    p = path or ARCHIVE
    new = rows_from_bootstrap(bootstrap, fetched_at)
    if new.empty:
        return 0
    merged = merge_rows(load(p), new)
    p.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(p, index=False, encoding="utf-8", lineterminator="\n")
    return int(len(new))


def recorded_fetch_time(meta_path: Path | None = None) -> float | None:
    """The bootstrap's recorded fetch time from the sidecar — never file mtime."""
    mp = meta_path or (Path(__file__).resolve().parents[1] / "output" / "fpl_cache_meta.json")
    try:
        v = json.loads(mp.read_text(encoding="utf-8")).get("fpl_bootstrap.json")
        return float(v) if v is not None else None
    except Exception:                                                # noqa: BLE001
        return None
