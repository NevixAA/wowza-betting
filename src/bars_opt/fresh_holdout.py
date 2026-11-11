"""FRESH HOLDOUT (owner 2026-11-10 "Fresh holdout"; pre-registered in config/bars_fresh_holdout_protocol.json).

A row is FRESH when its bet-time price was NEVER available to any earlier bars study run: it is priced only by
the bought OddsAPI history (price_source oddsapi_hist) AND no earlier run's universe held it priced.

PROOF OF "NEVER AVAILABLE" (earlier_runs), from git history, fail-closed:
  * every committed version of output/bars_opt/universe_{standard,new_format}.parquet (the universes the
    bars_opt SELECT / FINAL / monthly / serving runs read): its priced (row_id, market) keys are SEEN;
  * the league_bars runs (search_ledger kind league_bars) saved no universe; they priced with the same own
    loaders (football-data + the own snapshot files). Their snapshot files must be unchanged since those runs,
    or every older version's keys must still be in today's file - otherwise LedgerError (nothing is assumed);
  * every row our own sources price today is SEEN as well (own prices are never fresh).

THE HOLDOUT per family (track x market-side), holdout_window:
  1. the family's study-eligible rows (study.family_frame on the priced universe), grouped by match date;
  2. FRESH RUNS = maximal stretches of consecutive match dates on which EVERY eligible row is fresh;
  3. (protocol v2, owner 2026-11-10 "Fix the rule") the MOST RECENT fresh run of at least MIN_RUN_ROWS rows
     (= MIN_HOLDOUT_BETS / IN_PERIOD_CHECK_SHARE) whose HOLDOUT PART (step 4) gives at least one league
     MIN_HOLDOUT_BETS rows - a later run that cannot host a 150-bet test is skipped (counted);
  4. that run is split in time order at a month boundary by src.league_bars.in_period_split (the owner's
     2026-11-08 split rule, CHECK part ~40%): HOLDOUT = the run's rows dated >= the split;
  5. SELECTION = every eligible row of the family dated < the split. Rows dated after the run's end are used
     by neither. No overlap; nothing from the holdout is read before the single test.
A family without such a run gets NO historical PROVEN test (NO_FRESH_HOLDOUT): it waits for the live Clean
Record from 2026-11-21.
"""
from __future__ import annotations

import io
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src import league_bars as LB
from src.bars_opt import oddsapi_prices as P
from src.bars_opt import search_ledger as L
from src.bars_opt import study as S

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_FILE = ROOT / "config" / "bars_fresh_holdout_protocol.json"
UNIVERSE_PATHS = {"standard": "output/bars_opt/universe_standard.parquet",
                  "new_format": "output/bars_opt/universe_new_format.parquet"}
OWN_SNAPSHOT_FILES = ("output/standard_sidemarket_odds_history.csv", "output/newformat_odds_history.csv")
SNAP_KEY = ["match_date", "league", "match", "market"]
#: the owner's strict minimum, scaled to the check share of the owner's split rule
MIN_RUN_ROWS = int(math.ceil(LB.MIN_HOLDOUT_BETS / LB.IN_PERIOD_CHECK_SHARE))
NO_FRESH = "NO_FRESH_HOLDOUT"


def _blob_bytes(blob: str) -> bytes:
    r = subprocess.run(["git", "cat-file", "-p", blob], cwd=ROOT, capture_output=True)
    if r.returncode != 0:
        raise L.LedgerError(f"git cat-file {blob} failed")
    return r.stdout


def earlier_runs(ref: str = "HEAD") -> tuple[set, list[dict]]:
    """(SEEN priced (row_id, market) keys, per-run provenance rows). Fail-closed (LedgerError)."""
    seen, runs = set(), []
    for track, path in UNIVERSE_PATHS.items():
        vs = L.versions(path, ref)
        if not vs:
            raise L.LedgerError(f"{path}: no committed version - the earlier universes cannot be proven")
        for c, blob in vs:
            u = pd.read_parquet(io.BytesIO(_blob_bytes(blob)), columns=["row_id", "market", "yes", "no"])
            k = u[u["yes"].notna() & u["no"].notna()]
            keys = set(zip(k["row_id"].astype(str), k["market"].astype(str)))
            seen |= keys
            runs.append({"kind": "universe", "track": track, "commit": c[:9], "rows": int(len(u)),
                         "priced": len(keys), "price_source": "own (no history existed)"})
    lb_commits = sorted({r["commit"] for r in L.build(ref)["runs"] if r["kind"] == "league_bars"})
    for c in lb_commits:
        for f in OWN_SNAPSHOT_FILES:
            changed = L._git("log", "--format=%H", f"{c}..{ref}", "--", f).split()
            row = {"kind": "league_bars_price_files", "commit": c, "file": f, "changed_since": len(changed)}
            if changed:
                old = pd.read_csv(io.BytesIO(subprocess.run(["git", "show", f"{c}:{f}"], cwd=ROOT,
                                                             capture_output=True).stdout), usecols=SNAP_KEY)
                now = pd.read_csv(ROOT / f, usecols=SNAP_KEY)
                gone = len(set(map(tuple, old.astype(str).values)) - set(map(tuple, now.astype(str).values)))
                row["keys_missing_today"] = gone
                if gone:
                    raise L.LedgerError(f"{f}: {gone} snapshot key(s) the league_bars run {c} could price are gone "
                                        "today - its priced rows cannot be proven from today's file")
            runs.append(row)
    return seen, runs


def mark_fresh(priced: pd.DataFrame, seen: set) -> pd.Series:
    keys = list(zip(priced["row_id"].astype(str), priced["market"].astype(str)))
    unseen = np.array([k not in seen for k in keys], dtype=bool)
    return pd.Series((priced["price_source"] == P.SRC_HIST).to_numpy() & unseen, index=priced.index)


def _runs(day: pd.DataFrame) -> list[dict]:
    """Maximal stretches of consecutive match dates whose rows are ALL fresh."""
    out, cur = [], None
    for d, r in day.iterrows():
        if r["all_fresh"]:
            cur = cur or {"start": d, "end": d, "rows": 0, "days": 0}
            cur["end"], cur["rows"], cur["days"] = d, cur["rows"] + int(r["n"]), cur["days"] + 1
        elif cur:
            out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    return out


def holdout_window(f: pd.DataFrame) -> dict:
    """f: one family's eligible rows with a boolean 'fresh' column. Returns the window (or NO_FRESH)."""
    if f.empty:
        return {"status": NO_FRESH, "reason": "no study-eligible row"}
    day = f.groupby("date")["fresh"].agg(all_fresh="all", n="size").sort_index()
    runs = _runs(day)
    big = [r for r in runs if r["rows"] >= MIN_RUN_ROWS]
    if not big:
        return {"status": NO_FRESH, "reason": f"no fresh run with >= {MIN_RUN_ROWS} rows "
                f"(largest {max((r['rows'] for r in runs), default=0)})", "fresh_runs": len(runs)}
    # protocol v2 (owner 2026-11-10 "Fix the rule"): the most recent run whose HOLDOUT PART gives at least one
    # league MIN_HOLDOUT_BETS rows; a later run that cannot host a 150-bet test is skipped (counted)
    run = split = None
    for r in reversed(big):
        in_run = f[(f["date"] >= r["start"]) & (f["date"] <= r["end"])]
        sp = LB.in_period_split(in_run["date"])
        if sp is None:
            continue
        ho = in_run[in_run["date"] >= sp]
        if (ho.groupby("league").size() >= LB.MIN_HOLDOUT_BETS).any():
            run, split = r, sp
            break
    if run is None:
        return {"status": NO_FRESH, "reason": f"no fresh run of >= {MIN_RUN_ROWS} rows whose holdout part gives "
                f"a league {LB.MIN_HOLDOUT_BETS} rows", "fresh_runs": len(runs)}
    skipped = [r for r in runs if r["start"] > run["end"]]
    return {"status": "OK", "run_start": run["start"], "run_end": run["end"], "run_rows": run["rows"],
            "holdout_start": pd.Timestamp(split), "holdout_end": run["end"], "selection_before": pd.Timestamp(split),
            "later_short_runs_skipped": len(skipped), "fresh_runs": len(runs)}


def split_family(f: pd.DataFrame, w: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(selection rows, holdout rows) of a family under its window. Disjoint by construction; asserted."""
    if w.get("status") != "OK":
        return f.iloc[0:0], f.iloc[0:0]
    sel = f[f["date"] < w["selection_before"]]
    ho = f[(f["date"] >= w["holdout_start"]) & (f["date"] <= w["holdout_end"])]
    if not ho["fresh"].all():
        raise AssertionError("a holdout row is not fresh")
    if set(sel["row_id"]) & set(ho["row_id"]) or (len(sel) and sel["date"].max() >= w["holdout_start"]):
        raise AssertionError("selection and holdout overlap")
    return sel, ho


def families(priced_by_track: dict, seen: set) -> list[dict]:
    """[{track, mside, frame (eligible rows + fresh), window}] for every family of the priced universes."""
    out = []
    for track, priced in priced_by_track.items():
        fr = mark_fresh(priced, seen)
        for mside, (market, _side) in S.MARKET_SIDES.items():
            pm = priced[priced["market"] == market]
            if pm.empty:
                continue
            f = S.family_frame(priced, mside)
            fresh = dict(zip(pm["row_id"].astype(str), fr[pm.index]))
            f = f.assign(fresh=f["row_id"].astype(str).map(fresh).fillna(False).astype(bool))
            out.append({"track": track, "mside": mside, "frame": f, "window": holdout_window(f)})
    return out


def coverage(fams: list[dict]) -> tuple[list[dict], list[dict]]:
    """(per family, per family x league) fresh-holdout coverage: counts and dates only."""
    fam_rows, lg_rows = [], []
    for x in fams:
        f, w = x["frame"], x["window"]
        sel, ho = split_family(f, w)
        cells150 = int((ho.groupby("league").size() >= LB.MIN_HOLDOUT_BETS).sum()) if len(ho) else 0
        d = (lambda t: None if t is None or pd.isna(t) else str(pd.Timestamp(t).date()))
        fam_rows.append({"track": x["track"], "mside": x["mside"], "status": w["status"],
                         "reason": w.get("reason", ""), "eligible_rows": int(len(f)), "fresh_rows": int(f["fresh"].sum()),
                         "holdout_from": d(w.get("holdout_start")), "holdout_to": d(w.get("holdout_end")),
                         "holdout_rows": int(len(ho)), "selection_rows": int(len(sel)),
                         "holdout_leagues": int(ho["league"].nunique()) if len(ho) else 0,
                         "cells_ge_150": cells150})
        if len(ho):
            for lg, g in ho.groupby("league"):
                lg_rows.append({"track": x["track"], "mside": x["mside"], "league": lg, "holdout_rows": int(len(g)),
                                "selection_rows": int((sel["league"] == lg).sum()),
                                "ge_150": bool(len(g) >= LB.MIN_HOLDOUT_BETS)})
    return fam_rows, lg_rows


def protocol() -> tuple[dict, str]:
    """(protocol, sha256 of the file's text with LF line ends - the same on every checkout, CRLF or not)."""
    b = PROTOCOL_FILE.read_bytes().replace(b"\r\n", b"\n")
    import hashlib
    return json.loads(b.decode("utf-8")), hashlib.sha256(b).hexdigest()


def price_tiers() -> list[str]:
    """The collector tiers the protocol pins as BET-price sources (v4: exactly BT)."""
    t = protocol()[0]["data"].get("price_tiers")
    if not t:
        raise SystemExit("the fresh-holdout protocol pins no price_tiers")
    return list(t)


def close_tiers() -> list[str]:
    """The tiers read ONLY for the closing pair (CLV) - never a bet price (v4: B1). [] = none declared."""
    return list(protocol()[0]["data"].get("close_tiers") or [])


def own_backfill_excluded() -> bool:
    """v4: own 00:00-stamped backfill pairs are excluded from every bar / verdict."""
    return bool(protocol()[0]["data"].get("exclude_own_backfill"))


def price_stamp():
    """What prepare stamps into <priced>/price_tiers.json and the study requires: the bet tiers alone (v3) or, from
    v4, {bet_tiers, close_tiers, own_backfill_excluded}."""
    if close_tiers() or own_backfill_excluded():
        return {"bet_tiers": price_tiers(), "close_tiers": close_tiers(), "own_backfill_excluded": own_backfill_excluded()}
    return price_tiers()


def evidence_label() -> str:
    """The label every fresh-holdout verdict of this protocol version carries ("" = none)."""
    return str(protocol()[0].get("verdict_label") or "")


def protocol_committed() -> tuple[bool, str]:
    rel = PROTOCOL_FILE.relative_to(ROOT).as_posix()
    if not L._git("ls-files", "--", rel).strip():
        return False, f"{rel} is not committed"
    if L._git("status", "--porcelain", "--", rel).strip():
        return False, f"{rel} has uncommitted changes"
    return True, ""
