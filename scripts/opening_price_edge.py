"""Opening-price edge research (owner 2026-11-15 "yes"): does the CLEAN model beat the OPEN (T-48h) price where it
does not beat the BET-tier (BT) price, and is betting early worth it per league x market x side?

    python scripts/opening_price_edge.py prepare --local <dir outside the repo>      # prices + counts, no effects
    python scripts/opening_price_edge.py run     --local <dir>                      # refuses unless the protocol hash
                                                                                    # is registered in the ledger
    python scripts/opening_price_edge.py timing                                     # Q4: live first-sight timing

PROTOCOL: config/opening_price_edge_protocol.json, registered (sha256, LF) in
config/opening_price_edge_protocol_ledger.json BEFORE `run` computes any effect. `run` refuses a protocol whose
hash is not the registered one.

LICENCE: OddsAPI prices are read ONLY through src/oddsapi_history_adapter.py (attach / prices_for / attach_to_universe
with explicit roles) and every priced intermediate stays in --local (refused when inside this public repo). The repo
receives counts, ROI, CLV, CI and p-values only (output/research/opening_price_edge/*.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import oddsapi_history_adapter as A  # noqa: E402
from src.market_edge import power_devig  # noqa: E402

PROTOCOL = ROOT / "config" / "opening_price_edge_protocol.json"
LEDGER = ROOT / "config" / "opening_price_edge_protocol_ledger.json"
OUT = ROOT / "output" / "research" / "opening_price_edge"
MARKETS = ("over15", "over25", "over35", "btts")
SIDE_NAME = {("over15", "yes"): "over15_over", ("over15", "no"): "over15_under",
             ("over25", "yes"): "over25_over", ("over25", "no"): "over25_under",
             ("over35", "yes"): "over35_over", ("over35", "no"): "over35_under",
             ("btts", "yes"): "btts_yes", ("btts", "no"): "btts_no"}


def sha_lf(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def proto() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def local_dir(p: str) -> Path:
    d = Path(p).resolve()
    if ROOT.resolve() in (d, *d.parents):
        raise SystemExit("refused: priced intermediates may not be written inside the public repo")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── universe ─────────────────────────────────────────────────────────────────────────────────

def universe(cfg: dict) -> pd.DataFrame:
    """Clean walk-forward / model-holdout rows of the champion recipe, the four markets, dated before dev_end."""
    frames = []
    for track, meta in cfg["data"]["predictions"].items():
        d = pd.read_parquet(meta["path"])
        d = d[(d["candidate"] == cfg["data"]["candidate"]) & d["market"].isin(MARKETS)]
        d = d[pd.to_datetime(d["date"]) < pd.Timestamp(cfg["data"]["dev_end_exclusive"])]
        frames.append(d)
    u = pd.concat(frames, ignore_index=True)
    parts = u["row_id"].astype(str).str.split("|", expand=True)
    if parts.shape[1] != 4:
        raise SystemExit("a row_id does not split into date|league|home|away")
    u["home"], u["away"] = parts[2].to_numpy(), parts[3].to_numpy()
    u["date"] = pd.to_datetime(u["date"]).dt.normalize()
    return u.reset_index(drop=True)


# ── prepare: prices (LOCAL) + coverage counts ──────────────────────────────────────────────────

def prepare(cfg: dict, local: Path) -> dict:
    u = universe(cfg)
    store = Path(cfg["data"]["oddsapi_store"]["path"])
    fx = u.drop_duplicates("row_id")[["row_id", "league", "date", "home", "away"]].reset_index(drop=True)
    odds = A.load_store(store)
    ctx = A.load_context(store, fx)
    cov: dict = {"universe_rows": int(len(u)), "fixtures": int(len(fx))}

    main = odds[odds["tier"].isin(["BT", "C1", "B1"]) & odds["market"].isin(["btts", "alternate_totals"])]
    long, s, unmatched = A.attach(fx, main, ctx)
    cov["main_dropped"] = long.attrs.get("dropped", {})
    cov["main_unmatched"] = unmatched["kind"].value_counts().to_dict()
    s = s[(s["team"] == "") & (s["player"] == "") & s["market"].isin(MARKETS)]
    s.to_parquet(local / "summary_main.parquet")
    del long, main

    mv = odds[(odds["tier"] == "B4") & (odds["market"] == "totals")]
    del odds
    long4, s4, _ = A.attach(fx, mv, ctx)
    cov["b4_dropped"] = long4.attrs.get("dropped", {})
    s4 = s4[(s4["team"] == "") & (s4["player"] == "") & s4["market"].isin(MARKETS)]
    # hours before the fixture's own (actual) kickoff of each featured snapshot: the labels are day-timed
    ko = pd.to_datetime(long4["actual_kickoff"], utc=True)
    ts = pd.to_datetime(long4["snapshot_ts"], utc=True)
    hrs = long4.assign(h=(ko - ts).dt.total_seconds() / 3600.0).groupby(
        ["fixture_id", "snapshot_label"])["h"].median().rename("hours_before").reset_index()
    s4 = s4.merge(hrs, on=["fixture_id", "snapshot_label"], how="left")
    s4.to_parquet(local / "summary_b4.parquet")
    del long4, mv

    # football-data pre-match O/U 2.5 through the certified join (Standard only)
    from src import fd_prematch as F
    aliases = json.loads(Path(cfg["data"]["fd_prematch"]["aliases"]).read_text(encoding="utf-8"))
    o25 = u[(u["track"] == "standard") & (u["market"] == "over25")][["row_id", "league", "date"]].drop_duplicates("row_id")
    fdp, rep = F.certified_ou25(o25, aliases, ROOT / cfg["data"]["fd_prematch"]["fd_raw"])
    fdp[["row_id", "yes", "no", "yes_c", "no_c", "price_book"]].to_parquet(local / "fd_ou25.parquet")
    cov["fd_join"] = {lg: {k: r.get(k) for k in ("served", "joined", "join_rate")} for lg, r in rep.items()}

    # coverage counts per track x league x market x role (rows with a two-sided median pair)
    rows = []
    for role, label in (("bet", "bet"), ("open", "T-48h"), ("close", "close")):
        ss = A.prices_for(s, role)
        ss = ss[ss["snapshot_label"] == label]
        pr = A.attach_to_universe(u[["row_id", "league", "date", "home", "away", "market", "track"]], ss, purpose=role)
        ok = pr[f"oh_{role}_yes_median"].notna() & pr[f"oh_{role}_no_median"].notna()
        g = pr[ok].groupby(["track", "league", "market"]).size()
        rows += [{"track": k[0], "league": k[1], "market": k[2], "role": role, "rows": int(v)} for k, v in g.items()]
    fd_ok = fdp["yes"].notna() & fdp["no"].notna()
    for lg, v in fdp[fd_ok].groupby("league").size().items():
        rows.append({"track": "standard", "league": lg, "market": "over25", "role": "fd_pre", "rows": int(v)})
    cov["priced_rows"] = rows
    return cov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "run", "timing"])
    ap.add_argument("--local", default=None)
    a = ap.parse_args(argv)
    cfg = proto()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "prepare":
        cov = prepare(cfg, local_dir(a.local))
        (OUT / "coverage.json").write_text(json.dumps(cov, indent=1, default=str) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in cov.items() if k != "priced_rows"}, indent=1, default=str))
        return 0
    if a.cmd == "run":
        from scripts.opening_price_edge_run import run  # noqa: E402
        return run(cfg, local_dir(a.local))
    from scripts.opening_price_edge_run import timing  # noqa: E402
    return timing(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
