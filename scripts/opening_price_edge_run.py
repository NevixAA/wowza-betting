"""`run` and `timing` of scripts/opening_price_edge.py (protocol config/opening_price_edge_protocol.json).

Every number written here is a count, a mean of P/L or CLV, a CI or a p-value. No price leaves --local.
"""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import opening_price_edge as M  # noqa: E402
from src import oddsapi_history_adapter as A  # noqa: E402
from src.bars_opt import study as S  # noqa: E402
from src.market_edge import power_devig  # noqa: E402

Z95 = 1.96
MIN_BETS = 150
ALPHA = 0.025
ROLE_LABEL = {"open": "T-48h", "bet": "bet", "close": "close"}
ROLES = ("OPEN", "BET", "FD_PRE")
BUCKETS = ((36.0, 1e9, "mv_gt36h"), (18.0, 36.0, "mv_18_36h"), (8.0, 18.0, "mv_8_18h"),
           (3.0, 8.0, "mv_3_8h"), (0.5, 3.0, "mv_0.5_3h"))
GIT = "C:/Program Files/Git/cmd/git.exe"


def git(*args) -> str:
    return subprocess.run([GIT, "-C", str(ROOT), *args], check=True, capture_output=True, text=True,
                          encoding="utf-8").stdout


def blob_of(path: Path) -> str:
    return git("hash-object", str(path)).strip()


# ── guards ───────────────────────────────────────────────────────────────────────────────────

def check_pins(cfg: dict) -> list[str]:
    bad = []
    led = json.loads(M.LEDGER.read_text(encoding="utf-8")) if M.LEDGER.exists() else {}
    if led.get("protocol_sha256") != M.sha_lf(M.PROTOCOL):
        bad.append("protocol hash is not the registered one")
    d = cfg["data"]
    for t, meta in d["predictions"].items():
        import hashlib
        if hashlib.sha256(Path(meta["path"]).read_bytes()).hexdigest() != meta["sha256"]:
            bad.append(f"predictions {t} sha256 differs")
    st = subprocess.run([GIT, "-C", d["oddsapi_store"]["path"], "rev-parse", "HEAD"], capture_output=True,
                        text=True).stdout.strip()
    if st != d["oddsapi_store"]["commit"]:
        bad.append(f"store commit {st[:9]} is not the pinned one")
    if blob_of(ROOT / d["adapter"]["file"]) != d["adapter"]["git_blob"]:
        bad.append("adapter is not the pinned blob")
    if blob_of(ROOT / d["fd_prematch"]["file"]) != d["fd_prematch"]["git_blob"]:
        bad.append("fd_prematch is not the pinned blob")
    if blob_of(ROOT / "config" / "fd_raw_crosswalk.json") != d["fd_prematch"]["crosswalk_git_blob"]:
        bad.append("fd crosswalk is not the pinned blob")
    return bad


def load_top3(cfg: dict, local: Path):
    blob = cfg["data"]["calibration_code"]["git_blob"]
    git("fetch", "-q", "origin", cfg["data"]["calibration_code"]["branch"])
    src = git("cat-file", "-p", blob)
    f = local / f"top3_ranking_{blob[:12]}.py"
    f.write_text(src, encoding="utf-8", newline="\n")
    if blob_of(f) != blob:
        raise SystemExit("top3_ranking blob mismatch")
    spec = importlib.util.spec_from_file_location("top3_ranking_pinned", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── priced side frame ────────────────────────────────────────────────────────────────────────

def fair_yes(a, b) -> np.ndarray:
    out = np.full(len(a), np.nan)
    for i, (x, y) in enumerate(zip(np.asarray(a, float), np.asarray(b, float))):
        if np.isfinite(x) and np.isfinite(y):
            v = power_devig(float(x), float(y))
            out[i] = np.nan if v is None else float(v)
    return out


def wide(u: pd.DataFrame, local: Path) -> pd.DataFrame:
    s = pd.read_parquet(local / "summary_main.parquet")
    base = u[["row_id", "league", "date", "home", "away", "market", "track", "stage", "y", "p"]].copy()
    w = base.copy()
    for role, label in ROLE_LABEL.items():
        ss = A.prices_for(s, role)
        ss = ss[ss["snapshot_label"] == label]
        pr = A.attach_to_universe(base, ss, purpose=role)
        for side in ("yes", "no"):
            w[f"{role}_{side}"] = pr[f"oh_{role}_{side}_median"].to_numpy(float)
            w[f"{role}_{side}_best"] = pr[f"oh_{role}_{side}_best"].to_numpy(float)
        w[f"{role}_fair"] = fair_yes(w[f"{role}_yes"], w[f"{role}_no"])
    fd = pd.read_parquet(local / "fd_ou25.parquet")
    is_fd = (w["track"] == "standard") & (w["market"] == "over25")
    fdm = w.loc[is_fd, ["row_id"]].merge(fd, on="row_id", how="left")
    for c_out, c_in in (("fd_yes", "yes"), ("fd_no", "no"), ("fdc_yes", "yes_c"), ("fdc_no", "no_c")):
        w[c_out] = np.nan
        w.loc[is_fd, c_out] = fdm[c_in].to_numpy(float)
    w["fd_fair"] = fair_yes(w["fd_yes"], w["fd_no"])
    w["fdc_fair"] = fair_yes(w["fdc_yes"], w["fdc_no"])
    # MOVEMENT (B4 featured totals): fair per snapshot, averaged inside each hours-before bucket
    b4 = pd.read_parquet(local / "summary_b4.parquet")
    b4 = b4[b4["market"] == "over25"]
    pv = b4.pivot_table(index=["fixture_id", "snapshot_label", "hours_before"], columns="side", values="median",
                        aggfunc="first").reset_index()
    pv = pv.dropna(subset=["yes", "no"])
    pv["f"] = fair_yes(pv["yes"], pv["no"])
    for lo, hi, name in BUCKETS:
        g = pv[(pv["hours_before"] > lo) & (pv["hours_before"] <= hi)].groupby("fixture_id")["f"].mean()
        w[name] = np.where(w["market"] == "over25", w["row_id"].map(g), np.nan)
    return w


def sides(w: pd.DataFrame, role: str) -> pd.DataFrame:
    """One row per (universe row, side) priced in `role` (OPEN -> open, BET -> bet, FD_PRE -> fd)."""
    k = {"OPEN": "open", "BET": "bet", "FD_PRE": "fd"}[role]
    ok = np.isfinite(w[f"{k}_fair"])
    if role == "FD_PRE":
        ok &= (w["track"] == "standard") & (w["market"] == "over25")
    d = w[ok]
    out = []
    for side in ("yes", "no"):
        flip = side == "no"
        f = d[f"{k}_fair"].to_numpy(float)
        o = pd.DataFrame({
            "track": d["track"].to_numpy(), "league": d["league"].to_numpy(), "market": d["market"].to_numpy(),
            "side": side, "row_id": d["row_id"].to_numpy(), "date": d["date"].to_numpy(), "stage": d["stage"].to_numpy(),
            "p_used": (1 - d["p"] if flip else d["p"]).to_numpy(float),
            "fair": 1 - f if flip else f,
            "odds": d[f"{k}_{side}"].to_numpy(float),
            "won": ((d["y"] == 0) if flip else (d["y"] == 1)).to_numpy(bool)})
        o["odds_best"] = d[f"{k}_{side}_best"].to_numpy(float) if role != "FD_PRE" else np.nan
        cf = d["fdc_fair"] if role == "FD_PRE" else d["close_fair"]
        o["fair_close"] = (1 - cf if flip else cf).to_numpy(float)
        for other in ("bet", "open"):
            ff = d[f"{other}_fair"].to_numpy(float)
            o[f"fair_{other}"] = 1 - ff if flip else ff
        for _, _, name in BUCKETS:
            ff = d[name].to_numpy(float)
            o[name] = 1 - ff if flip else ff
        out.append(o)
    r = pd.concat(out, ignore_index=True)
    r["mside"] = [M.SIDE_NAME[(m, s)] for m, s in zip(r["market"], r["side"])]
    r["date"] = pd.to_datetime(r["date"]).dt.normalize()
    r["role"] = role
    return r


def calibrate(r: pd.DataFrame, T) -> pd.DataFrame:
    """Walk-forward R2 lam per (track, league, mside): month M uses rows dated before M's first day."""
    r = r.copy()
    r["month"] = r["date"].dt.to_period("M").dt.to_timestamp()
    r["lam"] = np.nan
    for m in sorted(r["month"].unique()):
        prior = r[r["date"] < m]
        if prior.empty:
            continue
        lam = T._lams(prior[["track", "league", "mside", "fair", "p_used", "won"]])
        if lam.empty:
            continue
        key = dict(zip(zip(lam["track"], lam["league"], lam["mside"]), lam["lam"]))
        idx = r.index[r["month"] == m]
        r.loc[idx, "lam"] = [key.get(k, np.nan) for k in zip(r.loc[idx, "track"], r.loc[idx, "league"],
                                                              r.loc[idx, "mside"])]
    r["raw_flag"] = r["lam"].isna()
    r["p_cal"] = np.where(r["raw_flag"], r["p_used"], r["fair"] + r["lam"] * (r["p_used"] - r["fair"]))
    r["cal_ev"] = r["p_cal"] * r["odds"] - 1.0
    r["bet"] = r["cal_ev"] > 0
    r["pnl"] = np.where(r["won"], r["odds"] - 1.0, -1.0)
    r["pnl_best"] = np.where(r["won"], r["odds_best"] - 1.0, -1.0)
    r["clv"] = r["odds"] * r["fair_close"] - 1.0
    return r


# ── statistics ───────────────────────────────────────────────────────────────────────────────

def stat(x: np.ndarray, days: np.ndarray) -> dict:
    x = np.asarray(x, float)
    keep = np.isfinite(x)
    x, days = x[keep], np.asarray(days)[keep]
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": None, "se": None, "lo": None, "hi": None, "p": 1.0}
    mu = float(x.mean())
    se = S.max_se(x, pd.to_datetime(days).strftime("%Y-%m-%d").to_numpy()) if n >= 2 else float("nan")
    if not np.isfinite(se) or se <= 0:
        return {"n": n, "mean": mu, "se": None, "lo": None, "hi": None, "p": 1.0}
    p = 0.5 * math.erfc((mu / se) / math.sqrt(2))
    return {"n": n, "mean": mu, "se": se, "lo": mu - Z95 * se, "hi": mu + Z95 * se, "p": p}


def holm(ps: list[float]) -> list[float]:
    m = len(ps)
    order = np.argsort(ps, kind="mergesort")
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[j]))
        adj[j] = run
    return adj.tolist()


def family(u: pd.DataFrame) -> list[tuple]:
    from src import league_registry as LR
    from src.standard_repair import FD_CODES
    fd_files = {p.stem.rsplit("_", 1)[0] for p in (ROOT / "output" / "collect" / "fd_raw").glob("*_*.csv")}
    mem = []
    for (tr, lg), _ in u.groupby(["track", "league"]):
        keyed = bool(LR.odds_api_sport_keys_for(lg))
        for mk in M.MARKETS:
            for sd in ("yes", "no"):
                if keyed:
                    mem += [(tr, lg, mk, sd, "OPEN"), (tr, lg, mk, sd, "BET")]
                if tr == "standard" and mk == "over25" and FD_CODES.get(lg) in fd_files:
                    mem.append((tr, lg, mk, sd, "FD_PRE"))
    return mem


def verdict(roi: dict, clv: dict, p_holm: float) -> str:
    n = roi["n"]
    if n < MIN_BETS:
        return "INSUFFICIENT"
    if p_holm < ALPHA and roi["lo"] is not None and roi["lo"] > 0:
        return "PROVEN"
    if roi["mean"] > 0 and ((roi["lo"] or -1) > 0 or (clv["lo"] is not None and clv["lo"] > 0)):
        return "PROMISING"
    return "UNPROVEN"


def r4(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 4)


def run(cfg: dict, local: Path) -> int:
    bad = check_pins(cfg)
    if bad:
        print("refused:", "; ".join(bad))
        return 2
    T = load_top3(cfg, local)
    u = M.universe(cfg)
    mem = family(u)
    w = wide(u, local)
    frames = {role: calibrate(sides(w, role), T) for role in ROLES}
    cells, tests = [], []
    for key in mem:
        tr, lg, mk, sd, role = key
        r = frames[role]
        c = r[(r["track"] == tr) & (r["league"] == lg) & (r["market"] == mk) & (r["side"] == sd)]
        b = c[c["bet"]]
        days = b["date"].to_numpy()
        roi = stat(b["pnl"].to_numpy(), days)
        clv = stat(b["clv"].to_numpy(), days)
        nb = c[~c["bet"]]
        rec = {"track": tr, "league": lg, "market": mk, "side": sd, "role": role,
               "priced_rows": int(len(c)), "bets": roi["n"],
               "raw_share": r4(b["raw_flag"].mean()) if len(b) else None,
               "holdout_stage_share": r4((b["stage"] == "holdout").mean()) if len(b) else None,
               "first_bet": str(b["date"].min().date()) if len(b) else None,
               "last_bet": str(b["date"].max().date()) if len(b) else None,
               "roi": {k: r4(v) for k, v in roi.items()}, "clv": {k: r4(v) for k, v in clv.items()},
               "roi_best_price": r4(b["pnl_best"].mean()) if len(b) and role != "FD_PRE" else None,
               "mean_cal_ev": r4(b["cal_ev"].mean()) if len(b) else None,
               "clv_drift_selected": r4((b["fair_close"] - b["fair"]).mean()) if len(b) else None,
               "clv_drift_not_selected": r4((nb["fair_close"] - nb["fair"]).mean()) if len(nb) else None,
               "shorten_share": r4((b["fair_close"] > b["fair"])[b["fair_close"].notna()].mean())
               if b["fair_close"].notna().any() else None}
        path = {}
        for col in ["fair_bet", "fair_open"] + [n for _, _, n in BUCKETS] + ["fair_close"]:
            v = b["odds"] * b[col] - 1.0
            v = v[np.isfinite(v)]
            path[col] = {"n": int(len(v)), "edge": r4(v.mean()) if len(v) else None}
        rec["movement"] = path
        tests += [(len(cells), "roi", roi["p"]), (len(cells), "clv", clv["p"])]
        cells.append(rec)
    adj = holm([t[2] for t in tests])
    for (i, which, _), pa in zip(tests, adj):
        cells[i][which]["p_holm"] = r4(pa) if pa >= 1e-4 else float(f"{pa:.3g}")
    for c in cells:
        c["verdict"] = verdict(c["roi"], c["clv"], c["roi"]["p_holm"])
    by = {(c["track"], c["league"], c["market"], c["side"], c["role"]): c for c in cells}
    for c in cells:
        if c["role"] == "OPEN":
            bt = by.get((c["track"], c["league"], c["market"], c["side"], "BET"))
            c["early_only"] = c["verdict"] in ("PROVEN", "PROMISING") and (
                bt is None or bt["verdict"] not in ("PROVEN", "PROMISING"))
    pooled = []
    for role, r in frames.items():
        for (tr, mk, sd), c in r.groupby(["track", "market", "side"]):
            b = c[c["bet"]]
            pooled.append({"track": tr, "market": mk, "side": sd, "role": role, "priced_rows": int(len(c)),
                           "roi": {k: r4(v) for k, v in stat(b["pnl"].to_numpy(), b["date"].to_numpy()).items()},
                           "clv": {k: r4(v) for k, v in stat(b["clv"].to_numpy(), b["date"].to_numpy()).items()},
                           "raw_share": r4(b["raw_flag"].mean()) if len(b) else None})
    res = {"protocol": cfg["protocol"], "protocol_sha256": M.sha_lf(M.PROTOCOL), "holm_family_tests": len(tests),
           "holm_family_cells": len(cells), "cells": cells, "pooled_context": pooled,
           "verdict_counts": pd.Series([c["verdict"] for c in cells]).value_counts().to_dict()}
    (M.OUT / "results.json").write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("holm_family_tests", "holm_family_cells", "verdict_counts")}, indent=1))
    return 0


# ── Q4: live first-sight timing ──────────────────────────────────────────────────────────────

def timing(cfg: dict) -> int:
    git("fetch", "-q", "origin", "main")
    sha = git("rev-parse", "origin/main").strip()
    a = pd.read_csv(StringIO(git("show", "origin/main:output/bets_ledger.csv")))
    a["market"] = "over25"
    b = pd.read_csv(StringIO(git("show", "origin/main:output/side_bets_ledger.csv")))
    led = pd.concat([a, b], ignore_index=True)
    led = led[led["source"].astype(str) == "live"].copy()
    led["track"] = led["model_type"].astype(str)
    fx = led.rename(columns={"match_date": "date", "home_team": "home", "away_team": "away"})[
        ["league", "date", "home", "away"]].drop_duplicates().reset_index(drop=True)
    fx["date"] = pd.to_datetime(fx["date"]).dt.strftime("%Y-%m-%d")
    store = Path(cfg["data"]["oddsapi_store"]["path"])
    ctx = A.load_context(store, fx)
    empty = pd.DataFrame(columns=["event_id", "snapshot_ts", "commence_time"])
    fmap, unm = A.match_universe(fx, empty, ctx)
    ko = A.actual_kickoffs(empty, ctx)
    fid = A.fixture_id(fx.assign(date=fx["date"]))
    fx["kickoff"] = [ko.get(fmap.get(f)) if fmap.get(f) else pd.NaT for f in fid]
    led["date"] = pd.to_datetime(led["match_date"]).dt.strftime("%Y-%m-%d")
    led = led.merge(fx.rename(columns={"home": "home_team", "away": "away_team"}),
                    on=["league", "date", "home_team", "away_team"], how="left")
    gen = pd.to_datetime(led["generated_at"], utc=True, errors="coerce")
    led["hours"] = (pd.to_datetime(led["kickoff"], utc=True) - gen).dt.total_seconds() / 3600
    out = {"ledger_commit": sha, "first_sight_rows": int(len(led)),
           "with_kickoff": int(led["hours"].notna().sum()),
           "negative_hours_rows": int((led["hours"] < 0).sum()), "groups": []}
    for (tr, mk), g in led.groupby(["track", "market"]):
        h = g["hours"].dropna()
        h = h[h >= 0]
        out["groups"].append({"track": tr, "market": mk, "first_sights": int(len(g)), "with_kickoff": int(len(h)),
                              "median_hours": r4(h.median()) if len(h) else None,
                              "share_ge_48h": r4((h >= 48).mean()) if len(h) else None,
                              "share_ge_125h": r4((h >= 125).mean()) if len(h) else None,
                              "share_ge_153h": r4((h >= 153).mean()) if len(h) else None,
                              "share_lt_24h": r4((h < 24).mean()) if len(h) else None})
    h = led["hours"].dropna()
    h = h[h >= 0]
    out["all"] = {"n": int(len(h)), "median_hours": r4(h.median()), "share_ge_48h": r4((h >= 48).mean()),
                  "share_ge_125h": r4((h >= 125).mean()), "share_ge_153h": r4((h >= 153).mean()),
                  "quantiles_hours": {str(q): r4(h.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}}
    out["generated_month_counts"] = gen.dt.strftime("%Y-%m").value_counts().sort_index().to_dict()
    (M.OUT / "timing.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "groups"}, indent=1, default=str))
    print(pd.DataFrame(out["groups"]).to_string(index=False))
    return 0
