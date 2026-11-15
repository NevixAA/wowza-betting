"""Sharp-agreement filter study (owner 2026-11-15: "yes"). Protocol: reports/sharp_filter/protocol.json.

QUESTION. Do the clean model's O/U tips (an approved cell's own bar, TP-2026-11-15-bade2c0b) perform better
when Pinnacle / the multi-book market moved the same way before the bet? And does following the sharp move
alone (no model) already do as well?

LICENCE. The Odds API history is PRIVATE (wowza-exec branch odds-history) and this repo is PUBLIC. Every
per-row frame goes to --work, which must be OUTSIDE this repo (assert_outside_repo). The repo receives only
counts, ROI, CI, p-values and lead-time statistics (results.json, the report).

READ RULES. Prices are read only through src.oddsapi_history_adapter.attach (post-kickoff quotes, quotes
<= 1.01 and quotes without a snapshot time are dropped there). Roles: B4's intermediate snapshots are
MOVEMENT, B4's last snapshot and B1 are CLOSE. A CLOSE snapshot is never a signal input and never a bet price;
it is read only for CLV. Tier BT (role BET) is read only for the timing analysis (question 4).

STEPS
    python scripts/sharp_filter_study.py build    --work DIR     # private frames (no outcomes read)
    python scripts/sharp_filter_study.py coverage --work DIR     # counts only, before registration
    python scripts/sharp_filter_study.py evaluate --work DIR     # refuses unless the protocol is committed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import oddsapi_history_adapter as A  # noqa: E402
from src.market_edge import power_devig  # noqa: E402

PROTOCOL_FILE = ROOT / "reports" / "sharp_filter" / "protocol.json"
RESULTS_FILE = ROOT / "reports" / "sharp_filter" / "results.json"
PROPOSAL_FILE = ROOT / "output" / "threshold_proposals" / "bars_oddsapi_hist" / "2026-11_bade2c0b.json"


def protocol() -> dict:
    return json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))


def protocol_sha() -> str:
    return hashlib.sha256(PROTOCOL_FILE.read_bytes()).hexdigest()


def _git(*a) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout


def protocol_committed() -> tuple[bool, str]:
    rel = PROTOCOL_FILE.relative_to(ROOT).as_posix()
    if not _git("ls-files", "--", rel).strip():
        return False, f"{rel} is not committed"
    if _git("status", "--porcelain", "--", rel).strip():
        return False, f"{rel} has uncommitted changes"
    return True, _git("log", "-1", "--format=%H", "--", rel).strip()


def assert_outside_repo(p: Path) -> Path:
    p = Path(p).resolve()
    if p == ROOT or ROOT in p.parents:
        raise SystemExit(f"--work must be outside the public repo ({ROOT}): licensed prices stay private")
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── universe ────────────────────────────────────────────────────────────────────────────────────

def load_universe(P: dict) -> pd.DataFrame:
    """Clean walk-forward predictions (both tracks), the protocol's candidate and markets, date < dev_end."""
    parts = []
    for f in P["data"]["predictions"].values():
        d = pd.read_parquet(f)
        parts.append(d[(d["candidate"] == P["data"]["candidate"]) & d["market"].isin(P["data"]["model_markets"])])
    u = pd.concat(parts, ignore_index=True)
    u = u[pd.to_datetime(u["date"]) < pd.Timestamp(P["data"]["dev_end_exclusive"])].copy()
    if u.duplicated(["row_id", "market"]).any():
        raise SystemExit("a fixture x market has two predictions: refusing")
    s = u["row_id"].str.split("|", expand=True)
    if s.shape[1] != 4:
        raise SystemExit("row_id is not date|league|home|away")
    u["home"], u["away"] = s[2], s[3]
    return u.reset_index(drop=True)


def load_results(fixtures: pd.DataFrame, P: dict) -> tuple[pd.DataFrame, dict]:
    """Full-time goals for the 1X2 market-only arm: v9 output/fd_history.parquet joined EXACTLY on
    (date, league, home, away). A joined row whose goals contradict the model universe's own over25 outcome
    is a data conflict: excluded and counted. No result -> no 1X2 settlement (counted)."""
    f = pd.read_parquet(ROOT / P["data"]["results"].split(" ")[0])     # the path the protocol names first
    f["date"] = pd.to_datetime(f["date"])
    f = f.rename(columns={"home_team": "home", "away_team": "away"})
    f = f[f["home_goals"].notna() & f["away_goals"].notna()]
    f = f.drop_duplicates(["date", "league", "home", "away"], keep=False)
    fx = fixtures.copy()
    fx["date"] = pd.to_datetime(fx["date"])
    m = fx.merge(f[["date", "league", "home", "away", "home_goals", "away_goals"]],
                 on=["date", "league", "home", "away"], how="left")
    st = {"fixtures": int(len(fx)), "with_result": int(m["home_goals"].notna().sum())}
    y25 = m["y_over25"]
    conflict = m["home_goals"].notna() & y25.notna() & (((m["home_goals"] + m["away_goals"]) > 2.5).astype(float) != y25)
    st["over25_conflict_excluded"] = int(conflict.sum())
    m.loc[conflict, ["home_goals", "away_goals"]] = np.nan
    hg, ag = m["home_goals"], m["away_goals"]
    m["result_1x2"] = np.where(hg.isna(), None, np.where(hg > ag, "home", np.where(hg < ag, "away", "draw")))
    st["settleable_1x2"] = int(m["result_1x2"].notna().sum())
    return m[["row_id", "result_1x2"]], st


# ── prices (private) ────────────────────────────────────────────────────────────────────────────

def attach_prices(u: pd.DataFrame, P: dict) -> tuple[pd.DataFrame, dict]:
    store = Path(P["data"]["store_root"])
    o = A.load_store(store)
    keep = pd.Series(False, index=o.index)
    for tier, markets in P["data"]["tiers"].items():
        keep |= (o["tier"] == tier) & o["market"].isin(markets)
    o = o[keep].reset_index(drop=True)
    fx = u.drop_duplicates("row_id")[["row_id", "date", "league", "home", "away"]]
    ctx = A.load_context(store, fx)
    long, _summary, unmatched = A.attach(fx, o, ctx)
    st = {"store_rows_read": int(len(o)), "dropped_by_adapter": long.attrs.get("dropped", {}),
          "fixtures": int(len(fx)), "fixtures_matched": int(long["fixture_id"].nunique()) if len(long) else 0,
          "unmatched": unmatched["kind"].value_counts().to_dict() if len(unmatched) else {}}
    return long, st


def _utc(s):
    return pd.to_datetime(s, utc=True, errors="coerce")


def ou_quotes(long: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One book's two-sided full-match total pair per (fixture, tier, role, snapshot, book, line); a book
    quoting two prices for one outcome in one snapshot is ambiguous (dropped, counted)."""
    d = long[long["provider_market"].isin(["totals", "alternate_totals"]) & long["team"].isna()
             & long["player"].isna() & long["side"].isin(["yes", "no"]) & long["line"].notna()].copy()
    d["price"] = pd.to_numeric(d["price"], errors="coerce")
    d["ts"], d["ko"] = _utc(d["snapshot_ts"]), _utc(d["actual_kickoff"])
    key = ["fixture_id", "tier", "role", "ts", "bookmaker", "line"]
    n = d.groupby(key + ["side"])["price"].transform("nunique")
    amb = d.loc[n > 1, key].drop_duplicates()
    if len(amb):
        d = d.merge(amb.assign(_a=1), on=key, how="left")
        d = d[d["_a"].isna()].drop(columns="_a")
    w = d.pivot_table(index=key + ["ko", "league"], columns="side", values="price", aggfunc="first").reset_index()
    w = w.dropna(subset=["yes", "no"])
    w["fair_over"] = [power_devig(a, b) for a, b in zip(w["yes"], w["no"])]
    w = w[w["fair_over"].notna()].copy()
    w["fair_over"] = w["fair_over"].astype(float)
    w["lead_h"] = (w["ko"] - w["ts"]).dt.total_seconds() / 3600
    return w.reset_index(drop=True), {"ambiguous_groups": int(len(amb))}


def x12_quotes(long: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One book's complete home / draw / away triple per (fixture, tier, role, snapshot, book); fair
    probabilities by proportional normalisation of that triple (protocol signal.devig_1x2)."""
    d = long[(long["market"] == "1x2") & long["side"].isin(["home", "draw", "away"])].copy()
    d["price"] = pd.to_numeric(d["price"], errors="coerce")
    d["ts"], d["ko"] = _utc(d["snapshot_ts"]), _utc(d["actual_kickoff"])
    key = ["fixture_id", "tier", "role", "ts", "bookmaker"]
    n = d.groupby(key + ["side"])["price"].transform("nunique")
    amb = d.loc[n > 1, key].drop_duplicates()
    if len(amb):
        d = d.merge(amb.assign(_a=1), on=key, how="left")
        d = d[d["_a"].isna()].drop(columns="_a")
    w = d.pivot_table(index=key + ["ko", "league"], columns="side", values="price", aggfunc="first").reset_index()
    w = w.dropna(subset=["home", "draw", "away"])
    inv = 1 / w[["home", "draw", "away"]].to_numpy(float)
    fair = inv / inv.sum(axis=1, keepdims=True)
    for i, s in enumerate(("home", "draw", "away")):
        w[f"fair_{s}"] = fair[:, i]
    w["lead_h"] = (w["ko"] - w["ts"]).dt.total_seconds() / 3600
    return w.reset_index(drop=True), {"ambiguous_groups": int(len(amb))}


# ── snapshots and signals ──────────────────────────────────────────────────────────────────────

def movement_snapshots(ou: pd.DataFrame, x12: pd.DataFrame) -> pd.DataFrame:
    """Every distinct B4 MOVEMENT snapshot per fixture (any book, either market), with its lead in hours."""
    parts = [q[(q["tier"] == "B4") & (q["role"] == "MOVEMENT")][["fixture_id", "ts", "lead_h"]] for q in (ou, x12)]
    s = pd.concat(parts, ignore_index=True).drop_duplicates(["fixture_id", "ts"])
    return s.sort_values(["fixture_id", "ts"]).reset_index(drop=True)


def pick(snaps: pd.DataFrame, lo: float, hi: float, target: float) -> pd.Series:
    """{fixture_id: ts} of the snapshot whose lead lies in [lo, hi] closest to target (tie: the later one)."""
    s = snaps[(snaps["lead_h"] >= lo) & (snaps["lead_h"] <= hi)].copy()
    s["dist"] = (s["lead_h"] - target).abs()
    s = s.sort_values(["fixture_id", "dist", "ts"], ascending=[True, True, False])
    return s.drop_duplicates("fixture_id").set_index("fixture_id")["ts"]


def next_snapshot(snaps: pd.DataFrame, after: pd.Series) -> pd.Series:
    """{fixture_id: ts} of the first MOVEMENT snapshot strictly after `after` (pre-kickoff by the adapter)."""
    s = snaps.merge(after.rename("after").rename_axis("fixture_id").reset_index(), on="fixture_id")
    s = s[s["ts"] > s["after"]].sort_values(["fixture_id", "ts"])
    return s.drop_duplicates("fixture_id").set_index("fixture_id")["ts"]


def window_times(snaps: pd.DataFrame, W: dict) -> pd.DataFrame:
    o = pick(snaps, *W["open_lead_h"], W["open_target_h"]).rename("ts_open")
    g = pick(snaps, *W["signal_lead_h"], W["signal_target_h"]).rename("ts_sig")
    t = pd.concat([o, g], axis=1).dropna()
    t = t[t["ts_open"] < t["ts_sig"]]
    if W["bet"] == "signal_snapshot":
        t["ts_bet"] = t["ts_sig"]
    elif W["bet"] == "next_snapshot":
        t["ts_bet"] = next_snapshot(snaps, t["ts_sig"])
    else:
        raise ValueError(W["bet"])
    t = t.dropna().rename_axis("fixture_id").reset_index()
    lead = snaps.set_index(["fixture_id", "ts"])["lead_h"]
    for c in ("ts_open", "ts_sig", "ts_bet"):
        t["lead_" + c[3:]] = lead.reindex(pd.MultiIndex.from_arrays([t["fixture_id"], t[c]])).to_numpy()
    return t


def _at(q: pd.DataFrame, times: pd.DataFrame, col: str) -> pd.DataFrame:
    x = q.merge(times[["fixture_id", col]], left_on=["fixture_id", "ts"], right_on=["fixture_id", col])
    return x.drop(columns=[col])


def _main_line(x: pd.DataFrame, ref: float) -> pd.Series:
    x = x.assign(d=(x["line"] - ref).abs()).sort_values(["fixture_id", "d", "line"])
    return x.drop_duplicates("fixture_id").set_index("fixture_id")["line"]


def ou_moves(ou: pd.DataFrame, times: pd.DataFrame, P: dict) -> pd.DataFrame:
    """Per fixture: PIN (the sharp book) and CONS (median of the other books) change of the fair Over
    probability, open -> signal snapshot. Positive = toward Over. pin_kind SAME (a line quoted at both
    snapshots: the common line nearest the reference line) or LINE (no common line, the main line moved:
    value +inf / -inf, a whole move beyond any threshold)."""
    S = P["signal"]
    ref = S["ou_reference_line"]
    q = ou[(ou["tier"] == "B4") & (ou["role"] == "MOVEMENT")]
    a, b = _at(q, times, "ts_open"), _at(q, times, "ts_sig")
    pin = S["sharp_book"]
    out = pd.DataFrame(index=pd.Index(times["fixture_id"], name="fixture_id"))
    out["pin"], out["pin_kind"] = np.nan, None
    pa, pb = a[a["bookmaker"] == pin], b[b["bookmaker"] == pin]
    common = pa.merge(pb, on=["fixture_id", "line"], suffixes=("_o", "_s"))
    common["d"] = (common["line"] - ref).abs()
    common = common.sort_values(["fixture_id", "d", "line"]).drop_duplicates("fixture_id")
    out.loc[common["fixture_id"].to_numpy(), "pin"] = (common["fair_over_s"] - common["fair_over_o"]).to_numpy()
    out.loc[common["fixture_id"].to_numpy(), "pin_kind"] = "SAME"
    lo_, ls_ = _main_line(pa, ref), _main_line(pb, ref)
    both = lo_.index.intersection(ls_.index).difference(pd.Index(common["fixture_id"]))
    mv = np.sign(ls_[both] - lo_[both])
    mv = mv[mv != 0]
    out.loc[mv.index, "pin"] = mv.to_numpy() * np.inf
    out.loc[mv.index, "pin_kind"] = "LINE"
    ra = a[(a["bookmaker"] != pin) & (a["line"] == ref)]
    rb = b[(b["bookmaker"] != pin) & (b["line"] == ref)]
    c = ra.merge(rb, on=["fixture_id", "bookmaker"], suffixes=("_o", "_s"))
    c["dd"] = c["fair_over_s"] - c["fair_over_o"]
    g = c.groupby("fixture_id")["dd"].agg(["median", "count"])
    out["cons_books"] = g["count"].reindex(out.index)
    g = g[g["count"] >= S["consensus_min_books"]]
    out["cons"] = g["median"].reindex(out.index)
    return out.reset_index()


def x12_moves(x12: pd.DataFrame, times: pd.DataFrame, P: dict) -> pd.DataFrame:
    """Per fixture x side (home / draw / away): PIN and CONS change of the fair probability, open -> signal."""
    S = P["signal"]
    q = x12[(x12["tier"] == "B4") & (x12["role"] == "MOVEMENT")]
    a, b = _at(q, times, "ts_open"), _at(q, times, "ts_sig")
    pin = S["sharp_book"]
    rows = []
    for side in ("home", "draw", "away"):
        f = f"fair_{side}"
        p = a[a["bookmaker"] == pin][["fixture_id", f]].merge(b[b["bookmaker"] == pin][["fixture_id", f]],
                                                               on="fixture_id", suffixes=("_o", "_s"))
        pin_s = pd.Series((p[f + "_s"] - p[f + "_o"]).to_numpy(), index=p["fixture_id"].to_numpy())
        c = a[a["bookmaker"] != pin][["fixture_id", "bookmaker", f]].merge(
            b[b["bookmaker"] != pin][["fixture_id", "bookmaker", f]], on=["fixture_id", "bookmaker"],
            suffixes=("_o", "_s"))
        c["dd"] = c[f + "_s"] - c[f + "_o"]
        g = c.groupby("fixture_id")["dd"].agg(["median", "count"])
        g = g[g["count"] >= S["consensus_min_books"]]["median"]
        d = pd.DataFrame({"pin": pin_s, "cons": g})
        d.index.name = "fixture_id"
        d = d.reset_index()
        d["side"] = side
        d["pin_kind"] = np.where(d["pin"].notna(), "SAME", None)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def thresholds(m: pd.DataFrame, col: str, group: list[str], P: dict) -> pd.Series:
    """Per row: the protocol quantile of |move| among the SAME league's (and group's) fixtures dated STRICTLY
    earlier (prices only, never an outcome); NaN (WARMUP) while fewer than min_history earlier measured,
    finite moves exist. m carries league, date and `col`."""
    S = P["signal"]
    out = pd.Series(np.nan, index=m.index)
    a = m[col].abs()
    for _, g in m.groupby(["league"] + group):
        gi = g.index
        fin = np.isfinite(a.loc[gi].to_numpy(float))
        vals = np.sort(np.array([]))
        f_dates = g["date"].to_numpy()[fin]
        f_vals = a.loc[gi].to_numpy(float)[fin]
        order = np.argsort(f_dates, kind="stable")
        f_dates, f_vals = f_dates[order], f_vals[order]
        for day in np.unique(g["date"].to_numpy()):
            k = np.searchsorted(f_dates, day, side="left")          # strictly earlier dates only
            if k >= S["min_history"]:
                vals = f_vals[:k]
                out.loc[gi[g["date"].to_numpy() == day]] = float(np.quantile(vals, S["quantile"]))
    return out


def direction(v: pd.Series, thr: pd.Series) -> pd.Series:
    """+1 moved toward the side beyond the threshold, -1 against, 0 within it; NaN = NO_DATA / WARMUP."""
    v, thr = v.astype(float), thr.astype(float)
    out = pd.Series(np.nan, index=v.index)
    ok = v.notna() & thr.notna()
    out[ok & (v > thr)] = 1.0
    out[ok & (v < -thr)] = -1.0
    out[ok & (v.abs() <= thr)] = 0.0
    return out


CLASS = {1.0: "AGREE", -1.0: "DISAGREE", 0.0: "NONE"}


def classify_signal(pin_dir: pd.Series, cons_dir: pd.Series, signal: str) -> pd.Series:
    """AGREE / DISAGREE / NONE / MIXED / NO_DATA of one signal definition, from the side's directions."""
    if signal == "PIN":
        d = pin_dir.map(CLASS)
    elif signal == "CONS":
        d = cons_dir.map(CLASS)
    elif signal == "BOTH":
        same = pin_dir.notna() & cons_dir.notna() & (pin_dir == cons_dir)
        d = pd.Series(np.where(same, pin_dir.map(CLASS), np.where(pin_dir.notna() & cons_dir.notna(), "MIXED", None)),
                      index=pin_dir.index)
    else:
        raise ValueError(signal)
    return d.fillna("NO_DATA")


# ── bet and close prices (Pinnacle, one book's real pair; never a CLOSE snapshot as a bet price) ───

LINE_OF = {"over15": 1.5, "over25": 2.5, "over35": 3.5}


def ou_price_at(ou: pd.DataFrame, times: pd.DataFrame, line: float, book: str) -> pd.DataFrame:
    """The book's exact-line pair in the bet snapshot (role MOVEMENT only)."""
    q = ou[(ou["tier"] == "B4") & (ou["role"] == "MOVEMENT") & (ou["bookmaker"] == book) & (ou["line"] == line)]
    x = q.merge(times[["fixture_id", "ts_bet"]], left_on=["fixture_id", "ts"], right_on=["fixture_id", "ts_bet"])
    return x[["fixture_id", "yes", "no", "fair_over"]].drop_duplicates("fixture_id")


def ou_close(ou: pd.DataFrame, line: float, book: str) -> pd.DataFrame:
    """CLV only: the book's exact-line pair in the LATEST CLOSE-role snapshot, tier B1 (the match's own close)
    first, else B4's last snapshot. Carries ts_c so a close not later than the bet is discarded."""
    out = []
    for tier in ("B1", "B4"):
        q = ou[(ou["tier"] == tier) & (ou["role"] == "CLOSE") & (ou["bookmaker"] == book) & (ou["line"] == line)]
        q = q.sort_values("ts").drop_duplicates("fixture_id", keep="last")
        out.append(q[["fixture_id", "ts", "yes", "no"]].assign(close_tier=tier))
    c = pd.concat(out, ignore_index=True).drop_duplicates("fixture_id", keep="first")
    return c.rename(columns={"ts": "ts_c", "yes": "yes_c", "no": "no_c"})


def x12_price_at(x12: pd.DataFrame, times: pd.DataFrame, book: str) -> pd.DataFrame:
    q = x12[(x12["tier"] == "B4") & (x12["role"] == "MOVEMENT") & (x12["bookmaker"] == book)]
    x = q.merge(times[["fixture_id", "ts_bet"]], left_on=["fixture_id", "ts"], right_on=["fixture_id", "ts_bet"])
    return x[["fixture_id", "home", "draw", "away"]].drop_duplicates("fixture_id")


def x12_close(x12: pd.DataFrame, book: str) -> pd.DataFrame:
    q = x12[(x12["tier"] == "B4") & (x12["role"] == "CLOSE") & (x12["bookmaker"] == book)]
    q = q.sort_values("ts").drop_duplicates("fixture_id", keep="last")
    return q[["fixture_id", "ts", "home", "draw", "away"]].rename(
        columns={"ts": "ts_c", "home": "home_c", "draw": "draw_c", "away": "away_c"})


# ── model tips (the approved cell's own bar, src.bars_opt.serve.score) ──────────────────────────

def model_cells(P: dict) -> tuple[list[dict], list[dict]]:
    """MODEL-mode cells of the approved proposal for the protocol's market-sides; LEAGUE_WIDE cells bet every
    match (no model signal) and are listed, not tested in the model arm."""
    prop = json.loads(PROPOSAL_FILE.read_text(encoding="utf-8"))
    if prop.get("proposal_id") != P["tips"]["proposal_id"]:
        raise SystemExit("proposal id differs from the protocol")
    cells, skipped = [], []
    for e in prop["entries"]:
        for side, c in ((e.get("proposed") or {}).get("bars_opt") or {}).items():
            if c["mside"] not in P["tips"]["msides"]:
                continue
            (cells if c["mode"] == "MODEL" else skipped).append(c)
    return cells, skipped


def tips_for_cell(cell: dict, u: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    from src.bars_opt import serve as SV
    from src.bars_opt import study as S
    market, side = S.MARKET_SIDES[cell["mside"]]
    r = u[(u["track"] == cell["track"]) & (u["league"] == cell["league"]) & (u["market"] == market)]
    r = r.merge(price, left_on="row_id", right_on="fixture_id")
    if r.empty:
        return r
    if side == "yes":
        r["p_side"], r["fair"], r["odds"] = r["p"], r["fair_over"], r["yes"]
        r["won"] = r["y"] == 1
    else:
        r["p_side"], r["fair"], r["odds"] = 1 - r["p"], 1 - r["fair_over"], r["no"]
        r["won"] = r["y"] == 0
    r = r[r["y"].notna()].reset_index(drop=True)
    sc = SV.score(r[["league", "p_side", "fair", "odds"]], cell)
    r["tip"] = sc["tier"].astype(str).ne("") & sc["tier"].notna()
    r["measure"] = sc["measure"]
    return r[r["tip"]].assign(mside=cell["mside"], track=cell["track"], dir_sign=1 if side == "yes" else -1)


# ── statistics ─────────────────────────────────────────────────────────────────────────────────

def cell_stats(b: pd.DataFrame, seed: int) -> dict:
    from src.bars_opt import study as S
    r = S.score(b.assign(date=pd.to_datetime(b["date"])), seed=seed)
    if len(b):
        r["win_rate"] = round(float(b["won"].mean()), 4)
        r["win_minus_implied"] = round(float(b["won"].mean() - (1 / b["odds"]).mean()), 4)
    return r


def diff_test(a: pd.DataFrame, b: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """ROI(a) - ROI(b): match dates resampled jointly (one block bootstrap over the union of dates)."""
    if len(a) < 2 or len(b) < 2:
        return {"diff": None, "ci95": [None, None], "p_one_sided": None}
    da, db = pd.to_datetime(a["date"]).dt.strftime("%Y-%m-%d"), pd.to_datetime(b["date"]).dt.strftime("%Y-%m-%d")
    days = np.unique(np.r_[da.to_numpy(), db.to_numpy()])
    ix = {d: i for i, d in enumerate(days)}
    ia, ib = da.map(ix).to_numpy(), db.map(ix).to_numpy()
    sa, na = np.bincount(ia, a["pnl"].to_numpy(float), len(days)), np.bincount(ia, minlength=len(days)).astype(float)
    sb, nb = np.bincount(ib, b["pnl"].to_numpy(float), len(days)), np.bincount(ib, minlength=len(days)).astype(float)
    rng = np.random.default_rng(seed)
    c = rng.multinomial(len(days), np.full(len(days), 1 / len(days)), size=n_boot).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (c @ sa) / (c @ na) - (c @ sb) / (c @ nb)
    d = d[np.isfinite(d)]
    obs = float(a["pnl"].mean() - b["pnl"].mean())
    return {"diff": round(obs, 4), "ci95": [round(float(np.quantile(d, 0.025)), 4), round(float(np.quantile(d, 0.975)), 4)],
            "p_one_sided": round(float((int((d <= 0).sum()) + 1) / (len(d) + 1)), 6)}


# ── build (private frames; no outcome is read) ─────────────────────────────────────────────────

def build(work: Path) -> dict:
    P = protocol()
    u = load_universe(P)
    long, st = attach_prices(u, P)
    ou, s1 = ou_quotes(long)
    x12, s2 = x12_quotes(long)
    del long
    u.to_parquet(work / "universe.parquet")
    ou.to_parquet(work / "ou_quotes.parquet")
    x12.to_parquet(work / "x12_quotes.parquet")
    st |= {"ou_quotes": int(len(ou)), "x12_quotes": int(len(x12)), "ou_ambiguous": s1["ambiguous_groups"],
           "x12_ambiguous": s2["ambiguous_groups"]}
    (work / "build_stats.json").write_text(json.dumps(st, indent=1, default=str), encoding="utf-8")
    return st


def _fixture_frame(u: pd.DataFrame) -> pd.DataFrame:
    """One row per fixture: league, track, date, the over25 outcome (for the O/U market-only arm)."""
    f = u.sort_values("market").drop_duplicates("row_id")[["row_id", "league", "track", "date", "home", "away"]]
    y = u[u["market"] == "over25"].set_index("row_id")["y"]
    f["y_over25"] = f["row_id"].map(y)
    return f.rename(columns={"row_id": "fixture_id"}).reset_index(drop=True)


# ── evaluate ───────────────────────────────────────────────────────────────────────────────────

def evaluate(work: Path, seed: int = 0) -> dict:
    ok, commit = protocol_committed()
    if not ok:
        raise SystemExit(f"refusing: {commit}. Register (commit) the protocol before computing any effect.")
    P = protocol()
    from src import league_bars as LB
    from src.bars_opt import study as S
    u = pd.read_parquet(work / "universe.parquet")
    ou = pd.read_parquet(work / "ou_quotes.parquet")
    x12 = pd.read_parquet(work / "x12_quotes.parquet")
    fx = _fixture_frame(u)
    res, rst = load_results(fx.rename(columns={"fixture_id": "row_id"}), P)
    fx["result_1x2"] = fx["fixture_id"].map(res.set_index("row_id")["result_1x2"])
    fdate = fx.set_index("fixture_id")["date"]
    flg = fx.set_index("fixture_id")["league"]
    book = P["signal"]["sharp_book"]
    snaps = movement_snapshots(ou, x12)
    cells, skipped = model_cells(P)
    bets, timing, cover, live = [], {}, {}, None

    for wname, W in P["windows"].items():
        t = window_times(snaps, W)
        timing[wname] = {k: _q(t["lead_" + k]) for k in ("open", "sig", "bet")} | {"fixtures": int(len(t))}
        # directions, per signal, O/U (toward Over) and 1X2 (toward the side)
        om = ou_moves(ou, t, P)
        om["league"], om["date"] = om["fixture_id"].map(flg), om["fixture_id"].map(fdate)
        xm = x12_moves(x12, t, P)
        xm["league"], xm["date"] = xm["fixture_id"].map(flg), xm["fixture_id"].map(fdate)
        for m, grp in ((om, []), (xm, ["side"])):
            m.drop(index=m.index[m["league"].isna()], inplace=True)
            for col in ("pin", "cons"):
                m[col + "_thr"] = thresholds(m, col, grp, P)
                m[col + "_dir"] = direction(m[col], m[col + "_thr"])
        cover[wname] = {"ou_pin_kind": om["pin_kind"].value_counts(dropna=False).to_dict(),
                        "ou_pin_dir": om["pin_dir"].value_counts(dropna=False).to_dict(),
                        "ou_cons_dir": om["cons_dir"].value_counts(dropna=False).to_dict(),
                        "x12_pin_dir": xm["pin_dir"].value_counts(dropna=False).to_dict(),
                        "x12_cons_dir": xm["cons_dir"].value_counts(dropna=False).to_dict()}
        odir = om.set_index("fixture_id")
        # ---- model arm
        for mk, line in LINE_OF.items():
            price = ou_price_at(ou, t, line, book)
            close = ou_close(ou, line, book)
            for cell in [c for c in cells if S.MARKET_SIDES[c["mside"]][0] == mk]:
                tp = tips_for_cell(cell, u, price)
                if tp.empty:
                    continue
                tp = tp.merge(close, on="fixture_id", how="left")
                tp = tp.merge(t[["fixture_id", "ts_bet"]], on="fixture_id")
                late = tp["ts_c"].notna() & (tp["ts_c"] > tp["ts_bet"])
                tp["odds_c"] = np.where(late, np.where(tp["dir_sign"] == 1, tp["yes_c"], tp["no_c"]), np.nan)
                tp["pnl"] = np.where(tp["won"], tp["odds"] - 1.0, -1.0)
                for sig in P["signals"]:
                    pd_ = tp["fixture_id"].map(odir["pin_dir"]) * tp["dir_sign"]
                    cd_ = tp["fixture_id"].map(odir["cons_dir"]) * tp["dir_sign"]
                    cls = classify_signal(pd_.replace(-0.0, 0.0), cd_.replace(-0.0, 0.0), sig)
                    bets.append(tp.assign(arm="model", window=wname, signal=sig, cls=cls.to_numpy(),
                                          side=cell["mside"])[BET_COLS])
        # ---- market-only arm: O/U 2.5, bet the side the sharp move points to
        price = ou_price_at(ou, t, 2.5, book).merge(ou_close(ou, 2.5, book), on="fixture_id", how="left")
        price = price.merge(t[["fixture_id", "ts_bet"]], on="fixture_id")
        mo = om.merge(price, on="fixture_id").merge(fx[["fixture_id", "y_over25", "track"]], on="fixture_id")
        mo = mo[mo["y_over25"].notna()]
        late = mo["ts_c"].notna() & (mo["ts_c"] > mo["ts_bet"])
        for sig in P["signals"]:
            cls_over = classify_signal(mo["pin_dir"], mo["cons_dir"], sig)
            cls_under = classify_signal(-mo["pin_dir"], -mo["cons_dir"], sig)
            for sd, cl, od, oc, won in (("over25_over", cls_over, mo["yes"], mo["yes_c"], mo["y_over25"] == 1),
                                        ("over25_under", cls_under, mo["no"], mo["no_c"], mo["y_over25"] == 0)):
                k = (cl == "AGREE").to_numpy()
                b = pd.DataFrame({"fixture_id": mo["fixture_id"], "league": mo["league"], "date": mo["date"],
                                  "track": mo["track"], "odds": od.astype(float), "won": won,
                                  "odds_c": np.where(late, oc, np.nan)})[k]
                b["pnl"] = np.where(b["won"], b["odds"] - 1.0, -1.0)
                bets.append(b.assign(arm="market", window=wname, signal=sig, cls="FOLLOW", side=sd)[BET_COLS])
        # ---- market-only arm: 1X2, bet each side the sharp move points to
        xp = x12_price_at(x12, t, book).merge(x12_close(x12, book), on="fixture_id", how="left")
        xp = xp.merge(t[["fixture_id", "ts_bet"]], on="fixture_id")
        xx = xm.merge(xp, on="fixture_id").merge(fx[["fixture_id", "result_1x2", "track"]], on="fixture_id")
        xx = xx[xx["result_1x2"].notna()]
        late = (xx["ts_c"].notna() & (xx["ts_c"] > xx["ts_bet"])).to_numpy()
        for sig in P["signals"]:
            cl = classify_signal(xx["pin_dir"], xx["cons_dir"], sig)
            for sd in ("home", "draw", "away"):
                k = ((cl == "AGREE") & (xx["side"] == sd)).to_numpy()
                b = pd.DataFrame({"fixture_id": xx["fixture_id"], "league": xx["league"], "date": xx["date"],
                                  "track": xx["track"], "odds": xx[sd].astype(float),
                                  "won": xx["result_1x2"] == sd,
                                  "odds_c": np.where(late, xx[sd + "_c"], np.nan)})[k]
                b["pnl"] = np.where(b["won"], b["odds"] - 1.0, -1.0)
                bets.append(b.assign(arm="market", window=wname, signal=sig, cls="FOLLOW", side="1x2_" + sd)[BET_COLS])
        timing[wname]["move_path"] = move_path(ou, t, P)
        if wname == P["timing"]["primary_window"]:
            live = live_tip_rate(ou, u, cells, P, odir, t)

    B = pd.concat(bets, ignore_index=True)
    B.to_parquet(work / "bets.parquet")                     # private: carries prices
    out = score_cells(B, P, seed)
    out["timing"] = {"windows": timing, "live_tips": live}
    out["coverage"] = cover
    out["results_join"] = rst
    out["cells_model_mode"] = len(cells)
    out["cells_league_wide_skipped"] = sorted(f"{c['track']}|{c['mside']}|{c['league']}" for c in skipped)
    out["protocol_sha256"] = protocol_sha()
    out["protocol_commit"] = commit
    return out


BET_COLS = ["arm", "window", "signal", "cls", "side", "track", "league", "fixture_id", "date", "odds", "odds_c",
            "won", "pnl"]


def _q(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    return {"n": int(len(s)), "median": round(float(s.median()), 2), "p25": round(float(s.quantile(.25)), 2),
            "p75": round(float(s.quantile(.75)), 2)}


def move_path(ou: pd.DataFrame, t: pd.DataFrame, P: dict) -> dict:
    """How much of Pinnacle's open -> last-movement-snapshot change of the fair Over probability (same line) is
    already present at each lead-time bucket: mean |change| and the share of fixtures whose change so far has
    the sign of the full change. Prices only."""
    book, ref = P["signal"]["sharp_book"], P["signal"]["ou_reference_line"]
    q = ou[(ou["tier"] == "B4") & (ou["bookmaker"] == book)]
    o = q.merge(t[["fixture_id", "ts_open"]], left_on=["fixture_id", "ts"], right_on=["fixture_id", "ts_open"])
    o = o.assign(d=(o["line"] - ref).abs()).sort_values(["fixture_id", "d"]).drop_duplicates("fixture_id")
    path = q.merge(o[["fixture_id", "line", "fair_over", "ts_open"]].rename(columns={"fair_over": "f0"}),
                   on=["fixture_id", "line"])
    path = path[path["ts"] >= path["ts_open"]]
    path["chg"] = path["fair_over"] - path["f0"]
    last_mv = path[path["role"] == "MOVEMENT"].sort_values("ts").drop_duplicates("fixture_id", keep="last")
    full = last_mv.set_index("fixture_id")["chg"]
    close = path[path["role"] == "CLOSE"].sort_values("ts").drop_duplicates("fixture_id", keep="last")
    full_c = close.set_index("fixture_id")["chg"]
    path["full"] = path["fixture_id"].map(full)
    edges = P["timing"]["lead_buckets_h"]
    out = []
    for hi, lo in zip(edges[:-1], edges[1:]):
        g = path[(path["role"] == "MOVEMENT") & (path["lead_h"] <= hi) & (path["lead_h"] > lo)]
        g = g.sort_values("ts").drop_duplicates("fixture_id", keep="last")
        nz = g[g["full"] != 0]
        out.append({"lead_h": f"({lo},{hi}]", "fixtures": int(len(g)),
                    "mean_abs_change_pp": round(float(g["chg"].abs().mean() * 100), 3) if len(g) else None,
                    "same_sign_as_final_movement": round(float((np.sign(nz["chg"]) == np.sign(nz["full"])).mean()), 4)
                    if len(nz) else None})
    out.append({"lead_h": "close (CLV snapshot)", "fixtures": int(len(full_c)),
                "mean_abs_change_pp": round(float(full_c.abs().mean() * 100), 3) if len(full_c) else None})
    return {"pinnacle_ou_path": out}


def score_cells(B: pd.DataFrame, P: dict, seed: int) -> dict:
    """Family F1 (the decision family): ROI > 0 of every configuration cell with at least one bet - model
    arm AGREE / DISAGREE / NONE / ALL and market-only FOLLOW - Holm over all of them, the owner's strict
    PROVEN rule. Family F2 (reported, not a decision): differences AGREE-NONE, AGREE-DISAGREE, and model AGREE
    minus market-only FOLLOW of the same window x signal x league x side; Holm over F2."""
    from src import league_bars as LB
    from src.bars_opt import study as S
    rows = []
    m = B[B["arm"] == "model"]
    key = ["window", "signal", "track", "side", "league"]
    for k, g in m.groupby(key):
        for cls in ("AGREE", "DISAGREE", "NONE", "MIXED", "NO_DATA"):
            gg = g[g["cls"] == cls]
            rows.append(dict(zip(key, k)) | {"arm": "model", "cls": cls, "tested": cls in P["tested_classes"],
                                             "stats": cell_stats(gg, seed) if len(gg) else {"bets": 0}})
    first = P["signals"][0]
    for k, g in m[m["signal"] == first].groupby(["window", "track", "side", "league"]):
        rows.append(dict(zip(["window", "track", "side", "league"], k)) |
                    {"signal": "-", "arm": "model", "cls": "ALL", "tested": True, "stats": cell_stats(g, seed)})
    mk = B[B["arm"] == "market"]
    for k, g in mk.groupby(["window", "signal", "side", "league"]):
        rows.append(dict(zip(["window", "signal", "side", "league"], k)) |
                    {"track": g["track"].iloc[0], "arm": "market", "cls": "FOLLOW", "tested": True,
                     "stats": cell_stats(g, seed)})
    tested = [r for r in rows if r["tested"] and r["stats"].get("bets", 0) > 0
              and r["stats"].get("p_one_sided") is not None]
    p = np.array([r["stats"]["p_one_sided"] for r in tested], float)
    adj = LB.holm(p)
    floor = round(1.0 / (S.N_BOOT + 1), 4)
    n_boot = int(math.ceil(S.P_RESOLUTION_HITS * len(tested) / S.HOLM_ALPHA))
    refined = 0
    for r in tested:
        if r["stats"]["p_one_sided"] <= floor and n_boot > S.N_BOOT:
            g = _cell_bets(B, r)
            r["stats"]["p_one_sided"] = S.fast_one_sided_p(g["pnl"].to_numpy(float),
                                                           pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d").to_numpy(),
                                                           n_boot, seed)
            r["stats"]["p_resamples"] = n_boot
            refined += 1
    if refined:
        adj = LB.holm(np.array([r["stats"]["p_one_sided"] for r in tested], float))
    for r, a in zip(tested, adj):
        s = r["stats"]
        r["p_holm"] = None if not np.isfinite(a) else round(float(a), 6)
        lo = s["ci95"][0]
        if s["bets"] >= P["proven"]["min_bets"] and r["p_holm"] is not None and r["p_holm"] < P["proven"]["holm_alpha_one_sided"] \
                and lo is not None and lo > 0:
            r["verdict"] = "PROVEN"
        elif s["roi"] is not None and s["roi"] > 0 and (s["clv"] is None or s["clv"] > 0):
            r["verdict"] = "PROMISING"
        else:
            r["verdict"] = "UNPROVEN"
    # F2
    diffs = []
    for k, g in m.groupby(key):
        d = dict(zip(key, k))
        ag = g[g["cls"] == "AGREE"]
        for other in ("NONE", "DISAGREE"):
            diffs.append(d | {"contrast": f"AGREE-{other}", "n_a": int(len(ag)), "n_b": int((g["cls"] == other).sum())}
                         | diff_test(ag, g[g["cls"] == other], P["stats"]["diff_boot"], seed))
        mo = mk[(mk["window"] == d["window"]) & (mk["signal"] == d["signal"]) & (mk["side"] == d["side"])
                & (mk["league"] == d["league"])]
        diffs.append(d | {"contrast": "AGREE-MARKET_FOLLOW", "n_a": int(len(ag)), "n_b": int(len(mo))}
                     | diff_test(ag, mo, P["stats"]["diff_boot"], seed))
    dt = [x for x in diffs if x["p_one_sided"] is not None]
    for x, a in zip(dt, LB.holm(np.array([x["p_one_sided"] for x in dt], float))):
        x["p_holm"] = round(float(a), 6)
    return {"cells": rows, "family_F1_size": len(tested), "p_refined_cells": refined,
            "p_refined_resamples": n_boot if refined else None,
            "diffs": diffs, "family_F2_size": len(dt),
            "verdict_counts": pd.Series([r.get("verdict") for r in tested]).value_counts().to_dict()}


def _cell_bets(B: pd.DataFrame, r: dict) -> pd.DataFrame:
    g = B[(B["arm"] == r["arm"]) & (B["window"] == r["window"]) & (B["side"] == r["side"]) & (B["league"] == r["league"])]
    if r["arm"] == "model":
        g = g[g["track"] == r["track"]]
        if r["cls"] == "ALL":
            return g[g["signal"] == g["signal"].iloc[0]] if len(g) else g
        return g[(g["signal"] == r["signal"]) & (g["cls"] == r["cls"])]
    return g[g["signal"] == r["signal"]]


def live_tip_rate(ou: pd.DataFrame, u: pd.DataFrame, cells: list[dict], P: dict, odir: pd.DataFrame,
                  t: pd.DataFrame) -> dict:
    """Question 4. A live tip is first made at the measured live lead time (tier BT, role BET: T-125 h
    Standard / T-153 h New-Format). Per MODEL cell: tips at the BT price on fixtures where the primary window
    can be read, how many are still a tip at the bet snapshot of the primary window, and how many of those have
    sharp AGREE (PIN signal) - per active week of the league (an ISO week with at least one such fixture).
    Prices only; no outcome."""
    from src.bars_opt import study as S
    book = P["signal"]["sharp_book"]
    out, tot_bt, tot_pass, tot_agree = [], 0.0, 0.0, 0.0
    for mk, line in LINE_OF.items():
        q = ou[(ou["tier"] == "BT") & (ou["role"] == "BET") & (ou["bookmaker"] == book) & (ou["line"] == line)]
        bt = q.sort_values("ts").drop_duplicates("fixture_id")[["fixture_id", "yes", "no", "fair_over", "lead_h"]]
        bt = bt[bt["fixture_id"].isin(t["fixture_id"])]
        late = ou_price_at(ou, t, line, book)
        for cell in [c for c in cells if S.MARKET_SIDES[c["mside"]][0] == mk]:
            r = u[(u["track"] == cell["track"]) & (u["league"] == cell["league"]) & (u["market"] == mk)
                  & u["row_id"].isin(t["fixture_id"])]
            weeks = pd.to_datetime(r["date"]).dt.strftime("%G-%V").nunique()
            if not weeks:
                continue
            a = tips_for_cell(cell, u, bt.drop(columns="lead_h"))
            b = tips_for_cell(cell, u, late)
            still = a[a["fixture_id"].isin(b["fixture_id"])]
            sgn = 1 if S.MARKET_SIDES[cell["mside"]][1] == "yes" else -1
            agree = still[(still["fixture_id"].map(odir["pin_dir"]) * sgn) == 1]
            sig = a["fixture_id"].map(odir["pin_dir"]) * sgn
            with_late = a[a["fixture_id"].isin(late["fixture_id"])]
            out.append({"track": cell["track"], "mside": cell["mside"], "league": cell["league"], "active_weeks": int(weeks),
                        "bt_priced_fixtures": int(r["row_id"].isin(bt["fixture_id"]).sum()),
                        "tips_at_bt": int(len(a)), "still_tip_at_bet_snapshot": int(len(still)),
                        "still_tip_and_pin_agree": int(len(agree)),
                        "bt_tips_with_signal": int(sig.notna().sum()), "bt_tips_pin_agree": int((sig == 1).sum()),
                        "bt_tips_pin_disagree": int((sig == -1).sum()), "bt_tips_pin_none": int((sig == 0).sum()),
                        "bt_tips_with_bet_snapshot_price": int(len(with_late)),
                        "pin_agree_per_active_week": round(int((sig == 1).sum()) / weeks, 3),
                        "tips_per_active_week_at_bt": round(len(a) / weeks, 3),
                        "passing_per_active_week": round(len(agree) / weeks, 3)})
            tot_bt += len(a) / weeks
            tot_pass += len(agree) / weeks
            tot_agree += int((sig == 1).sum()) / weeks
    return {"cells": out, "sum_tips_per_active_week_at_bt": round(tot_bt, 2),
            "sum_passing_per_active_week": round(tot_pass, 2),
            "sum_pin_agree_per_active_week": round(tot_agree, 2),
            "note": "sums add per-cell rates over each league's own active weeks (an upper bound for a calendar week "
                    "in which every league plays); BT is priced by Pinnacle only, so a cell's BT tips are the ones "
                    "Pinnacle quoted on the exact line at T-125/153 h"}


def coverage(work: Path) -> dict:
    """Counts only (no outcome): what the protocol can measure. Run before registration."""
    P = protocol()
    u = pd.read_parquet(work / "universe.parquet")
    ou = pd.read_parquet(work / "ou_quotes.parquet")
    x12 = pd.read_parquet(work / "x12_quotes.parquet")
    fx = _fixture_frame(u)
    flg, fdate = fx.set_index("fixture_id")["league"], fx.set_index("fixture_id")["date"]
    snaps = movement_snapshots(ou, x12)
    book = P["signal"]["sharp_book"]
    cells, _ = model_cells(P)
    out = {"snap_lead_h": _q(snaps["lead_h"]), "fixtures_with_b4": int(snaps["fixture_id"].nunique()), "windows": {}}
    for wname, W in P["windows"].items():
        t = window_times(snaps, W)
        om = ou_moves(ou, t, P)
        om["league"], om["date"] = om["fixture_id"].map(flg), om["fixture_id"].map(fdate)
        for col in ("pin", "cons"):
            om[col + "_thr"] = thresholds(om, col, [], P)
            om[col + "_dir"] = direction(om[col], om[col + "_thr"])
        price = ou_price_at(ou, t, 2.5, book)
        ntips = {}
        for cell in cells:
            mk = cell["mside"].split("_")[0]
            tp = tips_for_cell(cell, u, ou_price_at(ou, t, LINE_OF[mk], book))
            ntips[f"{cell['track']}|{cell['mside']}|{cell['league']}"] = int(len(tp))
        out["windows"][wname] = {"fixtures": int(len(t)), "lead_open": _q(t["lead_open"]), "lead_sig": _q(t["lead_sig"]),
                                 "lead_bet": _q(t["lead_bet"]), "pin_kind": om["pin_kind"].value_counts(dropna=False).to_dict(),
                                 "pin_dir": om["pin_dir"].value_counts(dropna=False).to_dict(),
                                 "cons_dir": om["cons_dir"].value_counts(dropna=False).to_dict(),
                                 "pinnacle_25_pair_at_bet": int(len(price)), "model_tips_per_cell": ntips,
                                 "model_tips_total": int(sum(ntips.values()))}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "coverage", "evaluate"])
    ap.add_argument("--work", required=True)
    a = ap.parse_args(argv)
    work = assert_outside_repo(Path(a.work))
    if a.step == "build":
        print(json.dumps(build(work), indent=1, default=str))
    elif a.step == "coverage":
        c = coverage(work)
        (work / "coverage.json").write_text(json.dumps(c, indent=1, default=str), encoding="utf-8")
        print(json.dumps(c, indent=1, default=str))
    else:
        r = evaluate(work)
        RESULTS_FILE.write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
        print(json.dumps({k: r[k] for k in ("family_F1_size", "family_F2_size", "verdict_counts")}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
