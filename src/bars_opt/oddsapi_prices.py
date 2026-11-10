"""Priced universes for the bars study: the study's universe rows plus the REAL bet-time price from the bought
OddsAPI history (PRIVATE store, wowza-exec branch odds-history) where our own captured price is missing.

LICENCE. The Odds API forbids redistribution and this repo is PUBLIC. Nothing in this module prints a price,
a club or an event id; priced universes and every per-row output go to a directory OUTSIDE this repo
(assert_outside_repo). Reports keep aggregate counts and ROI / CI / agreement statistics only.

RULES (each one is the study's own rule, read from src/league_bars.py and src/bars_opt/universe.py; nothing
new is invented here):
  * OWN ALWAYS WINS. A row whose universe already carries our own two-sided pair (yes AND no) keeps it, and
    its closing pair, untouched. price_source = "own" / "oddsapi_hist" / None (unpriced); close_source the
    same for the CLV pair (an own closing pair is never replaced).
  * BET-TIME TIMING = src.league_bars.snapshot_pairs: both sides from the SAME snapshot, captured BEFORE the
    kickoff (snapshot_ts < the provider's commence_time; snapshot_ts <= match day 00:00 UTC when no kickoff
    is known) and from match day - 1 day on; the EARLIEST such pair is the bet-time price and the LATEST is
    the closing pair (CLV only). B1 holds close snapshots (kickoff - 5 min) today; C1 (T-48h) is read by the
    same rule when it lands (a T-48h snapshot earlier than match day - 1 day is outside the window, exactly
    as an own capture would be). The provider always states the kickoff, so the no-kickoff fallback of the
    own rule never applies here: a pair without a kickoff inside the fixture's day window is not used.
  * ONE BOOK'S PAIR, NEVER BEST OR CONSENSUS. The study prices with one book's two-sided pair: football-data
    "Bet365 (B365>2.5 / B365<2.5), else Pinnacle (P>2.5 / P<2.5). Average / maximum columns are never used
    (not one book's price, not executable)"; the snapshot captures are Bet365 only. BOOK_ORDER is that same
    order as OddsAPI bookmaker keys; the provider's eu region carries no Bet365, so the history prices
    with Pinnacle or not at all.
  * EXACT LINE, PER SIDE. over15 / over25 / over35 take only the provider's 1.5 / 2.5 / 3.5 Over AND Under;
    a 2.25 / 2.75 (or any quarter / whole) line never stands in. BTTS takes Yes AND No. A book quoting two
    different prices for the same outcome in one snapshot is ambiguous and dropped (counted).
  * NEVER ACROSS LEAGUES OR FIXTURES. The event -> fixture map is the adapter's (src.oddsapi_history_adapter.
    match_universe: one league's sport keys, one match day, both clubs proven, one-to-one).
    verify_join re-checks every attached row: same fixture id, same league, a sport key of that league, one
    fixture per event, the provider kickoff inside the collector's day window of the fixture's date, and the
    snapshot before that kickoff. Any failure raises JoinIntegrityError; nothing is written.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

#: the study's own same-book order (src.league_bars.fd_ou25_pairs: Bet365, else Pinnacle), as OddsAPI keys
BOOK_ORDER = ("bet365", "pinnacle")
#: the exact provider line of each universe market (None = no line: BTTS)
LINE_OF = {"over15": 1.5, "over25": 2.5, "over35": 3.5, "btts": None}
SRC_OWN, SRC_HIST = "own", "oddsapi_hist"
HIST_COLS = ["oh_event_id", "oh_book", "oh_snapshot_ts", "oh_commence_time", "oh_snapshot_ts_close"]


class JoinIntegrityError(RuntimeError):
    pass


def assert_outside_repo(path: Path) -> Path:
    """Refuse any output path inside this PUBLIC repo (redistribution of the provider's data is prohibited)."""
    p = Path(path).resolve()
    try:
        p.relative_to(ROOT.resolve())
    except ValueError:
        return p
    raise SystemExit(f"{p} is inside the public wowza-betting tree: refused (OddsAPI redistribution prohibited)")


def _utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True, format="mixed")


# ── provider pairs ─────────────────────────────────────────────────────────────────────────────

def history_pairs(long: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Two-sided same-book same-snapshot pairs on the exact line, from the adapter's long frame.
    Returns (pairs: fixture_id, market, league, event_id, bookmaker, ts, ko, yes, no; drop counts)."""
    cols = ["fixture_id", "market", "league", "event_id", "bookmaker", "ts", "ko", "yes", "no"]
    st = {"quotes": 0 if long is None else int(len(long)), "kept_quotes": 0, "ambiguous_groups": 0,
          "one_sided_groups": 0, "other_line_total_quotes": 0, "inexact_line_quotes": 0}
    if long is None or long.empty:
        return pd.DataFrame(columns=cols), st
    full = long["team"].isna() & long["player"].isna()
    # a full-match total on any other line (2.25 / 2.75 / 3.0 / 4.5 ...) is never a stand-in: counted, dropped
    st["other_line_total_quotes"] = int((full & long["provider_market"].isin(["totals", "alternate_totals"])
                                         & ~long["market"].isin(LINE_OF)).sum())
    d = long[long["market"].isin(LINE_OF) & full & long["side"].isin(["yes", "no"])].copy()
    want = d["market"].map(LINE_OF)
    line = pd.to_numeric(d["line"], errors="coerce")
    exact = np.where(want.isna(), line.isna(), (line - want.astype(float)).abs() < 1e-9)
    st["inexact_line_quotes"] = int((~exact).sum())
    d = d[exact]
    d["price"] = pd.to_numeric(d["price"], errors="coerce")
    d = d[np.isfinite(d["price"]) & (d["price"] > 1)]
    d["ts"], d["ko"] = _utc(d["snapshot_ts"]), _utc(d["commence_time"])
    d = d[d["ts"].notna()]
    key = ["fixture_id", "market", "bookmaker", "ts", "side"]
    nprice = d.groupby(key)["price"].transform("nunique")
    amb = d.loc[nprice > 1, ["fixture_id", "market", "bookmaker", "ts"]].drop_duplicates()
    st["ambiguous_groups"] = int(len(amb))
    if len(amb):
        d = d.merge(amb.assign(_amb=1), on=["fixture_id", "market", "bookmaker", "ts"], how="left")
        d = d[d["_amb"].isna()].drop(columns="_amb")
    d = d.drop_duplicates(key)
    st["kept_quotes"] = int(len(d))
    if d.empty:
        return pd.DataFrame(columns=cols), st
    w = d.pivot_table(index=["fixture_id", "market", "league", "event_id", "bookmaker", "ts"], columns="side",
                      values="price", aggfunc="first").reset_index()
    for s in ("yes", "no"):
        if s not in w:
            w[s] = np.nan
    both = w["yes"].notna() & w["no"].notna()
    st["one_sided_groups"] = int((~both).sum())
    w = w[both].copy()
    if w.empty:
        return pd.DataFrame(columns=cols), st
    ko = d.drop_duplicates("event_id").set_index("event_id")["ko"]
    w["ko"] = w["event_id"].map(ko)
    return w[cols].reset_index(drop=True), st


def pick_pairs(pairs: pd.DataFrame, fixture_dates: pd.Series, stats: dict | None = None) -> pd.DataFrame:
    """Per (fixture_id, market): the first book of BOOK_ORDER with a provable pre-kickoff pair; its EARLIEST
    pair is the bet-time price, its LATEST the closing pair (src.league_bars.snapshot_pairs timing).
    fixture_dates: {fixture_id: our match date}. A pair whose provider kickoff is unknown or outside the
    collector's day window of OUR fixture date (src.oddsapi_history.day_window; the adapter also accepts an
    adjacent-day listing) is never used: the kickoff must match the fixture (counted in stats)."""
    from src import oddsapi_history as OH
    out_cols = ["fixture_id", "market", "league", "event_id", "bookmaker", "ts", "ko", "yes", "no",
                "ts_c", "yes_c", "no_c"]
    stats = stats if stats is not None else {}
    if pairs.empty:
        return pd.DataFrame(columns=out_cols)
    p = pairs.copy()
    p["fdate"] = p["fixture_id"].map(fixture_dates)
    p["day0"] = pd.to_datetime(p["fdate"]).dt.tz_localize("UTC")
    win = {d: OH.day_window(pd.Timestamp(d).strftime("%Y-%m-%d")) for d in p["fdate"].dropna().unique()}
    lo = p["fdate"].map(lambda d: win[d][0] if d in win else None)
    hi = p["fdate"].map(lambda d: win[d][1] if d in win else None)
    ko_ok = p["ko"].notna() & lo.notna() & (p["ko"] >= pd.to_datetime(lo, utc=True)) & (p["ko"] < pd.to_datetime(hi, utc=True))
    stats["pairs_kickoff_outside_fixture_window"] = int((~ko_ok).sum())
    stats["fixtures_kickoff_outside_fixture_window"] = int(p.loc[~ko_ok, "fixture_id"].nunique())
    p = p[ko_ok]
    pre = p["ts"] < p["ko"]
    p = p[pre & p["day0"].notna() & (p["ts"] >= p["day0"] - pd.Timedelta(days=1))]
    p = p[p["bookmaker"].isin(BOOK_ORDER)]
    if p.empty:
        return pd.DataFrame(columns=out_cols)
    p["rank"] = p["bookmaker"].map({b: i for i, b in enumerate(BOOK_ORDER)})
    best = p.groupby(["fixture_id", "market"])["rank"].transform("min")
    p = p[p["rank"] == best].sort_values("ts")
    g = p.groupby(["fixture_id", "market"], as_index=False)
    first, last = g.first(), g.last()
    last = last[["fixture_id", "market", "ts", "yes", "no"]].rename(columns={"ts": "ts_c", "yes": "yes_c",
                                                                             "no": "no_c"})
    return first.merge(last, on=["fixture_id", "market"])[out_cols]


# ── the priced universe ──────────────────────────────────────────────────────────────────────────

def build_priced(universe: pd.DataFrame, picked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(priced universe, overlap frame for the own-vs-history cross-check). Own pairs always win."""
    u = universe.copy()
    u["fixture_id"] = u["row_id"].astype(str)
    own = u["yes"].notna() & u["no"].notna()
    own_c = u["yes_c"].notna() & u["no_c"].notna()
    h = picked.rename(columns={"yes": "h_yes", "no": "h_no", "yes_c": "h_yes_c", "no_c": "h_no_c",
                               "event_id": "oh_event_id", "bookmaker": "oh_book", "ts": "oh_snapshot_ts",
                               "ko": "oh_commence_time", "ts_c": "oh_snapshot_ts_close", "league": "h_league"})
    u = u.merge(h, on=["fixture_id", "market"], how="left", validate="many_to_one")
    own, own_c = own.to_numpy(), own_c.to_numpy()
    for c in ("h_yes", "h_no", "h_yes_c", "h_no_c"):
        u[c] = pd.to_numeric(u[c], errors="coerce").astype(float)
    hist = (u["h_yes"].notna() & u["h_no"].notna()).to_numpy()
    overlap = u[own & hist][["track", "market", "league", "row_id", "yes", "no", "h_yes", "h_no"]].copy()
    fill = ~own & hist
    fill_c = fill & ~own_c & (u["h_yes_c"].notna() & u["h_no_c"].notna()).to_numpy()
    for c, f in (("yes", fill), ("no", fill), ("yes_c", fill_c), ("no_c", fill_c)):
        u[c] = np.where(f, u["h_" + c], u[c].astype(float))
    u["price_source"] = np.where(own, SRC_OWN, np.where(fill, SRC_HIST, None))
    u["close_source"] = np.where(own_c, SRC_OWN, np.where(fill_c, SRC_HIST, None))
    # history metadata stays only on the rows it priced (an own row carries none)
    for c in HIST_COLS + ["h_league"]:
        u[c] = u[c].where(fill, None)
    return u.drop(columns=["h_yes", "h_no", "h_yes_c", "h_no_c"]), overlap


def verify_join(priced: pd.DataFrame, fmap: dict, event_sport: dict) -> dict:
    """Every history-priced row: the adapter mapped THIS fixture id to THIS event, the event is mapped to no
    other fixture, the price's league is the row's league and the event's sport key is one of that league's,
    the provider kickoff lies in the collector's day window of the fixture's date, and the bet-time snapshot
    is before that kickoff. Raises JoinIntegrityError on any failure."""
    from src import league_registry as LR
    from src import oddsapi_history as OH
    h = priced[priced["price_source"] == SRC_HIST]
    inv: dict = {}
    for f, e in fmap.items():
        inv.setdefault(e, set()).add(f)
    bad = {"fixture_event": 0, "event_to_many_fixtures": 0, "league": 0, "sport_key": 0,
           "kickoff_window": 0, "snapshot_after_kickoff": 0}
    keys: dict = {}
    for r in h.itertuples(index=False):
        if fmap.get(r.fixture_id) != r.oh_event_id:
            bad["fixture_event"] += 1
        if len(inv.get(r.oh_event_id, ())) != 1:
            bad["event_to_many_fixtures"] += 1
        fx_league = str(r.fixture_id).split("|")[1]
        if not (r.league == fx_league == r.h_league):
            bad["league"] += 1
        if r.league not in keys:
            keys[r.league] = set(LR.odds_api_sport_keys_for(r.league))
        if event_sport.get(r.oh_event_id) not in keys[r.league]:
            bad["sport_key"] += 1
        ko = pd.Timestamp(r.oh_commence_time)
        lo, hi = OH.day_window(str(r.fixture_id).split("|")[0])
        if pd.isna(ko) or not (pd.Timestamp(lo) <= ko < pd.Timestamp(hi)):
            bad["kickoff_window"] += 1
        if pd.isna(ko) or not pd.Timestamp(r.oh_snapshot_ts) < ko:
            bad["snapshot_after_kickoff"] += 1
    n_bad = sum(bad.values())
    if n_bad:
        raise JoinIntegrityError(f"history join integrity failed on {n_bad} check(s): {bad}")
    return {"rows_checked": int(len(h)), "events": int(h["oh_event_id"].nunique()), "failures": bad}


# ── one track, from the store ────────────────────────────────────────────────────────────────────

def price_track(universe: pd.DataFrame, store_root: Path) -> dict:
    """The whole chain for one track's universe: adapter match + attach, pairs, pick, priced universe,
    integrity check. Returns {"priced", "overlap", "stats"}."""
    from src import oddsapi_history_adapter as AD
    odds = AD.load_store(store_root)
    return price_frames(universe, odds, AD.load_context(store_root, universe))


def price_frames(universe: pd.DataFrame, odds: pd.DataFrame, ctx: dict | None) -> dict:
    """price_track on frames (ctx None = the events the odds rows carry, the adapter's own fallback)."""
    from src import oddsapi_history_adapter as AD
    odds = odds[odds["league"].isin(set(universe["league"]))]
    fmap, unmatched = AD.match_universe(universe, odds, ctx)
    long, _summary, _ = AD.attach(universe, odds, ctx)
    pairs, pst = history_pairs(long)
    fx = universe.assign(fixture_id=universe["row_id"].astype(str)).drop_duplicates("fixture_id")
    picked = pick_pairs(pairs, fx.set_index("fixture_id")["date"], pst)
    priced, overlap = build_priced(universe, picked)
    event_sport = odds.drop_duplicates("event_id").set_index("event_id")["sport_key"].to_dict()
    integ = verify_join(priced, fmap, event_sport)
    kinds = unmatched["kind"].value_counts().to_dict() if not unmatched.empty else {}
    stats = {"store_rows_in_track_leagues": int(len(odds)), "fixtures_matched": len(fmap), "unmatched": kinds,
             "pairs": pst, "fixture_markets_with_history_pair": int(len(picked)),
             "books_used": picked["bookmaker"].value_counts().to_dict() if len(picked) else {},
             "integrity": integ}
    return {"priced": priced, "overlap": overlap, "stats": stats}


# ── coverage and agreement (counts / aggregate statistics only) ─────────────────────────────────

def _months(dates: pd.Series) -> int:
    return int(pd.to_datetime(dates).dt.to_period("M").nunique()) if len(dates) else 0


def _monthly_sim_months(sel: pd.DataFrame) -> int:
    """How many months study.monthly_sim_family would simulate on these SELECT rows (its own rule)."""
    from src.bars_opt import study as S
    if sel.empty:
        return 0
    first = (sel["date"].min() + pd.DateOffset(months=S.MONTHLY_MIN_HISTORY_MONTHS)).to_period("M").to_timestamp()
    return len(pd.date_range(first, pd.Timestamp("2026-08-01"), freq="MS"))


def coverage(priced: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """(per league x market rows, per family windows) - BEFORE = own prices only, AFTER = own + history.
    Priced counts are the study's own eligible rows (study.family_frame: a de-viggable pair, an outcome
    and a probability)."""
    from src.bars_opt import study as S
    track = str(priced["track"].iloc[0])
    before = priced.copy()
    hist = before["price_source"] == SRC_HIST
    before.loc[hist, ["yes", "no"]] = np.nan
    before.loc[before["close_source"] == SRC_HIST, ["yes_c", "no_c"]] = np.nan
    rows, fams = [], []
    for mside, (market, _side) in S.MARKET_SIDES.items():
        if not (priced["market"] == market).any():
            continue
        fb, fa = S.family_frame(before, mside), S.family_frame(priced, mside)
        src = priced.set_index(["row_id", "market"])["price_source"]
        fa_src = fa["row_id"].map(lambda r: src.get((r, market)))
        sel_b, sel_a = fb[fb["date"] < S.SELECT_END], fa[fa["date"] < S.SELECT_END]
        fin = lambda f: f[(f["date"] >= S.FINAL_START) & (f["date"] <= S.FINAL_END)]  # noqa: E731
        fams.append({"track": track, "mside": mside,
                     "select_before": len(sel_b), "select_after": len(sel_a),
                     "inner_blocks_before": max(len(S.inner_blocks(sel_b["date"])) - 1, 0),
                     "inner_blocks_after": max(len(S.inner_blocks(sel_a["date"])) - 1, 0),
                     "monthly_sim_months_before": _monthly_sim_months(sel_b),
                     "monthly_sim_months_after": _monthly_sim_months(sel_a),
                     "final_before": len(fin(fb)), "final_after": len(fin(fa))})
        if mside == "over25_under":          # same rows as over25_over: one coverage row per market
            continue
        for lg in sorted(priced.loc[priced["market"] == market, "league"].unique()):
            b, a = fb[fb["league"] == lg], fa[fa["league"] == lg]
            a_src = fa_src[fa["league"] == lg]
            sb, sa = b[b["date"] < S.SELECT_END], a[a["date"] < S.SELECT_END]
            rows.append({"track": track, "market": market, "league": lg,
                         "rows": int(((priced["market"] == market) & (priced["league"] == lg)).sum()),
                         "priced_before": len(b), "priced_after": len(a),
                         "after_own": int((a_src == SRC_OWN).sum()), "after_oddsapi_hist": int((a_src == SRC_HIST).sum()),
                         "select_before": len(sb), "select_after": len(sa),
                         "select_months_before": _months(sb["date"]), "select_months_after": _months(sa["date"]),
                         "final_before": len(fin(b)), "final_after": len(fin(a)),
                         "own_bars_possible_before": len(sb) >= S.MIN_OWN_BETS,
                         "own_bars_possible_after": len(sa) >= S.MIN_OWN_BETS,
                         "final_can_reach_min_before": len(fin(b)) >= S.MIN_FINAL_BETS,
                         "final_can_reach_min_after": len(fin(a)) >= S.MIN_FINAL_BETS})
    return rows, fams


def agreement(overlap: pd.DataFrame) -> list[dict]:
    """Own vs history on rows priced by BOTH (aggregate only): per track x market the count and the median
    absolute difference of each side's decimal price and of the de-vigged Yes/Over probability."""
    from src.market_edge import power_devig
    out = []
    if overlap.empty:
        return out
    for (track, market), g in overlap.groupby(["track", "market"]):
        fo = np.array([power_devig(float(a), float(b)) or np.nan for a, b in zip(g["yes"], g["no"])], float)
        fh = np.array([power_devig(float(a), float(b)) or np.nan for a, b in zip(g["h_yes"], g["h_no"])], float)
        dy, dn = (g["yes"] - g["h_yes"]).abs(), (g["no"] - g["h_no"]).abs()
        fd = np.abs(fo - fh)
        same = (g["yes"] == g["h_yes"]) & (g["no"] == g["h_no"])
        out.append({"track": track, "market": market, "rows_both": int(len(g)), "leagues": int(g["league"].nunique()),
                    # identical pairs: own BACKFILL rows (price_provenance BACKFILL_HISTORICAL; the new-format
                    # backfill is documented there as OddsAPI history) can share the provider's source, so an
                    # identical pair checks the join, not the provider's independence
                    "share_identical_pair": round(float(same.mean()), 4),
                    "mad_odds_yes": round(float(dy.median()), 4), "mad_odds_no": round(float(dn.median()), 4),
                    "median_rel_diff_yes": round(float((dy / g["yes"]).median()), 4),
                    "median_rel_diff_no": round(float((dn / g["no"]).median()), 4),
                    "mad_fair_prob": round(float(np.nanmedian(fd)), 4) if np.isfinite(fd).any() else None,
                    "p90_abs_fair_prob": round(float(np.nanpercentile(fd, 90)), 4) if np.isfinite(fd).any() else None})
    return out


# ── B1 completeness (the store's own run ledger) ────────────────────────────────────────────────

def b1_complete(store_root: Path) -> tuple[bool, str]:
    """True only when a collector run finished tier B1: exit code 0 (all planned tiers done) with B1 in it,
    or a run that went on to a LATER tier (tiers run strictly in order, src.oddsapi_history.TIERS)."""
    from src import oddsapi_history as OH
    later = set(OH.TIER_ORDER[OH.TIER_ORDER.index("B1") + 1:])
    runs = sorted((Path(store_root) / "ledger" / "runs").glob("*.json"))
    for f in runs:
        r = json.loads(f.read_text(encoding="utf-8"))
        tiers = set((r.get("stats") or {}).keys()) | set((r.get("credits_by_tier") or {}).keys())
        if "B1" in tiers and r.get("exit_code") == OH.EXIT_DONE:
            return True, f"run {r.get('run_id')} finished all planned tiers (B1 included)"
        if tiers & later:
            return True, f"run {r.get('run_id')} reached tier(s) {sorted(tiers & later)} after B1"
    return False, f"no run in {len(runs)} ledger run file(s) finished B1"
