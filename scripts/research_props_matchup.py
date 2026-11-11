"""research_props_matchup.py - RESEARCH ONLY (no serving change). Props matchup study, v9 side.

Owner idea (2026-11-04, queued): per player x league x market bars for props, plus an edge when the
opponent's regular goalkeeper or regular defenders are absent, for SOT 1+ and anytime scorer.
Design and the Pro scaffold: wowzaV9-Pro `research/props-matchup` (reports/R_PROPS_MATCHUP_SPEC.md).
This file answers the v9 questions of 2026-11-11 with the data that exists today:

  1. inventory   per league: settled ledger player-fixtures with (a) a real bet-time prop price,
                 (b) the opponent's absence list known BEFORE kickoff, (c) both.
  2. absence     do opponent GK / regular-defender absences raise SOT or scoring beyond the SERVED
                 model's probability? Residual test, per league, pooled only as context.
  3. bars        is any player x league x market cell big enough for a bar? Accrual and dates.

RULES IT FOLLOWS (owner): real data only, missing stays missing, nothing defaulted, per league, the
strict PROVEN rule (>= 150 holdout bets, CI > 0 after Holm over every configuration searched),
nothing trained, nothing written to a serving path, no prices written anywhere (counts only).

CLOCKS. An absence is decision-grade only when it was OBSERVED before kickoff:
  STRICT     Pro `output/studies/injury_snapshots.parquet`, rows with pre_kickoff, captured_at earlier
             than the fixture's FINAL kickoff (v9 output/collect/af_ht_scores.parquet; Pro's own
             fixture_ts when the fixture is not there). The latest such list per fixture is used.
  RETRO_INJ  v9 `output/collect/injuries.parquet` - fetched after the match (2026-11-05..09). Pro
             measured 18.8% of a final list added after kickoff, so this is hindsight-contaminated.
             CONTEXT ONLY, never a decision.
  RETRO_XI   the regular player is not in the starting XI (v9 lineups, fetched after the match). The
             XI is announced about an hour before kickoff, but no capture time proves it. CONTEXT ONLY.
Lineups with player ids observed before kickoff: none exist (scripts/capture_prematch.py has never
run: its workflow is not on the default branch), so there is no STRICT lineup clock.

REGULARS (fixed in config/experiments/props_matchup/protocol.json before any effect was computed):
from the team's last LOOKBACK fixtures with a known lineup dated strictly before the target fixture's
date, a goalkeeper is the regular GK when he started (pos G) in at least REGULAR_SHARE of them (the
one with most starts); a defender (pos D) is regular with the same share. Fewer than MIN_OBS
fixtures in the window -> the team's regulars are UNKNOWN and every exposure for that fixture is
missing (NaN), never 0. API-Football positions are G/D/M/F only: centre-backs and full-backs are both
'D' and cannot be split without inventing a role.

Usage (single-threaded, read-only on every input):
    python scripts/research_props_matchup.py --scored <props_harness_holdout_scored.csv.gz>
        --pro-injuries <Pro injury_snapshots.parquet> --ledger <player_ledger.csv>
        --prices <player_prop_odds_history.csv> --price-archive <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "config" / "experiments" / "props_matchup" / "protocol.json"

MARKETS = ("sot", "goals")                  # SOT 1+ and anytime scorer (owner scope)
INVENTORY_MARKETS = ("sot", "sot2", "goals")
RARE_MARKETS = ("goals2", "goals3", "sot3", "sot4")
EXPOSURES = ("gk_out", "def_out")
CLOCKS = ("STRICT", "RETRO_INJ", "RETRO_XI")
DECISION_CLOCKS = ("STRICT",)
ABSENT_TYPE = "Missing Fixture"


# ── protocol ────────────────────────────────────────────────────────────────────────────────────

def load_protocol(path: Path | None = None) -> dict:
    p = PROTOCOL_PATH if path is None else Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    body = {k: v for k, v in d.items() if k != "protocol_sha256"}
    sha = protocol_sha(body)
    if d.get("protocol_sha256") != sha:
        raise ValueError(f"protocol hash mismatch: stored {d.get('protocol_sha256')} != computed {sha}")
    return d


def protocol_sha(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


# ── small helpers ───────────────────────────────────────────────────────────────────────────────

def norm_name(s) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().replace(".", " ").replace("-", " ").split())


def _utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


# ── fixtures and regulars ───────────────────────────────────────────────────────────────────────

def fixture_table(ht: pd.DataFrame) -> pd.DataFrame:
    """fixture_id -> league, kickoff (UTC), date, home_id, away_id (provider ids, final kickoff)."""
    f = ht[["fixture_id", "league", "kickoff", "home_id", "away_id"]].copy()
    f["kickoff"] = _utc(f["kickoff"])
    f = f.dropna(subset=["fixture_id", "kickoff", "home_id", "away_id"])
    f = f.drop_duplicates("fixture_id", keep="last")
    f["fixture_id"] = f["fixture_id"].astype("int64")
    for c in ("home_id", "away_id"):
        f[c] = f[c].astype("int64")
    f["date"] = f["kickoff"].dt.tz_convert(None).dt.normalize()
    return f.reset_index(drop=True)


def team_starts(lineup_players: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """One row per (team_id, fixture_id, player_id) starter with pos and the fixture date."""
    lp = lineup_players[lineup_players["role"] == "start"][["fixture_id", "team_id", "player_id", "pos"]]
    lp = lp.dropna(subset=["fixture_id", "team_id", "player_id"]).copy()
    for c in ("fixture_id", "team_id", "player_id"):
        lp[c] = lp[c].astype("int64")
    lp = lp.drop_duplicates(["fixture_id", "team_id", "player_id"])
    return lp.merge(fixtures[["fixture_id", "date"]], on="fixture_id", how="inner")


def regulars(starts: pd.DataFrame, targets: pd.DataFrame, lookback: int, min_obs: int,
             share: float) -> pd.DataFrame:
    """For each target (fixture_id, team_id, date): the team's regular GK and regular defenders from
    its last `lookback` lineup fixtures dated strictly before `date`.
    Returns fixture_id, team_id, n_obs, reg_gk (player id or <NA>), reg_def (tuple of ids);
    reg_known False when n_obs < min_obs (then both are unknown, never empty)."""
    tf = (starts[["team_id", "fixture_id", "date"]].drop_duplicates(["team_id", "fixture_id"])
          .sort_values(["team_id", "date", "fixture_id"]))
    by_team = {t: g for t, g in tf.groupby("team_id", sort=False)}
    st_by_team = {t: g for t, g in starts.groupby("team_id", sort=False)}
    out = []
    for r in targets.itertuples(index=False):
        g = by_team.get(r.team_id)
        rec = {"fixture_id": r.fixture_id, "team_id": r.team_id, "n_obs": 0, "reg_known": False,
               "reg_gk": pd.NA, "reg_def": ()}
        if g is not None:
            prev = g[g["date"] < r.date].tail(lookback)
            n = len(prev)
            rec["n_obs"] = n
            if n >= min_obs:
                rec["reg_known"] = True
                s = st_by_team[r.team_id]
                s = s[s["fixture_id"].isin(prev["fixture_id"])]
                gk = s[s["pos"] == "G"]["player_id"].value_counts()
                if len(gk) and gk.iloc[0] / n >= share:
                    rec["reg_gk"] = int(gk.index[0])
                d = s[s["pos"] == "D"]["player_id"].value_counts()
                rec["reg_def"] = tuple(sorted(int(p) for p, c in d.items() if c / n >= share))
        out.append(rec)
    return pd.DataFrame(out, columns=["fixture_id", "team_id", "n_obs", "reg_known", "reg_gk", "reg_def"])


# ── absence lists per clock ─────────────────────────────────────────────────────────────────────

def strict_lists(pro: pd.DataFrame, fixtures: pd.DataFrame) -> tuple[pd.DataFrame, set]:
    """Latest pre-kickoff Pro list per fixture. Returns (absent rows: fixture_id, team_id, player_id)
    and the set of fixture ids whose list was observed before the FINAL kickoff."""
    p = pro.copy()
    p["captured_at"] = _utc(p["captured_at"])
    p["fixture_ts"] = _utc(p["fixture_ts"])
    p = p[p["pre_kickoff"].astype(bool)]
    p = p.rename(columns={"af_fixture_id": "fixture_id"})
    final = fixtures.set_index("fixture_id")["kickoff"]
    ko = p["fixture_id"].map(final)
    ko = ko.where(ko.notna(), p["fixture_ts"])           # fixture missing from v9: Pro's own time
    p = p[p["captured_at"] < ko]
    if p.empty:
        return pd.DataFrame(columns=["fixture_id", "team_id", "player_id"]), set()
    last = p.groupby("fixture_id")["captured_at"].transform("max")
    p = p[p["captured_at"] == last]
    observed = set(int(x) for x in p["fixture_id"].unique())
    a = p[p["type"] == ABSENT_TYPE][["fixture_id", "team_id", "player_id"]].drop_duplicates()
    return a.astype("int64"), observed


def retro_lists(inj: pd.DataFrame) -> tuple[pd.DataFrame, set]:
    """v9 injuries backfill (fetched after the match): CONTEXT ONLY."""
    observed = set(int(x) for x in inj["fixture_id"].dropna().unique())
    a = inj[inj["type"] == ABSENT_TYPE][["fixture_id", "team_id", "player_id"]].dropna().drop_duplicates()
    return a.astype("int64"), observed


def exposures(targets: pd.DataFrame, regs: pd.DataFrame, absent: pd.DataFrame, observed: set,
              clock: str, xi: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per (fixture_id, defending team_id): gk_out / def_out as 1.0 / 0.0, NaN when the list was not
    observed or the regulars are unknown (gk_out also NaN when the team has no regular GK).
    RETRO_XI: absent = regular not among the fixture's starters (`xi` = starts of that fixture)."""
    t = targets[["fixture_id", "team_id"]].drop_duplicates().merge(regs, on=["fixture_id", "team_id"], how="left")
    if clock == "RETRO_XI":
        if xi is None:
            raise ValueError("RETRO_XI needs the fixture starters")
        sets = {k: set(int(x) for x in g) for k, g in xi.groupby(["fixture_id", "team_id"])["player_id"]}
        obs = np.array([(f, tm) in sets for f, tm in zip(t["fixture_id"], t["team_id"])], dtype=bool)
    else:
        sets = {k: set(int(x) for x in g) for k, g in absent.groupby(["fixture_id", "team_id"])["player_id"]}
        obs = t["fixture_id"].isin(observed).to_numpy()
    gk, de, nd = [], [], []
    for i, r in enumerate(t.itertuples(index=False)):
        known = False if pd.isna(r.reg_known) else bool(r.reg_known)
        if not obs[i] or not known:
            gk.append(np.nan); de.append(np.nan); nd.append(np.nan)
            continue
        regdef = tuple(r.reg_def) if isinstance(r.reg_def, tuple) else ()
        s = sets.get((r.fixture_id, r.team_id), set())
        if clock == "RETRO_XI":
            def is_out(pid, s=s):
                return pid not in s
        else:
            def is_out(pid, s=s):
                return pid in s
        gk.append(np.nan if pd.isna(r.reg_gk) else float(is_out(int(r.reg_gk))))
        n_out = sum(1 for p in regdef if is_out(p))
        nd.append(float(n_out))
        de.append(float(n_out >= 1) if regdef else 0.0)
    t["gk_out"], t["def_out"], t["n_def_out"] = gk, de, nd
    t["clock"] = clock
    return t[["fixture_id", "team_id", "clock", "gk_out", "def_out", "n_def_out", "n_obs"]]


# ── residual test ───────────────────────────────────────────────────────────────────────────────

def residual_diff(df: pd.DataFrame, exposure: str, n_boot: int, seed: int) -> dict:
    """Mean residual (y - p_served) of exposed rows minus unexposed rows; percentile CI and one-sided
    p (H1: diff > 0) from a bootstrap that resamples whole fixtures within each group."""
    d = df.dropna(subset=[exposure, "y", "p"])
    e, u = d[d[exposure] == 1.0], d[d[exposure] == 0.0]
    res = {"n_rows_exposed": int(len(e)), "n_rows_unexposed": int(len(u)),
           "n_fix_exposed": int(e["fixture_id"].nunique()), "n_fix_unexposed": int(u["fixture_id"].nunique())}
    if len(e) == 0 or len(u) == 0:
        return {**res, "diff": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p_one_sided": np.nan,
                "oe_exposed": np.nan, "oe_unexposed": np.nan}
    r_e = (e["y"] - e["p"]).to_numpy(); r_u = (u["y"] - u["p"]).to_numpy()
    diff = float(r_e.mean() - r_u.mean())
    rng = np.random.default_rng(seed)

    def _cluster(g):
        s = g.assign(r=g["y"] - g["p"]).groupby("fixture_id")["r"].agg(["sum", "count"])
        return s["sum"].to_numpy(), s["count"].to_numpy()
    se, ce = _cluster(e); su, cu = _cluster(u)
    ie = rng.integers(0, len(se), size=(n_boot, len(se)))
    iu = rng.integers(0, len(su), size=(n_boot, len(su)))
    boot = se[ie].sum(1) / ce[ie].sum(1) - su[iu].sum(1) / cu[iu].sum(1)
    p1 = float((1 + (boot <= 0).sum()) / (n_boot + 1))
    return {**res, "diff": diff, "ci_lo": float(np.quantile(boot, 0.025)),
            "ci_hi": float(np.quantile(boot, 0.975)), "p_one_sided": p1,
            "oe_exposed": float(e["y"].sum() / e["p"].sum()),
            "oe_unexposed": float(u["y"].sum() / u["p"].sum())}


def holm(pvals: pd.Series) -> pd.Series:
    """Holm step-down adjusted p-values over the non-missing entries (every test counted)."""
    p = pvals.dropna().sort_values()
    m = len(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p.to_numpy())
    out = pd.Series(np.minimum(adj, 1.0), index=p.index)
    return out.reindex(pvals.index)


# ── inventory ───────────────────────────────────────────────────────────────────────────────────

def load_prices(history: str, archive_dir: str | None) -> pd.DataFrame:
    parts = [pd.read_csv(history, low_memory=False)]
    if archive_dir:
        parts += [pd.read_csv(f, low_memory=False) for f in sorted(glob.glob(os.path.join(archive_dir, "*.csv")))]
    p = pd.concat(parts, ignore_index=True)
    p = p.drop_duplicates(["snapshot_ts", "match_date", "player", "market", "odds", "source", "bookmaker"])
    p["snapshot_ts"] = _utc(p["snapshot_ts"])
    p["pkey"] = p["player"].map(norm_name)
    return p


def corroborated_price(ledger: pd.DataFrame, prices: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Per ledger row: price_real (a timestamped snapshot BEFORE kickoff for the same match date,
    player, market carries exactly the ledger's market_odds), price_source (api_football / oddsapi /
    both), and kickoff_known. Kickoff = the provider's final kickoff via fixture_id; without one, only
    snapshots dated before the match day count. Counts only - no price is returned."""
    L = ledger.reset_index(drop=True).copy()
    L["row"] = np.arange(len(L))
    L["pkey"] = L["player_name"].map(norm_name)
    ko = pd.to_numeric(L["fixture_id"], errors="coerce").map(fixtures.set_index("fixture_id")["kickoff"])
    L["kickoff"] = ko
    L["kickoff_known"] = ko.notna()
    m = L[["row", "match_date", "pkey", "market", "market_odds", "kickoff"]].merge(
        prices[["match_date", "pkey", "market", "odds", "snapshot_ts", "snapshot_date", "source"]],
        on=["match_date", "pkey", "market"], how="inner")
    before = np.where(m["kickoff"].notna(), m["snapshot_ts"] < m["kickoff"],
                      pd.to_datetime(m["snapshot_date"]) < pd.to_datetime(m["match_date"]))
    same = (m["odds"] - m["market_odds"]).abs() < 1e-9
    ok = m[before & same.to_numpy()]
    src = ok.groupby("row")["source"].agg(lambda s: "both" if s.nunique() > 1 else s.iloc[0])
    L["price_real"] = L["row"].isin(ok["row"])
    L["price_source"] = L["row"].map(src)
    return L.drop(columns=["row", "pkey"])


def opponent_team_id(L: pd.DataFrame, fixtures: pd.DataFrame) -> pd.Series:
    """The defending (opponent) provider team id of a ledger row; NaN when the side is not proven
    (no fixture_id, or the row's team is neither the home nor the away name)."""
    fx = fixtures.set_index("fixture_id")
    fid = pd.to_numeric(L["fixture_id"], errors="coerce")
    home = L["team"].map(norm_name) == L["home_team"].map(norm_name)
    away = L["team"].map(norm_name) == L["away_team"].map(norm_name)
    hid = fid.map(fx["home_id"]); aid = fid.map(fx["away_id"])
    return pd.Series(np.where(home & ~away, aid, np.where(away & ~home, hid, np.nan)), index=L.index)


# ── accrual projection ──────────────────────────────────────────────────────────────────────────

def project_date(n_now: int, weekly_rate: float, target: int, today: pd.Timestamp) -> str:
    if n_now >= target:
        return "reached"
    if not weekly_rate or weekly_rate <= 0 or np.isnan(weekly_rate):
        return "never at the current rate"
    weeks = (target - n_now) / weekly_rate
    return (today + pd.Timedelta(weeks=weeks)).strftime("%Y-%m-%d")


# ── main ────────────────────────────────────────────────────────────────────────────────────────

def _read_lineups(lineup_dir: Path) -> pd.DataFrame:
    parts = []
    for f in sorted(lineup_dir.glob("players_*.parquet")):
        d = pd.read_parquet(f, columns=["fixture_id", "team_id", "role", "player_id", "pos"])
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def run(args) -> dict:
    proto = load_protocol()
    P = proto["regulars"]
    S = proto["statistics"]
    today = pd.Timestamp(args.today)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    fixtures = fixture_table(pd.read_parquet(ROOT / "output/collect/af_ht_scores.parquet"))
    lineups = _read_lineups(ROOT / "output/collect/lineups")
    starts = team_starts(lineups, fixtures)
    pro = pd.read_parquet(args.pro_injuries)
    strict_abs, strict_obs = strict_lists(pro, fixtures)
    retro_abs, retro_obs = retro_lists(pd.read_parquet(ROOT / "output/collect/injuries.parquet"))

    # served-model holdout rows (run 42855107089: hgb_nan_native fitted through 2026-08-31 = served)
    sc = pd.read_csv(args.scored, low_memory=False,
                     usecols=["market", "date", "player_id", "fixture_id", "league", "is_home", "y",
                              "p_hgb_nan_native"])
    sc = sc[sc["market"].isin(MARKETS)].rename(columns={"p_hgb_nan_native": "p"})
    sc["fixture_id"] = sc["fixture_id"].astype("int64")
    import pyarrow.dataset as ds
    ph = ds.dataset(str(ROOT / "player_history.parquet"), format="parquet").to_table(
        columns=["fixture_id", "player_id", "position"],
        filter=ds.field("date") >= "2026-08-01").to_pandas()
    ph = ph.dropna(subset=["fixture_id", "player_id"]).astype({"fixture_id": "int64", "player_id": "int64"})
    ph = ph.drop_duplicates(["fixture_id", "player_id"])
    sc = sc.merge(ph, on=["fixture_id", "player_id"], how="left")
    sc = sc[sc["position"].isin(["D", "M", "F"])]          # outfield attackers of the cell
    fx = fixtures.set_index("fixture_id")
    sc["def_team_id"] = np.where(sc["is_home"] == 1, sc["fixture_id"].map(fx["away_id"]),
                                 sc["fixture_id"].map(fx["home_id"]))
    sc = sc.dropna(subset=["def_team_id"]).astype({"def_team_id": "int64"})
    sc["date"] = pd.to_datetime(sc["date"])

    targets = sc[["fixture_id", "def_team_id"]].drop_duplicates().rename(columns={"def_team_id": "team_id"})
    targets = targets.merge(fixtures[["fixture_id", "date"]], on="fixture_id")
    regs = regulars(starts, targets, P["lookback_fixtures"], P["min_obs_fixtures"], P["regular_share"])
    xi = starts[["fixture_id", "team_id", "player_id"]]
    ex = pd.concat([exposures(targets, regs, strict_abs, strict_obs, "STRICT"),
                    exposures(targets, regs, retro_abs, retro_obs, "RETRO_INJ"),
                    exposures(targets, regs, retro_abs, retro_obs, "RETRO_XI", xi=xi)], ignore_index=True)

    rows = []
    for clock in CLOCKS:
        e = ex[ex["clock"] == clock].rename(columns={"team_id": "def_team_id"})
        d = sc.merge(e, on=["fixture_id", "def_team_id"], how="left")
        for market in MARKETS:
            dm = d[d["market"] == market]
            for exp in EXPOSURES:
                for lg, g in [("POOLED (context)", dm)] + sorted(dm.groupby("league")):
                    if g[exp].notna().sum() == 0:
                        continue
                    seed = S["seed"] + len(rows)
                    r = residual_diff(g, exp, S["n_boot"], seed)
                    rows.append({"clock": clock, "market": market, "exposure": exp, "league": lg, **r})
    res = pd.DataFrame(rows)
    per_league = res["league"] != "POOLED (context)"
    tested = per_league & (res["n_fix_exposed"] >= 1)
    res["holm_p"] = np.nan
    res.loc[tested, "holm_p"] = holm(res.loc[tested, "p_one_sided"]).to_numpy()
    res["n_tests_in_family"] = int(tested.sum())
    res["thin"] = res["n_fix_exposed"] < S["min_exposed_fixtures_to_read"]
    res.to_csv(out / "absence_effects.csv", index=False)

    # ── inventory on the live ledger (counts only) ──
    L = pd.read_csv(args.ledger, low_memory=False)
    L = L[L["result"].isin(["WIN", "LOSS"]) & L["market"].isin(INVENTORY_MARKETS + RARE_MARKETS)].copy()
    prices = load_prices(args.prices, args.price_archive)
    L = corroborated_price(L, prices, fixtures)
    L["opp_team_id"] = opponent_team_id(L, fixtures)
    fid = pd.to_numeric(L["fixture_id"], errors="coerce")
    L["absence_strict"] = fid.isin(strict_obs) & L["opp_team_id"].notna()
    L["absence_retro"] = fid.isin(retro_obs) & L["opp_team_id"].notna()
    L["both_strict"] = L["price_real"] & L["absence_strict"]
    L["match_date"] = pd.to_datetime(L["match_date"])
    since = L["match_date"] >= pd.Timestamp(proto["inventory"]["price_era_start"])
    inv = (L[since].groupby(["league"]).agg(
        settled=("result", "size"),
        a_price_real=("price_real", "sum"),
        a_price_af=("price_source", lambda s: int(s.isin(["api_football", "both"]).sum())),
        b_absence_strict=("absence_strict", "sum"),
        b_absence_retro_context=("absence_retro", "sum"),
        c_both_strict=("both_strict", "sum")).reset_index())
    inv_m = (L[since].groupby(["league", "market"]).agg(
        settled=("result", "size"), a_price_real=("price_real", "sum"),
        c_both_strict=("both_strict", "sum")).reset_index())
    inv.to_csv(out / "inventory_by_league.csv", index=False)
    inv_m.to_csv(out / "inventory_by_league_market.csv", index=False)

    # ── bars feasibility: player x league x market cells of settled priced tips ──
    pr = L[since & L["price_real"]].copy()
    pr["pid"] = pd.to_numeric(pr["player_id"], errors="coerce")
    cells = pr.dropna(subset=["pid"]).groupby(["league", "market", "pid"]).size().rename("n").reset_index()
    win = proto["accrual"]["rate_window_days"]
    recent = pr[pr["match_date"] > today - pd.Timedelta(days=win)]
    lm = pr.groupby(["league", "market"]).size().rename("n_now").reset_index()
    rate = (recent.groupby(["league", "market"]).size() / (win / 7)).rename("per_week").reset_index()
    lm = lm.merge(rate, on=["league", "market"], how="left")
    lm["per_week"] = lm["per_week"].fillna(0.0)
    tgt = S["min_holdout_bets"]
    lm["date_150"] = [project_date(n, w, tgt, today) for n, w in zip(lm["n_now"], lm["per_week"])]
    lm.to_csv(out / "accrual_league_market.csv", index=False)
    pc = recent.dropna(subset=["pid"]).groupby(["league", "market", "pid"]).size() / (win / 7)
    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol_sha256": proto["protocol_sha256"],
        "inputs": {"scored": os.path.basename(args.scored), "ledger_rows_settled": int(len(L)),
                   "strict_fixtures_observed": len(strict_obs), "retro_fixtures_observed": len(retro_obs)},
        "cells": {"n_cells": int(len(cells)), "max_n": int(cells["n"].max()) if len(cells) else 0,
                  "n_ge_30": int((cells["n"] >= 30).sum()), "n_ge_150": int((cells["n"] >= 150).sum()),
                  "max_player_rate_per_week": float(pc.max()) if len(pc) else 0.0},
        "league_market_reached_150": int((lm["n_now"] >= tgt).sum()),
        "n_tests_in_family": int(tested.sum()),
        "any_holm_significant": bool((res["holm_p"] < S["alpha"]).any()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scored", required=True)
    ap.add_argument("--pro-injuries", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--price-archive", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--today", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    s = run(ap.parse_args(argv))
    print(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
