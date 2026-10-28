"""
ONE canonical feature implementation for training AND serving (recipe "v9_repaired").
=====================================================================================

Standard forensic repair, 2026-10-28. RESEARCH RECIPE ONLY — nothing in production calls this
unless `recipe="v9_repaired"` is passed explicitly; the legacy builders in
`src/feature_engineering.py` are untouched and remain what serves.

WHY A SECOND IMPLEMENTATION AND NOT A PATCH. The legacy recipe computes every feature twice:
`build_features` (training, vectorised over history) and `build_upcoming_features` (live, a
per-fixture loop). The two disagree on most of the 42 Standard serving features — measured by
`scripts/standard_feature_audit.py` — for reasons that cannot be patched one at a time:

  * home_advantage           training: rolling per team; live: the constant 1.0 (P0 bug 2).
                             Training's own version also aligns a league-grouped expanding mean
                             onto date-sorted rows (`groupby().expanding()` returns league order),
                             so its league average belongs to some other match.
  * *_last5 form             training: last 5 ROWS incl. gaps, keyed by team name ACROSS leagues;
                             live: league-scoped, `list(home_tail)+list(away_tail)` then `[-5:]`,
                             i.e. mostly the last five AWAY matches, not the last five matches.
  * away_ht_*_last5          live: not league-scoped at all (P0 bug 4).
  * *_ht_attack_str          training divides by a constant 0.65 (league_avg_goals is computed
                             AFTER it is read); live divides by the all-time league half-average.
  * *_attack/defense_str     training: as-of league average; live: all-time league average.
  * season venue stats       training: (league, season-LABEL) groups, and season labels are mixed
                             ("2025/26", "2025/2026", "2025") so a season can split in two; live:
                             the team's LATEST season (last season at a season start), then
                             overwritten by API-Football /teams/statistics.
  * rest days                live: `tail(1000)` of the whole history then a POSITIONAL tail that
                             misaligns whenever the board is not date-sorted.
  * bookmaker_overround      training: football-data CLOSING average (AvgC); live: one book's
                             current price (P0 bug 5).

So the repair is structural: ONE function computes every feature from a frame that may contain
unplayed rows. Training runs it over history; serving appends the board's fixtures as unplayed
rows and runs the SAME function. Parity holds by construction and the audit measures it.

INVARIANTS (all tested in tests/test_standard_forensic_repair.py)
  * as-of: a row's features use only PLAYED rows with a strictly earlier position in the
    (league, team) timeline, and league averages use only strictly earlier DATES;
  * batch invariance: unplayed rows contribute nothing, so a fixture's features do not depend on
    which other fixtures are on the board;
  * league scope: every team statistic is keyed (league, team) — a same-named club in another
    competition can never leak in;
  * prices: `price_mode` in {"none", "asof", "closing"}. "asof" reads ONLY `odds_over25_entry` /
    `odds_under25_entry` (never a *C* closing column); "closing" reproduces the legacy
    contamination for the CLOSING_CONTAMINATED_REPRODUCTION arm; "none" drops price features.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

log = logging.getLogger(__name__)

FEATURE_RECIPE_VERSION = "v9r-features-1.0"
PRICE_MODES = ("none", "asof", "closing")
TEAM_SCOPES = ("league", "club")
#: The price-derived features of the Standard serving payload (models/model_v9_standard.pkl).
#: p_over25_poisson_dc is NOT price-derived (it only falls back to a price if poisson is absent,
#: which the repaired recipe never does).
PRICE_FEATURES = ("bookmaker_overround",)

#: Exact feature list of the serving Standard payload as of 2026-10-28 (42 columns).
STANDARD_SERVING_FEATURES = [
    "home_scored_last5", "away_scored_last5", "home_conceded_last5", "away_conceded_last5",
    "home_over25_last5", "away_over25_last5", "home_attack_str", "away_attack_str",
    "home_defense_str", "away_defense_str", "home_advantage", "away_home_adv_factor",
    "home_rest_days", "away_rest_days", "home_corners_pg_roll", "away_corners_pg_roll",
    "home_fouls_pg_roll", "away_fouls_pg_roll", "combined_sot_ratio", "home_xg_last5",
    "away_xg_last5", "home_insidebox_last5", "away_insidebox_last5", "home_season_goals_h",
    "away_season_goals_a", "home_season_conceded_h", "away_season_conceded_a", "home_cs_rate_h",
    "away_cs_rate_a", "home_ht_scored_last5", "away_ht_scored_last5", "home_ht_conceded_last5",
    "away_ht_conceded_last5", "home_ht_attack_str", "away_ht_attack_str", "combined_ht_goals_avg",
    "home_ht_over05_rate", "away_ht_over05_rate", "home_ht_over15_rate", "away_ht_over15_rate",
    "bookmaker_overround", "p_over25_poisson_dc",
]

_HA_HOME_DEFAULT = 0.55
_HA_AWAY_DEFAULT = 0.45
_LG_AWAY_DEFAULT = 1.1


def feature_cols_for(price_mode: str = "none", base: list | None = None) -> list:
    """The repaired recipe's model inputs for a price mode ("none" drops price features)."""
    if price_mode not in PRICE_MODES:
        raise ValueError(f"price_mode must be one of {PRICE_MODES}, got {price_mode!r}")
    cols = list(base or STANDARD_SERVING_FEATURES)
    if price_mode == "none":
        cols = [c for c in cols if c not in PRICE_FEATURES]
    return cols


def season_key(dates: pd.Series, leagues: pd.Series) -> pd.Series:
    """Canonical season, DERIVED from the date (CLAUDE.md: never trust a typed season label).

    Standard-format leagues run Aug-May -> "YYYY/YY" with a July roll; every other league is
    keyed by calendar year. The stored `season` column mixes "2025/26", "2025/2026" and "2025"
    for the same competition, which is why it is not used.
    """
    d = pd.to_datetime(dates, errors="coerce")
    std = leagues.isin(config.STANDARD_FORMAT_LEAGUES)
    start = d.dt.year.where(d.dt.month >= 7, d.dt.year - 1)
    std_lbl = start.astype("Int64").astype(str) + "/" + ((start + 1) % 100).astype("Int64").map(
        lambda v: f"{int(v):02d}" if pd.notna(v) else "")
    cal_lbl = d.dt.year.astype("Int64").astype(str)
    return pd.Series(np.where(std, std_lbl, cal_lbl), index=dates.index)


def _roll_before(df: pd.DataFrame, keys: list, col: str, n: int | None) -> pd.Series:
    """Mean of the last `n` NON-NULL values of `col` strictly before each row, per group.

    `df` must already be sorted in timeline order. n=None -> expanding (all prior values).
    Unplayed rows carry NaN and therefore never contribute — the batch-invariance guarantee.
    """
    def _one(x: pd.Series) -> pd.Series:
        obs = x.dropna()
        if obs.empty:
            return pd.Series(np.nan, index=x.index)
        r = (obs.expanding().mean() if n is None
             else obs.rolling(n, min_periods=1).mean())
        return r.reindex(x.index).ffill().shift(1)
    return df.groupby(keys, sort=False)[col].transform(_one)


def _league_asof_mean(df: pd.DataFrame, col: str) -> pd.Series:
    """Per-league mean of `col` over played rows on STRICTLY EARLIER DATES (same-day excluded)."""
    played = df[col].notna()
    agg = (df[played].groupby(["league", "date"])[col].agg(["sum", "count"])
           .reset_index().sort_values(["league", "date"]))
    agg["m_through"] = (agg.groupby("league")["sum"].cumsum()
                        / agg.groupby("league")["count"].cumsum())
    result = np.full(len(df), np.nan)
    qdates = df["date"].to_numpy()
    for lg, idx in df.groupby("league", sort=False).indices.items():
        a = agg[agg["league"] == lg]
        if a.empty:
            continue
        dts = a["date"].to_numpy()
        vals = a["m_through"].to_numpy()
        # index of the last played date STRICTLY before the row's date
        pos = np.searchsorted(dts, qdates[idx], side="left") - 1
        result[idx] = np.where(pos >= 0, vals[np.clip(pos, 0, None)], np.nan)
    return pd.Series(result, index=df.index)


def _price_overround(over, under) -> pd.Series:
    o = 1.0 / pd.to_numeric(over, errors="coerce").replace(0, np.nan)
    u = 1.0 / pd.to_numeric(under, errors="coerce").replace(0, np.nan)
    return (o + u - 1.0).clip(lower=0.0)


def build_features_repaired(matches: pd.DataFrame, n: int | None = None,
                            price_mode: str = "none", team_scope: str = "league") -> pd.DataFrame:
    """Canonical as-of features. Rows with no full-time score are UNPLAYED: they receive
    features and contribute nothing. Returns the frame sorted by (date, league, home, away)
    with a fresh RangeIndex and the original row position in `_src_pos`.

    team_scope="league" (default, the brief's rule): team rolling stats keyed (league, team).
    team_scope="club": keyed by club name ACROSS competitions but strictly CHRONOLOGICAL — the
    explicit cross-league variant (legacy crossed leagues too, but in alphabetical league order,
    so Standard fixtures of clubs that changed league had a LATER match inside their window).
    Use only on an identity-repaired frame (src/identity_repair) so one name = one club."""
    if team_scope not in TEAM_SCOPES:
        raise ValueError(f"team_scope must be one of {TEAM_SCOPES}")
    if price_mode not in PRICE_MODES:
        raise ValueError(f"price_mode must be one of {PRICE_MODES}, got {price_mode!r}")
    n = n or config.ROLLING_N
    df = matches.copy()
    df["_src_pos"] = np.arange(len(df))
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if getattr(df["date"].dt, "tz", None) is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df["league"] = df["league"].astype(str)
    df["home_team"] = df["home_team"].astype(str)
    df["away_team"] = df["away_team"].astype(str)
    for c in ("home_goals", "away_goals"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    # Upcoming rows first within a date: a same-day played match is never "before" a fixture.
    df["_upc"] = (~(df["home_goals"].notna() & df["away_goals"].notna())).astype(int)
    df = (df.sort_values(["date", "_upc", "league", "home_team", "away_team", "_src_pos"],
                         ascending=[True, False, True, True, True, True], kind="mergesort")
            .reset_index(drop=True))
    played = df["_upc"] == 0
    df["_season_key"] = season_key(df["date"], df["league"])

    def _num(c):
        return pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)

    hg, ag = df["home_goals"].where(played), df["away_goals"].where(played)
    hth, hta = _num("ht_home_goals").where(played), _num("ht_away_goals").where(played)
    tot_ht = hth + hta
    df["_total_goals"] = hg + ag
    df["_away_goals_played"] = ag

    # ── league as-of averages (strictly earlier dates) ───────────────────────────────
    df["league_avg_goals"] = _league_asof_mean(df, "_total_goals")
    lg_away = _league_asof_mean(df, "_away_goals_played")
    home_flag = (hg > lg_away.fillna(_LG_AWAY_DEFAULT)).astype(float).where(played)

    # ── team-centric long frame, keyed (league, team) ────────────────────────────────
    def _side(is_home: bool) -> pd.DataFrame:
        p, o = ("home", "away") if is_home else ("away", "home")
        s = pd.DataFrame({
            "_row": df.index, "date": df["date"], "league": df["league"],
            "team": df[f"{p}_team"], "is_home": int(is_home), "_played": played,
            "scored": df[f"{p}_goals"].where(played), "conceded": df[f"{o}_goals"].where(played),
            "over25": ((hg + ag) > 2.5).astype(float).where(played),
            "shots": _num(f"{p}_shots").where(played), "sot": _num(f"{p}_sot").where(played),
            "corners": _num(f"{p}_corners").where(played), "fouls": _num(f"{p}_fouls").where(played),
            "xg": _num(f"{p}_xg").where(played), "insidebox": _num(f"{p}_insidebox").where(played),
            "ht_scored": (hth if is_home else hta), "ht_conceded": (hta if is_home else hth),
            "ht_o05": (tot_ht >= 1).astype(float).where(tot_ht.notna()),
            "ht_o15": (tot_ht >= 2).astype(float).where(tot_ht.notna()),
            "home_flag": home_flag if is_home else np.nan,
            "played_date": df["date"].where(played),
        })
        return s
    tc = pd.concat([_side(True), _side(False)], ignore_index=True)
    keys = ["league", "team"] if team_scope == "league" else ["team"]
    tc = tc.sort_values(keys + ["date", "_row"], kind="mergesort").reset_index(drop=True)
    for stat in ("scored", "conceded", "over25", "shots", "sot", "corners", "fouls", "xg",
                 "insidebox", "ht_scored", "ht_conceded", "ht_o05", "ht_o15", "home_flag"):
        tc[f"r_{stat}"] = _roll_before(tc, keys, stat, n)
    prev_played = tc.groupby(keys, sort=False)["played_date"].transform(lambda x: x.ffill().shift(1))
    tc["r_rest"] = (tc["date"] - prev_played).dt.days

    names = {"r_scored": "scored_last5", "r_conceded": "conceded_last5",
             "r_over25": "over25_last5", "r_shots": "shots_last5", "r_sot": "sot_last5",
             "r_corners": "corners_pg_roll", "r_fouls": "fouls_pg_roll", "r_xg": "xg_last5",
             "r_insidebox": "insidebox_last5", "r_ht_scored": "ht_scored_last5",
             "r_ht_conceded": "ht_conceded_last5", "r_ht_o05": "ht_over05_rate",
             "r_ht_o15": "ht_over15_rate", "r_rest": "rest_days", "r_home_flag": "_ha"}
    for is_home, pfx in ((1, "home_"), (0, "away_")):
        part = tc[tc["is_home"] == is_home].set_index("_row")
        for src, dst in names.items():
            df[pfx + dst] = part[src].reindex(df.index).to_numpy()
    df["home_advantage"] = df.pop("home__ha").fillna(_HA_HOME_DEFAULT)
    df["away_home_adv_factor"] = df.pop("away__ha").fillna(_HA_AWAY_DEFAULT)

    # ── season-to-date venue stats, canonical season key ────────────────────────────
    hcs = (ag == 0).astype(float).where(played)
    acs = (hg == 0).astype(float).where(played)
    tmp = pd.DataFrame({"league": df["league"], "s": df["_season_key"], "ht": df["home_team"],
                        "at": df["away_team"], "hg": hg, "ag": ag, "hcs": hcs, "acs": acs})
    df["home_season_goals_h"] = _roll_before(tmp, ["league", "s", "ht"], "hg", None)
    df["home_season_conceded_h"] = _roll_before(tmp, ["league", "s", "ht"], "ag", None)
    df["home_cs_rate_h"] = _roll_before(tmp, ["league", "s", "ht"], "hcs", None)
    df["away_season_goals_a"] = _roll_before(tmp, ["league", "s", "at"], "ag", None)
    df["away_season_conceded_a"] = _roll_before(tmp, ["league", "s", "at"], "hg", None)
    df["away_cs_rate_a"] = _roll_before(tmp, ["league", "s", "at"], "acs", None)

    # ── derived (same formulas as legacy where legacy was internally consistent) ─────
    half = (df["league_avg_goals"] / 2).replace(0, np.nan)
    df["home_attack_str"] = df["home_scored_last5"] / half
    df["away_attack_str"] = df["away_scored_last5"] / half
    df["home_defense_str"] = df["home_conceded_last5"] / half
    df["away_defense_str"] = df["away_conceded_last5"] / half
    df["home_ht_attack_str"] = df["home_ht_scored_last5"] / half
    df["away_ht_attack_str"] = df["away_ht_scored_last5"] / half
    df["combined_ht_goals_avg"] = (df["home_ht_scored_last5"].fillna(0)
                                   + df["away_ht_scored_last5"].fillna(0))
    df["home_sot_ratio_last5"] = df["home_sot_last5"] / df["home_shots_last5"].replace(0, np.nan)
    df["away_sot_ratio_last5"] = df["away_sot_last5"] / df["away_shots_last5"].replace(0, np.nan)
    df["combined_sot_ratio"] = (df["home_sot_ratio_last5"].fillna(0.35)
                                + df["away_sot_ratio_last5"].fillna(0.33)) / 2
    from src.poisson import dixon_coles_p_over25 as _dc
    lh = df["home_scored_last5"].clip(lower=0.3, upper=4.0)
    la = df["away_scored_last5"].clip(lower=0.3, upper=4.0)
    df["p_over25_poisson_dc"] = [(_dc(h, a) if (pd.notna(h) and pd.notna(a)) else np.nan)
                                 for h, a in zip(lh, la)]

    # ── prices ───────────────────────────────────────────────────────────────────────
    if price_mode == "closing":
        df["bookmaker_overround"] = _price_overround(df.get("odds_over25"), df.get("odds_under25"))
        df["_price_class"] = "LEGACY_CLOSING_OR_LIVE"
    elif price_mode == "asof":
        if "odds_over25_entry" in df.columns and "odds_under25_entry" in df.columns:
            df["bookmaker_overround"] = _price_overround(df["odds_over25_entry"],
                                                         df["odds_under25_entry"])
        else:
            df["bookmaker_overround"] = np.nan
        df["_price_class"] = df.get("entry_price_class", "ENTRY_ASOF")
    else:
        df["bookmaker_overround"] = np.nan
        df["_price_class"] = "NONE"

    df = df.drop(columns=["_total_goals", "_away_goals_played"])
    df["feature_recipe_version"] = FEATURE_RECIPE_VERSION
    return df


def _resolve_names(up: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """OddsAPI spelling -> history spelling, league-scoped, refusing ambiguity (invariant 11)."""
    from src.team_names import resolve
    up = up.copy()
    clubs = {}
    for lg, g in hist.groupby("league"):
        clubs[str(lg)] = sorted(set(g["home_team"].astype(str)) | set(g["away_team"].astype(str)))
    for c in ("home_team", "away_team"):
        out = []
        for lg, name in zip(up["league"].astype(str), up[c].astype(str)):
            pool = clubs.get(lg, [])
            if name in pool or not pool:
                out.append(name)
            else:
                hit = resolve(name, pool)
                out.append(hit if hit else name)
        up[c] = out
    return up


def build_upcoming_features_repaired(upcoming: pd.DataFrame, historical: pd.DataFrame,
                                     n: int | None = None, price_mode: str = "none",
                                     cutoff=None, team_scope: str = "league") -> pd.DataFrame:
    """Serve-time features through the SAME function training uses.

    `upcoming` needs date, league, home_team, away_team (+ odds_over25/odds_under25 if a price
    mode needs them). Live prices are the ENTRY price at scoring time, so in "asof" mode they
    are written to the *_entry columns. `cutoff` (optional Timestamp) drops history on/after it —
    used by the replay audit; live callers pass nothing because history is all in the past.
    Returns one row per upcoming fixture, in input order, with the upcoming columns preserved.
    """
    up = upcoming.copy().reset_index(drop=True)
    up["_upc_id"] = np.arange(len(up))
    up["date"] = pd.to_datetime(up["date"], errors="coerce")
    if getattr(up["date"].dt, "tz", None) is not None:
        up["date"] = up["date"].dt.tz_convert(None)
    up["date"] = up["date"].dt.normalize()
    hist = historical.copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.normalize()
    if cutoff is not None:
        hist = hist[hist["date"] < pd.Timestamp(cutoff)]
    hist = hist[hist["home_goals"].notna() & hist["away_goals"].notna()]
    leagues = set(up["league"].astype(str))
    in_lg = hist["league"].astype(str).isin(leagues)
    up_named = _resolve_names(up, hist[in_lg])
    if team_scope == "league":
        hist = hist[in_lg]                                     # every feature is league-scoped
    else:
        clubs = set(up_named["home_team"]) | set(up_named["away_team"])
        hist = hist[in_lg | hist["home_team"].isin(clubs) | hist["away_team"].isin(clubs)]
    core = up_named[["date", "league", "home_team", "away_team", "_upc_id"]].copy()
    core["home_goals"] = np.nan
    core["away_goals"] = np.nan
    if price_mode == "asof":
        core["odds_over25_entry"] = pd.to_numeric(up.get("odds_over25"), errors="coerce")
        core["odds_under25_entry"] = pd.to_numeric(up.get("odds_under25"), errors="coerce")
        core["entry_price_class"] = "LIVE_ENTRY_AT_SCORING_TIME"
    elif price_mode == "closing":
        core["odds_over25"] = pd.to_numeric(up.get("odds_over25"), errors="coerce")
        core["odds_under25"] = pd.to_numeric(up.get("odds_under25"), errors="coerce")
    full = build_features_repaired(pd.concat([hist, core], ignore_index=True), n=n,
                                   price_mode=price_mode, team_scope=team_scope)
    feats =full[full["_upc_id"].notna()].sort_values("_upc_id").copy()
    feats["_upc_id"] = feats["_upc_id"].astype(int)
    produced = [c for c in feats.columns
                if c not in hist.columns and c not in core.columns and c != "_src_pos"]
    base = up.drop(columns=[c for c in produced if c in up.columns])
    out = base.merge(feats[["_upc_id"] + produced], on="_upc_id", how="left").drop(columns=["_upc_id"])
    out["history_team_home"] = up_named["home_team"].to_numpy()
    out["history_team_away"] = up_named["away_team"].to_numpy()
    return out
