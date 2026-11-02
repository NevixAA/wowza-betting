"""
Player Feature Engineering v3 — match-level rolling features
=============================================================
build_features(match_rows)
  Input:  flat list of player-match dicts from collect_match_history()
          Each dict = one player in one specific completed match.
  Output: DataFrame with proper rolling features (shift=1, no leakage)
          and per-match binary targets.

build_upcoming_features(players, history_df)
  Uses actual rolling history from build_features() output.

DATA TRUTH (owner rule, 2026-11-01). No feature is ever a typed value:
  * every opponent / keeper / referee / league quantity is COMPUTED from the matches we hold
    (the per-(fixture, team) table `team_match_table`), with an explicit minimum sample
    (config.MIN_*); below it the value is NaN;
  * "league average" normalisers are the league's OWN measured average from its prior fixtures
    (`_league_asof`), never 1.3 / 6.0 / 3.5 / 6.8 / 0.12;
  * a missing value stays NaN (no fillna(0.0) / fillna(<prior>)); the serving model's imputer
    (model.predict_proba) decides what a missing input means, and logs it;
  * training and serving share ONE definition: `team_match_table` + `_team_rolling` for training,
    `_team_now` for serving, and `_add_composites` for every derived feature, so a composite can
    no longer be computed one way in training and another way at serve time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from . import league_quality as _LQ
from .setpiece import features as _SP

NAN = float("nan")

# Retired set-piece composites (typed position weights over set-piece goal classifications that
# API-Football does not provide; 100% missing in the 2026-11-01 rebuild). Kept as NaN columns for
# the legacy serving payloads; the measured replacements are _SP.SETPIECE_COLUMNS and
# corner_matchup_index. defender_sot_role_index is DROPPED on the same terms: its real-data
# counterpart is us_sp_sot_p90 (set-piece shots on target per 90, every position, Understat).
LEGACY_SETPIECE_COLUMNS = ("sp_taker_score", "sp_receiver_score", "set_piece_threat_score",
                           "set_piece_corner_matchup", "defender_sot_role_index")

# season_* feature -> raw per-match source column, for AS-OF-DATE recomputation.
_SEASON_SRC = {
    "season_goals_pg":       "goals",
    "season_assists_pg":     "assists",
    "season_shots_pg":       "shots_total",
    "season_sot_pg":         "shots_on_target",
    "season_cards_pg":       "yellow_cards",
    "season_minutes_pg":     "minutes",
    "season_dribble_pg":     "dribbles_success",
    "season_fouls_pg":       "fouls_committed",
    "season_fouls_drawn_pg": "fouls_drawn",
    "season_start_rate":     "started",
}

# The values the pre-2026-11-01 build_features WROTE into a profile column that the collect did not
# provide (`df["height_cm"] = 180.0`, weight 75, age 25). Measured 2026-11-01: 100% of the 876,130
# player_history rows carry exactly these three values (enrich-profiles has never populated them).
# They are recognised here only to be treated as MISSING; a real 180 cm player is lost to NaN until
# enrich-profiles writes real heights, which is the honest side of that trade.
_PROFILE_PLACEHOLDERS = {"height_cm": 180.0, "weight_kg": 75.0, "age": 25.0}


# ── small helpers ─────────────────────────────────────────────────────────────────────────────────

def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric column, or an all-NaN column when absent (never a typed default)."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _ratio(a, b) -> pd.Series:
    """a / b with an undefined ratio (b <= 0 or missing) left NaN."""
    b = pd.to_numeric(b, errors="coerce").astype(float)
    return pd.to_numeric(a, errors="coerce") / b.where(b > 0)


def _profile(df: pd.DataFrame, col: str) -> pd.Series:
    s = _num(df, col)
    return s.where(s != _PROFILE_PLACEHOLDERS[col]) if col in _PROFILE_PLACEHOLDERS else s


# ── collected real data (scripts/collect_*.py -> output/collect, owner decision C/D 2026-11-02) ──

COLLECT_DIR = config.OUTPUT_DIR / "collect"
_COLLECTED: dict = {}


def _collected(name: str) -> pd.DataFrame:
    """output/collect/<name>.parquet (cached on path + mtime); empty frame when absent."""
    p = COLLECT_DIR / f"{name}.parquet"
    stamp = (str(p), p.stat().st_mtime if p.exists() else None)
    hit = _COLLECTED.get(name)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    try:
        df = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    except Exception as e:                                              # noqa: BLE001
        print(f"[features] {p.name} unreadable ({e}): not used")
        df = pd.DataFrame()
    _COLLECTED[name] = (stamp, df)
    return df


def _season_year(s) -> pd.Series:
    """API-Football season label -> its start year ('2024' / '2024/25' -> 2024), NaN if none."""
    return pd.to_numeric(pd.Series(s).astype(str).str.extract(r"^(\d{4})")[0], errors="coerce")


def collected_profiles() -> pd.DataFrame:
    """One row per player_id from /players?id&season (status ok): height_cm and weight_kg of the
    LATEST season that reports them, birth_date when every season that reports one agrees (two
    different provider birth dates -> refused, NaN)."""
    p = _collected("player_profiles")
    memo = _COLLECTED.get("_profiles_derived")
    if memo is not None and memo[0] is p:
        return memo[1]
    out = _derive_profiles(p)
    _COLLECTED["_profiles_derived"] = (p, out)
    return out


def _derive_profiles(p: pd.DataFrame) -> pd.DataFrame:
    cols = ["player_id", "height_cm", "weight_kg", "birth_date"]
    if p.empty or "status" not in p.columns:
        return pd.DataFrame(columns=cols)
    p = p[p["status"] == "ok"].copy()                     # the collector's own status label
    p["_y"] = _season_year(p["season"]).to_numpy()
    p = p.sort_values(["player_id", "_y"])
    out = pd.DataFrame(index=pd.Index(p["player_id"].unique(), name="player_id"))
    for c in ("height_cm", "weight_kg"):
        v = pd.to_numeric(p[c], errors="coerce")
        out[c] = v.groupby(p["player_id"]).last()           # last non-null by season order
    bd = p.dropna(subset=["birth_date"]).groupby("player_id")["birth_date"].agg(
        lambda s: s.iloc[0] if s.nunique() == 1 else None)
    out["birth_date"] = bd
    return out.reset_index()


def collected_sidelined() -> dict:
    """{player_id: [{start, end}]} for every player whose /sidelined answer was collected (an empty
    list is a real 'no recorded spell'); players never asked are absent."""
    players, spells = _collected("sidelined_players"), _collected("sidelined_spells")
    if players.empty:
        return {}
    out = {int(p): [] for p in pd.to_numeric(players["player_id"], errors="coerce").dropna()}
    if not spells.empty:
        for pid, s, e in zip(spells["player_id"], spells["start"], spells["end"]):
            if int(pid) in out:
                out[int(pid)].append({"start": s, "end": e})
    return out


def _season_pen_totals(hist: pd.DataFrame) -> pd.DataFrame:
    """Per (player_id, season start year): penalty scored / attempts. The collected season record
    (/players statistics.penalty, all competitions) where the provider reported it, else the sum of
    the per-match rows we hold for that season. Real counts only."""
    cols = ["player_id", "_y", "pen_s", "pen_a"]
    parts = []
    if {"penalty_scored", "penalty_missed", "season", "player_id"} <= set(hist.columns):
        ps, pm = _num(hist, "penalty_scored"), _num(hist, "penalty_missed")
        h = pd.DataFrame({"player_id": hist["player_id"], "_y": _season_year(hist["season"]).to_numpy(),
                          "pen_s": ps, "pen_a": ps + pm}).dropna(subset=["player_id", "_y"])
        h = h.groupby(["player_id", "_y"]).sum(min_count=1).reset_index()
        parts.append(h.assign(_src=1))
    prof = _collected("player_profiles")
    if not prof.empty and "pen_entries" in prof.columns:
        p = prof[pd.to_numeric(prof["pen_entries"], errors="coerce") > 0]
        if len(p):
            parts.append(pd.DataFrame({
                "player_id": p["player_id"].to_numpy(), "_y": _season_year(p["season"]).to_numpy(),
                "pen_s": pd.to_numeric(p["penalty_scored"], errors="coerce").to_numpy(),
                "pen_a": (pd.to_numeric(p["penalty_scored"], errors="coerce")
                          + pd.to_numeric(p["penalty_missed"], errors="coerce")).to_numpy(),
                "_src": 0}))
    if not parts:
        return pd.DataFrame(columns=cols)
    t = pd.concat(parts, ignore_index=True).dropna(subset=["pen_a"])
    t = t.sort_values(["player_id", "_y", "_src"]).drop_duplicates(["player_id", "_y"], keep="first")
    return t[cols]


def _penalty_asof(df: pd.DataFrame) -> pd.Series:
    """penalty_conversion_rate per row of a frame sorted by (player_id, date): earlier seasons'
    totals (_season_pen_totals) + this season's per-match attempts strictly before the row; NaN
    below config.MIN_PEN_ATTEMPTS attempts."""
    if not {"penalty_scored", "penalty_missed", "player_id"} <= set(df.columns):
        return pd.Series(np.nan, index=df.index)
    ps, pm = _num(df, "penalty_scored"), _num(df, "penalty_missed")
    att = ps + pm
    if "season" not in df.columns:                  # no season: the per-match record only
        cs = ps.groupby(df["player_id"]).cumsum() - ps
        ca = att.groupby(df["player_id"]).cumsum() - att
        return (cs / ca).where(ca >= config.MIN_PEN_ATTEMPTS)
    y = pd.Series(_season_year(df["season"]).to_numpy(), index=df.index)
    key = [df["player_id"], y]
    in_s = ps.groupby(key).cumsum() - ps
    in_a = att.groupby(key).cumsum() - att
    tot = _season_pen_totals(df)
    prior_s = pd.Series(0.0, index=df.index)
    prior_a = pd.Series(0.0, index=df.index)
    if len(tot):
        tot = tot.sort_values(["player_id", "_y"])
        g = tot.groupby("player_id")
        tot = tot.assign(_cs=g["pen_s"].cumsum(), _ca=g["pen_a"].cumsum())
        q = pd.DataFrame({"player_id": pd.to_numeric(df["player_id"], errors="coerce").to_numpy(float),
                          "_y": y.to_numpy(float), "_i": np.arange(len(df))}).dropna(subset=["_y", "player_id"])
        q = q.sort_values("_y")
        t = tot[["player_id", "_y", "_cs", "_ca"]].astype({"_y": float, "player_id": float}).sort_values("_y")
        # cumulative totals of the seasons strictly BEFORE the row's season
        hit = pd.merge_asof(q, t, on="_y", by="player_id", allow_exact_matches=False)
        hit = hit.dropna(subset=["_ca"])
        prior_s.iloc[hit["_i"].to_numpy()] = hit["_cs"].to_numpy()
        prior_a.iloc[hit["_i"].to_numpy()] = hit["_ca"].to_numpy()
    s, a = prior_s + in_s, prior_a + in_a
    return (s / a.where(a > 0)).where(a >= config.MIN_PEN_ATTEMPTS)


def _penalty_now(tot: pd.DataFrame, pid, phist_all: pd.DataFrame, season) -> float:
    """Serving counterpart of _penalty_asof for an upcoming fixture of `season` (None = unknown:
    every stored season counts, all of them are before the fixture)."""
    t = tot[pd.to_numeric(tot["player_id"], errors="coerce") == pid] if len(tot) else tot
    if season is None:
        s, a = (float(t["pen_s"].sum()), float(t["pen_a"].sum())) if len(t) else (0.0, 0.0)
    else:
        y = float(_season_year([season]).iloc[0])
        prior = t[t["_y"] < y] if len(t) else t
        s, a = (float(prior["pen_s"].sum()), float(prior["pen_a"].sum())) if len(prior) else (0.0, 0.0)
        if "season" in phist_all.columns:
            cur = phist_all[_season_year(phist_all["season"]).to_numpy() == y]
            ps, pm = _num(cur, "penalty_scored"), _num(cur, "penalty_missed")
            s += float(ps.sum())
            a += float((ps + pm).sum())
    return s / a if a >= config.MIN_PEN_ATTEMPTS else NAN


def _height_series(d: pd.DataFrame) -> pd.Series:
    """Player height per row: the collected profile first (a real 180 cm stays 180), else a stored
    value that is not the old placeholder."""
    stored = _profile(d, "height_cm")
    prof = collected_profiles()
    if prof.empty or "player_id" not in d.columns:
        return stored
    h = d["player_id"].map(prof.set_index("player_id")["height_cm"]).astype(float)
    return h.where(h.notna(), stored)


def _attach_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """height_cm / weight_kg / age per row: collected profile first, else a REAL stored value
    (the old placeholders count as missing). Age at the row's match date from the birth date."""
    prof = collected_profiles()
    stored = {c: _profile(df, c) for c in ("age", "height_cm", "weight_kg")}
    if prof.empty:
        for c, v in stored.items():
            df[c] = v
        return df
    p = prof.set_index("player_id")
    pid = df["player_id"]
    for c in ("height_cm", "weight_kg"):
        df[c] = pid.map(p[c]).astype(float).where(lambda s: s.notna(), stored[c])
    bd = pd.to_datetime(pid.map(p["birth_date"]), errors="coerce")
    age = (pd.to_datetime(df["date"], errors="coerce") - bd).dt.days / 365.25
    df["age"] = age.where(age.notna(), stored["age"])
    return df


def _grouped_prev(grp, col: str):
    obj, keys = grp.obj, grp.keys
    by = [obj[k] for k in keys] if isinstance(keys, list) else obj[keys]
    return obj, by, pd.to_numeric(obj[col], errors="coerce").groupby(by).shift(1)


def _realign(out: pd.Series, obj: pd.DataFrame) -> pd.Series:
    return out.reset_index(level=list(range(out.index.nlevels - 1)), drop=True).reindex(obj.index)


def _roll(grp, col: str, n: int, min_n: int = 1) -> pd.Series:
    """Mean of the previous n values per group (shift(1): pre-match knowledge only), NaN unless at
    least min_n of them are present. Vectorised (groupby.rolling), same result as the old per-group
    lambda."""
    obj, by, prev = _grouped_prev(grp, col)
    return _realign(prev.groupby(by).rolling(n, min_periods=min_n).mean(), obj)


def _expand(grp, col: str, min_n: int = 1) -> pd.Series:
    """Mean of ALL previous values per group (shift(1) + expanding)."""
    obj, by, prev = _grouped_prev(grp, col)
    return _realign(prev.groupby(by).expanding(min_periods=min_n).mean(), obj)


def _first_letter(pos: pd.Series) -> pd.Series:
    return pos.astype(str).str.strip().str.upper().str[:1].where(pos.notna(), "")


# ── team identity (no raw string matching across competitions) ──────────────────────────────────

_SCOPE_CACHE: dict = {}


def team_scope(league) -> str | None:
    """Identity scope of the clubs/teams in `league`, from the registry: a club league's COUNTRY (so
    a promoted club keeps one identity across divisions and two same-named clubs in different
    countries never merge), 'national_teams', or 'club_cup'. None for a competition the registry
    does not know (Friendlies, WC qualifiers, AFCON in the current history): its rows feed no
    opponent table, because nothing tells us which teams they are."""
    if league in _SCOPE_CACHE:
        return _SCOPE_CACHE[league]
    reg = config.league_registry()
    ctype = reg.competition_type(league) if isinstance(league, str) else None
    if ctype == "club_league":
        country = (reg.entry(league) or {}).get("country")
        scope = f"club_league:{country}" if country else None
    elif ctype in ("national_teams", "club_cup"):
        scope = ctype
    else:
        scope = None
    _SCOPE_CACHE[league] = scope
    return scope


def _team_keys(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(own team key, opponent key) per player row: '<API-Football team name>|<scope>'. Within one
    API-Football competition scope the team name IS the identity (the provider names a club the same
    way in every fixture); history carries no team id. NaN where the scope is unknown."""
    scope = df["league"].map(team_scope) if "league" in df.columns else pd.Series(None, index=df.index)
    if "opponent" in df.columns:
        opp = df["opponent"]
    else:
        opp = df["away_team"].where(df["is_home"].astype(bool), df["home_team"])
    own = df["team"].astype(str) + "|" + scope
    oth = opp.astype(str) + "|" + scope
    return own.where(scope.notna()), oth.where(scope.notna())


# ── one row per (fixture, team): what the team did and what it conceded ─────────────────────────

# Per-fixture team quantities whose rolling mean becomes an opponent feature, with the minimum
# number of prior fixtures that must carry the quantity.
_TEAM_ROLL = {
    "goals_conceded":     ("opp_goals_conceded_pg",        "MIN_OPP_MATCHES"),
    "sot_conceded":       ("opp_sot_conceded_pg",          "MIN_OPP_MATCHES"),
    "goals_scored":       ("team_goals_pg_roll",           "MIN_OPP_MATCHES"),
    "def_aerial":         ("opp_def_aerial_win_rate",      "MIN_OPP_MATCHES"),
    "def_fouls_pg":       ("opp_def_fouls_pg",             "MIN_OPP_MATCHES"),
    "def_cards_pg":       ("opp_def_cards_pg",             "MIN_OPP_MATCHES"),
    "def_rating":         ("opp_def_player_rating_pg",     "MIN_OPP_MATCHES"),
    "def_rating_top":     ("opp_top_def_rating",           "MIN_OPP_MATCHES"),
    "def_height":         ("opp_def_mean_height",          "MIN_OPP_MATCHES"),
    "sp_conceded":        ("opp_sp_goals_conceded_pg",     "MIN_OPP_MATCHES"),
    "gk_save_rate":       ("opp_gk_save_rate",             "MIN_GK_MATCHES"),
    "gk_saves":           ("opp_gk_saves_pg",              "MIN_GK_MATCHES"),
    "conc_fwd_goals":     ("opp_goals_conceded_vs_fwd_pg", "MIN_OPP_MATCHES"),
    "conc_fwd_sot":       ("opp_sot_conceded_vs_fwd_pg",   "MIN_OPP_MATCHES"),
    "conc_mid_goals":     ("opp_goals_conceded_vs_mid_pg", "MIN_OPP_MATCHES"),
    "conc_mid_sot":       ("opp_sot_conceded_vs_mid_pg",   "MIN_OPP_MATCHES"),
    "mid_fouls_pg":       ("opp_mid_fouls_pg",             "MIN_OPP_MATCHES"),
    "mid_cards_pg":       ("opp_mid_cards_pg",             "MIN_OPP_MATCHES"),
    "corners_conceded":   ("opp_corners_conceded_pg",      "MIN_OPP_MATCHES"),
}


def team_match_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (fixture_id, team key) from the collected player rows of that fixture.

    goals/sot conceded are what the OTHER side's players scored in the same fixture; defender,
    keeper and midfielder quantities are this team's OWN players of that role. A quantity that the
    collected rows cannot measure (no keeper row, no duels, no SP classification) is NaN, never 0.

    Semantic fixes vs the pre-2026-11-01 code, stated so a reviewer can check them:
      * opp_mid_fouls_pg / opp_mid_cards_pg are the opponent's OWN midfielders' fouls/cards (the
        docstring's meaning); the old self-join rolled the fouls of the opponent's OPPONENTS.
      * opp_strength_index divides the opponent's goals CONCEDED (was goals scored, against its own
        comment) by the league's measured goals per team-match.
    """
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    d["_tkey"], d["_okey"] = _team_keys(d)
    n_unscoped = int(d["_tkey"].isna().sum())
    if n_unscoped:
        lgs = sorted(d.loc[d["_tkey"].isna(), "league"].astype(str).unique())[:8]
        print(f"[features] {n_unscoped:,} player row(s) without a team or in a competition the "
              f"registry does not know ({lgs}) feed no opponent table")
    d = d[d["_tkey"].notna()]
    if d.empty:
        return pd.DataFrame()
    pos = _first_letter(d["position"]) if "position" in d.columns else pd.Series("", index=d.index)
    for c in ("goals", "shots_on_target", "duels_won", "duels_total", "fouls_committed",
              "yellow_cards", "saves", "goals_conceded", "sp_goal", "team_corners"):
        d[c] = _num(d, c)
    d["_rating"] = _num(d, "rating").where(lambda s: s > 0)
    d["_height"] = _height_series(d)
    key = ["fixture_id", "_tkey"]

    t = d.groupby(key).agg(date=("date", "first"), league=("league", "first"),
                           _okey=("_okey", "first"),
                           goals_scored=("goals", "sum"), sot_scored=("shots_on_target", "sum"),
                           team_corners=("team_corners", "first"),
                           sp_scored=("sp_goal", "sum"), _sp_n=("sp_goal", "count"))
    t["sp_scored"] = t["sp_scored"].where(t.pop("_sp_n") > 0)     # unclassified goals -> NaN

    def _role(letter, **aggs):
        sub = d[pos == letter]
        if sub.empty:
            return pd.DataFrame(index=t.index)
        return sub.groupby(key).agg(**aggs)

    dd = _role("D", _dw=("duels_won", "sum"), _dt=("duels_total", "sum"),
               def_fouls_pg=("fouls_committed", "mean"), def_cards_pg=("yellow_cards", "mean"),
               def_rating=("_rating", "mean"), def_rating_top=("_rating", "max"),
               def_height=("_height", "mean"))
    if not dd.empty:
        dd["def_aerial"] = _ratio(dd.pop("_dw"), dd.pop("_dt"))
    gk = _role("G", gk_saves=("saves", "sum"), _sn=("saves", "count"),
               _gc=("goals_conceded", "sum"), _gn=("goals_conceded", "count"))
    if not gk.empty:
        gk["gk_saves"] = gk["gk_saves"].where(gk.pop("_sn") > 0)
        _gc = gk.pop("_gc").where(gk.pop("_gn") > 0)
        gk["gk_save_rate"] = _ratio(gk["gk_saves"], gk["gk_saves"] + _gc)
    fw = _role("F", fwd_goals=("goals", "sum"), fwd_sot=("shots_on_target", "sum"))
    md = _role("M", mid_goals=("goals", "sum"), mid_sot=("shots_on_target", "sum"),
               mid_fouls_pg=("fouls_committed", "mean"), mid_cards_pg=("yellow_cards", "mean"))
    t = t.join([x for x in (dd, gk, fw, md) if not x.empty], how="left")
    for c in ("def_aerial", "def_fouls_pg", "def_cards_pg", "def_rating", "def_rating_top",
              "def_height", "gk_saves", "gk_save_rate", "fwd_goals", "fwd_sot", "mid_goals",
              "mid_sot", "mid_fouls_pg", "mid_cards_pg"):
        if c not in t.columns:
            t[c] = np.nan
    t = t.reset_index()

    # What the OTHER side did in the same fixture = what this team conceded.
    other = t[["fixture_id", "_tkey", "goals_scored", "sot_scored", "sp_scored",
               "fwd_goals", "fwd_sot", "mid_goals", "mid_sot", "team_corners"]].rename(columns={
        "_tkey": "_okey", "goals_scored": "goals_conceded", "sot_scored": "sot_conceded",
        "sp_scored": "sp_conceded", "fwd_goals": "conc_fwd_goals", "fwd_sot": "conc_fwd_sot",
        "mid_goals": "conc_mid_goals", "mid_sot": "conc_mid_sot", "team_corners": "corners_conceded"})
    t = t.merge(other, on=["fixture_id", "_okey"], how="left")
    # Corners for / against from the API-Football team statistics. A fixture where BOTH sides show 0
    # is an unrecorded statistic, not a result (see _blank_unrecorded_corners): NaN.
    both0 = (t["team_corners"] == 0) & (t["corners_conceded"] == 0)
    t.loc[both0, ["team_corners", "corners_conceded"]] = np.nan
    # Only the side we hold rows for can be measured: a fixture whose opponent was not collected
    # leaves every conceded quantity NaN (it used to drop out of an inner self-join silently).
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t["_day"] = t["date"].dt.normalize()
    return t.sort_values(["_tkey", "date", "fixture_id"]).reset_index(drop=True)


def _blank_unrecorded_corners(df: pd.DataFrame) -> pd.DataFrame:
    """team_corners -> NaN on a fixture where we hold BOTH teams' rows and every one shows 0.
    data_fetcher wrote `int(stats.get("Corner Kicks") or 0)`, so a fixture without statistics
    looks like 0-0 corners; a real 0-0 is rare (each side has 0 corners in ~2% of team-matches),
    while the share of 0s reaches 6-11% in leagues with patchy statistics (Finland, Sweden, Serie
    A, Bundesliga; measured 2026-11-02 on player_history). Returns the frame (modified copy)."""
    if not {"fixture_id", "team", "team_corners"} <= set(df.columns):
        return df
    df = df.copy()
    tc = _num(df, "team_corners")
    g = df.assign(_tc=tc).groupby("fixture_id")
    both = (g["team"].transform("nunique") >= 2) & (g["_tc"].transform("max") == 0)
    df["team_corners"] = tc.where(~both)
    return df


def _team_rolling(t: pd.DataFrame, n: int) -> pd.DataFrame:
    """Pre-match rolling value of each _TEAM_ROLL quantity per team (shift(1), min sample)."""
    g = t.groupby("_tkey", group_keys=False)
    out = t[["fixture_id", "_tkey"]].copy()
    for src, (feat, min_attr) in _TEAM_ROLL.items():
        out[feat] = _roll(g, src, n, getattr(config, min_attr))
    out["_opp_team_corners_pg"] = _roll(g, "team_corners", n, config.MIN_OPP_MATCHES)
    return out


def _team_now(rows: pd.DataFrame, n: int) -> dict:
    """Serving counterpart of _team_rolling: the team's state BEFORE its next match = the mean of
    its last n fixtures we hold, same minimum samples."""
    last = rows.tail(n)
    out = {}
    for src, (feat, min_attr) in _TEAM_ROLL.items():
        s = pd.to_numeric(last[src], errors="coerce")
        out[feat] = float(s.mean()) if s.notna().sum() >= getattr(config, min_attr) else NAN
    return out


# ── league averages, measured from the league's own prior fixtures ──────────────────────────────

# League normaliser -> the team-table column it averages (per team-match)
_LEAGUE_NORMS = {
    "lg_goals_per_team":   "goals_scored",     # goals per team-match (the opp_strength_index base)
    "lg_corners_per_team": "team_corners",
    "lg_sp_per_team":      "sp_scored",
    "lg_def_rating":       "def_rating",
    "lg_def_rating_top":   "def_rating_top",
    "lg_def_aerial":       "def_aerial",
}


def _league_asof(t: pd.DataFrame) -> pd.DataFrame:
    """Per (league, date): each normaliser = mean over the league's team-rows on EARLIER dates
    (never the same day, so a match never normalises itself), NaN until the league has
    config.MIN_LEAGUE_MATCHES prior fixtures (2 team-rows each) carrying the quantity."""
    if t.empty:
        return pd.DataFrame(columns=["league", "_day", *_LEAGUE_NORMS])
    out = None
    for name, col in _LEAGUE_NORMS.items():
        day = (t.assign(_v=pd.to_numeric(t[col], errors="coerce"))
                .groupby(["league", "_day"])["_v"].agg(["sum", "count"]).sort_index())
        cum = day.groupby(level="league").cumsum() - day        # strictly earlier dates
        mean = (cum["sum"] / cum["count"]).where(cum["count"] >= 2 * config.MIN_LEAGUE_MATCHES)
        out = mean.rename(name).to_frame() if out is None else out.join(mean.rename(name))
    return out.reset_index()


_CANON_CACHE: dict = {}


def _canon(league):
    """Registry canonical name (aliases such as 'World Cup' -> 'World Cup 2026'), None if unknown."""
    if league not in _CANON_CACHE:
        _CANON_CACHE[league] = (config.league_registry().canonical_name(league)
                                if isinstance(league, str) else None)
    return _CANON_CACHE[league]


def _same_league(col: pd.Series, league) -> pd.Series:
    """Rows of `col` naming `league` (registry identity, never raw string equality)."""
    target = _canon(league)
    if target is None:
        return pd.Series(False, index=col.index)
    return col.map(_canon) == target


def _league_now(t: pd.DataFrame, league) -> dict:
    """Serving counterpart of _league_asof: the league's averages over every fixture we hold."""
    rows = t[_same_league(t["league"], league)] if not t.empty else t
    out = {}
    for name, col in _LEAGUE_NORMS.items():
        s = pd.to_numeric(rows[col], errors="coerce") if len(rows) else pd.Series(dtype=float)
        out[name] = (float(s.mean()) if s.notna().sum() >= 2 * config.MIN_LEAGUE_MATCHES else NAN)
    return out


# ── referee, measured ────────────────────────────────────────────────────────────────────────────

def _referee_key(name) -> str | None:
    if name is None or (isinstance(name, float) and np.isnan(name)) or not str(name).strip():
        return None
    from .api_football import _referee_key as _rk
    return _rk(name) or None


def _fixture_yellows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per fixture: date, league, scope, referee key, total yellows of the collected rows."""
    f = (df.assign(_yc=_num(df, "yellow_cards"))
           .groupby("fixture_id")
           .agg(date=("date", "first"), league=("league", "first"),
                referee=("referee", "first"), yellows=("_yc", "sum")))
    f["date"] = pd.to_datetime(f["date"], errors="coerce")
    f["_scope"] = f["league"].map(team_scope)
    f["_rkey"] = f["referee"].map(_referee_key)
    return f.reset_index().sort_values(["date", "fixture_id"])


def _referee_asof(df: pd.DataFrame) -> pd.DataFrame:
    """Per fixture: the referee's yellows/match over his PRIOR fixtures in the same scope (min
    config.MIN_REF_MATCHES) and its z-score against the league's prior yellows/match mean and sd
    (min config.MIN_LEAGUE_MATCHES). Replaces the old typed 3.5 / 1.0 / cross-league fallback."""
    f = _fixture_yellows(df)
    known = f[f["_rkey"].notna() & f["_scope"].notna()]
    f["referee_yellows_pg"] = np.nan
    if len(known):
        g = known.groupby(["_scope", "_rkey"], group_keys=False)
        f.loc[known.index, "referee_yellows_pg"] = _expand(g, "yellows", config.MIN_REF_MATCHES)
    # League mean / sd per date from strictly earlier dates.
    day = f.groupby(["league", "date"])["yellows"].agg(["sum", "count"])
    day["sq"] = f.assign(_sq=f["yellows"] ** 2).groupby(["league", "date"])["_sq"].sum()
    cum = day.groupby(level="league").cumsum() - day
    mu = cum["sum"] / cum["count"]
    sd = np.sqrt((cum["sq"] / cum["count"] - mu ** 2).clip(lower=0) * cum["count"] / (cum["count"] - 1))
    ok = cum["count"] >= config.MIN_LEAGUE_MATCHES
    lg = pd.DataFrame({"_lg_mu": mu.where(ok), "_lg_sd": sd.where(ok & (sd > 0))}).reset_index()
    f = f.merge(lg, on=["league", "date"], how="left")
    f["referee_strictness"] = (f["referee_yellows_pg"] - f["_lg_mu"]) / f["_lg_sd"]
    f["lg_yellows_per_match"] = f["_lg_mu"]
    return f[["fixture_id", "referee_yellows_pg", "referee_strictness", "lg_yellows_per_match"]]


def league_yellow_stats(history: pd.DataFrame, league) -> tuple[float, float]:
    """(mean, sd) of total yellows per fixture in `league` over every fixture we hold, NaN below
    config.MIN_LEAGUE_MATCHES fixtures. For callers building a referee z-score at serve time."""
    if history is None or history.empty or "fixture_id" not in history.columns:
        return NAN, NAN
    f = _fixture_yellows(history[_same_league(history["league"], league)])
    if len(f) < config.MIN_LEAGUE_MATCHES:
        return NAN, NAN
    sd = float(f["yellows"].std())
    return float(f["yellows"].mean()), (sd if sd > 0 else NAN)


def referee_profile_from_history(history: pd.DataFrame, referee_name, league) -> dict:
    """Serving referee profile with EXACTLY the training definition (_referee_asof): the referee's
    yellows/match over every fixture we hold for him in the league's scope, z-scored against the
    league's measured mean/sd. NaN below the minimum samples. Keys match build_upcoming_features."""
    out = {"yellows_per_game": NAN, "strictness_score": NAN, "n_games": 0,
           "lg_yellows_per_match": NAN}
    rkey, scope = _referee_key(referee_name), team_scope(league)
    if history is None or history.empty or rkey is None or scope is None:
        return out
    f = _fixture_yellows(history[history["league"].map(team_scope) == scope])
    mine = f[f["_rkey"] == rkey]
    out["n_games"] = int(len(mine))
    mu, sd = league_yellow_stats(history, league)
    out["lg_yellows_per_match"] = mu
    if len(mine) >= config.MIN_REF_MATCHES:
        out["yellows_per_game"] = float(mine["yellows"].mean())
        if sd == sd and mu == mu:
            out["strictness_score"] = (out["yellows_per_game"] - mu) / sd
    return out


# ── season, as of date ───────────────────────────────────────────────────────────────────────────

def _resolve_seasons(df: pd.DataFrame) -> pd.Series:
    """API-Football season per row via src.season_resolver (the league's registry rule), NaN where
    the league is unknown. Replaces `date.year` (wrong for every Jul-Jun league)."""
    sr = config._v9_src("season_resolver")
    day = pd.to_datetime(df["date"], errors="coerce").dt.date
    pairs = pd.DataFrame({"league": df["league"].astype(str), "day": day})
    uniq = pairs.drop_duplicates()
    uniq["season"] = [sr.resolve_api_season(lg, d) if d == d and d is not None else sr.SEASON_UNKNOWN
                      for lg, d in zip(uniq["league"], uniq["day"])]
    s = pairs.merge(uniq, on=["league", "day"], how="left")["season"]
    s.index = df.index
    unknown = s == sr.SEASON_UNKNOWN
    if unknown.any():
        print(f"[features] season unknown for {int(unknown.sum()):,} row(s) "
              f"({sorted(df.loc[unknown, 'league'].astype(str).unique())[:6]}): season_* stay NaN")
    return s.where(~unknown)


def compute_season_asof(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute season_* features AS-OF-DATE — within-season expanding mean of PRIOR
    games only (shift(1)), mirroring the clean career_* pattern.

    FIXES a data leak: season_* used to be a WHOLE-SEASON /players/statistics average
    broadcast onto every match row (pipeline.mode_enrich_season), so a row's
    season_goals_pg included the very match being predicted (and all later matches of
    that season) -> inflated AUC. Caller must have df sorted by [player_id, date]
    (build_features already is). Operates in place and returns df.
    A player's first game of a season has no prior game: its season_* are NaN (were 0.0).
    """
    if "player_id" not in df.columns or "date" not in df.columns:
        return df
    if "season" not in df.columns:
        df["season"] = _resolve_seasons(df) if "league" in df.columns else np.nan
    g = df.groupby(["player_id", "season"], group_keys=False)
    for feat, raw in _SEASON_SRC.items():
        if raw in df.columns:
            df[feat] = _expand(g, raw)
    df["season_appearances"] = g.cumcount().astype(float).where(df["season"].notna())
    # No per-match pass-accuracy source exists (config._DATA_GAP_FEATURES): explicitly missing.
    df["season_pass_accuracy"] = np.nan
    return df


# ── composites: ONE definition for training and serving ─────────────────────────────────────────

def _add_composites(df: pd.DataFrame) -> pd.DataFrame:
    """Every derived feature, from base columns that are already present on `df` (training frame
    or the serving frame). League normalisers (lg_*) are the league's measured averages.
    NaN in -> NaN out; nothing is filled."""
    f = df.copy()                      # de-fragmented working frame (many columns are added)
    pf, pm, pd_ = f["pos_forward"], f["pos_midfielder"], f["pos_defender"]

    f["sot_rate"] = _ratio(f["sot_pg"], f["shots_pg"])
    f["shooting_efficiency_index"] = _ratio(f["goals_pg"], f["sot_pg"]).clip(upper=1.0)
    f["card_exposure_index"] = f["cards_pg"] * (f["minutes_pg"] / 90.0) * (1 - pf)
    f["sot_quality_score"] = f["shot_accuracy_rate"] * f["sot_pg"]
    f["opp_adjusted_shot_threat"] = f["shots_pg"] * f["opp_sot_conceded_pg"]
    f["creative_playmaker_score"] = f["kp_per90"] * (pm + 0.5 * pf)

    corners_rel = _ratio(f["team_corners_pg"], f["lg_corners_per_team"])
    sp_rel = _ratio(f["opp_sp_goals_conceded_pg"], f["lg_sp_per_team"])

    # Set-piece role (owner decision B, 2026-11-02): REAL data only. The legacy scores were typed
    # position weights (0.45 / 0.30 / 0.05 ...) over goal-event classifications API-Football does not
    # provide; they are retired (NaN, LEGACY_SETPIECE_COLUMNS). Their measured replacements are the
    # Understat features us_sp_taker_share / us_sp_receiver_share / us_sp_xg_p90 / us_sp_sot_p90 /
    # us_dfk_shots_p90 (player_model/setpiece/features.py; NaN outside the Understat leagues), and
    # the corner matchup below.
    for c in LEGACY_SETPIECE_COLUMNS:
        f[c] = np.nan
    # Corner matchup: the player's team's corners per match (his last n games) relative to its
    # league, times the opponent's corners CONCEDED per match relative to the league. Both from the
    # API-Football team statistics (team_corners), every league.
    f["corner_matchup_index"] = corners_rel * _ratio(f["opp_corners_conceded_pg"], f["lg_corners_per_team"])

    f["aerial_matchup_score"] = (f["aerial_won_rate"] * (1.0 - f["opp_def_aerial_win_rate"])).clip(lower=0.0)
    f["foul_draw_matchup_score"] = f["fouls_drawn_per90"] * f["opp_def_fouls_pg"]
    f["opp_def_aggression"] = f["opp_def_fouls_pg"] * (1.0 + f["opp_def_cards_pg"])

    save_gap = 1.0 - f["opp_gk_save_rate"]
    f["att_vs_gk_threat"] = (f["sot_pg"] * save_gap).clip(lower=0.0)
    f["clinical_vs_gk"] = (f["shot_accuracy_rate"] * save_gap).clip(lower=0.0)
    f["dribble_creativity_score"] = f["dribbles_pg"] * f["dribble_success_rate"]
    f["defensive_solidity"] = (f["defensive_actions_per90"]
                               * (1.0 - f["dribbled_past_pg"] / (f["dribbled_past_pg"] + 1.0))).clip(lower=0.0)
    f["penalty_threat_score"] = (f["penalties_won_pg"] * 3.0 + f["fouls_drawn_per90"] * 0.08).clip(upper=1.0)
    f["card_risk_index"] = (f["cards_pg"] * 2.0 + f["red_cards_pg"] * 3.0
                            + f["foul_committer_ratio"] * 0.5).clip(upper=1.0)
    f["midfield_engine_score"] = ((f["passes_pg"] / 50.0).clip(upper=1.0) * 0.4
                                  + (f["kp_per90"] / 3.0).clip(upper=1.0) * 0.4
                                  + (f["tackles_per90"] / 5.0).clip(upper=1.0) * 0.2) * (pm + 0.3 * pd_)
    f["offside_aggressiveness"] = (f["offsides_pg"] * pf).clip(upper=2.0)
    f["volume_shot_penetration"] = (f["shots_pg"] * f["shot_accuracy_rate"] * save_gap).clip(lower=0.0)
    f["finishing_threat_index"] = (f["sot_pg"] * f["shooting_efficiency_index"] * save_gap).clip(lower=0.0)
    f["box_dominance_vs_gk"] = ((f["box_actions_per90"] + f["dribbles_pg"]) * save_gap).clip(lower=0.0)
    f["aggression_adjusted_threat"] = (f["sot_pg"] * save_gap / (1.0 + f["offsides_pg"] * 0.3)).clip(lower=0.0)
    f["defensive_vulnerability_index"] = (
        (f["dribbled_past_pg"] * 1.5 + (1.0 - f["dribble_success_rate"]) * 0.5
         + (1.0 - f["aerial_won_rate"]) * 0.5).clip(upper=2.0)
        * (pf * 1.0 + pm * 0.7 + pd_ * 0.3))
    f["progressive_carrier_score"] = (
        ((f["dribbles_pg"] / 3.0).clip(upper=1.0) * 0.45 + (f["passes_pg"] / 60.0).clip(upper=1.0) * 0.35
         + f["dribble_success_rate"] * 0.20).clip(upper=1.0)
        * (pm * 1.0 + pf * 0.8 + pd_ * 0.5))
    f["disciplinary_pressure_index"] = (f["fouls_drawn_per90"] * 0.20 + f["fouls_per90"] * 0.25
                                        + f["cards_pg"] * 1.5 + f["duel_intensity_per90"] * 0.10).clip(upper=2.0)
    f["foul_magnet_score"] = (f["fouls_drawn_per90"] * 0.30 + f["penalty_threat_score"] * 0.40
                              + f["progressive_carrier_score"] * 0.30).clip(upper=1.0)
    f["forward_matchup_score"] = (f["sot_pg"] * f["opp_goals_conceded_vs_fwd_pg"] * pf).clip(lower=0.0)
    f["mid_threat_vs_defense"] = (f["kp_per90"] * f["opp_goals_conceded_vs_mid_pg"] * pm).clip(lower=0.0)
    f["box_threat_vs_leaky_defense"] = (f["box_actions_per90"] * f["opp_goals_conceded_pg"]).clip(lower=0.0)
    f["efficiency_vs_leaky_keeper"] = (f["shooting_efficiency_index"] * f["opp_sot_conceded_pg"]).clip(lower=0.0)
    f["kp_vs_aggressive_defense"] = (f["kp_per90"] * f["opp_def_cards_pg"]).clip(lower=0.0)
    f["team_momentum_forward_matchup"] = (f["team_goals_pg_roll"] * f["opp_goals_conceded_pg"] * pf).clip(lower=0.0)

    # Height: the player's and the opponent defenders' measured heights (NaN while unprofiled).
    f["height_diff_vs_opp_def"] = f["height_cm"] - f["opp_def_mean_height"]
    f["height_advantage_score"] = (f["height_diff_vs_opp_def"] / 10.0).clip(-2.0, 2.0)
    f["height_aerial_interaction"] = ((f["height_cm"].clip(160, 200) - 160) / 40) * f["aerial_won_rate"]
    f["age_peak_delta"] = (f["age"] - 27).abs()
    _h_adj = (f["height_advantage_score"] + 2.0) / 4.0
    f["defender_sp_edge"] = (f["sp_receiver_score"] * _h_adj * sp_rel * corners_rel.clip(0.3, 2.0)).clip(0, 4.0)
    _sp_sot_proxy = (f["sp_goals_pg"] * 5.0 + f["headed_goals_pg"] * 4.0).clip(0, 0.6)
    f["defender_sot_edge"] = (f["sp_receiver_score"] * _h_adj * corners_rel.clip(0.2, 3.0) * sp_rel
                              * (1.0 + _sp_sot_proxy * 4.0)).clip(0, 5.0)
    f["sp_threat_vs_weak_sp_defense"] = (f["sp_goals_pg"] * sp_rel * (1.0 - f["sp_taker_score"])).clip(0, 1.0)
    f["aerial_height_sp_composite"] = (f["aerial_won_rate"] * _h_adj.clip(0.3, 1.0) * sp_rel
                                       * (pd_ + 0.5 * pm)).clip(0, 2.0)
    f["sp_goal_probability_composite"] = (f["career_sp_goals_rate"] * 0.30 + f["sp_goals_pg"] * 0.30
                                          + f["aerial_won_rate"] * _h_adj * 0.20
                                          + (sp_rel - 1.0).clip(-0.5, 1.0) * 0.20).clip(0, 1.0)
    f["sp_taker_assist_edge"] = (f["sp_taker_score"] * (f["career_sp_assist_rate"].clip(0, 0.10) / 0.10)
                                 * corners_rel.clip(0.3, 2.0)).clip(0, 1.0)
    f["creative_pressure_matchup"] = (f["creative_playmaker_score"] * f["opp_def_fouls_pg"]).clip(lower=0.0)
    f["dribbler_vs_defensive_line"] = (f["dribble_creativity_score"] * (1.0 - f["opp_def_aerial_win_rate"])).clip(lower=0.0)
    f["carrier_vs_press"] = (f["progressive_carrier_score"] * f["opp_def_aggression"]).clip(lower=0.0)
    f["dribble_contact_rate"] = (f["dribbles_pg"] * (1.0 - f["dribble_success_rate"])).clip(lower=0.0)
    f["tackle_dribble_clash"] = (f["dribble_contact_rate"] * f["opp_mid_fouls_pg"]).clip(lower=0.0)
    # The referee's yellows relative to the league's MEASURED yellows/match (was / 3.5).
    f["card_clash_index"] = ((f["fouls_per90"] + f["dribble_contact_rate"]) * f["opp_mid_cards_pg"]
                             * _ratio(f["referee_yellows_pg"], f["lg_yellows_per_match"]).clip(lower=0.1)
                             ).clip(lower=0.0)
    f["opp_mid_discipline"] = (f["opp_mid_fouls_pg"] * 0.6 + f["opp_mid_cards_pg"] * 2.0).clip(lower=0.0)

    # Opposition strength: the opponent's goals CONCEDED per match relative to its league's goals per
    # team-match, both measured (was goals scored over a typed/whole-frame mean, fillna 1.3).
    f["opp_strength_index"] = _ratio(f["opp_goals_conceded_pg"], f["lg_goals_per_team"]).clip(0.3, 3.0)
    f["quality_adj_goals_pg"] = (f["goals_pg"] * f["opp_strength_index"]).clip(0.0, 5.0)
    f["quality_adj_sot_pg"] = (f["sot_pg"] * f["opp_strength_index"]).clip(0.0, 10.0)

    # Opponent defensive quality: each factor relative to the league's measured average (was 6.8 /
    # 7.0 / 0.50).
    f["opp_def_player_quality"] = (_ratio(f["opp_def_player_rating_pg"], f["lg_def_rating"])
                                   * _ratio(f["opp_top_def_rating"], f["lg_def_rating_top"])
                                   * _ratio(f["opp_def_aerial_win_rate"], f["lg_def_aerial"])).clip(0.2, 4.0)
    # Measured strength (owner decision A; league_quality.py): the Elo odds ratio between the
    # opposition the player's record was earned against and this opponent. Unclipped: a clip would
    # manufacture a value. The legacy hand-tier ratios are NaN (league_quality.LEGACY_COLUMNS).
    f["context_elo_odds"] = np.exp(f["player_career_avg_opp_elo"] - f["opp_club_elo"])
    f["elo_mismatch_goals"] = f["goals_pg"] * f["context_elo_odds"]
    f["elo_mismatch_sot"] = f["sot_pg"] * f["context_elo_odds"]
    for c in _LQ.LEGACY_COLUMNS:
        f[c] = np.nan
    return f


# Base (non-composite) columns _add_composites reads; any absent one is created as NaN.
_COMPOSITE_INPUTS = (
    "goals_pg", "sot_pg", "shots_pg", "cards_pg", "minutes_pg", "shot_accuracy_rate", "kp_per90",
    "team_corners_pg", "aerial_won_rate", "fouls_drawn_per90", "fouls_per90", "foul_committer_ratio",
    "duel_intensity_per90", "box_actions_per90", "dribbles_pg", "dribble_success_rate",
    "dribbled_past_pg", "defensive_actions_per90", "penalties_won_pg", "red_cards_pg", "passes_pg",
    "tackles_per90", "offsides_pg", "height_cm", "age", "sp_goals_pg", "headed_goals_pg",
    "fk_goals_pg", "career_sp_goals_rate", "career_sp_assist_rate", "referee_yellows_pg",
    *(feat for feat, _ in _TEAM_ROLL.values()), *_LEAGUE_NORMS,
    "lg_yellows_per_match", "pos_forward", "pos_midfielder", "pos_defender",
    *_LQ.MEASURED_COLUMNS, *_SP.SETPIECE_COLUMNS,
)


def _ensure(df: pd.DataFrame, cols) -> None:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")


# ── training ─────────────────────────────────────────────────────────────────────────────────────

def build_features(match_rows: list[dict], n: int = None) -> pd.DataFrame:
    """
    Build ML training data from per-match player stats.

    Features: rolling averages of player's PREVIOUS N matches (shift=1, zero leakage).
    Targets:  actual outcome in THIS match (not the same as features).

    Also computes opponent defensive rolling features per fixture.
    """
    if not match_rows:
        return pd.DataFrame()

    n = n or config.ROLLING_N

    df = pd.DataFrame(match_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
    # Before the minutes filter: the rule needs both teams' rows of a fixture.
    df = _blank_unrecorded_corners(df)

    # Drop bench/DNP rows (< 10 min)
    df = df[df["minutes"] >= 10].copy()

    # ── Per-player rolling features — shift(1) prevents any leakage ──────────
    grp = df.groupby("player_id", group_keys=False)
    for feat, raw in (("goals_pg", "goals"), ("assists_pg", "assists"), ("shots_pg", "shots_total"),
                      ("sot_pg", "shots_on_target"), ("cards_pg", "yellow_cards"),
                      ("minutes_pg", "minutes"), ("key_passes_pg", "key_passes")):
        df[feat] = _roll(grp, raw, n) if raw in df.columns else np.nan

    # API-Football rating 0 = not rated; no prior rated match -> NaN (was 7.0).
    df["_rating"] = _num(df, "rating").where(lambda s: s > 0)
    grp = df.groupby("player_id", group_keys=False)
    df["rating_pg"] = _roll(grp, "_rating", n)

    # ── League quality: the legacy hand-tier columns stay NaN; the measured strength columns are
    # attached below once the opponent is known (league_quality.attach_strength).
    _ensure(df, ("league_tier", "league_quality", "player_career_avg_quality"))

    # Count of previous games this player has in the dataset
    df["n_prev_games"] = grp["date"].transform("cumcount")

    # ── Venue-split rolling rates — the player's own previous n HOME (or AWAY) games ──
    # No venue history -> NaN (was the overall rate pasted in, or 0.0).
    # Every row gets the player's state BEFORE it: the mean of his last n games at that venue
    # (serving computes exactly this for the upcoming fixture).
    for _raw, _pref in (("goals", "goals"), ("shots_total", "shots"), ("shots_on_target", "sot")):
        for _venue, _is_h in (("home", True), ("away", False)):
            col = f"{_pref}_{_venue}_pg"
            if "is_home" not in df.columns or _raw not in df.columns:
                df[col] = np.nan
                continue
            m = df["is_home"].astype(bool) == _is_h
            sub = df.loc[m]
            after = _realign(pd.to_numeric(sub[_raw], errors="coerce")
                             .groupby(sub["player_id"]).rolling(n, min_periods=1).mean(), df)
            prev = after.groupby(df["player_id"]).shift(1)
            df[col] = prev.groupby(df["player_id"]).ffill()

    df["starter_rate"] = _roll(grp, "started", n) if "started" in df.columns else np.nan

    # ── Phase 1: ratio and per-90 rolling features (zero leakage) ────────────
    mins = _num(df, "minutes")
    df["_shot_acc"] = _ratio(_num(df, "shots_on_target"), _num(df, "shots_total"))
    df["_kp90"] = _ratio(_num(df, "key_passes"), mins) * 90
    df["_gi90"] = _ratio(_num(df, "goals") + _num(df, "assists"), mins) * 90
    df["_box90"] = _ratio(_num(df, "shots_total") + _num(df, "duels_won"), mins) * 90
    df["_aerial"] = _ratio(_num(df, "duels_won"), _num(df, "duels_total"))
    df["_duel90"] = _ratio(_num(df, "duels_total"), mins) * 90
    df["_fd90"] = _ratio(_num(df, "fouls_drawn"), mins) * 90
    df["_fp90"] = _ratio(_num(df, "fouls_committed"), mins) * 90
    _fc, _fd = _num(df, "fouls_committed"), _num(df, "fouls_drawn")
    df["_fcr"] = (_fc / (_fc + _fd + 0.01)).where((_fc + _fd) > 0)    # no foul either way -> undefined
    df["_tack90"] = _ratio(_num(df, "tackles_total"), mins) * 90
    df["_def90"] = _ratio(_num(df, "tackles_total") + _num(df, "interceptions"), mins) * 90
    df["_drib_acc"] = _ratio(_num(df, "dribbles_success"), _num(df, "dribbles_attempted"))
    grp = df.groupby("player_id", group_keys=False)
    for feat, raw in (("shot_accuracy_rate", "_shot_acc"), ("kp_per90", "_kp90"),
                      ("goal_involvement_rate", "_gi90"), ("box_actions_per90", "_box90"),
                      ("aerial_won_rate", "_aerial"), ("duel_intensity_per90", "_duel90"),
                      ("fouls_drawn_per90", "_fd90"), ("fouls_per90", "_fp90"),
                      ("foul_committer_ratio", "_fcr"), ("tackles_per90", "_tack90"),
                      ("defensive_actions_per90", "_def90"), ("dribble_success_rate", "_drib_acc")):
        df[feat] = _roll(grp, raw, n)
    for feat, raw in (("team_corners_pg", "team_corners"), ("tackles_pg", "tackles_total"),
                      ("interceptions_pg", "interceptions"), ("dribbles_pg", "dribbles_success"),
                      ("dribbled_past_pg", "dribbles_past"), ("red_cards_pg", "red_cards"),
                      ("passes_pg", "passes_total"), ("offsides_pg", "offsides"),
                      ("penalties_won_pg", "penalty_won"), ("saves_pg", "saves"),
                      ("goals_conceded_gk_pg", "goals_conceded")):
        df[feat] = _roll(grp, raw, n) if raw in df.columns else np.nan

    # Career-to-date averages — stable priors complementing the volatile last-5 rolling stats.
    for feat, raw in (("career_goals_pg", "goals"), ("career_sot_pg", "shots_on_target"),
                      ("career_shots_pg", "shots_total"), ("career_assists_pg", "assists")):
        df[feat] = _expand(grp, raw)

    # Penalty conversion from the player's OWN penalty record, NaN below config.MIN_PEN_ATTEMPTS
    # (was a last-5 mean of per-match rates filled with 0.75: 96% of rows carried the 0.75):
    # earlier SEASONS from the collected /players season record (all competitions) where the
    # provider reported it, else from our per-match rows of that season; the CURRENT season from
    # our per-match rows strictly before this match.
    df["penalty_conversion_rate"] = _penalty_asof(df)

    # The keeper's own save rate: saves / shots faced on target, NaN when he faced none.
    df["_gk_sr"] = _ratio(_num(df, "saves"), _num(df, "saves") + _num(df, "goals_conceded"))
    grp = df.groupby("player_id", group_keys=False)
    df["gk_save_rate"] = _roll(grp, "_gk_sr", n, config.MIN_GK_MATCHES)

    # ── Season-to-date features — AS-OF-DATE (fixes the whole-season broadcast leak).
    df = compute_season_asof(df)

    # ── Age and physical profile: the collected /players record first (height / weight of the
    # latest season reporting them; age at the match date from the provider birth date), else a
    # real stored value; the old code's placeholders count as missing.
    df = _attach_profiles(df)

    # Season-level fields absent from the frame stay missing (were 0.7 / 0.75 / 0.3 / 1.0 / 0.8).
    _ensure(df, ("season_start_rate", "season_pass_accuracy", "season_dribble_pg",
                 "season_fouls_pg", "season_fouls_drawn_pg"))
    # Injury features as of the match date from the collected /sidelined answer
    # (history_integrity.injury_features_asof: only spells that STARTED before the match). Players
    # never asked keep the injury refresh's stored values; absent -> NaN (were 0 / 365 / 0).
    _ensure(df, ("chronic_injury_risk", "days_since_last_injury", "return_from_injury_flag"))
    _sid = collected_sidelined()
    if _sid:
        from .history_integrity import apply_injury_pit
        apply_injury_pit(df, _sid)

    # ── Referee strictness, measured (see _referee_asof) ────────────────────────
    df = df.copy()                     # de-fragment after the column-by-column build above
    # Rows read back from player_history carry the previous build's derived columns: recompute
    # them rather than letting a merge suffix them (_x/_y).
    df = df.drop(columns=[c for c in ("referee_yellows_pg", "referee_strictness", "lg_yellows_per_match",
                                      *(feat for feat, _ in _TEAM_ROLL.values()), *_LEAGUE_NORMS)
                          if c in df.columns])
    if "referee" in df.columns and "fixture_id" in df.columns and "yellow_cards" in df.columns:
        df = df.merge(_referee_asof(df), on="fixture_id", how="left")
    else:
        df["referee_yellows_pg"] = np.nan
        df["referee_strictness"] = np.nan
        df["lg_yellows_per_match"] = np.nan

    # ── Rest days: days since the player's previous match. A first match has none (NaN, was 6.0);
    # a same-day duplicate (0 days) is not a rest period (NaN, was clipped up to 1). Capped at 30.
    _d = pd.to_datetime(df["date"], errors="coerce")
    _gap = _d.groupby(df["player_id"]).diff().dt.days
    df["rest_days"] = _gap.where(_gap >= 1).clip(upper=30)

    # ── Set-piece event rolling features (data_fetcher.enrich_sp_events; NaN where unclassified)
    grp = df.groupby("player_id", group_keys=False)
    for _raw_col, _feat_name in (("sp_goal", "sp_goals_pg"), ("headed_goal", "headed_goals_pg"),
                                 ("fk_goal", "fk_goals_pg"), ("sp_assist", "sp_assist_pg")):
        df[_feat_name] = _roll(grp, _raw_col, n) if _raw_col in df.columns else np.nan
    for _raw_col, _feat_name in (("sp_goal", "career_sp_goals_rate"),
                                 ("sp_assist", "career_sp_assist_rate")):
        df[_feat_name] = _expand(grp, _raw_col) if _raw_col in df.columns else np.nan
    df["sp_goals_share"] = _ratio(df["sp_goals_pg"], df["goals_pg"]).clip(0, 1)
    df["headed_goals_share"] = _ratio(df["headed_goals_pg"], df["goals_pg"]).clip(0, 1)

    # ── Opponent features, measured from the opponent's prior matches (team_match_table) ──────
    df["opponent"] = df["away_team"].where(df["is_home"].astype(bool), df["home_team"])
    # Measured club / league strength as of the match date (league_quality.attach_strength).
    df = _LQ.attach_strength(df)
    # Set-piece roles from mapped Understat shots, as of the match date (setpiece/features.py).
    _sp = _SP.asof(_SP.load(history_dates=df[["fixture_id", "date"]]), df["player_id"], df["date"])
    for c in _SP.SETPIECE_COLUMNS:
        df[c] = _sp[c].to_numpy()
    _, df["_okey"] = _team_keys(df)
    t = team_match_table(df)
    if not t.empty:
        roll = _team_rolling(t, n).rename(columns={"_tkey": "_okey"})
        df = df.merge(roll.drop(columns=["_opp_team_corners_pg"]), on=["fixture_id", "_okey"], how="left")
        df["_day"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.merge(_league_asof(t), on=["league", "_day"], how="left")
    _ensure(df, [feat for feat, _ in _TEAM_ROLL.values()] + list(_LEAGUE_NORMS))

    # ── Position encoding ─────────────────────────────────────────────────────
    pos = _first_letter(df["position"]) if "position" in df.columns else pd.Series("", index=df.index)
    df["pos_forward"] = (pos == "F").astype(int)
    df["pos_midfielder"] = (pos == "M").astype(int)
    df["pos_defender"] = (pos == "D").astype(int)

    _ensure(df, _COMPOSITE_INPUTS)
    df = _add_composites(df)

    # ── Target variables — actual outcome in THIS match ───────────────────────
    df["target_goals"]   = (df["goals"]             >= 1).astype(int)
    df["target_sot"]     = (df["shots_on_target"]   >= 1).astype(int)
    df["target_cards"]   = (df["yellow_cards"]       >= 1).astype(int)
    df["target_assists"] = (df["assists"]             >= 1).astype(int)
    df["target_goals2"]  = (df["goals"]             >= 2).astype(int)
    df["target_sot2"]    = (df["shots_on_target"]   >= 2).astype(int)
    df["target_sot3"]    = (df["shots_on_target"]   >= 3).astype(int)
    df["target_sot4"]    = (df["shots_on_target"]   >= 4).astype(int)

    # Drop rows with no prior history (rolling features would all be NaN)
    _n0 = len(df)
    df = df[df["n_prev_games"] >= 1].copy()
    if _n0 - len(df):
        print(f"[features] {(_n0 - len(df)):,} first-appearance row(s) dropped (no prior match)")

    # Feature columns numeric; a missing value STAYS missing (was fillna(0.0) over every column).
    for col in config.PLAYER_FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True, errors="ignore")
    return df.reset_index(drop=True)


# ── serving ──────────────────────────────────────────────────────────────────────────────────────

def _build_history_index(history: pd.DataFrame):
    """Index match-history ONCE for O(1) per-player lookup in build_upcoming_features.
    Returns (by_player_id, {}) of date-sorted per-player frames. Replaces the full 230k-row scan
    that ran once PER PLAYER PER MATCH. Lookup is by player_id ONLY (Data Truth: the old by-name
    index matched two players sharing a lower-cased name); the second element is kept empty for
    callers that unpack two values."""
    if history is None or history.empty:
        return {}, {}
    h = history.sort_values("date")
    by_pid = {pid: g for pid, g in h.groupby("player_id")} if "player_id" in h.columns else {}
    return by_pid, {}


_SERVE_CACHE: dict = {}


def _serving_context(history: pd.DataFrame) -> dict:
    """Team table and per-scope team lists over the WHOLE history, built once per history frame."""
    key = (id(history), len(history))
    hit = _SERVE_CACHE.get("key") == key
    if not hit:
        h = history.copy()
        h["date"] = pd.to_datetime(h["date"], errors="coerce")
        if "opponent" not in h.columns:
            h["opponent"] = h["away_team"].where(h["is_home"].astype(bool), h["home_team"])
        t = team_match_table(h)
        teams_by_scope: dict = {}
        if not t.empty:
            names = t["_tkey"].str.rsplit("|", n=1, expand=True)
            for scope, sub in names.groupby(1):
                teams_by_scope[scope] = sorted(sub[0].unique())
        # Measured strength over the whole history (career averages) and the collected real data.
        hs = _LQ.attach_strength(h) if "team" in h.columns else h
        by_pid_strength = ({p: g[["league_elo", "opp_club_elo", "_scope"]] for p, g in hs.groupby("player_id")}
                           if "player_id" in hs.columns else {})
        prof = collected_profiles()
        blank = set()
        if {"fixture_id", "team_corners"} <= set(h.columns):
            hb = _blank_unrecorded_corners(h)
            blank = set(hb.loc[hb["team_corners"].isna() & _num(h, "team_corners").notna(), "fixture_id"])
        _SERVE_CACHE.clear()
        _SERVE_CACHE.update({"key": key, "table": t,
                             "by_team": ({k: g for k, g in t.groupby("_tkey")} if not t.empty else {}),
                             "teams_by_scope": teams_by_scope,
                             "blank_corner_fixtures": blank,
                             "strength_by_pid": by_pid_strength,
                             "profiles": prof.set_index("player_id") if not prof.empty else prof,
                             "sidelined": collected_sidelined(),
                             "pen_totals": _season_pen_totals(h),
                             "sp_app": (_SP.load(history_dates=h[["fixture_id", "date"]])
                                        if "fixture_id" in h.columns else pd.DataFrame())})
    return _SERVE_CACHE


def _opponent_state(ctx: dict, opponent: str, league, n: int) -> dict:
    """Opponent features for the upcoming fixture: the opponent resolved (src.team_names.resolve)
    among the teams we hold in the fixture league's identity scope, then its last n fixtures.
    Unresolvable -> every opponent feature NaN (logged by the caller)."""
    nan_state = {feat: NAN for feat, _ in _TEAM_ROLL.values()}
    scope = team_scope(league)
    if not opponent or scope is None:
        return nan_state
    cands = ctx["teams_by_scope"].get(scope, [])
    hit = config._v9_src("team_names").resolve(str(opponent), cands) if cands else None
    if not hit:
        return nan_state
    rows = ctx["by_team"].get(f"{hit}|{scope}")
    return _team_now(rows, n) if rows is not None and len(rows) else nan_state


def _f(x) -> float:
    """float(x), or NaN for None / non-numeric."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def _mean_last(s: pd.Series, n: int, min_n: int = 1) -> float:
    s = pd.to_numeric(s, errors="coerce").tail(n)
    return float(s.mean()) if s.notna().sum() >= min_n else NAN


def build_upcoming_features(
    upcoming: list[dict],
    history: pd.DataFrame,
    referee_profile: dict | None = None,
    match_context:   dict | None = None,
    history_index: tuple | None = None,
) -> pd.DataFrame:
    """
    Build feature rows for upcoming player predictions.
    Uses the player's last N matches from match-level history as rolling features, and the SAME
    definitions as build_features for everything else (team table, league averages, composites).

    upcoming: list of player dicts (player_id, player_name, team, opponent, is_home, and the
              fixture's `fixture_league` / `fixture_date` when the caller knows them)
    history:  match-level DataFrame from build_features()
    referee_profile: {yellows_per_game, strictness_score, lg_yellows_per_match} for THIS fixture
              (referee_profile_from_history); absent keys are NaN.
    match_context: kept for callers; no feature is read from it any more (the opponent's concede
              rates used to come from it with 1.3 / 4.5 defaults and were never supplied).
    history_index: optional pre-built (by_pid, _) from _build_history_index(history).
    """
    if not upcoming or history.empty:
        return pd.DataFrame()

    by_pid, _ = history_index if history_index is not None else _build_history_index(history)
    n = config.ROLLING_N
    ref = referee_profile or {}
    ctx = _serving_context(history)
    rows = []
    no_pid = no_hist = 0
    opp_cache: dict = {}
    lg_cache: dict = {}

    for p in upcoming:
        pid = p.get("player_id")
        # player_id only: a player without one is not identifiable (skipped, counted).
        if pid is None or (isinstance(pid, float) and np.isnan(pid)):
            no_pid += 1
            continue
        phist_all = by_pid.get(pid)
        if phist_all is None or phist_all.empty:
            no_hist += 1
            continue
        phist = phist_all.tail(n)
        league = p.get("fixture_league") or p.get("league")
        fdate = pd.to_datetime(p.get("fixture_date"), errors="coerce")

        r: dict = {
            "player_id": pid, "player_name": p.get("player_name", ""),
            "team": p.get("team", ""), "opponent": p.get("opponent", ""),
            "position": p.get("position", ""),
            "n_games": len(phist), "n_prev_games": len(phist_all), "data_source": "match_level",
        }
        mins = _num(phist, "minutes")
        r["minutes_est"] = p.get("minutes", NAN)
        if "fixture_id" in phist.columns and ctx["blank_corner_fixtures"]:
            phist = phist.assign(team_corners=_num(phist, "team_corners").where(
                ~phist["fixture_id"].isin(ctx["blank_corner_fixtures"])))
        for feat, raw in (("goals_pg", "goals"), ("assists_pg", "assists"), ("shots_pg", "shots_total"),
                          ("sot_pg", "shots_on_target"), ("cards_pg", "yellow_cards"),
                          ("minutes_pg", "minutes"), ("key_passes_pg", "key_passes"),
                          ("starter_rate", "started"), ("team_corners_pg", "team_corners"),
                          ("tackles_pg", "tackles_total"), ("interceptions_pg", "interceptions"),
                          ("dribbles_pg", "dribbles_success"), ("dribbled_past_pg", "dribbles_past"),
                          ("red_cards_pg", "red_cards"), ("passes_pg", "passes_total"),
                          ("offsides_pg", "offsides"), ("penalties_won_pg", "penalty_won"),
                          ("saves_pg", "saves"), ("goals_conceded_gk_pg", "goals_conceded"),
                          ("sp_goals_pg", "sp_goal"), ("headed_goals_pg", "headed_goal"),
                          ("fk_goals_pg", "fk_goal"), ("sp_assist_pg", "sp_assist")):
            r[feat] = _mean_last(_num(phist, raw), n)
        r["rating_pg"] = _mean_last(_num(phist, "rating").where(lambda s: s > 0), n)
        _fc, _fd = _num(phist, "fouls_committed"), _num(phist, "fouls_drawn")
        for feat, ser in (
                ("shot_accuracy_rate", _ratio(_num(phist, "shots_on_target"), _num(phist, "shots_total"))),
                ("kp_per90", _ratio(_num(phist, "key_passes"), mins) * 90),
                ("goal_involvement_rate", _ratio(_num(phist, "goals") + _num(phist, "assists"), mins) * 90),
                ("box_actions_per90", _ratio(_num(phist, "shots_total") + _num(phist, "duels_won"), mins) * 90),
                ("aerial_won_rate", _ratio(_num(phist, "duels_won"), _num(phist, "duels_total"))),
                ("duel_intensity_per90", _ratio(_num(phist, "duels_total"), mins) * 90),
                ("fouls_drawn_per90", _ratio(_fd, mins) * 90),
                ("fouls_per90", _ratio(_fc, mins) * 90),
                ("foul_committer_ratio", (_fc / (_fc + _fd + 0.01)).where((_fc + _fd) > 0)),
                ("tackles_per90", _ratio(_num(phist, "tackles_total"), mins) * 90),
                ("defensive_actions_per90",
                 _ratio(_num(phist, "tackles_total") + _num(phist, "interceptions"), mins) * 90),
                ("dribble_success_rate", _ratio(_num(phist, "dribbles_success"), _num(phist, "dribbles_attempted")))):
            r[feat] = _mean_last(ser, n)
        r["gk_save_rate"] = _mean_last(_ratio(_num(phist, "saves"),
                                              _num(phist, "saves") + _num(phist, "goals_conceded")),
                                       n, config.MIN_GK_MATCHES)

        # Venue split: the player's last n games AT THIS VENUE (training definition).
        if "is_home" in phist_all.columns:
            _h = phist_all["is_home"].astype(bool)
            for raw, pref in (("goals", "goals"), ("shots_total", "shots"), ("shots_on_target", "sot")):
                r[f"{pref}_home_pg"] = _mean_last(_num(phist_all[_h], raw), n)
                r[f"{pref}_away_pg"] = _mean_last(_num(phist_all[~_h], raw), n)

        # Career-to-date (full history).
        for feat, raw in (("career_goals_pg", "goals"), ("career_sot_pg", "shots_on_target"),
                          ("career_shots_pg", "shots_total"), ("career_assists_pg", "assists"),
                          ("career_sp_goals_rate", "sp_goal"), ("career_sp_assist_rate", "sp_assist")):
            s = _num(phist_all, raw)
            r[feat] = float(s.mean()) if s.notna().any() else NAN
        # Season to date: the player's games in the UPCOMING fixture's season (as-of definition).
        season = config.prop_season(league, fdate.date()) if (league and fdate == fdate) else None
        _sr = config._v9_src("season_resolver")
        r["penalty_conversion_rate"] = _penalty_now(ctx["pen_totals"], pid, phist_all,
                                                    season if season and _sr.is_known(season) else None)
        if season and _sr.is_known(season) and "season" in phist_all.columns:
            cur = phist_all[phist_all["season"].astype(str) == str(season)]
        else:
            cur = phist_all.iloc[0:0]
        for feat, raw in _SEASON_SRC.items():
            s = _num(cur, raw)
            r[feat] = float(s.mean()) if s.notna().any() else NAN
        r["season_appearances"] = float(len(cur)) if season and _sr.is_known(season) else NAN

        # Profile: a real age from the caller (live squad list) first, then the collected /players
        # record (age at the fixture date from the birth date; latest height / weight), else the
        # history's measured value; the old code's placeholders count as missing.
        prof = ctx["profiles"]
        prow = prof.loc[pid] if (len(prof) and pid in prof.index) else None
        for c in ("age", "height_cm", "weight_kg"):
            given = _f(p.get(c)) if c == "age" else NAN
            if given == given and given > 0:
                r[c] = given
                continue
            if prow is not None:
                if c == "age":
                    bd = pd.to_datetime(prow["birth_date"], errors="coerce")
                    given = (fdate - bd).days / 365.25 if (bd == bd and fdate == fdate) else NAN
                else:
                    given = _f(prow[c])
                if given == given:
                    r[c] = given
                    continue
            s = _profile(phist_all, c).dropna()
            r[c] = float(s.iloc[-1]) if len(s) else NAN

        # Injury: the collected /sidelined answer as of the fixture date; else the LATEST stored
        # row's values (NaN stays NaN; never an older row's value).
        _sid = ctx["sidelined"].get(int(pid)) if isinstance(pid, (int, np.integer, float)) else None
        if _sid is not None and fdate == fdate:
            from .history_integrity import injury_features_asof
            r.update(injury_features_asof(_sid, fdate))
        else:
            for c in ("chronic_injury_risk", "days_since_last_injury", "return_from_injury_flag"):
                v = _num(phist_all, c)
                r[c] = float(v.iloc[-1]) if len(v) else NAN
            # Rows written before 2026-11-01 encode "no recorded spell" as the 365 sentinel: decode
            # it to the current meaning (NaN days, history_integrity.injury_features_asof).
            from .history_integrity import NO_INJURY_DAYS as _LEGACY_NONE
            if r["days_since_last_injury"] == _LEGACY_NONE:
                r["days_since_last_injury"] = NAN

        # Set-piece roles from mapped Understat shots, as of the fixture date.
        if fdate == fdate and len(ctx["sp_app"]):
            _sp = _SP.asof(ctx["sp_app"], [pid], [fdate]).iloc[0]
            r.update({c: float(_sp[c]) for c in _SP.SETPIECE_COLUMNS})
        else:
            r.update({c: NAN for c in _SP.SETPIECE_COLUMNS})

        # Referee, measured (pipeline passes referee_profile_from_history).
        r["referee_yellows_pg"] = _f(ref.get("yellows_per_game"))
        r["referee_strictness"] = _f(ref.get("strictness_score"))

        # Venue, position.
        r["is_home"] = _f(p.get("is_home"))
        _pos = str(p.get("position") or (phist["position"].iloc[-1] if "position" in phist.columns else "")
                   ).strip().upper()[:1]
        r["pos_forward"], r["pos_midfielder"], r["pos_defender"] = (
            int(_pos == "F"), int(_pos == "M"), int(_pos == "D"))

        # Rest days: from the player's last match to THIS fixture (NaN without a fixture date).
        last = pd.to_datetime(phist_all["date"], errors="coerce").max()
        gap = (fdate.normalize() - last.normalize()).days if (fdate == fdate and last == last) else NAN
        r["rest_days"] = float(min(gap, 30)) if gap == gap and gap >= 1 else NAN

        # Opponent (measured) and league averages.
        okey = (str(p.get("opponent", "")), league)
        if okey not in opp_cache:
            opp_cache[okey] = _opponent_state(ctx, okey[0], league, n)
        r.update(opp_cache[okey])
        if league not in lg_cache:
            lg_cache[league] = _league_now(ctx["table"], league) if not ctx["table"].empty else {}
        r.update(lg_cache[league])
        r["lg_yellows_per_match"] = _f(ref.get("lg_yellows_per_match"))

        # Measured strength (league_quality.py): the fixture's clubs and league as of its date, and
        # the player's career averages over his stored rows. Legacy hand-tier columns: NaN.
        skey = (str(p.get("team", "")), okey[0], league, fdate)
        if skey not in lg_cache:
            lg_cache[skey] = _LQ.fixture_strength(skey[0], skey[1], league, fdate)
        r.update(lg_cache[skey])
        _rows = ctx["strength_by_pid"].get(pid)
        r.update(_LQ.career_now(_rows, league) if _rows is not None else
                 {"player_career_avg_league_elo": NAN, "player_career_avg_opp_elo": NAN})
        r["league_tier"] = r["league_quality"] = r["player_career_avg_quality"] = NAN
        rows.append(r)

    if no_pid or no_hist:
        print(f"[features] upcoming: {no_pid} player(s) without player_id and {no_hist} without "
              f"history skipped")
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    _ensure(out, _COMPOSITE_INPUTS)
    out = _add_composites(out)
    n_no_opp = int(out["opp_goals_conceded_pg"].isna().sum())
    if n_no_opp:
        print(f"[features] upcoming: {n_no_opp}/{len(out)} row(s) without measured opponent "
              f"concede rates (opponent unresolved or < {config.MIN_OPP_MATCHES} prior matches)")
    return out


def build_rolling_features(match_rows: list[dict], n: int = None) -> dict:
    """Compute rolling feature dict from a player's recent match rows (predict path). A column the
    rows do not carry is NaN (was 0.0)."""
    if n is None:
        n = config.ROLLING_N
    if not match_rows:
        return {}

    rows  = match_rows[:n]
    df    = pd.DataFrame(rows)
    apps  = len(rows)

    def _pg(col):
        return float(df[col].sum() / apps) if col in df.columns else NAN

    goals_pg   = _pg("goals")
    assists_pg = _pg("assists")
    shots_pg   = _pg("shots_total")
    sot_pg     = _pg("shots_on_target")
    cards_pg   = _pg("yellow_cards")
    minutes_pg = _pg("minutes")
    kp_pg      = _pg("key_passes")
    sot_rate   = sot_pg / shots_pg if shots_pg > 0 else NAN

    return {
        "goals_pg":      round(goals_pg,   4),
        "assists_pg":    round(assists_pg,  4),
        "shots_pg":      round(shots_pg,    4),
        "sot_pg":        round(sot_pg,      4),
        "cards_pg":      round(cards_pg,    4),
        "minutes_pg":    round(minutes_pg,  1),
        "key_passes_pg": round(kp_pg,       4),
        "sot_rate":      round(sot_rate,    4),
        "n_games":       apps,
        "data_source":   "match_level",
    }


def compute_ges(row: dict, opp_weakness: float = NAN, penalty_duty: bool = False) -> float:
    """Goal Edge Score (0.0-1.0). Gates goals/SOT signals.

    For forwards: replaces xg_form with clinical efficiency (sot_rate * opp_weakness).
    For non-forwards: uses legacy goals_pg form (header/set-piece scorers).

    Any input it needs that is missing -> NaN (predict then gives the goals/SOT tip AVOID). The
    old .get(..., 0) defaults and opp_weakness=1.0 manufactured a score for unknown players.
    """
    def _v(k):
        v = row.get(k)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return NAN
        return v

    goals_pg, shots_pg, minutes_pg = _v("goals_pg"), _v("shots_pg"), _v("minutes_pg")
    is_fwd, sot_rate = _v("pos_forward"), _v("sot_rate")
    try:
        opp_weakness = float(opp_weakness)
    except (TypeError, ValueError):
        opp_weakness = NAN
    need = [goals_pg, shots_pg, minutes_pg, is_fwd, opp_weakness] + ([sot_rate] if is_fwd == 1 else [])
    if any(v != v for v in need):
        return NAN

    xg_form  = min(goals_pg / 0.50, 1.0)
    shot_vol = min(shots_pg / 3.5, 1.0)
    pen      = 1.0 if penalty_duty else 0.0
    opp      = min(opp_weakness / 1.5, 1.0)
    min_sec  = min(minutes_pg / 85.0, 1.0)
    # clinical: forward accuracy × defensive leakiness, normalised to [0,1]
    form = min(sot_rate * opp_weakness / (0.35 * 1.5), 1.0) if is_fwd else xg_form

    return round(0.40 * form + 0.20 * shot_vol + 0.15 * pen + 0.15 * opp + 0.10 * min_sec, 3)
