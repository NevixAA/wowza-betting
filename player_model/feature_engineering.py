"""
Player Feature Engineering v3 — match-level rolling features
=============================================================
build_features(match_rows)
  Input:  flat list of player-match dicts from collect_match_history()
          Each dict = one player in one specific completed match.
  Output: DataFrame with proper rolling features (shift=1, no leakage)
          and per-match binary targets.

build_upcoming_features(upcoming, history_df)
  Runs build_features ITSELF on the history plus the upcoming rows (no result yet) and returns
  those rows: every serving input is the training definition, value for value (2026-11-09).

DATA TRUTH (owner rule, 2026-11-01). No feature is ever a typed value:
  * every opponent / keeper / referee / league quantity is COMPUTED from the matches we hold
    (the per-(fixture, team) table `team_match_table`), with an explicit minimum sample
    (config.MIN_*); below it the value is NaN;
  * "league average" normalisers are the league's OWN measured average from its prior fixtures
    (`_league_asof`), never 1.3 / 6.0 / 3.5 / 6.8 / 0.12;
  * a missing value stays NaN (no fillna(0.0) / fillna(<prior>)); the serving model's imputer
    (model.predict_proba) decides what a missing input means, and logs it;
  * training and serving share ONE code path: serving rows are computed by build_features on the
    same history (build_upcoming_features), so no feature can be computed one way in training and
    another way at serve time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from . import league_quality as _LQ
from .setpiece import features as _SP

NAN = float("nan")


def _label(df: pd.DataFrame, col: str, k: int) -> pd.Series:
    """1.0 when the recorded count >= k, 0.0 when it is below, NaN when no count was recorded (or the
    column is absent) - a missing count is never read as 0 (owner 2026-11-04)."""
    v = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(NAN, index=df.index)
    return (v >= k).astype(float).where(v.notna())

# PURGED set-piece features (owner rule 2026-11-02, "no fake data, no proxies"). None of them is
# set-piece data: they were typed position weights (0.45 / 0.30 ...), typed rates (x 0.30 "sp
# rate", x 5.0 / x 4.0 "SOT proxy"), clip bounds (.clip(0.3, 2.0), .clip(0, 0.10) / 0.10) and
# arithmetic on aerial-duel / corner / height numbers, or rolling means of `sp_goal` / `sp_assist`,
# which data_fetcher._parse_sp_events sets for ANY header or free-kick goal (a header from open
# play is not a set piece; API-Football labels goals "Normal Goal" so the split is unknown anyway).
# They are no longer built for training OR serving (no NaN placeholder column either); a legacy
# serving payload that lists one reads it as absent -> NaN, exactly what it read before (they were
# all-NaN columns). The ONLY set-piece features are the Understat per-shot situations
# (_SP.SETPIECE_COLUMNS, us_sp_*: NaN where Understat has no data).
REMOVED_SETPIECE_FEATURES = (
    "sp_taker_score", "sp_receiver_score", "set_piece_threat_score", "set_piece_corner_matchup",
    "defender_sot_role_index", "defender_sp_edge", "defender_sot_edge",
    "sp_threat_vs_weak_sp_defense", "sp_taker_assist_edge", "aerial_height_sp_composite",
    "sp_goal_probability_composite", "opp_sp_goals_conceded_pg", "lg_sp_per_team",
    "sp_goals_pg", "sp_assist_pg", "career_sp_goals_rate", "career_sp_assist_rate", "sp_goals_share",
)

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


def profile_composites(f: pd.DataFrame) -> pd.DataFrame:
    """The age / height composites, IN PLACE (NaN wherever an input is missing). One definition for
    build_features/_add_composites and scripts/rebuild_player_history.py."""
    f["height_diff_vs_opp_def"] = f["height_cm"] - f["opp_def_mean_height"]
    f["height_advantage_score"] = (f["height_diff_vs_opp_def"] / 10.0).clip(-2.0, 2.0)
    f["height_aerial_interaction"] = ((f["height_cm"].clip(160, 200) - 160) / 40) * f["aerial_won_rate"]
    f["age_peak_delta"] = (f["age"] - 27).abs()
    return f


def minutes_ok(minutes) -> pd.Series:
    """Minutes played, NaN above config.MAX_MATCH_MINUTES (a provider error, not a measurement).
    ONE definition for build_features and serving."""
    m = pd.to_numeric(pd.Series(minutes), errors="coerce")
    return m.where(m <= config.MAX_MATCH_MINUTES)


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


def _known_before(v: pd.Series, key) -> pd.Series:
    """Per group: the sum of the RECORDED values on earlier rows (an unrecorded value adds nothing;
    the row's own value is never read, so an upcoming row - no result yet - and a played row get the
    same pre-match value). The pattern of league_quality.career averages."""
    k = v.notna()
    vz = v.where(k, 0.0)                                    # unrecorded -> contributes no attempt
    # the prefix sum up to the PREVIOUS row (exact: (S + x) - x is not always S in floating point)
    prev = vz.groupby(key).cumsum().groupby(key).shift(1)
    return prev.where(prev.notna(), 0.0)                    # no earlier row: an empty sum


def _penalty_asof(df: pd.DataFrame) -> pd.Series:
    """penalty_conversion_rate per row of a frame sorted by (player_id, date): earlier seasons'
    totals (_season_pen_totals) + this season's per-match attempts strictly before the row; NaN
    below config.MIN_PEN_ATTEMPTS attempts."""
    if not {"penalty_scored", "penalty_missed", "player_id"} <= set(df.columns):
        return pd.Series(np.nan, index=df.index)
    ps, pm = _num(df, "penalty_scored"), _num(df, "penalty_missed")
    att = ps + pm
    if "season" not in df.columns:                  # no season: the per-match record only
        cs = _known_before(ps, [df["player_id"]])
        ca = _known_before(att, [df["player_id"]])
        return (cs / ca).where(ca >= config.MIN_PEN_ATTEMPTS)
    y = pd.Series(_season_year(df["season"]).to_numpy(), index=df.index)
    key = [df["player_id"], y]
    in_s = _known_before(ps, key)
    in_a = _known_before(att, key)
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
    """(own team key, opponent key) per player row.

    A CLUB is 'af:<API-Football team id>' in EVERY competition (player_model.team_identity): the
    row's own ids, else its fixture's ids (side by club identity), else the club's name resolved
    among the clubs of the row's competition. So an opponent in a Champions League tie is the same
    key as in its domestic league, and its opponent features come from ALL its recent matches
    (owner question 2026-11-02: the cup's own scope held no history, every cup opponent was NaN).
    A national team, or a club no id source resolves, keeps '<name>|<scope>'. NaN where the
    competition is unknown to the registry (Friendlies, qualifiers) and no fixture id proves it."""
    from . import team_identity as TI
    scope = df["league"].map(team_scope) if "league" in df.columns else pd.Series(None, index=df.index)
    if "opponent" in df.columns:
        opp = df["opponent"]
    else:
        opp = df["away_team"].where(df["is_home"].astype(bool), df["home_team"])
    own_n = (df["team"].astype(str) + "|" + scope).where(scope.notna())
    oth_n = (opp.astype(str) + "|" + scope).where(scope.notna())
    tid, oid = TI.row_team_ids(df)
    own, oth = tid.map(TI.key), oid.map(TI.key)
    club = scope.notna() & (scope != "national_teams")
    if "league" in df.columns:
        for ser, names in ((own, df["team"]), (oth, opp)):
            m = ser.isna() & club
            if m.any():
                pairs = pd.Series(list(zip(names[m].astype(str), df.loc[m, "league"])), index=names[m].index)
                cache = {p: TI.club_key(p[0], p[1]) for p in pd.unique(pairs)}
                ser.loc[m] = pairs.map(cache)
    return own.where(own.notna(), own_n), oth.where(oth.notna(), oth_n)


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
              "yellow_cards", "saves", "goals_conceded", "team_corners"):
        d[c] = _num(d, c)
    d["_rating"] = _num(d, "rating").where(lambda s: s > 0)
    d["_height"] = _height_series(d)
    key = ["fixture_id", "_tkey"]

    t = d.groupby(key).agg(date=("date", "first"), league=("league", "first"),
                           _okey=("_okey", "first"),
                           goals_scored=("goals", "sum"), _gs_n=("goals", "count"),
                           sot_scored=("shots_on_target", "sum"), _ss_n=("shots_on_target", "count"),
                           team_corners=("team_corners", "first"))
    # A sum over rows that carry NO count is missing, never 0 (an unplayed upcoming fixture, or a
    # fixture the provider sent without counts; owner 2026-11-09: no defaults).
    t["goals_scored"] = t["goals_scored"].where(t.pop("_gs_n") > 0)
    t["sot_scored"] = t["sot_scored"].where(t.pop("_ss_n") > 0)

    def _role(letter, **aggs):
        sub = d[pos == letter]
        if sub.empty:
            return pd.DataFrame(index=t.index)
        return sub.groupby(key).agg(**aggs)

    dd = _role("D", _dw=("duels_won", "sum"), _dw_n=("duels_won", "count"),
               _dt=("duels_total", "sum"), _dt_n=("duels_total", "count"),
               def_fouls_pg=("fouls_committed", "mean"), def_cards_pg=("yellow_cards", "mean"),
               def_rating=("_rating", "mean"), def_rating_top=("_rating", "max"),
               def_height=("_height", "mean"))
    if not dd.empty:
        dd["def_aerial"] = _ratio(dd.pop("_dw").where(dd.pop("_dw_n") > 0),
                                  dd.pop("_dt").where(dd.pop("_dt_n") > 0))
    gk = _role("G", gk_saves=("saves", "sum"), _sn=("saves", "count"),
               _gc=("goals_conceded", "sum"), _gn=("goals_conceded", "count"))
    if not gk.empty:
        gk["gk_saves"] = gk["gk_saves"].where(gk.pop("_sn") > 0)
        _gc = gk.pop("_gc").where(gk.pop("_gn") > 0)
        gk["gk_save_rate"] = _ratio(gk["gk_saves"], gk["gk_saves"] + _gc)
    fw = _role("F", fwd_goals=("goals", "sum"), _fg_n=("goals", "count"),
               fwd_sot=("shots_on_target", "sum"), _fs_n=("shots_on_target", "count"))
    md = _role("M", mid_goals=("goals", "sum"), _mg_n=("goals", "count"),
               mid_sot=("shots_on_target", "sum"), _ms_n=("shots_on_target", "count"),
               mid_fouls_pg=("fouls_committed", "mean"), mid_cards_pg=("yellow_cards", "mean"))
    for _x, _pairs in ((fw, (("fwd_goals", "_fg_n"), ("fwd_sot", "_fs_n"))),
                       (md, (("mid_goals", "_mg_n"), ("mid_sot", "_ms_n")))):
        for _c, _n in _pairs:
            if _c in _x.columns:
                _x[_c] = _x[_c].where(_x.pop(_n) > 0)
    t = t.join([x for x in (dd, gk, fw, md) if not x.empty], how="left")
    for c in ("def_aerial", "def_fouls_pg", "def_cards_pg", "def_rating", "def_rating_top",
              "def_height", "gk_saves", "gk_save_rate", "fwd_goals", "fwd_sot", "mid_goals",
              "mid_sot", "mid_fouls_pg", "mid_cards_pg"):
        if c not in t.columns:
            t[c] = np.nan
    t = t.reset_index()

    # What the OTHER side did in the same fixture = what this team conceded.
    other = t[["fixture_id", "_tkey", "goals_scored", "sot_scored",
               "fwd_goals", "fwd_sot", "mid_goals", "mid_sot", "team_corners"]].rename(columns={
        "_tkey": "_okey", "goals_scored": "goals_conceded", "sot_scored": "sot_conceded",
        "fwd_goals": "conc_fwd_goals", "fwd_sot": "conc_fwd_sot",
        "mid_goals": "conc_mid_goals", "mid_sot": "conc_mid_sot", "team_corners": "corners_conceded"})
    t = t.merge(other, on=["fixture_id", "_okey"], how="left")
    # Corners for / against from the API-Football team statistics. A fixture where BOTH sides show 0
    # is an unrecorded statistic, not a result (see _blank_unrecorded_corners): NaN.
    both0 = (t["team_corners"] == 0) & (t["corners_conceded"] == 0)
    t.loc[both0, ["team_corners", "corners_conceded"]] = np.nan
    # Only the side we hold rows for can be measured: a fixture whose opponent was not collected
    # leaves every conceded quantity NaN (it used to drop out of an inner self-join silently).
    t = _with_fixture_scores(t)
    # An UPCOMING fixture (serving: no result yet) has no score row. The opponent side's row is
    # added with every quantity missing, so the opponent's pre-match rolling is read at this
    # fixture exactly as training reads it from the score-only row of the played fixture.
    if UPCOMING_COL in d.columns and d[UPCOMING_COL].astype(bool).any():
        u = (d[d[UPCOMING_COL].astype(bool)].groupby(["fixture_id", "_tkey"])
             .agg(date=("date", "first"), league=("league", "first"), _okey=("_okey", "first"))
             .reset_index())
        opp = u.rename(columns={"_tkey": "_okey", "_okey": "_tkey"})
        opp = opp[opp["_tkey"].notna()]
        held = set(zip(pd.to_numeric(t["fixture_id"], errors="coerce"), t["_tkey"]))
        opp = opp[[(f, k) not in held for f, k in zip(pd.to_numeric(opp["fixture_id"], errors="coerce"),
                                                      opp["_tkey"])]]
        if len(opp):
            t = pd.concat([t, opp.reindex(columns=t.columns)], ignore_index=True)
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t["_day"] = t["date"].dt.normalize()
    return t.sort_values(["_tkey", "date", "fixture_id"]).reset_index(drop=True)


def _with_fixture_scores(t: pd.DataFrame) -> pd.DataFrame:
    """Goals scored / conceded per team-fixture from the API-Football 90-MINUTE score
    (output/strength/competition_fixtures.parquet, score.fulltime), and one row per side of every
    collected fixture we hold NO player rows for (goals only, every other quantity NaN).

    Why: a club's recent matches are ALL the competitions it played. A Champions League opponent
    from a league whose players we do not collect (Club Brugge, Sporting, Kairat) has its domestic
    and cup results in the collected fixtures, so its goals-conceded rate is measured from those;
    its shots / keeper / defender / midfield quantities exist only where we hold its player rows
    (its cup matches, once scripts/collect_cup_player_stats.py has run). The player-row goal sum
    it replaces missed own goals and counted extra-time goals; the fixture score is the 90-minute
    figure (props settle on 90 minutes). A fixture without a stored score keeps the player sum."""
    from . import team_identity as TI
    fx = TI.fixture_results()
    if fx.empty or t.empty:
        return t
    fx = fx.dropna(subset=["home_id", "away_id"])
    sides = []
    for own, oth, gs, gc in (("home_id", "away_id", "ft_home", "ft_away"),
                             ("away_id", "home_id", "ft_away", "ft_home")):
        sides.append(pd.DataFrame({"fixture_id": fx["fixture_id"].to_numpy(),
                                   "_tkey": fx[own].map(TI.key).to_numpy(),
                                   "_okey": fx[oth].map(TI.key).to_numpy(),
                                   "date": fx["date"].to_numpy(), "league": fx["competition"].to_numpy(),
                                   "_gs": fx[gs].to_numpy(dtype=float), "_gc": fx[gc].to_numpy(dtype=float)}))
    s = pd.concat(sides, ignore_index=True).dropna(subset=["_tkey", "_okey"])
    s = s[s["_gs"].notna() & s["_gc"].notna()]
    t = t.assign(fixture_id=pd.to_numeric(t["fixture_id"], errors="coerce"))
    t = t.merge(s[["fixture_id", "_tkey", "_gs", "_gc"]], on=["fixture_id", "_tkey"], how="left")
    has = t["_gs"].notna()
    t.loc[has, "goals_scored"] = t.loc[has, "_gs"]
    t.loc[has, "goals_conceded"] = t.loc[has, "_gc"]
    t = t.drop(columns=["_gs", "_gc"])
    held = pd.MultiIndex.from_frame(t[["fixture_id", "_tkey"]])
    new = s[~pd.MultiIndex.from_frame(s[["fixture_id", "_tkey"]]).isin(held)]
    if len(new):
        add = new.rename(columns={"_gs": "goals_scored", "_gc": "goals_conceded"})
        t = pd.concat([t, add.reindex(columns=t.columns)], ignore_index=True)
    return t


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


def _roll_measured(t: pd.DataFrame, col: str, n: int, min_n: int) -> pd.Series:
    """Per team (_tkey), BEFORE each fixture: the mean of the team's last n fixtures that MEASURE
    `col` (unmeasured fixtures skipped, not counted), NaN below min_n. A team's fixtures include
    every competition (the score-only rows of _with_fixture_scores carry goals but no shots /
    keeper / defender figures): skipping the unmeasured ones keeps those features on the team's
    last n measured matches instead of losing them to a tie we hold no player rows for.
    `t` sorted by (_tkey, date)."""
    v = pd.to_numeric(t[col], errors="coerce")
    ok = v.notna()
    key = t["_tkey"]
    # measured fixtures strictly before each row, within its team
    before = ok.astype(int).groupby(key).cumsum() - ok.astype(int)
    out = pd.Series(np.nan, index=t.index, dtype=float)
    if not ok.any():
        return out
    sub = pd.DataFrame({"_tkey": key[ok], "v": v[ok]})
    sub["_j"] = sub.groupby("_tkey").cumcount() + 1
    sub["after"] = (sub.groupby("_tkey")["v"].rolling(n, min_periods=min_n).mean()
                    .reset_index(level=0, drop=True).reindex(sub.index))
    q = pd.DataFrame({"_tkey": key, "_j": before, "_i": np.arange(len(t))})
    hit = q[q["_j"] > 0].merge(sub[["_tkey", "_j", "after"]], on=["_tkey", "_j"], how="left")
    out.iloc[hit["_i"].to_numpy()] = hit["after"].to_numpy()
    return out


def _team_rolling(t: pd.DataFrame, n: int) -> pd.DataFrame:
    """Pre-match rolling value of each _TEAM_ROLL quantity per team: the mean of its last n
    fixtures that measure it (strictly before the fixture; min sample; _roll_measured)."""
    out = t[["fixture_id", "_tkey"]].copy()
    for src, (feat, min_attr) in _TEAM_ROLL.items():
        out[feat] = _roll_measured(t, src, n, getattr(config, min_attr))
    out["_opp_team_corners_pg"] = _roll_measured(t, "team_corners", n, config.MIN_OPP_MATCHES)
    return out


# ── league averages, measured from the league's own prior fixtures ──────────────────────────────

# League normaliser -> the team-table column it averages (per team-match)
_LEAGUE_NORMS = {
    "lg_goals_per_team":   "goals_scored",     # goals per team-match (the opp_strength_index base)
    "lg_corners_per_team": "team_corners",
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
        # strictly earlier dates: the prefix sum up to the previous date (exact in floating point;
        # cumsum - day is not, so serving and training could differ in the last bit)
        cum = day.groupby(level="league").cumsum().groupby(level="league").shift(1)
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
                referee=("referee", "first"), yellows=("_yc", "sum"), _yn=("_yc", "count")))
    f["yellows"] = f["yellows"].where(f.pop("_yn") > 0)     # no recorded card count -> missing, not 0
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
    cum = day.groupby(level="league").cumsum().groupby(level="league").shift(1)   # exact prefix
    mu = cum["sum"] / cum["count"]
    sd = np.sqrt((cum["sq"] / cum["count"] - mu ** 2).clip(lower=0) * cum["count"] / (cum["count"] - 1))
    ok = cum["count"] >= config.MIN_LEAGUE_MATCHES
    lg = pd.DataFrame({"_lg_mu": mu.where(ok), "_lg_sd": sd.where(ok & (sd > 0))}).reset_index()
    f = f.merge(lg, on=["league", "date"], how="left")
    f["referee_strictness"] = (f["referee_yellows_pg"] - f["_lg_mu"]) / f["_lg_sd"]
    f["lg_yellows_per_match"] = f["_lg_mu"]
    return f[["fixture_id", "referee_yellows_pg", "referee_strictness", "lg_yellows_per_match"]]


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


# ── previous season, longer-window shot accuracy (owner-approved 2026-11-02; ONE definition) ─────

# prev_season_* player feature -> per-match source column: the mean per appearance over the
# player's rows of the PREVIOUS season (season start year - 1, every competition we hold), dated
# strictly before the row. Own names: they never fill a season_* (this-season) value.
PREV_SEASON_PLAYER = {
    "prev_season_goals_pg":    "goals",
    "prev_season_assists_pg":  "assists",
    "prev_season_shots_pg":    "shots_total",
    "prev_season_sot_pg":      "shots_on_target",
    "prev_season_cards_pg":    "yellow_cards",
    "prev_season_minutes_pg":  "_minutes_ok",
    "prev_season_start_rate":  "started",
}
# prev_season_* team feature -> (side, team-table column): the player's own team ('own') or the
# opponent ('opp'), mean per match over that team's previous-season fixtures (90-minute scores).
PREV_SEASON_TEAM = {
    "prev_season_team_goals_pg":     ("own", "goals_scored"),
    "prev_season_team_conceded_pg":  ("own", "goals_conceded"),
    "prev_season_opp_goals_pg":      ("opp", "goals_scored"),
    "prev_season_opp_conceded_pg":   ("opp", "goals_conceded"),
}
PREV_SEASON_COLUMNS = (*PREV_SEASON_PLAYER, "prev_season_appearances", "prev_season_shot_accuracy",
                       *PREV_SEASON_TEAM)


def _prev_season_sums(src: pd.DataFrame, key: str, cols, q: pd.DataFrame) -> pd.DataFrame:
    """For each query row of `q` (columns key, _y, date): over the `src` rows with the same key whose
    season start year (_y) is the query's _y - 1 and whose date is STRICTLY before the query date,
    the row count (_n) and, per col, the sum (<col>) and the number of rows carrying it (<col>__n).
    Returned aligned with q's index. Used by training and serving alike."""
    out_cols = ["_n", *cols, *(f"{c}__n" for c in cols)]
    out = pd.DataFrame(np.nan, index=q.index, columns=out_cols)
    if src.empty or q.empty:
        return out
    s = src[[key, "_y", "date", *cols]].copy()
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    s["_y"] = pd.to_numeric(s["_y"], errors="coerce").astype(float)
    s = s.dropna(subset=[key, "_y", "date"]).sort_values("date", kind="stable")
    if s.empty:
        return out
    gk = [s[key], s["_y"]]
    cum = pd.DataFrame({key: s[key].to_numpy(), "_y": s["_y"].to_numpy(), "date": s["date"].to_numpy(),
                        "_n": s.groupby(gk, sort=False).cumcount().to_numpy() + 1.0})
    for c in cols:
        v = pd.to_numeric(s[c], errors="coerce")
        cum[c] = v.fillna(0.0).groupby(gk).cumsum().to_numpy()
        cum[f"{c}__n"] = v.notna().astype(float).groupby(gk).cumsum().to_numpy()
    qq = pd.DataFrame({key: q[key].to_numpy(),
                       "_y": pd.to_numeric(q["_y"], errors="coerce").to_numpy(dtype=float) - 1.0,
                       "date": pd.to_datetime(q["date"], errors="coerce").to_numpy(), "_i": np.arange(len(q))})
    qq = qq.dropna(subset=[key, "_y", "date"]).sort_values("date", kind="stable")
    if qq.empty:
        return out
    if cum[key].dtype != qq[key].dtype:
        cum[key], qq[key] = cum[key].astype(str), qq[key].astype(str)
    hit = pd.merge_asof(qq, cum, on="date", by=[key, "_y"], allow_exact_matches=False)
    out.iloc[hit["_i"].to_numpy(), :] = hit[out_cols].to_numpy(dtype=float)
    return out


def _prev_season_player(src: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """prev_season_* player features for the queries q (player_id, _y, date) from the player rows
    `src` (player_id, _y, date + raw columns). NaN below config.MIN_PREV_SEASON_MATCHES rows; the
    shot accuracy is sum(sot) / sum(shots), NaN below config.MIN_SHOTS_FOR_ACCURACY shots."""
    raws = sorted(set(PREV_SEASON_PLAYER.values()))
    sh, so = _num(src, "shots_total"), _num(src, "shots_on_target")
    both = sh.notna() & so.notna()
    src = src.assign(_minutes_ok=minutes_ok(_num(src, "minutes")).to_numpy(),
                     **{r: _num(src, r) for r in raws if r != "_minutes_ok"},
                     _acc_sh=sh.where(both), _acc_so=so.where(both))
    s = _prev_season_sums(src, "player_id", [*raws, "_acc_sh", "_acc_so"], q)
    enough = s["_n"] >= config.MIN_PREV_SEASON_MATCHES
    out = pd.DataFrame(index=q.index)
    for feat, raw in PREV_SEASON_PLAYER.items():
        out[feat] = _ratio(s[raw], s[f"{raw}__n"]).where(enough)
    out["prev_season_appearances"] = s["_n"]            # NaN = no row of that season held
    out["prev_season_shot_accuracy"] = _ratio(s["_acc_so"], s["_acc_sh"]).where(
        s["_acc_sh"] >= config.MIN_SHOTS_FOR_ACCURACY)
    return out


def _team_seasons(t: pd.DataFrame) -> pd.Series:
    """Season start year of every team-table row (the registry season rule of its competition via
    src.season_resolver; NaN when the competition is unknown to the registry)."""
    if t.empty:
        return pd.Series(dtype=float)
    return pd.Series(_season_year(_resolve_seasons(t[["league", "date"]])).to_numpy(), index=t.index)


def _prev_season_team(t: pd.DataFrame, q: pd.DataFrame, side: str) -> pd.DataFrame:
    """Team prev_season_* of `side` ('own' / 'opp') for the queries q (_tkey, _y, date): mean per
    fixture over the team's previous-season fixtures (team table `t` with _y), NaN below
    config.MIN_PREV_SEASON_MATCHES fixtures carrying the quantity."""
    cols = sorted({c for _, c in PREV_SEASON_TEAM.values()})
    s = _prev_season_sums(t, "_tkey", cols, q)
    out = pd.DataFrame(index=q.index)
    for feat, (sd, col) in PREV_SEASON_TEAM.items():
        if sd == side:
            out[feat] = _ratio(s[col], s[f"{col}__n"]).where(s[f"{col}__n"] >= config.MIN_PREV_SEASON_MATCHES)
    return out


def shot_accuracy_before(df: pd.DataFrame, n: int | None = None) -> pd.Series:
    """shot_accuracy_l10: per row of `df` (sorted by player, date), BEFORE it: sum(shots on target) /
    sum(shots) over the player's previous n matches (config.SHOT_ACC_WINDOW; a match without both
    counts recorded adds nothing), NaN below config.MIN_SHOTS_FOR_ACCURACY shots. A separate
    feature from the 5-match shot_accuracy_rate (mean of per-match ratios). Serving calls it on the
    player's history plus one empty row for the fixture: the same definition."""
    n = n or config.SHOT_ACC_WINDOW
    sh, so = _num(df, "shots_total"), _num(df, "shots_on_target")
    both = sh.notna() & so.notna()
    g = df["player_id"]
    sums = {}
    for name, v in (("so", so.where(both)), ("sh", sh.where(both))):
        prev = v.groupby(g).shift(1)
        sums[name] = (prev.groupby(g).rolling(n, min_periods=1).sum()
                      .reset_index(level=0, drop=True).reindex(df.index))
    return _ratio(sums["so"], sums["sh"]).where(sums["sh"] >= config.MIN_SHOTS_FOR_ACCURACY)


# ── rest days on the COMPLETE match history (league + collected cup rows; ONE definition) ────────

def played_days(frame: pd.DataFrame) -> pd.DataFrame:
    """(player_id, _day) of every match a player PLAYED (minutes > 0) in `frame` and in the collected
    UEFA cup rows (output/collect/cup_player_rows.parquet, every status incl. extra time), one row
    per player and day. A row without a minutes value is not proof of a match played: excluded."""
    parts = []
    for src in (frame, _collected("cup_player_rows")):
        if src is None or src.empty or not {"player_id", "date", "minutes"} <= set(src.columns):
            continue
        m = _num(src, "minutes") > 0
        parts.append(pd.DataFrame({
            "player_id": pd.to_numeric(src.loc[m, "player_id"], errors="coerce").to_numpy(dtype=float),
            "_day": pd.to_datetime(src.loc[m, "date"], errors="coerce").dt.normalize().to_numpy()}))
    if not parts:
        return pd.DataFrame(columns=["player_id", "_day"])
    out = pd.concat(parts, ignore_index=True).dropna()
    return out.drop_duplicates().sort_values("_day", kind="stable").reset_index(drop=True)


def rest_days_before(played: pd.DataFrame, q: pd.DataFrame) -> pd.Series:
    """rest_days for each query row (player_id, date): days from the player's last PLAYED day strictly
    before the query day (played_days: every competition we hold) to the query day. NaN without one.
    Uncapped (was clipped at 30: a 60-day gap is 60). Training and serving both call this."""
    out = pd.Series(np.nan, index=q.index, dtype=float)
    if played.empty or q.empty:
        return out
    qq = pd.DataFrame({"player_id": pd.to_numeric(q["player_id"], errors="coerce").to_numpy(dtype=float),
                       "_day": pd.to_datetime(q["date"], errors="coerce").dt.normalize().to_numpy(),
                       "_i": np.arange(len(q))}).dropna(subset=["player_id", "_day"])
    if qq.empty:
        return out
    p = played.assign(_last=played["_day"]).sort_values("_day", kind="stable")
    hit = pd.merge_asof(qq.sort_values("_day", kind="stable"), p, on="_day", by="player_id",
                        allow_exact_matches=False)
    out.iloc[hit["_i"].to_numpy()] = (hit["_day"] - hit["_last"]).dt.days.to_numpy(dtype=float)
    return out


# ── composites: ONE definition for training and serving ─────────────────────────────────────────

def _add_composites(df: pd.DataFrame) -> pd.DataFrame:
    """Every derived feature, from base columns that are already present on `df` (training frame
    or the serving frame). League normalisers (lg_*) are the league's measured averages.
    NaN in -> NaN out; nothing is filled."""
    f = df.copy()                      # de-fragmented working frame (many columns are added)
    pf, pm, pd_ = f["pos_forward"], f["pos_midfielder"], f["pos_defender"]

    f["sot_rate"] = _ratio(f["sot_pg"], f["shots_pg"])
    # Unclipped (was .clip(upper=1.0)): a goal is a shot on target, so the measured ratio is <= 1;
    # a value above 1 is a provider inconsistency and stays visible instead of being rewritten.
    f["shooting_efficiency_index"] = _ratio(f["goals_pg"], f["sot_pg"])
    f["card_exposure_index"] = f["cards_pg"] * (f["minutes_pg"] / 90.0) * (1 - pf)
    f["sot_quality_score"] = f["shot_accuracy_rate"] * f["sot_pg"]
    f["opp_adjusted_shot_threat"] = f["shots_pg"] * f["opp_sot_conceded_pg"]
    f["creative_playmaker_score"] = f["kp_per90"] * (pm + 0.5 * pf)

    corners_rel = _ratio(f["team_corners_pg"], f["lg_corners_per_team"])

    # Set pieces: REAL data only (owner rule 2026-11-02). The only set-piece features are the
    # Understat per-shot situations us_sp_* (player_model/setpiece/features.py; NaN outside the
    # Understat leagues). Nothing here builds a set-piece feature (REMOVED_SETPIECE_FEATURES).
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
    f["offside_aggressiveness"] = f["offsides_pg"] * pf               # unclipped (was upper=2.0)
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
    profile_composites(f)
    f["creative_pressure_matchup"] = (f["creative_playmaker_score"] * f["opp_def_fouls_pg"]).clip(lower=0.0)
    f["dribbler_vs_defensive_line"] = (f["dribble_creativity_score"] * (1.0 - f["opp_def_aerial_win_rate"])).clip(lower=0.0)
    f["carrier_vs_press"] = (f["progressive_carrier_score"] * f["opp_def_aggression"]).clip(lower=0.0)
    f["dribble_contact_rate"] = (f["dribbles_pg"] * (1.0 - f["dribble_success_rate"])).clip(lower=0.0)
    f["tackle_dribble_clash"] = (f["dribble_contact_rate"] * f["opp_mid_fouls_pg"]).clip(lower=0.0)
    # The referee's yellows relative to the league's MEASURED yellows/match (was / 3.5). Unclipped
    # (was a 0.1 floor that manufactured a ratio for a referee with ~no yellows).
    f["card_clash_index"] = ((f["fouls_per90"] + f["dribble_contact_rate"]) * f["opp_mid_cards_pg"]
                             * _ratio(f["referee_yellows_pg"], f["lg_yellows_per_match"])
                             ).clip(lower=0.0)
    f["opp_mid_discipline"] = (f["opp_mid_fouls_pg"] * 0.6 + f["opp_mid_cards_pg"] * 2.0).clip(lower=0.0)

    # Opposition strength: the opponent's goals CONCEDED per match relative to its league's goals per
    # team-match, both measured (was goals scored over a typed/whole-frame mean, fillna 1.3).
    # Unclipped (were .clip(0.3, 3.0), .clip(0, 5), .clip(0, 10)): a bound rewrites a measured ratio
    # into a typed one. The serving HistGradientBoosting payloads were trained on the clipped values,
    # so a value beyond the old bound falls in the same edge bin as the bound: their scores do not move.
    f["opp_strength_index"] = _ratio(f["opp_goals_conceded_pg"], f["lg_goals_per_team"])
    f["quality_adj_goals_pg"] = f["goals_pg"] * f["opp_strength_index"]
    f["quality_adj_sot_pg"] = f["sot_pg"] * f["opp_strength_index"]

    # Opponent defensive quality: each factor relative to the league's measured average (was 6.8 /
    # 7.0 / 0.50). Unclipped (was .clip(0.2, 4.0)).
    f["opp_def_player_quality"] = (_ratio(f["opp_def_player_rating_pg"], f["lg_def_rating"])
                                   * _ratio(f["opp_top_def_rating"], f["lg_def_rating_top"])
                                   * _ratio(f["opp_def_aerial_win_rate"], f["lg_def_aerial"]))
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
    "tackles_per90", "offsides_pg", "height_cm", "age", "headed_goals_pg",
    "fk_goals_pg", "referee_yellows_pg",
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

#: marks a serving row (an upcoming fixture, no result yet) inside build_features
UPCOMING_COL = "is_upcoming"     # kept in the output (columns starting with _ are dropped)
#: position_source values
POSITION_FROM_LINEUP = "LINEUP_ANNOUNCED"
POSITION_FROM_LATEST = "LATEST_EARLIER_MATCH"
POSITION_NONE = "NONE"
_LINEUP_CACHE: dict = {}


def collected_lineup_positions() -> pd.DataFrame:
    """(fixture_id, player_id, pos) from the collected API-Football lineups
    (output/collect/lineups/players_*.parquet, scripts/collect_lineups.py): the ANNOUNCED position of
    every listed player, starters and substitutes. Empty when none are collected."""
    d = COLLECT_DIR / "lineups"
    files = sorted(d.glob("players_*.parquet")) if d.exists() else []
    key = tuple((f.name, f.stat().st_mtime_ns) for f in files)
    if _LINEUP_CACHE.get("key") != key:
        parts = [pd.read_parquet(f, columns=["fixture_id", "player_id", "pos"]) for f in files]
        lp = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["fixture_id", "player_id", "pos"])
        lp["fixture_id"] = pd.to_numeric(lp["fixture_id"], errors="coerce")
        lp["player_id"] = pd.to_numeric(lp["player_id"], errors="coerce")
        lp["pos"] = lp["pos"].map(_pos_letter)
        lp = lp[lp["fixture_id"].notna() & lp["player_id"].notna() & (lp["pos"] != "")]
        _LINEUP_CACHE.clear()
        _LINEUP_CACHE.update(key=key, frame=lp.drop_duplicates(["fixture_id", "player_id"], keep="last"))
    return _LINEUP_CACHE["frame"]


def _pos_letter(v) -> str:
    s = "" if v is None or (isinstance(v, float) and v != v) else str(v).strip().upper()
    return s[:1] if s[:1] in ("G", "D", "M", "F") else ""


def _prematch_position(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(position letter or '', source) per row from pre-match evidence: the row's lineup_position
    (serving: the live announced lineup), else the collected announced lineup of the row's fixture,
    else the position of the player's latest EARLIER row in this frame (rows are ordered by player,
    date, fixture). One definition for training and serving."""
    idx = df.index
    lineup = (df["lineup_position"].map(_pos_letter) if "lineup_position" in df.columns
              else pd.Series("", index=idx))
    lp = collected_lineup_positions()
    if len(lp) and "fixture_id" in df.columns:
        k = pd.DataFrame({"fixture_id": pd.to_numeric(df["fixture_id"], errors="coerce").to_numpy(),
                          "player_id": pd.to_numeric(df["player_id"], errors="coerce").to_numpy()})
        got = k.merge(lp, on=["fixture_id", "player_id"], how="left")["pos"].fillna("").to_numpy()
        lineup = lineup.where(lineup != "", pd.Series(got, index=idx))
    own = df["position"].map(_pos_letter) if "position" in df.columns else pd.Series("", index=idx)
    if UPCOMING_COL in df.columns:                      # an upcoming row has no match position yet
        own = own.where(~df[UPCOMING_COL].astype(bool), "")
    prev = own.replace("", np.nan).groupby(df["player_id"]).shift(1)
    prev = prev.groupby(df["player_id"]).ffill().fillna("")
    pos = lineup.where(lineup != "", prev)
    src = pd.Series(np.where(lineup != "", POSITION_FROM_LINEUP,
                             np.where(prev != "", POSITION_FROM_LATEST, POSITION_NONE)), index=idx)
    return pos.astype(str), src


def build_features(match_rows: list[dict], n: int = None) -> pd.DataFrame:
    """
    Build ML training data from per-match player stats.

    Features: rolling averages of player's PREVIOUS N matches (shift=1, zero leakage).
    Targets:  actual outcome in THIS match (not the same as features).

    Also computes opponent defensive rolling features per fixture.
    """
    if match_rows is None or len(match_rows) == 0:
        return pd.DataFrame()

    n = n or config.ROLLING_N

    # A list of row dicts, or a DataFrame (pipeline.merge_cup_rows passes the history frame: a
    # 900k-row list of dicts would cost several GB).
    df = match_rows.copy() if isinstance(match_rows, pd.DataFrame) else pd.DataFrame(match_rows)
    df["date"] = pd.to_datetime(df["date"])
    # A row without a real player id has no player to build a history for: player_id 0 pooled
    # 153 different players into ONE rolling history (props_truth 2026-11-02). Dropped, logged.
    _pid = pd.to_numeric(df["player_id"], errors="coerce")
    if (~(_pid > 0)).any():
        print(f"[features] {int((~(_pid > 0)).sum()):,} row(s) without a real player_id dropped")
        df = df[(_pid > 0).to_numpy()]
    # Stable order with a tie-break (two fixtures of one player on one day; owner 2026-11-09): the
    # training frame and the serving rows (build_upcoming_features) order history identically.
    _order = ["player_id", "date"] + (["fixture_id"] if "fixture_id" in df.columns else [])
    df = df.sort_values(_order, kind="mergesort").reset_index(drop=True)
    if UPCOMING_COL not in df.columns:
        df[UPCOMING_COL] = False
    df[UPCOMING_COL] = df[UPCOMING_COL].fillna(False).astype(bool)
    # Before the minutes filter: the rule needs both teams' rows of a fixture.
    df = _blank_unrecorded_corners(df)
    # Every match each player played (this frame before the minutes filter + the collected cup rows):
    # rest_days is measured on the complete history, not on the training rows.
    _played = played_days(df)

    # Drop bench/DNP rows (< 10 min)
    # (an UPCOMING row - serving, no result yet - has no minutes; it is kept: build_upcoming_features)
    df = df[(df["minutes"] >= 10) | df[UPCOMING_COL]].copy()
    # Minutes above a match's maximum are a provider error: the raw value stays, every
    # minutes-derived feature reads it as missing (_minutes_ok).
    df["_minutes_ok"] = minutes_ok(df["minutes"])

    # ── Per-player rolling features — shift(1) prevents any leakage ──────────
    grp = df.groupby("player_id", group_keys=False)
    for feat, raw in (("goals_pg", "goals"), ("assists_pg", "assists"), ("shots_pg", "shots_total"),
                      ("sot_pg", "shots_on_target"), ("cards_pg", "yellow_cards"),
                      ("minutes_pg", "_minutes_ok"), ("key_passes_pg", "key_passes")):
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
    mins = _num(df, "_minutes_ok")
    df["_shot_acc"] = _ratio(_num(df, "shots_on_target"), _num(df, "shots_total"))
    df["_kp90"] = _ratio(_num(df, "key_passes"), mins) * 90
    df["_gi90"] = _ratio(_num(df, "goals") + _num(df, "assists"), mins) * 90
    df["_box90"] = _ratio(_num(df, "shots_total") + _num(df, "duels_won"), mins) * 90
    df["_aerial"] = _ratio(_num(df, "duels_won"), _num(df, "duels_total"))
    df["_duel90"] = _ratio(_num(df, "duels_total"), mins) * 90
    df["_fd90"] = _ratio(_num(df, "fouls_drawn"), mins) * 90
    df["_fp90"] = _ratio(_num(df, "fouls_committed"), mins) * 90
    _fc, _fd = _num(df, "fouls_committed"), _num(df, "fouls_drawn")
    df["_fcr"] = _ratio(_fc, _fc + _fd)      # exact share (was + 0.01 typed); no foul -> undefined
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

    # ── Rest days: days since the player's previous match in ANY competition we hold (league rows,
    # rows under 10 minutes, collected cup rows: rest_days_before). No earlier match -> NaN (was
    # 6.0); uncapped (was clipped at 30).
    df["rest_days"] = rest_days_before(_played, df[["player_id", "date"]]).to_numpy()

    # ── Goal-type rolling features under their own names (data_fetcher.enrich_sp_events: the goal
    # event's detail says header / free kick; NaN where the fixture's goals are unclassified). The
    # `sp_goal` / `sp_assist` columns (header OR free kick, called "set piece") feed nothing: a
    # header is not a set piece (REMOVED_SETPIECE_FEATURES).
    grp = df.groupby("player_id", group_keys=False)
    for _raw_col, _feat_name in (("headed_goal", "headed_goals_pg"), ("fk_goal", "fk_goals_pg")):
        df[_feat_name] = _roll(grp, _raw_col, n) if _raw_col in df.columns else np.nan
    df["headed_goals_share"] = _ratio(df["headed_goals_pg"], df["goals_pg"])      # unclipped

    # ── Opponent features, measured from the opponent's prior matches (team_match_table) ──────
    df["opponent"] = df["away_team"].where(df["is_home"].astype(bool), df["home_team"])
    # API-Football team ids of the row's club and opponent (team_identity: the row's own, else its
    # fixture's), resolved ONCE and kept on the row: the opponent tables and the strength lookup
    # key clubs by id across every competition.
    from . import team_identity as _TI
    df["team_id"], df["opponent_id"] = _TI.row_team_ids(df)
    # Measured club / league strength as of the match date (league_quality.attach_strength).
    df = _LQ.attach_strength(df)
    # Set-piece roles from mapped Understat shots, as of the match date (setpiece/features.py).
    _sp = _SP.asof(_SP.load(history_dates=df[["fixture_id", "date"]]), df["player_id"], df["date"])
    for c in _SP.SETPIECE_COLUMNS:
        df[c] = _sp[c].to_numpy()
    df["_tkey"], df["_okey"] = _team_keys(df)
    t = team_match_table(df)
    if not t.empty:
        roll = _team_rolling(t, n).rename(columns={"_tkey": "_okey"})
        df = df.merge(roll.drop(columns=["_opp_team_corners_pg"]), on=["fixture_id", "_okey"], how="left")
        df["_day"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.merge(_league_asof(t), on=["league", "_day"], how="left")
    _ensure(df, [feat for feat, _ in _TEAM_ROLL.values()] + list(_LEAGUE_NORMS))

    # ── Previous season (player and both teams) and the longer-window shot accuracy: real values
    # under their own names, NaN below their minimum samples (one definition, also used by serving).
    df["_y"] = _season_year(df["season"]).to_numpy() if "season" in df.columns else np.nan
    ps = _prev_season_player(df, df[["player_id", "_y", "date"]])
    for c in ps.columns:
        df[c] = ps[c].to_numpy()
    if not t.empty:
        t["_y"] = _team_seasons(t).to_numpy()
        for side, k in (("own", "_tkey"), ("opp", "_okey")):
            pt = _prev_season_team(t, df[[k, "_y", "date"]].rename(columns={k: "_tkey"}), side)
            for c in pt.columns:
                df[c] = pt[c].to_numpy()
    _ensure(df, PREV_SEASON_COLUMNS)
    df["shot_accuracy_l10"] = shot_accuracy_before(df).to_numpy()

    # ── Position encoding: PRE-MATCH evidence only (owner 2026-11-09) ─────────
    # The match-statistics position is published with the match (after kickoff), so it is not an
    # input. The row's position is the ANNOUNCED lineup position for this fixture (the caller's
    # lineup_position, else the collected API-Football lineups), else the player's latest EARLIER
    # match position; position_source says which. No real source -> the flags are missing (never
    # 0/0/0, which reads as a goalkeeper).
    df["position_prematch"], df["position_source"] = _prematch_position(df)
    _pp = df["position_prematch"]
    _known = _pp != ""
    for _flag, _letter in (("pos_forward", "F"), ("pos_midfielder", "M"), ("pos_defender", "D")):
        df[_flag] = (_pp == _letter).astype(float).where(_known)

    _ensure(df, _COMPOSITE_INPUTS)
    df = _add_composites(df)

    # ── Target variables — actual outcome in THIS match ───────────────────────
    # A count the provider did not send is NaN (data_fetcher, owner 2026-11-04), and so is its
    # label: `(NaN >= 1).astype(int)` read a missing count as "did not score" - the 'or 0' bug
    # one step later. model.train drops unlabelled rows; a recorded count labels as before.
    df["target_goals"]   = _label(df, "goals", 1)
    df["target_sot"]     = _label(df, "shots_on_target", 1)
    df["target_cards"]   = _label(df, "yellow_cards", 1)
    df["target_assists"] = _label(df, "assists", 1)
    df["target_goals2"]  = _label(df, "goals", 2)
    df["target_sot2"]    = _label(df, "shots_on_target", 2)
    df["target_sot3"]    = _label(df, "shots_on_target", 3)
    df["target_sot4"]    = _label(df, "shots_on_target", 4)

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


# ── Serving: the TRAINING code on history + the upcoming rows (owner 2026-11-09) ───────────────

#: fields a serving row carries (predict.run_player_predictions); everything else is computed
UPCOMING_FIELDS = ("player_id", "player_name", "fixture_id", "date", "league", "team", "home_team",
                   "away_team", "is_home", "referee", "lineup_position")


def upcoming_rows(upcoming: list[dict]) -> pd.DataFrame:
    """Serving rows in the training row shape: identity and fixture context only, no result (every
    count missing), marked UPCOMING_COL. `fixture_date` / `fixture_league` are accepted for `date` /
    `league`. The season is the registry rule (_resolve_seasons), the one the collector's history
    rows carry. A row without a real player id or fixture date is dropped and counted."""
    rows = pd.DataFrame(upcoming)
    if rows.empty:
        return rows
    if "date" not in rows.columns and "fixture_date" in rows.columns:
        rows["date"] = rows["fixture_date"]
    if "league" not in rows.columns and "fixture_league" in rows.columns:
        rows["league"] = rows["fixture_league"]
    for alias, col in (("fixture_date", "date"), ("fixture_league", "league")):
        if alias in rows.columns:
            rows[col] = rows[alias].where(rows[alias].notna(), rows[col])
    rows = rows[[c for c in UPCOMING_FIELDS if c in rows.columns]].copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    ok = (pd.to_numeric(rows["player_id"], errors="coerce") > 0) & rows["date"].notna()
    if (~ok).any():
        print(f"[features] upcoming: {int((~ok).sum())} row(s) without a player id or fixture date dropped")
    rows = rows[ok.to_numpy()].copy()
    rows["player_id"] = pd.to_numeric(rows["player_id"]).astype("int64")
    rows["season"] = _resolve_seasons(rows)
    rows[UPCOMING_COL] = True
    return rows


def build_upcoming_features(upcoming: list[dict], history: pd.DataFrame, n: int = None) -> pd.DataFrame:
    """Feature rows for upcoming player-fixtures, built by build_features ITSELF on the history plus
    the upcoming rows, so every input is the training definition, value for value (shift-based
    rolling, strictly-earlier team / league / referee tables, the pre-match position). The rows
    returned are the upcoming ones; `position` is their pre-match position (position_source says
    from where). A player with no earlier row has no features (counted, as build_features drops
    first appearances)."""
    up = upcoming_rows(upcoming)
    if up.empty or history is None or history.empty:
        return pd.DataFrame()
    n = n or config.ROLLING_N
    h = history.copy()
    h[UPCOMING_COL] = False
    if "fixture_id" in h.columns and "fixture_id" in up.columns:
        h = h[~pd.to_numeric(h["fixture_id"], errors="coerce").isin(pd.to_numeric(up["fixture_id"], errors="coerce"))]
    frame = build_features(pd.concat([h, up], ignore_index=True, sort=False), n)
    if frame.empty or UPCOMING_COL not in frame.columns:
        return pd.DataFrame()
    out = frame[frame[UPCOMING_COL].astype(bool)].copy()
    n_none = len(up) - len(out)
    if n_none:
        print(f"[features] upcoming: {n_none} player(s) without an earlier match: no features")
    out["position"] = out["position_prematch"]
    out["n_games"] = np.minimum(pd.to_numeric(out["n_prev_games"], errors="coerce"), n)
    out["data_source"] = "match_level"
    return out.reset_index(drop=True)


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
