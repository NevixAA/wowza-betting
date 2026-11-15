"""FANTASY tips family — FPL-style expected-points projections from the prop models.

Premier League only (v1). Fantasy is a PREDICTION CONTEST (no bookmaker / vig / edge),
so the prop model's accuracy (AUC 0.69-0.86) converts directly to value — unlike prop
BETTING, which is efficiently priced (see memory project_prop_edge_backtest).

This family is fully separate from the SNIPER/MARKSMAN betting family: no odds, no edge,
no tiers — just the model's raw probabilities ranked into fantasy points.

v1 uses each player's LATEST form row (parquet). When PL resumes (~Aug) the upcoming-
fixture / opponent-adjusted path can be layered on via build_upcoming_features.
Enhancement TODO: clean-sheet points for DEF/GK from the team O/U model (v1 = attack only).
"""
from __future__ import annotations
import logging
import unicodedata
from pathlib import Path
import os as _os
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

from player_model import config
# NOT IMPORTED AT MODULE LEVEL ON PURPOSE. player_model.model pulls in scikit-learn (and
# lightgbm once a model is unpickled), and the Streamlit dashboard imports THIS module only for
# `rank_and_captains` / `MIN_CAPTAIN_START`, which are pure pandas. Loading the model stack for
# that cost the deployment tens of megabytes of module code it never used, on a host capped at
# about 1 GB that had already gone over. The import now happens inside the one function that
# scores players, so the pipeline pays for it and the viewer does not.

_V9_SRC: dict = {}       # v9 src modules privately loaded under a foreign `src` (see below)


def _v9_src(name: str):
    """v9's src.<name>, even when the importer has its OWN `src` package (wowzaV9-Pro imports
    player_model from _v9/, where `src` is Pro's). Same approach as config._season_resolver: a
    foreign `src` and sys.path are restored afterwards."""
    import importlib
    import sys
    root = Path(__file__).resolve().parents[1]
    cur = sys.modules.get("src")
    foreign = cur is not None and (root / "src").resolve() not in {
        Path(p).resolve() for p in getattr(cur, "__path__", [])}
    if not foreign:
        if str(root) not in sys.path:
            sys.path.append(str(root))
        return importlib.import_module(f"src.{name}")
    if name in _V9_SRC:
        return _V9_SRC[name]
    saved_path = list(sys.path)
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "src" or k.startswith("src.")}
    sys.path.insert(0, str(root))
    try:
        _V9_SRC[name] = importlib.import_module(f"src.{name}")
        return _V9_SRC[name]
    finally:
        sys.path[:] = saved_path
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            sys.modules.pop(k)
        sys.modules.update(saved)


def _registry():
    """src.league_registry (config/leagues.json) - the source of every league fact used here."""
    return _v9_src("league_registry")


# The one league FPL scores, from config/leagues.json (fpl: true) - never typed here. None when the
# registry flags no FPL league: every builder below then returns empty and logs why.
FANTASY_LEAGUE = _registry().fpl_league()
if FANTASY_LEAGUE is None:
    log.error("fantasy: config/leagues.json flags no fpl league - fantasy projections disabled")


def _fantasy_league_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of `df` whose league resolves (registry canonical_name) to FANTASY_LEAGUE. Matched per
    distinct spelling through the registry, never by a raw == on the history's label."""
    if FANTASY_LEAGUE is None or df.empty or "league" not in df.columns:
        return df.iloc[0:0]
    reg = _registry()
    hit = [lg for lg in df["league"].dropna().unique() if reg.canonical_name(lg) == FANTASY_LEAGUE]
    return df[df["league"].isin(hit)]


def _fantasy_league_spellings() -> list[str]:
    """FANTASY_LEAGUE plus its registered aliases (for pushdown filters on a parquet dataset)."""
    if FANTASY_LEAGUE is None:
        return []
    return [FANTASY_LEAGUE] + [a for a, c in _registry().alias_map().items() if c == FANTASY_LEAGUE]


def _fantasy_league_id() -> int | None:
    """API-Football id of FANTASY_LEAGUE from the registry; None (logged) when it has none."""
    e = _registry().entry(FANTASY_LEAGUE) if FANTASY_LEAGUE else None
    lid = (e or {}).get("api_football_id")
    if lid is None:
        log.warning("fantasy: no api_football_id registered for %r - skipped", FANTASY_LEAGUE)
    return lid


GOAL_PTS = {"F": 4.0, "M": 5.0, "D": 6.0, "G": 10.0}  # legacy parquet-position goal pts (GKP 10, FPL)
# FPL goal points keyed by FPL position codes (GKP/DEF/MID/FWD)
# GKP 10, from FPL's own game_config.scoring.goals_scored (was 6 - Data Truth / R2-D fantasy bug #2).
FPL_GOAL_PTS = {"FWD": 4.0, "MID": 5.0, "DEF": 6.0, "GKP": 10.0}
# Clean-sheet points by position (DEF/GK 4, MID 1, FWD 0)
FPL_CS_PTS   = {"DEF": 4.0, "GKP": 4.0, "MID": 1.0, "FWD": 0.0}
# Defensive-contribution thresholds = FPL's REAL ones, on FPL's OWN action count
# (elements.defensive_contribution_per_90). Derived 2026-11-01 from FPL's live scoring for GW6-8
# (2,004 player-gameweeks): DEF scored 2 pts from exactly 10 actions (never at 9), MID from 12
# (never at 11), FWD never below 12. The old 5 / 4.5 / 4 were applied to our own different count
# (tackles+interceptions+blocks+duels_won) and gave FWDs more DC pts than DEFs (R2-D fantasy bug #1).
# Goalkeepers have no DC threshold: FPL awards them no defensive-contribution points (was a
# sentinel 999). _DC_NONE lists the positions that score 0 DC points by rule.
_DC_THRESH = {"DEF": 10.0, "MID": 12.0, "FWD": 12.0}
_DC_NONE = ("GKP",)
_POS_ALIAS = {"F": "FWD", "M": "MID", "D": "DEF", "G": "GKP",
              "FWD": "FWD", "MID": "MID", "DEF": "DEF", "GKP": "GKP", "GK": "GKP"}
_PARQUET = Path(__file__).resolve().parents[1] / "player_history.parquet"

# ── Availability-aware ranking and captaincy ─────────────────────────────────
# A player who will not play is worth ZERO points, however good he is. Ranking by CONDITIONAL
# points -- what he scores IF he plays -- therefore puts men who cannot kick a ball at the top
# of the board. Measured on the 2026-09-27 projection: 8 of the top 20 could not play, including
# Ekitiké at #2 and Romero at #3, both with p_start 0.00; and 65 of 170 players with p_start=0
# outranked someone fully available.
#
# The correct number already existed and nothing used it. `xpts_rot` is exactly
# fantasy_pts x p_start -- the UNCONDITIONAL expectation -- so this is a change of which column
# decides the order, not new modelling.
#
# Conditional points are KEPT, not deleted. They are the right number for "how good is he when
# he plays", which is a real question; they are simply the wrong number for "who should I pick".

#: p_start at or above which a player is treated as a dependable starter.
NAILED_START = 0.85
#: p_start below which a player is a rotation risk and must never be a captain suggestion.
MIN_CAPTAIN_START = 0.60
_OUT_STATUSES = ("injured", "unavailable", "suspended")


def start_confidence(p_start, availability=None, chance_of_playing=None) -> str:
    """An interpretable label, not an invented 0-100 score.

    The brief is explicit that a number like "AI confidence 92%" is worthless unless it is
    calibrated, so this says only what the data supports: a status, a start probability, and
    FPL's own published chance of playing.
    """
    av = str(availability or "").lower()
    if av in _OUT_STATUSES:
        return "Unavailable"
    # Unknown P(start) is reported as unknown - it used to read as 0 and label the player
    # 'Unavailable', which is a claim the data does not make.
    p = float(p_start) if p_start is not None and p_start == p_start else None
    if p is None:
        return "Start probability unknown"
    if p <= 0:
        return "Unavailable"
    if av == "doubtful":
        return "Doubtful"
    if p < MIN_CAPTAIN_START:
        return "Rotation risk"
    if p < NAILED_START:
        return "Likely starter"
    return "Nailed"


def rank_and_captains(df, base_col: str = "fantasy_pts", n_captains: int = 3):
    """Sort by UNCONDITIONAL expected points and pick captains who will actually start.

    Adds/overwrites: xpts_uncond, start_confidence, overall_rank, pos_rank, captain_pick.
    `base_col` is the conditional per-game number to discount, so the dashboard can rank a
    fixture-adjusted or multi-gameweek view through the same rule.
    """
    import numpy as _np
    import pandas as _pd
    d = df.copy()
    nan = _pd.Series(_np.nan, index=d.index)
    # Unknown points or P(start) stay NaN (ranked last, labelled) - never 0, which ranked an
    # unscored player as a known blank and labelled him 'Unavailable'.
    base = _pd.to_numeric(d[base_col], errors="coerce") if base_col in d.columns else nan
    p = (_pd.to_numeric(d["p_start"], errors="coerce").clip(0, 1)
         if "p_start" in d.columns else nan)
    av = d.get("availability", _pd.Series("", index=d.index)).astype(str).str.lower()

    # Unconditional = conditional x P(start), and hard zero for anyone ruled out, so an
    # "available" flag and a stale non-zero p_start cannot resurrect a ruled-out player.
    d["xpts_uncond"] = (base * p).where(~av.isin(_OUT_STATUSES), 0.0).round(3)
    d["start_confidence"] = [
        start_confidence(pp, aa, cc)
        for pp, aa, cc in zip(p, av, d.get("chance_of_playing", _pd.Series(None, index=d.index)))
    ]

    d = d.sort_values(["xpts_uncond", base_col], ascending=False,
                      na_position="last").reset_index(drop=True)
    d["overall_rank"] = _np.arange(1, len(d) + 1)
    d["pos_rank"] = (d.groupby("position")["xpts_uncond"]
                     .rank(ascending=False, method="first", na_option="bottom").astype(int))

    # CAPTAINCY IS ITS OWN DECISION, not "the top of the list". Excluding only `injured` let
    # doubtful players through -- on 2026-09-27 both João Pedro and Pedro Porro were suggested
    # as captain at a 75% chance of playing. A captain is doubled, so a blank costs twice.
    # Recomputed on the SORTED frame: `av`/`p` above carry the pre-sort index, and masking the
    # re-indexed `d` with them applied row i's flags to whoever now sits at position i.
    av_s = d.get("availability", _pd.Series("", index=d.index)).astype(str).str.lower()
    p_s = (_pd.to_numeric(d["p_start"], errors="coerce")
           if "p_start" in d.columns else _pd.Series(_np.nan, index=d.index))
    eligible = ((~av_s.isin(_OUT_STATUSES)) & (av_s != "doubtful") & (p_s >= MIN_CAPTAIN_START)
                & d["xpts_uncond"].notna())
    d["captain_pick"] = False
    if eligible.any():
        d.loc[d[eligible].head(n_captains).index, "captain_pick"] = True
    return d
_OUT = Path(__file__).resolve().parents[1] / "output" / "fantasy_tips.csv"


def _ascii(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))


def _recent_def_actions(pl_all: pd.DataFrame, n: int = 10) -> dict:
    """Recent mean defensive actions/game per player (tackles+interceptions+blocks+duels_won).
    Approximates FPL CBIT — source lacks clearances/ball-recoveries (flagged at call site)."""
    cols = [c for c in ("tackles_total", "interceptions", "blocks", "duels_won") if c in pl_all.columns]
    if not cols:
        return {}
    recent = pl_all.sort_values("date").groupby("player_id").tail(n)
    da = recent.groupby("player_id")[cols].mean().sum(axis=1)
    return {k: float(v) for k, v in da.items()}


def _cs_prob(fdr_list: list) -> float:
    """P(clean sheet) from the team's next fixtures' official FDR, via a Poisson on opponent
    xG (harder fixture -> higher opp xG -> lower CS prob). NaN when no fixtures (off-season)."""
    import math
    if not fdr_list:
        return float("nan")
    # A fixture with no FDR is skipped, not read as a neutral 3.
    ps = [math.exp(-(0.5 + 0.42 * float(f["fdr"]))) for f in fdr_list if f.get("fdr") is not None]
    return float(np.mean(ps)) if ps else float("nan")


def _poisson_dc_pts(mean_actions: float, threshold: float) -> float:
    """Expected defensive-contribution pts = 2 · P(actions >= threshold), actions ~ Poisson(mean).
    Better than a linear clip: a player averaging near the bar hits it ~half the time. (APPROX —
    our action count lacks clearances/ball-recoveries, so thresholds are pre-lowered.)"""
    import math
    if mean_actions <= 0:
        return 0.0
    k = int(math.ceil(threshold))
    term = math.exp(-mean_actions)
    cdf = term
    for i in range(1, k):
        term *= mean_actions / i
        cdf += term
    return round(2.0 * max(0.0, min(1.0, 1.0 - cdf)), 3)


def _expected_bonus(p_goal: float, p_assist: float, p_sot2: float,
                    def_pg: float, cs: float) -> float:
    """Rough expected FPL BONUS (0-3). BPS is dominated by goals/assists, then defensive volume
    + clean sheets. We can't see rivals' BPS to rank the top-3, so this is a calibrated-ish
    heuristic on the drivers we do model. Any unknown driver -> NaN (never read as 0)."""
    if any(v is None or v != v for v in (p_goal, p_assist, p_sot2, def_pg, cs)):
        return float("nan")
    b = 1.5 * p_goal + 0.9 * p_assist + 0.3 * p_sot2 + 0.8 * min(def_pg / 12.0, 1.0) + 0.25 * cs
    b *= 0.66   # CALIBRATED to actual bonus mean (0.213) vs uncalibrated (0.321) — backtest
    return round(float(min(max(b, 0.0), 3.0)), 3)


def _pen_taker_ids(pl_all: pd.DataFrame, min_pens: float = 2.0) -> set:
    """player_ids with real penalty history (won+scored over their record) — likely pen takers.
    A pen taker's goal expectation is materially higher; flagged for display."""
    cols = [c for c in ("penalty_scored", "penalty_won") if c in pl_all.columns]
    if not cols:
        return set()
    pens = pl_all.groupby("player_id")[cols].sum().sum(axis=1)
    return set(pens[pens >= min_pens].index)


def _resolve_club(name, candidates) -> str | None:
    """A club name onto one of `candidates` (club names of the SAME league) via the canonical
    league-scoped resolver src.team_names.resolve. None when unmatched or ambiguous."""
    resolve = _v9_src("team_names").resolve
    return resolve(str(name), [str(c) for c in candidates]) if name == name and name else None


#: API-Football finished fixtures with their 90-minute score (scripts/collect_competition_fixtures).
_FIXTURE_RESULTS = Path(__file__).resolve().parents[1] / "output" / "strength" / "competition_fixtures.parquet"
_GC_COLS = ["team", "fixture_id", "date", "gc"]


def _team_goals_conceded(pl_all: pd.DataFrame, results: pd.DataFrame | None = None) -> pd.DataFrame:
    """TEAM goals conceded per fixture: [team, fixture_id, date, gc], from the FIXTURE SCORE.

    The old map read the history's per-player `goals_conceded`. API-Football fills that for
    goalkeepers only - every outfielder row carries 0 (measured 2026-11-02 on the 45,230 Premier
    League rows: D/M/F mean exactly 0.000) - and the map took the FIRST row of each team-fixture,
    almost always an outfielder's 0. A club's mean came out ~0.0-0.1, below the 0.3 plausibility
    floor, so the clean-sheet model NEVER fitted: 19 of 20 clubs fell back to FDR alone. Even the
    goalkeeper rows are not a team figure (22 of 3,085 team-fixtures disagree with the real score,
    e.g. Arsenal 2026-09-19: 0 on the keeper row, 3 conceded).

    The team figure is the opponent's 90-minute goals in the API-Football fixture result, joined
    on the API-Football fixture_id; the history's club spelling is resolved onto that fixture's
    two clubs with src.team_names.resolve. A fixture with no stored result, or whose clubs do not
    resolve, has no figure (dropped, counted in the log), never 0."""
    if results is None:
        try:
            results = pd.read_parquet(_FIXTURE_RESULTS, columns=["fixture_id", "home_name",
                                                                 "away_name", "ft_home", "ft_away"])
        except Exception as e:                                                # noqa: BLE001
            log.warning("fantasy: no fixture results (%s) - team goals conceded unknown", e)
            return pd.DataFrame(columns=_GC_COLS)
    if pl_all.empty or not {"team", "fixture_id"} <= set(pl_all.columns) or results.empty:
        return pd.DataFrame(columns=_GC_COLS)
    tf = (pl_all.dropna(subset=["team", "fixture_id"])
                .groupby(["team", "fixture_id"], as_index=False).agg(date=("date", "first")))
    tf["fixture_id"] = pd.to_numeric(tf["fixture_id"], errors="coerce")
    res = (results.assign(fixture_id=pd.to_numeric(results["fixture_id"], errors="coerce"))
                  .drop_duplicates("fixture_id").set_index("fixture_id"))
    rows, unresolved, no_result = [], 0, 0
    for t, fid, d in tf[["team", "fixture_id", "date"]].itertuples(index=False):
        if fid != fid or fid not in res.index:
            no_result += 1
            continue
        r = res.loc[fid]
        side = _resolve_club(t, [r["home_name"], r["away_name"]])
        if side == r["home_name"]:
            conc = r["ft_away"]
        elif side == r["away_name"]:
            conc = r["ft_home"]
        else:
            unresolved += 1
            continue
        conc = pd.to_numeric(conc, errors="coerce")
        if conc != conc:
            unresolved += 1
            continue
        rows.append({"team": t, "fixture_id": fid, "date": d, "gc": float(conc)})
    if no_result or unresolved:
        log.info("fantasy: team goals conceded unknown for %d team-fixture(s) with no stored result "
                 "and %d whose club did not resolve onto the fixture", no_result, unresolved)
    return pd.DataFrame(rows, columns=_GC_COLS)


def _team_conceded_map(pl_all: pd.DataFrame, n: int = 8, results: pd.DataFrame | None = None) -> dict:
    """{history team name: recent mean TEAM goals-conceded/game} — the team clean-sheet model
    input: _team_goals_conceded per fixture, then the team's last-n mean. Keyed by the history's
    own spelling; callers resolve other sources' names onto these keys with _resolve_club."""
    fx = _team_goals_conceded(pl_all, results)
    if fx.empty:
        return {}
    fx = fx.sort_values(["team", "date"])
    m = fx.groupby("team").tail(n).groupby("team")["gc"].mean()
    return {str(k): float(v) for k, v in m.items() if pd.notna(v)}


#: profile columns derived from age / height; unknown whenever their source is a placeholder.
_AGE_DERIVED = ("age_peak_delta",)
_HEIGHT_DERIVED = ("height_aerial_interaction", "opp_def_mean_height", "height_diff_vs_opp_def",
                   "height_advantage_score")


def _mask_profile_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """A copy of `df` with the history's profile PLACEHOLDERS read as missing: age 25.0, height
    180 cm and weight 75 kg (the values player_model.feature_engineering._PROFILE_PLACEHOLDERS
    records the old build wrote into every row) become NaN, and so does every column derived from
    a masked age / height. A real 25-year-old or 180 cm player is lost to NaN until the profile
    collector stores real values - the honest side of that trade (same rule as the feature build).

    NOT applied to the prop-model input: the serving payloads were fitted on the constant, their
    gradient-boost member drops all seven columns as constant (_UsableColumns) and their logistic
    member standardises the constant to 0; a NaN there would instead knock every row out of the
    logistic member and change every probability - a serving change only an approved props
    retrain may make (V3 note). Fantasy's own reads are masked."""
    from player_model.feature_engineering import _PROFILE_PLACEHOLDERS
    d = df.copy()
    for col, ph in _PROFILE_PLACEHOLDERS.items():
        if col not in d.columns:
            continue
        v = pd.to_numeric(d[col], errors="coerce")
        bad = v.isna() | (v == ph)
        d[col] = v.where(~bad)
        derived = _AGE_DERIVED if col == "age" else _HEIGHT_DERIVED if col == "height_cm" else ()
        for c in derived:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce").where(~bad)
    return d


def fpl_ids(df: pd.DataFrame) -> pd.Series:
    """FPL element id per row ('fpl_id' or 'element'), NaN where the row carries none. Every
    squad / projection / ownership join in the fantasy modules is on this id, never on a name."""
    for c in ("fpl_id", "element"):
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def fpl_crosswalk(players: pd.DataFrame, fpl: pd.DataFrame, norm=None,
                  crosswalk: pd.DataFrame | None = None) -> pd.Series:
    """FPL element id for each API-Football player row in `players`, NaN when not established.

    THE ONLY BRIDGE between the two sources, and it is an ID JOIN: players.player_id (API-Football)
    -> the stored crosswalk output/fpl_af_crosswalk.csv (player_model.fantasy_crosswalk: name +
    club + date of birth, ambiguity-refusing) -> FPL element id, kept only when that element is in
    the live FPL set `fpl`. The old bridge matched NAMES (FPL full name, then web_name): 101 of the
    318 current-season Premier League players, and it paired different people that share a short
    name - Leicester's Tetê with Kenny Tete, Leeds' Rodrigo with Rodri, Bournemouth's goalkeeper
    Neto with Pedro Neto (Data Truth round 2, 2026-11-02). `norm` is accepted and ignored (old
    signature). Downstream joins use the returned id, never a name.
    """
    from player_model import fantasy_crosswalk as _CW
    out = pd.Series(np.nan, index=players.index, dtype="float64")
    if fpl.empty or "fpl_id" not in fpl.columns or "player_id" not in players.columns:
        return out
    ids = _CW.af_to_fpl(players["player_id"], fpl["fpl_id"], crosswalk)
    n_miss = int(ids.isna().sum())
    if n_miss:
        log.info("fantasy: %d of %d player(s) have no established FPL element id (crosswalk)",
                 n_miss, len(ids))
    return ids


def build_fantasy_projections(parquet_path: Path | None = None, min_minutes: float = 45.0,
                              use_fpl: bool = True, next_n: int = 5) -> pd.DataFrame:
    """Per-player FPL expected-points projections.

    fantasy_pts = appearance + attack (P(goal)*pos_pts + P(assist)*3 + P(sot2)*1 bonus proxy)
                  + defensive-contribution (approx) + clean-sheet (DEF/GK/MID, from FDR).

    When use_fpl, joins to the LIVE official FPL squad (transfer-accurate team/position/price)
    and attaches availability/injury flags. Without the FPL API there is no FPL defensive-
    contribution figure, so no outfield player gets a complete projection (they are dropped,
    logged) rather than a board of guesses. A player missing any required input is dropped, logged with
    the reason; one with an unknown P(start) is kept but ranked last and never captained.
    """
    df = pd.read_parquet(parquet_path or _PARQUET)
    pl_all = _fantasy_league_rows(df).copy()
    if pl_all.empty:
        return pd.DataFrame()
    pl = pl_all.sort_values("date").groupby("player_id", as_index=False).tail(1)
    if "minutes_pg" not in pl.columns:
        log.error("fantasy: history has no minutes_pg column - no projections")
        return pd.DataFrame()
    mins = pd.to_numeric(pl["minutes_pg"], errors="coerce")
    if mins.isna().any():
        # Unknown minutes are not 60 (that earned the 2-pt appearance): the player is skipped.
        log.info("fantasy: %d player(s) skipped - minutes_pg unknown", int(mins.isna().sum()))
    pl = pl[mins >= min_minutes].copy()
    if pl.empty:
        return pd.DataFrame()

    from player_model.model import load_model, predict_proba
    for market, col in [("goals", "p_goal"), ("assists", "p_assist"), ("sot2", "p_sot2")]:
        payload = load_model(market)
        if payload is None:
            # A missing model is not a 0.0 probability (that silently erased every attacking
            # point). No projections are published; the previous CSV is left untouched.
            log.error("fantasy: prop model %r not found - fantasy projections not built", market)
            return pd.DataFrame()
        pl[col] = predict_proba(pl, payload).values
    # Scored on the frame the payload was fitted on; from here on the profile placeholders are
    # missing values (see _mask_profile_placeholders for why the model input is left as is).
    pl = _mask_profile_placeholders(pl)

    m = pd.to_numeric(pl["minutes_pg"], errors="coerce")
    pl["appearance"] = np.where(m >= 60, 2.0, 1.0)

    # No record -> NaN (not 0 defensive actions); the bonus driver then reads as unknown.
    da_map = _recent_def_actions(pl_all)
    pl["def_actions_pg"] = pl["player_id"].map(da_map) if da_map else np.nan

    # ── LIVE FPL layer: transfer-accurate team/position/price + availability + FDR ──
    fpl, fdr_map = pd.DataFrame(), {}
    if use_fpl:
        try:
            from player_model import fpl_api
            fpl = fpl_api.players_df()
            fdr_map = fpl_api.upcoming_fdr(next_n=next_n)
            norm = fpl_api._norm
        except Exception as e:                                              # noqa: BLE001
            log.warning("fantasy: FPL API unavailable (%s: %s) - no live FPL layer",
                        type(e).__name__, e)
            fpl, fdr_map = pd.DataFrame(), {}

    # Parquet values until FPL matches the player. Without an FPL match there is no live team,
    # position, price or DC figure - those stay missing (NaN), never a guess.
    pl["live_team"]         = pl["team"] if "team" in pl.columns else np.nan
    pl["price"]             = np.nan
    pl["availability"]      = "unknown"
    pl["injured"]           = False
    pl["doubtful"]          = False
    pl["chance_of_playing"] = np.nan
    pl["fpl_matched"]       = False
    pl["fpl_id"]            = np.nan
    pl["_fpl_pos"]          = ""

    if not fpl.empty:
        # API-Football player -> FPL element through the one ambiguity-refusing name bridge;
        # everything after this joins on the FPL element id.
        pl["fpl_id"] = fpl_crosswalk(pl, fpl, norm)
        by_id = fpl.drop_duplicates("fpl_id").set_index("fpl_id")
        for i, r in pl.iterrows():
            fid = r["fpl_id"]
            rec = by_id.loc[int(fid)] if fid == fid and int(fid) in by_id.index else None
            if rec is not None:
                pl.at[i, "live_team"]         = rec["team"]
                pl.at[i, "price"]             = rec["price"]
                pl.at[i, "availability"]      = rec["availability"]
                pl.at[i, "injured"]           = bool(rec["injured"])
                pl.at[i, "doubtful"]          = bool(rec["doubtful"])
                pl.at[i, "chance_of_playing"] = rec["chance_of_playing"]
                pl.at[i, "fpl_matched"]       = True
                pl.at[i, "_fpl_pos"]          = rec["position"]
                # FPL's own published numbers. players_df already fetches these; they were
                # being discarded. fpl_ep_next is the BENCHMARK THAT MATTERS: a projection that
                # cannot beat the expected-points figure printed free on the FPL website adds
                # nothing, however well calibrated it looks against itself. fpl_ppg is actual
                # points per game to date, i.e. the only "real score" available without pulling
                # per-gameweek history.
                pl.at[i, "fpl_ep_next"]       = rec.get("fpl_ep_next")
                pl.at[i, "fpl_ppg"]           = rec.get("fpl_ppg")
                pl.at[i, "fpl_form"]          = rec.get("fpl_form")
                pl.at[i, "total_points"]      = rec.get("total_points")
                pl.at[i, "owned_pct"]         = rec.get("owned_pct")
                pl.at[i, "fpl_dc_per90"]      = rec.get("dc_per90")
        # A player NOT in the current FPL set has LEFT the PL (e.g. Salah -> Besiktas,
        # De Bruyne -> abroad) or isn't fantasy-relevant -> DROP them. The parquet holds many
        # past-season players who've since transferred out, so a LOW match rate is EXPECTED and
        # CORRECT (those are exactly the players to drop) — so we drop unmatched whenever the
        # FPL bootstrap is valid. Only the pathological ZERO-match case (matching totally
        # broken) keeps all, to avoid an empty board falling back to a stale CSV.
        n_matched = int(pl["fpl_matched"].sum())
        if n_matched > 0:
            log.info("fantasy: %d/%d players matched current FPL squad — dropping %d departed/"
                     "non-FPL players", n_matched, len(pl), len(pl) - n_matched)
            pl = pl[pl["fpl_matched"]].copy()
        else:
            log.warning("fantasy: 0 players matched FPL (matching broken) — keeping all to avoid "
                        "empty board; departed players may show until matching is fixed")

    # scoring position: FPL position if matched, else parquet position (aliased to FPL codes).
    # An unknown position is NOT 'MID': goal/CS/DC points depend on it, so the player is dropped.
    parq_pos = pl["position"] if "position" in pl.columns else pd.Series(np.nan, index=pl.index)
    pos = pl["_fpl_pos"].where(pl["_fpl_pos"].astype(bool), parq_pos)
    pl["position"] = pos.map(lambda p: _POS_ALIAS.get(str(p)) if p == p and p is not None else None)
    no_pos = pl["position"].isna()
    if no_pos.any():
        log.info("fantasy: %d player(s) dropped - position unknown: %s", int(no_pos.sum()),
                 ", ".join(pl.loc[no_pos, "player_name"].astype(str).head(20)))
        pl = pl[~no_pos].copy()
        m = m.reindex(pl.index)
        if pl.empty:
            return pd.DataFrame()
    pl["team"] = pl["live_team"]   # `team` is now the LIVE club (fixes stale Salah etc.)

    pl["goal_pts"]   = pl["position"].map(FPL_GOAL_PTS)
    # sot2 weight CALIBRATED 0.45 (was 1.0): backtest vs actual FPL points showed 2+SOT is worth
    # only ~0.43 pts beyond goals/assists (they're correlated → +1 double-counted). scripts/fantasy_backtest.py
    pl["attack_pts"] = (pl["p_goal"] * pl["goal_pts"] + pl["p_assist"] * 3.0 + pl["p_sot2"] * 0.45).round(3)

    # Defensive-contribution pts — Poisson P(actions >= threshold), on FPL's OWN action count per
    # game (defensive_contribution_per_90 x expected minutes / 90) against FPL's real thresholds.
    # A player with no FPL figure gets NaN dc_pts (never a guess), so his total is unknown too.
    # Goalkeepers cannot score DC points under FPL's rules, so theirs is 0 whatever the count.
    # def_actions_pg (our proxy count) stays the BONUS input only, where its weights were fitted.
    thr = pl["position"].map(_DC_THRESH)
    if "fpl_dc_per90" in pl.columns:
        _dc_pg = pd.to_numeric(pl["fpl_dc_per90"], errors="coerce") * (m.reindex(pl.index) / 90.0)
    else:
        _dc_pg = pd.Series(np.nan, index=pl.index)
    pl["dc_actions_pg"] = _dc_pg.round(3)
    pl["dc_pts"] = [0.0 if p in _DC_NONE else
                    _poisson_dc_pts(float(a), float(t)) if a == a else np.nan
                    for a, t, p in zip(_dc_pg, thr, pl["position"])]

    # Clean-sheet pts — TEAM goals-conceded model blended with fixture difficulty.
    # P(CS) = exp(-lambda); lambda = 0.6·(team recent conceded/game) + 0.4·(FDR-implied opp xG).
    # Falls back to pure FDR when the club isn't matched in the parquet, then to NaN (unknown).
    conc_map = _team_conceded_map(pl_all)
    _conc_clubs = list(conc_map)
    _conc_key: dict = {}          # live (FPL) team -> history team, via the canonical resolver

    def _conc_for(team):
        if team not in _conc_key:
            _conc_key[team] = _resolve_club(team, _conc_clubs) if _conc_clubs else None
        k = _conc_key[team]
        return conc_map.get(k) if k is not None else None

    def _cs_row(r, fixtures=None):
        # NEXT fixture only by default: fantasy_pts is a NEXT-GAMEWEEK projection and is settled
        # against one gameweek, but this averaged the next 5 fixtures' FDR (R2-D fantasy bug
        # 'v1_next_gw_is_5_fixture_average'). The multi-GW total passes each fixture in turn.
        fl = fixtures if fixtures is not None else ((fdr_map.get(r["team"], []) if fdr_map else [])[:1])
        avg_fdr = np.mean([f["fdr"] for f in fl if f.get("fdr")]) if fl else np.nan
        fdr_xg  = (0.5 + 0.42 * avg_fdr) if avg_fdr == avg_fdr else np.nan   # NaN-safe
        conc = _conc_for(r["team"])
        # Reject implausible conceded rates (missing/zero-filled goals_conceded -> would
        # inflate CS). No PL side truly concedes <0.3 or >3.5/game over a window -> use FDR.
        if conc is not None and not (0.3 <= conc <= 3.5):
            conc = None
        if conc is not None and fdr_xg == fdr_xg:
            lam = 0.6 * conc + 0.4 * fdr_xg
        elif conc is not None:
            lam = conc
        elif fdr_xg == fdr_xg:
            lam = fdr_xg
        else:
            return np.nan
        return float(np.exp(-lam))

    # P(clean sheet) kept as a PROBABILITY: _expected_bonus expects one, and was being handed the
    # clean-sheet POINTS (0-4) - a DEF/GKP's bonus driver inflated up to 4x (R2-D fantasy bug
    # 'bonus_cs_points_as_probability'). No CS estimate -> NaN CS points (was 0), except where
    # FPL awards none for the position (FWD), which is 0 by rule.
    pl["cs_prob"] = pl.apply(_cs_row, axis=1)
    _cs_w = pl["position"].map(FPL_CS_PTS)
    pl["cs_pts"] = (pl["cs_prob"] * _cs_w).where(_cs_w != 0, 0.0).round(3)

    # Expected bonus (from BPS drivers); any unknown driver -> NaN bonus.
    pl["bonus_pts"] = [_expected_bonus(g, a, s, d, c) for g, a, s, d, c in
                       zip(pl["p_goal"], pl["p_assist"], pl["p_sot2"], pl["def_actions_pg"],
                           pl["cs_prob"])]

    pl["fantasy_pts"] = (pl["appearance"] + pl["attack_pts"] + pl["dc_pts"]
                         + pl["cs_pts"] + pl["bonus_pts"]).round(3)
    # A player missing a required input gets no projection and is not recommended (logged with
    # the reason) - the old 0-fills made an unknown look like a known blank.
    _need = {"dc_pts": "no FPL defensive_contribution_per_90", "cs_pts": "no clean-sheet estimate",
             "bonus_pts": "a bonus driver unknown (defensive actions / CS probability)"}
    unscored = pl["fantasy_pts"].isna()
    if unscored.any():
        for c, why in _need.items():
            k = int((unscored & pl[c].isna()).sum())
            if k:
                log.info("fantasy: %d player(s) not projected - %s", k, why)
        pl = pl[~unscored].copy()
        if pl.empty:
            return pd.DataFrame()

    # Minutes / rotation risk — P(start) from recent + season start rates, killed by injury.
    # Both rates unknown -> P(start) unknown (NaN), not 0.6 (which sat exactly on the captain bar).
    _nan = pd.Series(np.nan, index=pl.index)
    _sr  = pd.to_numeric(pl["starter_rate"], errors="coerce") if "starter_rate" in pl.columns else _nan
    _ssr = (pd.to_numeric(pl["season_start_rate"], errors="coerce")
            if "season_start_rate" in pl.columns else _nan)
    pl["p_start"] = (0.6 * _sr.fillna(_ssr) + 0.4 * _ssr.fillna(_sr)).clip(0, 1)
    if pl["p_start"].isna().any():
        log.info("fantasy: %d player(s) with no start-rate record - P(start) unknown, not ranked "
                 "as starters or captains", int(pl["p_start"].isna().sum()))
    # float64 explicitly: the start rates arrive as float32, and pandas 3 refuses to upcast on a
    # masked assignment (TypeError at the doubtful-player line below).
    pl["p_start"] = pl["p_start"].astype("float64")
    pl.loc[pl["injured"] == True, "p_start"] = 0.0                                    # noqa: E712
    _chance = pd.to_numeric(pl.get("chance_of_playing", np.nan), errors="coerce")
    _dmask = (pl["doubtful"] == True) & _chance.notna()                              # noqa: E712
    pl.loc[_dmask, "p_start"] = (pl.loc[_dmask, "p_start"] * _chance[_dmask] / 100.0)
    pl["p_start"] = pl["p_start"].round(2)
    pl["xpts_rot"] = (pl["fantasy_pts"] * pl["p_start"]).round(3)   # rotation-adjusted (uncond.)

    # Penalty-taker flag — materially higher goal expectation
    pen_ids = _pen_taker_ids(pl_all)
    pl["is_pen_taker"] = pl["player_id"].isin(pen_ids) if "player_id" in pl.columns else False

    pl["value"] = (pl["fantasy_pts"] / pl["price"]).round(3)

    # FPL fixture context (official FDR) — drives display + the dashboard "fixtures" banner.
    if fdr_map:
        def _fx_str(team):
            fl = fdr_map.get(team, [])[:next_n]
            return " · ".join(f"{f.get('opp','')}({'H' if f.get('home') else 'A'}{f.get('fdr','')})" for f in fl)
        def _fx_avg(team):
            vals = [f["fdr"] for f in fdr_map.get(team, [])[:next_n] if f.get("fdr")]
            return round(float(np.mean(vals)), 2) if vals else np.nan
        pl["next_fixtures"]      = pl["team"].map(_fx_str)
        pl["avg_fdr"]           = pl["team"].map(_fx_avg)
        pl["fixtures_available"] = True
    else:
        pl["next_fixtures"]      = ""
        pl["avg_fdr"]           = np.nan
        pl["fixtures_available"] = False
    # ── PER-FIXTURE opponent adjustment ──────────────────────────────────────────────────
    # These two lines used to be:
    #     fixture_adj_pts = fantasy_pts                     # i.e. no adjustment at all
    #     total_xpts_next = fantasy_pts * n_fixtures_next    # i.e. per-game points x N
    #
    # So the "opponent-adjusted" projection was IDENTICAL to the unadjusted one — FDR was fetched,
    # averaged into avg_fdr for display, and then never applied to any number — and the multi-GW
    # total was a flat multiplication. Selecting "Total next 5" simply showed one game's points
    # five times over, regardless of whether a club faced the top four or the bottom four.
    #
    # WHY ONLY attack_pts IS SCALED. fantasy_pts = appearance + attack_pts + dc_pts + cs_pts +
    # bonus_pts, and cs_pts is ALREADY FDR-adjusted inside _cs_row — which is exactly why the
    # original author left the total alone rather than risk double-counting. Scaling the whole
    # figure would count opponent difficulty twice for the clean-sheet component. So:
    #   appearance  FDR-independent: you are paid for minutes whoever you play
    #   attack_pts  FDR-sensitive: goals and assists are harder against better defences  <- scaled
    #   dc_pts      ambiguous: a harder opponent means MORE defending to do, not less. Left neutral
    #   cs_pts      already FDR-adjusted per _cs_row. NOT scaled again
    #   bonus_pts   derived from the components above; left with them
    #
    # THE COEFFICIENTS ARE A CONVENTION, NOT A FIT. 0.10 per FDR step from neutral (FDR 3) and a
    # 5% home edge are conventional FPL values, not values estimated from our own data. They are
    # env-overridable and should be replaced by a fitted mapping once enough gameweeks of
    # projected-vs-actual exist to estimate them. Stated plainly so nobody mistakes the shape of
    # this curve for something measured.
    _FDR_STEP = float(_os.getenv("FANTASY_FDR_STEP", "0.10"))
    _HOME_EDGE = float(_os.getenv("FANTASY_HOME_EDGE", "0.05"))

    _no_fdr = [0]

    def _fx_mult(f: dict) -> float:
        # A fixture with no official FDR gets NO opponent adjustment (x1.0, counted and logged)
        # rather than an invented neutral FDR of 3 plus a home/away edge.
        try:
            fdr = float(f["fdr"])
        except (KeyError, TypeError, ValueError):
            _no_fdr[0] += 1
            return 1.0
        m = 1.0 + (3.0 - fdr) * _FDR_STEP
        m *= (1.0 + _HOME_EDGE) if f.get("home") else (1.0 - _HOME_EDGE)
        return max(0.60, min(1.40, m))       # never invert or double a projection

    def _fixture_list(team):
        return fdr_map.get(team, [])[:next_n] if fdr_map else []

    _atk = pl["attack_pts"].astype(float)
    _cs_next = pl["cs_pts"].astype(float)
    # The FDR-neutral remainder: appearance + dc + bonus. Clean-sheet points are fixture-specific
    # and added per fixture below (they were averaged over 5 fixtures into one figure).
    _fixed = pl["fantasy_pts"].astype(float) - _atk - _cs_next

    def _cs_pts_for(i, f):
        w = FPL_CS_PTS[pl["position"].iloc[i]]
        if w == 0:
            return 0.0
        p = _cs_row(pl.iloc[i], [f])
        return np.nan if p != p else float(p) * w       # unknown CS -> unknown, not 0

    def _per_game(i, team):
        # NEXT gameweek: the next fixture only (it is settled against one gameweek).
        if team not in fdr_map:
            return np.nan
        fl = _fixture_list(team)[:1]
        if not fl:
            return np.nan
        return float(_fixed.iloc[i] + _cs_next.iloc[i] + _atk.iloc[i] * _fx_mult(fl[0]))

    def _total(i, team):
        # No fixture list for the club (absent map, or a club FPL lists no fixtures for) is
        # MISSING data -> NaN; it used to be one notional game, i.e. an invented fixture.
        if team not in fdr_map:
            return np.nan
        fl = _fixture_list(team)
        return float(sum(_fixed.iloc[i] + _cs_pts_for(i, f) + _atk.iloc[i] * _fx_mult(f) for f in fl))

    _teams = pl["team"].tolist()
    pl["fixture_adj_pts"] = [round(_per_game(i, tm), 3) for i, tm in enumerate(_teams)]
    pl["total_xpts_next"] = [round(_total(i, tm), 2) for i, tm in enumerate(_teams)]
    if _no_fdr[0]:
        log.info("fantasy: %d fixture(s) without an official FDR - attack points left unadjusted",
                 _no_fdr[0])
    _nofx = pl["fixture_adj_pts"].isna()
    if _nofx.any():
        log.info("fantasy: %d player(s) with no FPL fixture list - fixture-adjusted and window "
                 "totals unknown", int(_nofx.sum()))

    # Multi-GW planner — actual # of upcoming fixtures in the window, which is what captures
    # DOUBLE and BLANK gameweeks. Unknown (no fixture list) stays NaN, never 1.
    pl["n_fixtures_next"] = pl["team"].map(
        lambda t: len(fdr_map[t][:next_n]) if t in fdr_map else np.nan)
    # Rotation-adjusted total: same per-fixture weighting, scaled by P(start).
    pl["total_xpts_rot"] = (pd.Series(pl["total_xpts_next"], index=pl.index)
                            .astype(float) * pl["p_start"].astype(float)).round(2)

    keep = [c for c in ["fpl_id", "player_name", "team", "position", "price", "minutes_pg",
                        "p_goal", "p_assist", "p_sot2", "attack_pts", "dc_pts", "dc_actions_pg",
                        "cs_prob", "cs_pts",
                        "bonus_pts", "def_actions_pg", "fantasy_pts", "xpts_rot", "p_start",
                        "is_pen_taker", "value", "availability", "injured",
                        "doubtful", "chance_of_playing", "fpl_matched",
                        "fpl_ep_next", "fpl_ppg", "fpl_form", "total_points", "owned_pct",
                        "next_fixtures", "avg_fdr", "fixtures_available", "fixture_adj_pts",
                        "n_fixtures_next", "total_xpts_next", "total_xpts_rot"]
            if c in pl.columns]
    out = pl[keep].reset_index(drop=True)
    out = rank_and_captains(out, base_col="fantasy_pts")
    for c in ["p_goal", "p_assist", "p_sot2"]:
        out[c] = out[c].round(3)
    return out


# The legacy opponent-adjusted path (build_fantasy_projections_fixtures + _team_defense_map) was
# removed 2026-11-05 (propsdef): no caller anywhere, and it scaled points by a TYPED home edge
# (x1.10 home / x0.92 away) clipped to 0.6-1.5. The live next-gameweek fixture adjustment is the
# FDR step inside build_fantasy_projections (reported in reports/P2_PROPS_DEFAULTS_AND_FANTASY.md).


def generate_fantasy_tips(write: bool = True) -> pd.DataFrame:
    """Build projections and (optionally) write output/fantasy_tips.csv. Returns the df."""
    proj = build_fantasy_projections()
    if write and not proj.empty:
        _OUT.parent.mkdir(parents=True, exist_ok=True)
        proj.to_csv(_OUT, index=False)

        # Append-only projection log, so "projected vs actual" is answerable later.

        # fantasy_tips.csv is OVERWRITTEN each run and carries no date or gameweek,

        # so without this there is no record of what was projected WHEN.

        try:

            from player_model.fantasy_log import append as _log_proj

            # `proj`, not `out`. This said `_log_proj(out)` -- a name that does not exist in
            # this function -- so every run raised NameError, the except below swallowed it,
            # and output/fantasy_projection_log.csv was NEVER written. Three days of daily
            # runs, each reporting success, with the file absent from the repo entirely.
            #
            # The dashboard's 'Projected vs actual' section reads that log, so the one
            # question the feature exists to answer -- is this projection better than FPL's
            # free number -- was unanswerable, and it looked like missing data, not a typo.
            # STAMP THE GAMEWEEK. This said `_log_proj(proj)`, with no gw=, so every one of the
            # 8,366 rows written across 31 days carried an empty `gw` -- append() defaults the
            # keyword to None and writes "". Without that tag a projection cannot be joined to
            # the points actually scored, so settling forecasts against real gameweeks, and
            # therefore answering "is Wowza better than FPL's own ep_next", was impossible no
            # matter how long the log ran. The previous fix here got the file written at all
            # (it used to raise NameError); this makes what it writes usable.
            from player_model.fpl_api import gameweek_context as _gw_ctx
            _ctx = _gw_ctx()
            _log_proj(proj, gw=_ctx.get("gw"))
            if _ctx.get("gw") is None:
                log.warning("[fantasy_log] no target gameweek resolved "
                            f"({'season complete' if _ctx.get('season_complete') else 'FPL events unavailable'})"
                            " — rows logged without a gw tag and cannot be settled")

        except Exception as _e:

            log.warning(f"[fantasy_log] skipped ({type(_e).__name__}: {_e})")
    return proj


# Squad-view stats are recomputed over a USER-SELECTABLE window (last-N games) from the
# RAW per-game columns, so the dashboard can show 5/10/20/38/multi-season form dynamically.
# The history's age / height_cm are NOT data: 100% of rows carry the placeholders age 25.0 and
# height 180 cm (measured 2026-11-02 on all 45,230 Premier League rows; the comment here used to
# say "~84%"). Fantasy reads them as MISSING (_mask_profile_placeholders) and never shows them.
_RAW_MAP = {  # display "_pg" stat -> raw per-game source column
    "goals_pg": "goals", "assists_pg": "assists", "sot_pg": "shots_on_target",
    "shots_pg": "shots_total", "cards_pg": "yellow_cards", "minutes_pg": "minutes",
    "rating_pg": "rating", "saves_pg": "saves",
}
_SQUADS_OUT = Path(__file__).resolve().parents[1] / "output" / "pl_squads.csv"
_OFFICIAL_SQUADS = Path(__file__).resolve().parents[1] / "output" / "pl_squads_official.csv"
# ALL leagues we tip in, for predict.py's current-club overlay. Deliberately separate from
# _OFFICIAL_SQUADS, which build_squads() uses as the Premier League roster universe.
_ALL_SQUADS = Path(__file__).resolve().parents[1] / "output" / "squads_official.csv"
# Competitions whose "teams" are countries are identified by the registry's competition_type
# ('national_teams'), not a hand list: a squad row from one says nothing about a player's CLUB.


def build_squads(n: int = 5, parquet_path: Path | None = None, write: bool = False) -> pd.DataFrame:
    """Per-club PL squad view with each player's form over the LAST `n` games (recomputed
    from raw per-game columns, so n is fully dynamic: 5/10/20/38/multi-season). Roster comes
    from the official squad feed when present (transfer-accurate, all 20 clubs; new signings
    show with blank stats until they have PL history), else from parquet most-recent team.
    Adds `games_used` (actual games available <= n). write=True snapshots to pl_squads.csv."""
    df = pd.read_parquet(parquet_path or _PARQUET)
    pl = _fantasy_league_rows(df).copy()
    if pl.empty:
        return pd.DataFrame()
    pl = _mask_profile_placeholders(pl).sort_values("date")
    raw_cols = [r for r in _RAW_MAP.values() if r in pl.columns]

    recent = pl.groupby("player_id").tail(n)                       # last n games per player
    stats = recent.groupby("player_id")[raw_cols].mean().rename(
        columns={v: k for k, v in _RAW_MAP.items() if v in raw_cols})
    stats["games_used"] = recent.groupby("player_id").size()
    latest = pl.groupby("player_id").tail(1).set_index("player_id")
    if "player_career_avg_quality" in latest.columns:
        stats["player_career_avg_quality"] = latest["player_career_avg_quality"]
    stats = stats.reset_index()

    if _OFFICIAL_SQUADS.exists():
        try:
            base = pd.read_csv(_OFFICIAL_SQUADS)[["player_id", "player_name", "team", "position"]]
        except Exception:
            base = latest.reset_index()[["player_id", "player_name", "team", "position"]]
    else:
        base = latest.reset_index()[["player_id", "player_name", "team", "position"]]
    sq = base.merge(stats, on="player_id", how="left")

    for c in [c for c in sq.columns if c not in ("player_id", "player_name", "team", "position")]:
        sq[c] = pd.to_numeric(sq[c], errors="coerce").round(2)
    sort_key = "goals_pg" if "goals_pg" in sq.columns else "player_name"
    sq = sq.sort_values(["team", "position", sort_key], ascending=[True, True, False])
    if write:
        _SQUADS_OUT.parent.mkdir(parents=True, exist_ok=True)
        sq.to_csv(_SQUADS_OUT, index=False)
    return sq


def player_form(n: int = 5, parquet_path: Path | None = None) -> pd.DataFrame:
    """Per-player last-N-game form KPIs + a `form` list (last-N goals, for a sparkline).
    Recomputed from raw per-game columns so n is fully dynamic. Keyed by player_id."""
    df = pd.read_parquet(parquet_path or _PARQUET)
    pl = _fantasy_league_rows(df).sort_values("date")
    if pl.empty:
        return pd.DataFrame(columns=["player_id"])
    pl = _mask_profile_placeholders(pl)
    recent = pl.groupby("player_id").tail(n)
    # A missing source column gives a missing KPI - it used to be filled with the GOALS mean
    # under the SOT / minutes label.
    spec = {"f_goals_pg": ("goals", "mean"), "f_games": ("goals", "size")}
    if "shots_on_target" in pl.columns:
        spec["f_sot_pg"] = ("shots_on_target", "mean")
    if "minutes" in pl.columns:
        spec["f_min_pg"] = ("minutes", "mean")
    agg = recent.groupby("player_id").agg(**spec).round(2)
    for c in ("f_sot_pg", "f_min_pg"):
        if c not in agg.columns:
            agg[c] = np.nan
    agg = agg[["f_goals_pg", "f_sot_pg", "f_min_pg", "f_games"]]
    agg["form"] = recent.groupby("player_id")["goals"].apply(lambda s: [int(x) for x in s.tolist()])
    return agg.reset_index()


def upcoming_by_team(next_n: int = 5) -> dict:
    """{team_name: [next opponents]} for upcoming PL fixtures. Empty off-season (no key /
    no fixtures) — the caller renders a graceful placeholder until the season starts."""
    try:
        from player_model.api_football import get_upcoming_fixtures
        lid = _fantasy_league_id()
        season = config.prop_season(FANTASY_LEAGUE) if FANTASY_LEAGUE else None
        if lid is None or not str(season).isdigit():
            log.warning("fantasy: no league id / season for %r - no upcoming fixtures", FANTASY_LEAGUE)
            return {}
        fx = get_upcoming_fixtures(lid, season, next_n=next_n * 12)  # a few GWs to cover all clubs
        out: dict[str, list] = {}
        for f in fx:
            h = f.get("teams", {}).get("home", {}).get("name")
            a = f.get("teams", {}).get("away", {}).get("name")
            if h and a:
                out.setdefault(h, []).append(f"{a} (H)")
                out.setdefault(a, []).append(f"{h} (A)")
        return {t: v[:next_n] for t, v in out.items()}
    except Exception as e:                                                  # noqa: BLE001
        log.warning("fantasy: upcoming fixtures unavailable (%s: %s)", type(e).__name__, e)
        return {}


def _season_candidates(league: str) -> tuple[str, ...]:
    """The league's CURRENT API season (registry rules), then the previous one. Replaces the
    literal ('2026', '2025') - silently last season's squads from mid-2027, and J1 (2027) never
    right (R2-D fantasy bug 'season_hardcoded')."""
    cur = config.prop_season(league)
    if not str(cur).isdigit():
        log.warning("fantasy: no API season resolved for %r (%s) - skipped", league, cur)
        return ()
    return (cur, str(int(cur) - 1))


def refresh_official_squads(league_id: int | None = None, seasons=None) -> int:
    """Fetch official current squads via API-Football -> output/pl_squads_official.csv
    (transfer-window accurate; run daily in CI). Picks the first season that has teams.
    Returns player count, or 0 if unavailable (no key / off-season / endpoint)."""
    from player_model.api_football import get_league_teams, get_pl_squads
    lid = league_id or _fantasy_league_id()
    if lid is None:
        return 0
    seasons = seasons or _season_candidates(FANTASY_LEAGUE)
    season = next((s for s in seasons if len(get_league_teams(lid, s)) >= 15), None)
    if not season:
        return 0
    rows = get_pl_squads(lid, season)
    if not rows:
        return 0
    df = pd.DataFrame(rows)[["player_id", "player_name", "team", "position", "number", "age"]]
    _OFFICIAL_SQUADS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_OFFICIAL_SQUADS, index=False)
    return len(df)


def refresh_all_squads(leagues: dict | None = None,
                       seasons=None) -> int:
    """Official current squads for EVERY league we tip in -> output/squads_official.csv.

    WHY A SECOND FILE AND NOT A WIDER pl_squads_official.csv. That file is the PREMIER LEAGUE
    ROSTER UNIVERSE for `build_squads()` (line ~538: it is read as `base`, the list of players
    the FPL squad view is built from). Widening it in place would take pl_squads.csv from ~674
    Premier League players to ~10,000 across 26 competitions and put every league on the Fantasy
    dashboard page. Different consumers, different files.

    WHAT IT FIXES. `predict.py` resolves a player's current club from his latest appearance, and
    an appearance is only evidence of where he played THAT DAY. The live-squad overlay corrects
    that, but it has only ever been fed a Premier League file, so a summer transfer inside the
    EFL, Bundesliga 2, La Liga 2 or Serie B stayed invisible and the player kept being tipped for
    a club he had left. The join is on `player_id`, so it is immune to the club-name spelling
    differences that break everything else here.

    COST. One teams call plus one squad call per team, per league, all cached 24h in
    api_football. 26 leagues x ~19 clubs is roughly 500 calls — daily, against ~60,000/day of
    headroom. Do not move this into a 5-minute loop; it belongs in a daily batch job.

    A league that returns nothing is SKIPPED, not fatal: off-season, or a competition whose
    squads API-Football has not populated yet. Returns the total player count written.
    """
    from player_model.api_football import get_league_teams, get_pl_squads
    # The registry's prop leagues (props: true) are name -> API-Football id for every competition
    # the prop model scores - the same facts config.PROP_LEAGUES is derived from, read directly.
    reg = _registry()
    if leagues is None:
        leagues = dict(sorted(reg.prop_leagues().items()))
    # NATIONAL-TEAM COMPETITIONS ARE NOT CLUBS, and including them re-creates the exact bug the
    # current-club rule exists to prevent. Measured on the first full fetch: the World Cup entry
    # contributed 1,013 players whose "team" is Algeria, Argentina, Australia, Austria, Brazil...
    # Feeding that into the overlay would set a player's CURRENT CLUB to his country — which is
    # where the `Australia`/`Austria` teams in an earlier player_tips.csv came from. The kind of
    # competition comes from config/leagues.json competition_type; a league the registry cannot
    # classify is skipped (logged), since it might be a national-team competition.
    ctype = {n: reg.competition_type(n) for n in leagues}
    unknown = [n for n, t in ctype.items() if t is None]
    if unknown:
        print(f"[squads] skipped {len(unknown)} league(s) with no registry competition_type: "
              f"{', '.join(unknown)}")
    leagues = {n: i for n, i in leagues.items() if ctype[n] in ("club_league", "club_cup")}

    frames, skipped = [], []
    for name, lid in leagues.items():
        try:
            season = next((s for s in (seasons or _season_candidates(name))
                           if len(get_league_teams(lid, s)) >= 8), None)
            rows = get_pl_squads(lid, season) if season else []
        except Exception as e:                                   # noqa: BLE001
            print(f"[squads] {name} ({lid}) failed: {type(e).__name__}: {e}")
            rows = []
        if not rows:
            skipped.append(name)
            continue
        f = pd.DataFrame(rows)[["player_id", "player_name", "team", "position", "number", "age"]]
        f["league"] = name
        f["league_id"] = lid
        f["season"] = season
        frames.append(f)
        print(f"[squads] {name}: {len(f)} player(s) across "
              f"{f['team'].nunique()} club(s) (season {season})")

    if not frames:
        print("[squads] nothing fetched — file left untouched")
        return 0
    df = pd.concat(frames, ignore_index=True)
    # A ZERO OR NEGATIVE ID IS NOT AN ID. API-Football returned one squad row as player_id 0
    # ("Guillem Badia", Girona), and player_history.parquet holds 9 rows with the same non-id.
    # Joining them would map all 9 to Girona — one club, chosen by a collision, presented with
    # exactly as much confidence as a real answer.
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    bad = df["player_id"].isna() | (df["player_id"] <= 0)
    if bad.any():
        print(f"[squads] dropped {int(bad.sum())} row(s) with a non-id player_id")
        df = df[~bad]
    df["player_id"] = df["player_id"].astype("int64")
    # A player can appear in a domestic league AND a UEFA competition. Keep the DOMESTIC row:
    # the club is the same either way, but the domestic entry is the one that gets refreshed
    # every day of the season. Sorting club cups (registry competition_type 'club_cup': UCL, UEL,
    # UECL) last makes keep="first" do that. The old hand list spelled 'Europa Conference
    # League', which the registry calls 'Conference League', so Conference rows never sorted last.
    df["_uefa"] = (df["league"].map(ctype) == "club_cup").astype(int)
    df = (df.sort_values(["_uefa"]).drop_duplicates("player_id", keep="first")
            .drop(columns="_uefa").sort_values(["league", "team", "player_name"]))
    _ALL_SQUADS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_ALL_SQUADS, index=False)
    if skipped:
        print(f"[squads] no squads returned for {len(skipped)}: {', '.join(skipped)}")
    print(f"[squads] wrote {_ALL_SQUADS.name}: {len(df):,} player(s), "
          f"{df['team'].nunique()} club(s), {df['league'].nunique()} league(s)")
    return len(df)


if __name__ == "__main__":
    p = generate_fantasy_tips(write=True)
    if p.empty:
        print("no fantasy projections (no PL data / models missing)")
    else:
        print(f"fantasy projections: {len(p)} players -> {_OUT.name}")
        print("TOP 10:")
        for r in p.head(10).itertuples():
            print(f"  {r.overall_rank:>2}. {_ascii(r.player_name):<24} {r.position} -> {r.fantasy_pts}")
    sq = build_squads(write=True)
    print(f"squads: {len(sq)} players across {sq['team'].nunique() if not sq.empty else 0} clubs -> {_SQUADS_OUT.name}")
