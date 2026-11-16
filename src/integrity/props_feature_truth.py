"""Props feature truth - Data truth 7 (fix/props-feature-bugs, 2026-11-11).

    python -m src.integrity.props_feature_truth            # both checks, printed + JSON on stdout

Two defects reached the SERVED props features before these checks existed
(reports/R_COMBINED_MODELS_2026-11-11.md):
  * a goalkeeper's goals_conceded read 0 beside a real score (provider null read as 0 by the null
    rule; 14% / 47% / 84% of mapped fixtures in 2026-09 / 10 / 11) -> opp_gk_save_rate toward 1.0;
  * team_goals_pg_roll (own team, by name and contract) was merged on the OPPONENT key;
  * penalty_missed: the same provider change as goals_conceded (null read as 0), inflating the served
    penalty_conversion_rate (owner 2026-11-11, "Fix penalties too, then promote").

CHECKS
  gk_conceded   On the counts the props features read (props_harness.source_frame: re-fetch applied,
                then history_refetch.with_gk_conceded_checked), every keeper row that is checkable -
                his side's only keeper, >= 90 minutes, match finished in 90 minutes, score stored -
                must equal the opponent's 90-minute goals or be missing. RED on any disagreement.
                The value also counts what the provider data holds BEFORE the reader's check
                (contradictions blanked, per month) so a provider regression stays visible.
  side_keys     Every rolling team feature reads the side its name says: _TEAM_ROLL 'opp' entries are
                named opp_*, 'own' entries are not; PREV_SEASON_TEAM 'opp' entries are named
                prev_season_opp_*, 'own' entries prev_season_team_*. And BEHAVIOUR: on a synthetic
                three-club table where every club has its own constant per quantity,
                feature_engineering.attach_team_rolls must give each player row its own club's value
                for an 'own' feature and its opponent's value for an 'opp' feature. RED otherwise.
  null_missing  Every NULL_IS_MISSING count (goals_conceded, penalty_missed) the reader marked as a
                provider null (history_refetch.NULL_BLANKED_COL) is missing on the frame the features
                read - never 0. RED otherwise, or when the frame carries no provenance. The value also
                shows, per month, the provider nulls and the recorded penalty misses per 1,000 scored
                penalties on rows with known attempts, so a provider regression stays visible.

PROPS FEATURE AUDIT 2026-11-15 (owner-approved fixes; each check below would have caught its defect):
  started_drift   (7d) `started` as the features read it (lineup_started: the collected lineups) agrees
                with the lineup role on every row, and its share per league-month (>= DRIFT_MIN_ROWS
                rows) stays within STARTED_SHARE_BAND of the league's own share over its previous 12
                months. RED otherwise. The provider's own flag (started_provider) is measured beside:
                agreement with the lineups per month - the 2026-09..11 drift (0.92 / 0.81 / 0.81) shows
                as AMBER, since no feature reads that flag any more.
  null_as_zero  (7e) For EVERY per-player count: in a fixture whose answer sends explicit zeros
                (history_refetch.explicit_zero_fixtures) a count the provider left null for every
                played player (the field unpublished for the match) must not reach the features as 0.
                RED on any such fixture, and on a placeholder answer (every played player with 0 passes)
                whose detailed counts reach the frame. Per month the fixture-wide unpublished share is shown.
  stale_sources (7f) Every collected source a props feature reads keeps up with the matches: per
                league, the latest fixture covered by the injury lists (covered league-seasons) and by
                the lineups must be no more than STALE_DAYS before the latest match the props frame
                holds for that league. RED otherwise. The retired /sidelined source is measured beside
                (AMBER only: no feature reads it).
  shots_level   (7g) A shots definition change cancels out of the *_lgrel features: per league-month
                (>= LEVEL_MIN_TEAM_MATCHES team-matches with a level) the mean team shots relative to
                the league's as-of level (feature_engineering.league_shots_level) stays within
                SHOTS_REL_BAND of 1, outside the month of a sudden level move (>= SHOTS_JUMP vs the previous 3
                months) and the month after it (the as-of window passing). RED otherwise. Raw definition jumps (a league-month's team shots
                per match >= SHOTS_JUMP vs the league's previous 3 months while shots on target move
                < SOT_STABLE) are listed beside (AMBER: the raw features are withheld from training).

Nothing here changes data.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

GREEN, AMBER, RED, GREY = "GREEN", "AMBER", "RED", "NO_DATA"
GK_COLUMNS = ["fixture_id", "player_id", "date", "position", "minutes", "is_home", "team", "home_team",
              "away_team", "opponent", "league", "goals_conceded", "saves", "penalty_scored",
              "penalty_missed"]
#: 7d-7g (2026-11-15): thresholds of the checks (a check's tolerance, not a feature value)
DRIFT_MIN_ROWS = 200            # league-month rows before a starter share is compared
STARTED_SHARE_BAND = 0.10       # |month share - league's previous-12-month share| allowed
PROVIDER_AGREEMENT_MIN = 0.98   # provider started flag vs lineups, per month (AMBER below)
STALE_DAYS = 14                 # a source may trail the props frame's latest match by this much
LEVEL_MIN_TEAM_MATCHES = 40     # league-month team-matches before its relative shots are judged
SHOTS_REL_BAND = 0.15           # |mean relative team shots - 1| allowed per league-month
SHOTS_JUMP = 0.15               # raw team shots per match vs the previous 3 months
SOT_STABLE = 0.05               # ... while shots on target moved less than this
MIN_FIXTURE_ROWS = 10           # played rows of a fixture before "null for every player" is judged
_FRAME: dict = {}


def _check_columns() -> list[str]:
    from player_model.player_stat_nulls import HISTORY_COUNT_COLS
    return list(dict.fromkeys(GK_COLUMNS + ["started", "season", *HISTORY_COUNT_COLS]))


def _frame():
    """props_harness.source_frame (the columns the checks read), built once per process."""
    if "raw" not in _FRAME:
        from player_model.props_harness import source_frame
        _FRAME["raw"] = source_frame(columns=_check_columns())
    return _FRAME["raw"]


def _c(check, status, value, rule, detail=""):
    return {"check": check, "status": status, "value": value, "rule": rule, "detail": detail}


def gk_conceded(raw: pd.DataFrame | None = None, fx: pd.DataFrame | None = None) -> dict:
    """Keeper goals_conceded vs the 90-minute score on the frame the props features read."""
    from player_model import history_refetch as HR
    label = "Data truth 7a: keeper goals_conceded = match score"
    rule = ("a keeper who was his side's only keeper for the full 90 minutes of a match finished in 90 "
            "minutes has goals_conceded = the opponent's 90-minute goals, or missing - never another value")
    if raw is None:
        raw = _frame()
    if raw is None or raw.empty or HR.GK_CHECK_COL not in raw.columns:
        return _c(label, RED, "the props source frame carries no keeper check", rule,
                  "props_harness.source_frame must apply history_refetch.with_gk_conceded_checked")
    g = HR.gk_conceded_frame(raw, fx)
    chk = g[g["checkable"].astype(bool)]
    held = chk["gc"].notna()
    bad = held & (chk["gc"] != chk["conc"])
    # what the provider data held before the reader's check (kept beside in goals_conceded_provider)
    before = pd.to_numeric(raw.loc[chk.index, HR.GK_ORIG_COL], errors="coerce")
    pre_bad = before.notna() & (before != chk["conc"])
    month = pd.to_datetime(raw.loc[chk.index, "date"], errors="coerce").dt.strftime("%Y-%m")
    by_month = (pd.DataFrame({"m": month, "pre_bad": pre_bad, "missing": chk["gc"].isna()})
                .groupby("m").agg(rows=("pre_bad", "size"), contradicted_before_check=("pre_bad", "sum"),
                                  missing_after=("missing", "sum")))
    recent = by_month.tail(6)
    detail = "; ".join(f"{m}: {int(r.contradicted_before_check)}/{int(r.rows)} contradicted before, "
                       f"{int(r.missing_after)} missing after" for m, r in recent.iterrows())
    value = (f"{int(bad.sum())} disagreement(s) in {int(held.sum()):,} checkable keeper value(s) read by the "
             f"features; {int(pre_bad.sum()):,} provider value(s) contradicted the score and are missing")
    return {**_c(label, RED if bad.any() else GREEN, value, rule, detail),
            "by_month": {m: {k: int(v) for k, v in r.items()} for m, r in by_month.iterrows()}}


def null_missing(raw: pd.DataFrame | None = None) -> dict:
    """Provider-null counts of NULL_IS_MISSING columns stay missing on the frame the features read."""
    from player_model import history_refetch as HR
    from player_model.player_stat_nulls import NULL_IS_MISSING
    label = "Data truth 7c: provider-null counts stay missing"
    rule = ("a NULL_IS_MISSING count (" + ", ".join(sorted(NULL_IS_MISSING)) + ") the provider sent as "
            "null is missing on the props frame - never 0; a real provider 0 stays 0")
    raw = _frame() if raw is None else raw
    if raw is None or raw.empty or HR.NULL_BLANKED_COL not in raw.columns:
        return _c(label, RED, "the props source frame carries no provider-null provenance", rule,
                  f"history_refetch.with_refetched must write {HR.NULL_BLANKED_COL}")
    marks = raw[HR.NULL_BLANKED_COL].fillna("").astype(str).str.split(",")
    bad, nulls = {}, {}
    for c in sorted(NULL_IS_MISSING):
        m = marks.map(lambda xs, c=c: c in xs).to_numpy(dtype=bool)
        nulls[c] = int(m.sum())
        if c in raw.columns:
            bad[c] = int(pd.to_numeric(raw.loc[m, c], errors="coerce").notna().sum())
        else:
            bad[c] = 0
    detail = ""
    if {"penalty_scored", "penalty_missed", "date"} <= set(raw.columns):
        ps = pd.to_numeric(raw["penalty_scored"], errors="coerce")
        pm = pd.to_numeric(raw["penalty_missed"], errors="coerce")
        known = ps.notna() & pm.notna()
        mon = pd.to_datetime(raw["date"], errors="coerce").dt.strftime("%Y-%m")
        t = (pd.DataFrame({"m": mon, "s": ps.where(known), "x": pm.where(known), "n": pm.isna() & ps.notna()})
             .groupby("m").agg(s=("s", "sum"), x=("x", "sum"), n=("n", "sum")).tail(6))
        detail = "; ".join(f"{m}: {int(r.x)} missed / {int(r.s)} scored on known rows, {int(r.n)} null miss "
                           f"count(s)" for m, r in t.iterrows())
    value = (", ".join(f"{c}: {nulls[c]:,} provider null(s), {bad[c]} read as a value" for c in sorted(nulls)))
    return _c(label, RED if any(bad.values()) else GREEN, value, rule, detail)


# ── 7d-7g: props feature audit 2026-11-15 ────────────────────────────────────────────────────────

def _month(dates) -> pd.Series:
    return pd.to_datetime(dates, errors="coerce").dt.strftime("%Y-%m")


def started_drift(raw: pd.DataFrame | None = None, roles: pd.DataFrame | None = None) -> dict:
    """7d: `started` read by the features = the lineup role; its league-month share is stable."""
    from player_model import lineup_started as LS
    label = "Data truth 7d: started from lineups, no flag drift"
    rule = (f"started (the features' input) equals the collected lineup role on every row, and per "
            f"league-month (>= {DRIFT_MIN_ROWS} rows) its share stays within {STARTED_SHARE_BAND:.2f} of the "
            f"league's previous 12 months")
    raw = _frame() if raw is None else raw
    if raw is None or raw.empty or LS.ORIG_COL not in raw.columns or "started" not in raw.columns:
        return _c(label, RED, "the props source frame does not read started from the lineups", rule,
                  "props_harness.source_frame must apply lineup_started.with_lineup_started")
    roles = LS.lineup_roles() if roles is None else roles
    k = pd.DataFrame({"fixture_id": pd.to_numeric(raw["fixture_id"], errors="coerce").to_numpy(),
                      "player_id": pd.to_numeric(raw["player_id"], errors="coerce").to_numpy()})
    want = k.merge(roles, on=["fixture_id", "player_id"], how="left")["started"].to_numpy(dtype=float)
    got = pd.to_numeric(raw["started"], errors="coerce").to_numpy(dtype=float)
    mismatch = int((~((np.isnan(want) & np.isnan(got)) | (want == got))).sum())
    played = ((pd.to_numeric(raw["minutes"], errors="coerce") >= 10).to_numpy() if "minutes" in raw.columns
              else np.ones(len(raw), dtype=bool))
    d = pd.DataFrame({"m": _month(raw["date"]).to_numpy(), "lg": raw["league"].astype(str).to_numpy(),
                      "s": got, "p": pd.to_numeric(raw[LS.ORIG_COL], errors="coerce").to_numpy(dtype=float),
                      "mins": pd.to_numeric(raw["minutes"], errors="coerce").to_numpy()
                      if "minutes" in raw.columns else np.nan})[played]
    lm = d.dropna(subset=["s"]).groupby(["lg", "m"])["s"].agg(["sum", "count"]).reset_index().sort_values(["lg", "m"])
    drift = []
    for lg, g in lm.groupby("lg"):
        g = g.reset_index(drop=True)
        cs, cn = g["sum"].cumsum().shift(1), g["count"].cumsum().shift(1)
        cs12 = cs - g["sum"].cumsum().shift(13).fillna(0.0)
        cn12 = cn - g["count"].cumsum().shift(13).fillna(0.0)
        for i, r in g.iterrows():
            if r["count"] < DRIFT_MIN_ROWS or not (cn12.iloc[i] >= DRIFT_MIN_ROWS):
                continue
            share, prior = r["sum"] / r["count"], cs12.iloc[i] / cn12.iloc[i]
            if abs(share - prior) > STARTED_SHARE_BAND:
                drift.append(f"{lg} {r['m']}: {share:.2f} vs {prior:.2f}")
    both = d.dropna(subset=["s", "p"])
    agree = both.assign(ok=(both["s"] == both["p"])).groupby("m")["ok"].mean()
    short = d[(d["mins"] < 45) & d["p"].notna()].groupby("m")["p"].mean()
    recent = agree.tail(6)
    detail = "; ".join(f"{m}: provider flag agrees {v:.3f}, provider 'started' on {short.get(m, np.nan):.2f} of "
                       f"<45-min rows" for m, v in recent.items())
    status = RED if (mismatch or drift) else (AMBER if (recent < PROVIDER_AGREEMENT_MIN).any() else GREEN)
    value = (f"{mismatch} row(s) where the features' started differs from the lineup role; "
             f"{len(drift)} league-month(s) out of band" + (f" ({'; '.join(drift[:5])})" if drift else ""))
    return {**_c(label, status, value, rule, detail),
            "provider_agreement_by_month": {m: round(float(v), 4) for m, v in agree.items()}}


def null_as_zero(raw: pd.DataFrame | None = None, refetch: pd.DataFrame | None = None) -> dict:
    """7e: no count the provider left unpublished for a whole explicit-zero fixture is read as 0."""
    from player_model import history_refetch as HR
    from player_model.player_stat_nulls import COUNT_FIELDS, NULL_IS_NONE_WITHOUT_EXPLICIT_ZEROS
    label = "Data truth 7e: no null-as-0 in any count"
    rule = ("in a fixture whose answer sends explicit zeros, a count the provider left null for every played "
            "player is missing on the props frame, never 0 (every per-player count)")
    raw = _frame() if raw is None else raw
    rf = HR.load_refetch() if refetch is None else refetch
    if raw is None or raw.empty or rf is None or rf.empty:
        return _c(label, RED, "no props frame or re-fetch sidecar to check", rule)
    rf = rf.assign(fixture_id=pd.to_numeric(rf["fixture_id"], errors="coerce"),
                   player_id=pd.to_numeric(rf["player_id"], errors="coerce"))
    rf = rf.dropna(subset=["fixture_id", "player_id"]).drop_duplicates(["fixture_id", "player_id"], keep="last")
    ez = HR.explicit_zero_fixtures(rf)
    group_unpub = HR.group_unpublished_fixtures(rf, ez)
    played = ((pd.to_numeric(rf["minutes_refetch"], errors="coerce") > 0) if "minutes_refetch" in rf.columns
              else pd.Series(True, index=rf.index))
    rf = rf[played & rf["fixture_id"].map(ez).fillna(False).astype(bool)]
    filled = rf["filled_cols"].fillna("").astype(str).str.split(",")
    n_rows = rf.groupby("fixture_id")["player_id"].transform("size")
    rf = rf[n_rows >= MIN_FIXTURE_ROWS]
    filled = filled.loc[rf.index]
    fr = raw.assign(fixture_id=pd.to_numeric(raw["fixture_id"], errors="coerce"),
                    player_id=pd.to_numeric(raw["player_id"], errors="coerce"))
    fr = fr.drop_duplicates(["fixture_id", "player_id"]).set_index(["fixture_id", "player_id"])
    key = pd.MultiIndex.from_frame(rf[["fixture_id", "player_id"]])
    month = _month(fr["date"]).reindex(key).to_numpy()
    bad, unpublished = {}, {}
    for c in COUNT_FIELDS:
        if c not in rf.columns:
            continue
        isnull = filled.map(lambda xs, c=c: c in xs)
        fx_all = isnull.groupby(rf["fixture_id"]).transform("all").to_numpy(dtype=bool)
        if c in NULL_IS_NONE_WITHOUT_EXPLICIT_ZEROS:
            # published or not as a GROUP (player_stat_nulls.fixture_group_unpublished): a group published
            # for the match leaves a null of one of them the provider's none
            fx_all = rf["fixture_id"].map(group_unpub).eq(True).to_numpy(dtype=bool)
        if not fx_all.any():
            continue
        unpublished[c] = pd.Series(fx_all).groupby(month).mean()
        if c not in fr.columns:
            continue
        v = pd.to_numeric(fr[c], errors="coerce").reindex(key).to_numpy(dtype=float)
        zero = fx_all & (v == 0)
        if zero.any():
            bad[c] = int(pd.Series(rf["fixture_id"].to_numpy()[zero]).nunique())
    # a placeholder answer: every played player of the fixture with 0 passes (impossible in a played match)
    # must reach the features with no detailed count (history_refetch.with_placeholder_answers_blanked)
    fr2 = raw.assign(fixture_id=pd.to_numeric(raw["fixture_id"], errors="coerce"))
    if {"passes_total", "minutes"} <= set(fr2.columns):
        pl = fr2[pd.to_numeric(fr2["minutes"], errors="coerce") > 0]
        pz = pd.to_numeric(pl["passes_total"], errors="coerce").eq(0)
        gsz = pz.groupby(pl["fixture_id"]).agg(["all", "size"])
        ph = set(gsz.index[gsz["all"] & (gsz["size"] >= MIN_FIXTURE_ROWS)])
        if ph:
            from player_model.history_refetch import PLACEHOLDER_DETAIL_COLS
            rows = fr2[fr2["fixture_id"].isin(ph)]
            held = [c for c in PLACEHOLDER_DETAIL_COLS if c in rows.columns
                    and pd.to_numeric(rows[c], errors="coerce").notna().any()]
            if held:
                bad["placeholder_answer"] = len(ph)
    detail = "; ".join(f"{c}: unpublished in " + ", ".join(f"{m} {v:.0%}" for m, v in s.tail(4).items())
                       for c, s in sorted(unpublished.items()))
    if bad:
        value = (f"{len(bad)} count(s) read as 0 where the provider published nothing: "
                 + ", ".join(f"{c}: {n} fixture(s)" for c, n in sorted(bad.items())))
    else:
        value = (f"0 counts read as 0 where the provider published nothing; {len(unpublished)} count(s) "
                 f"unpublished fixture-wide in some explicit-zero fixtures, all missing on the frame")
    return {**_c(label, RED if bad else GREEN, value, rule, detail[:900]), "read_as_zero_fixtures": bad}


def _latest_by_league(df: pd.DataFrame, date_col: str = "date") -> pd.Series:
    d = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    return d.groupby(df["league"].astype(str)).max()


def stale_sources(raw: pd.DataFrame | None = None, inj: pd.DataFrame | None = None,
                  seasons: pd.DataFrame | None = None, roles: pd.DataFrame | None = None,
                  spells: pd.DataFrame | None = None, fx: pd.DataFrame | None = None) -> dict:
    """7f: the injury lists and the lineups keep up with the matches of every league they cover."""
    from player_model import injury_features as IF
    from player_model import lineup_started as LS
    from player_model import team_identity as TI
    from player_model.feature_engineering import _collected
    label = "Data truth 7f: no stale props source"
    rule = (f"per league, the latest fixture covered by the injury lists (covered league-seasons) and by the "
            f"collected lineups is at most {STALE_DAYS} days before the latest match the props frame holds")
    raw = _frame() if raw is None else raw
    if raw is None or raw.empty:
        return _c(label, RED, "no props frame", rule)
    inj = _collected("injuries") if inj is None else inj
    seasons = _collected("injuries_seasons") if seasons is None else seasons
    roles = LS.lineup_roles() if roles is None else roles
    fx = TI.fixture_results() if fx is None else fx
    frame_last = _latest_by_league(raw)
    stale, notes = [], []
    # injury lists: leagues whose LATEST answered league-season has coverage
    if inj is not None and not inj.empty and seasons is not None and not seasons.empty:
        cov = IF.coverage_table(seasons)
        latest = cov.sort_values("_y").groupby("league_id").tail(1)
        covered_ids = set(latest.loc[latest["covered"], "league_id"].astype(int))
        comp = fx.dropna(subset=["league_id"]).drop_duplicates("league_id").set_index("league_id")["competition"]
        idate = pd.to_datetime(inj["date"], errors="coerce").dt.normalize()
        if "fetched_at" in inj.columns:     # a listing of a fixture still to be played is no coverage
            fetched = pd.to_datetime(inj["fetched_at"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
            idate = idate.where(idate <= fetched)
        last_inj = idate.groupby(pd.to_numeric(inj["league_id"], errors="coerce")).max()
        for lid in sorted(covered_ids):
            lg = comp.get(float(lid))
            if lg is None or lg not in frame_last.index or float(lid) not in last_inj.index:
                continue
            gap = (frame_last[lg] - last_inj[float(lid)]).days
            if gap > STALE_DAYS:
                stale.append(f"injuries {lg}: {gap} days behind")
        notes.append(f"injuries: latest listed fixture {str(last_inj.max())[:10]}")
    else:
        stale.append("injuries: source absent")
    # lineups: the frame's fixtures with a collected lineup, latest per league
    fid = pd.to_numeric(raw["fixture_id"], errors="coerce")
    has = fid.isin(set(pd.to_numeric(roles["fixture_id"], errors="coerce")))
    lu_last = _latest_by_league(raw[has.to_numpy()])
    for lg, last in frame_last.items():
        if lg in lu_last.index and (last - lu_last[lg]).days > STALE_DAYS:
            stale.append(f"lineups {lg}: {(last - lu_last[lg]).days} days behind")
    notes.append(f"lineups: latest covered match {str(lu_last.max())[:10]}")
    # the retired /sidelined source (no feature reads it): AMBER only
    spells = _collected("sidelined_spells") if spells is None else spells
    amber = []
    if spells is not None and not spells.empty and "start" in spells.columns:
        st = pd.to_datetime(spells["start"], errors="coerce")
        recent = st[st <= frame_last.max()]
        months = recent.dt.to_period("M").value_counts().sort_index()
        last_busy = months[months >= 30].index.max() if (months >= 30).any() else None
        if last_busy is not None and (frame_last.max().to_period("M") - last_busy).n > 1:
            amber.append(f"/sidelined (retired, not read): last month with >= 30 spells starting {last_busy}")
    value = (f"{len(stale)} stale source-league(s)" + (f": {'; '.join(stale[:8])}" if stale else "")
             + (f"; {'; '.join(amber)}" if amber else ""))
    status = RED if stale else (AMBER if amber else GREEN)
    return _c(label, status, value, rule, "; ".join(notes))


def shots_level(raw: pd.DataFrame | None = None) -> dict:
    """7g: the league-relative shots stay at their own level through a definition change."""
    from player_model import feature_engineering as FE
    label = "Data truth 7g: shots definition drift cancels out"
    rule = (f"per league-month (>= {LEVEL_MIN_TEAM_MATCHES} team-matches with an as-of level) the mean team "
            f"shots relative to the league's own as-of level stays within {SHOTS_REL_BAND:.2f} of 1")
    raw = _frame() if raw is None else raw
    if raw is None or raw.empty or "shots_total" not in raw.columns:
        return _c(label, RED, "no props frame with shots_total", rule)
    lvl = FE.league_shots_level(raw)
    tm = pd.DataFrame({"fixture_id": pd.to_numeric(raw["fixture_id"], errors="coerce"),
                       "side": pd.to_numeric(raw["is_home"], errors="coerce"),
                       "league": raw["league"].astype(str), "m": _month(raw["date"]),
                       "sh": pd.to_numeric(raw["shots_total"], errors="coerce"),
                       "sot": pd.to_numeric(raw["shots_on_target"], errors="coerce"), "lvl": lvl})
    t = (tm.groupby(["fixture_id", "side"])
           .agg(league=("league", "first"), m=("m", "first"), sh=("sh", "sum"), sh_n=("sh", "count"),
                sot=("sot", "sum"), sot_n=("sot", "count"), lvl=("lvl", "first")).reset_index())
    t["sh"] = t["sh"].where(t["sh_n"] > 0)
    t["sot"] = t["sot"].where(t["sot_n"] > 0)
    t["rel"] = t["sh"] / t["lvl"]
    raw_m = t.dropna(subset=["sh", "sot"]).groupby(["league", "m"])[["sh", "sot"]].agg(["mean", "count"])
    jumps, transition = [], set()
    for lg, gg in raw_m.groupby(level=0):
        gg = gg.droplevel(0)
        sh, so, n = gg[("sh", "mean")], gg[("sot", "mean")], gg[("sh", "count")]
        sh3 = sh.shift(1).rolling(3, min_periods=3).mean()
        so3 = so.shift(1).rolling(3, min_periods=3).mean()
        for m in gg.index:
            if n[m] < LEVEL_MIN_TEAM_MATCHES or not (sh3[m] > 0) or not (so3[m] > 0):
                continue
            if abs(sh[m] / sh3[m] - 1) >= SHOTS_JUMP:
                # any sudden level move is a transition of the as-of window; a definition jump when SOT held
                nxt = (pd.Period(m, "M") + 1).strftime("%Y-%m")
                transition |= {(lg, m), (lg, nxt)}
                if abs(so[m] / so3[m] - 1) < SOT_STABLE:
                    jumps.append(f"{lg} {m}: {sh3[m]:.1f} -> {sh[m]:.1f}")
    g = t.dropna(subset=["rel"]).groupby(["league", "m"])["rel"].agg(["mean", "count"])
    g = g[g["count"] >= LEVEL_MIN_TEAM_MATCHES]
    # a sudden jump needs the as-of window (SHOTS_LEVEL_TEAM_MATCHES team-matches) to pass: its month and
    # the next are the transition (shown, not judged); every other league-month must sit in the band
    judged = g[[k not in transition for k in g.index]]
    out_band = judged[(judged["mean"] - 1).abs() > SHOTS_REL_BAND]
    rec = t.dropna(subset=["rel"]).groupby("m")["rel"].mean().tail(6)
    detail = ("relative team shots by month (all leagues): " + ", ".join(f"{m} {v:.3f}" for m, v in rec.items())
              + (f"; raw definition jumps: {'; '.join(jumps[-8:])}" if jumps else ""))
    value = (f"{len(out_band)} league-month(s) of {len(g)} out of band"
             + (": " + "; ".join(f"{lg} {m} {r['mean']:.2f}" for (lg, m), r in out_band.head(6).iterrows())
                if len(out_band) else "")
             + f"; {len(jumps)} raw definition jump(s) seen")
    status = RED if len(out_band) else (AMBER if jumps else GREEN)
    return {**_c(label, status, value, rule, detail[:900]), "raw_jumps": jumps}


def _synthetic_team_table() -> tuple[pd.DataFrame, dict]:
    """Three clubs playing each other in turn; club k's every quantity is a constant of its own."""
    from player_model import feature_engineering as FE
    clubs = ["af:1", "af:2", "af:3"]
    const = {c: {src: float(10 * (i + 1) + j) for j, src in enumerate(FE._TEAM_ROLL)}
             for i, c in enumerate(clubs)}
    pairs = [("af:1", "af:2"), ("af:2", "af:3"), ("af:3", "af:1")] * 6
    rows = []
    for k, (a, b) in enumerate(pairs):
        d = pd.Timestamp("2026-01-01") + pd.Timedelta(days=7 * k)
        for own, opp in ((a, b), (b, a)):
            rows.append({"fixture_id": 1000 + k, "_tkey": own, "_okey": opp, "date": d,
                         **const[own], "team_corners": np.nan})
    t = pd.DataFrame(rows).sort_values(["_tkey", "date", "fixture_id"]).reset_index(drop=True)
    return t, const


def side_keys() -> dict:
    """Name <-> side rule for every rolling team feature, and the attach behaviour on synthetic clubs."""
    from player_model import config
    from player_model import feature_engineering as FE
    label = "Data truth 7b: team features keyed to their side"
    rule = ("an opp_* feature reads the opponent's rolling, a team/own feature the player's own team's; "
            "the merge gives each row exactly that side's value")
    issues = []
    for src, (feat, _min, side) in FE._TEAM_ROLL.items():
        if side not in ("own", "opp"):
            issues.append(f"{feat}: side {side!r} is neither own nor opp")
        elif (side == "opp") != feat.startswith("opp_"):
            issues.append(f"{feat}: declared side {side} but named {'opp_*' if feat.startswith('opp_') else 'own'}")
    for feat, (side, _src) in FE.PREV_SEASON_TEAM.items():
        want = "prev_season_opp_" if side == "opp" else "prev_season_team_"
        if not feat.startswith(want):
            issues.append(f"{feat}: declared side {side} but not named {want}*")
    t, const = _synthetic_team_table()
    n = config.ROLLING_N
    roll = FE._team_rolling(t, n)
    players = t[["fixture_id", "_tkey", "_okey", "date"]].copy()
    got = FE.attach_team_rolls(players, roll)
    if len(got) != len(players):
        issues.append(f"attach_team_rolls changed the row count {len(players)} -> {len(got)}")
    checked = 0
    for src, (feat, _min, side) in FE._TEAM_ROLL.items():
        key = "_tkey" if side == "own" else "_okey"
        expect = got[key].map(lambda c, s=src: const[c][s])
        v = pd.to_numeric(got[feat], errors="coerce")
        m = v.notna()
        checked += int(m.sum())
        wrong = int((v[m] != expect[m]).sum())
        if not m.any():
            issues.append(f"{feat}: no value on the synthetic table (cannot verify its side)")
        elif wrong:
            issues.append(f"{feat}: {wrong} of {int(m.sum())} row(s) carry the other side's value (declared {side})")
    value = (f"{len(FE._TEAM_ROLL)} rolling + {len(FE.PREV_SEASON_TEAM)} previous-season team features; "
             f"{checked:,} synthetic value(s) checked; {len(issues)} issue(s)")
    return _c(label, RED if issues else GREEN, value, rule, "; ".join(issues)[:600])


def evaluate(raw: pd.DataFrame | None = None) -> list[dict]:
    out = []
    for label, fn in (("Data truth 7a: keeper goals_conceded = match score", lambda: gk_conceded(raw)),
                      ("Data truth 7b: team features keyed to their side", side_keys),
                      ("Data truth 7c: provider-null counts stay missing", lambda: null_missing(raw)),
                      ("Data truth 7d: started from lineups, no flag drift", lambda: started_drift(raw)),
                      ("Data truth 7e: no null-as-0 in any count", lambda: null_as_zero(raw)),
                      ("Data truth 7f: no stale props source", lambda: stale_sources(raw)),
                      ("Data truth 7g: shots definition drift cancels out", lambda: shots_level(raw))):
        try:
            out.append(fn())
        except Exception as e:                                         # noqa: BLE001
            out.append(_c(label, RED, "check failed", "must run", f"{type(e).__name__}: {e}"[:200]))
    return out


def main(argv=None) -> int:
    checks = evaluate()
    for c in checks:
        print(f"{c['status']:7} {c['check']}: {c['value']}" + (f"  [{c['detail']}]" if c["detail"] else ""))
    print(json.dumps(checks, indent=1, default=str))
    return 1 if any(c["status"] == RED for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
