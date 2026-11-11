"""Props feature truth - Data truth 6 (fix/props-feature-bugs, 2026-11-11).

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
_FRAME: dict = {}


def _frame():
    """props_harness.source_frame (the columns the checks read), built once per process."""
    if "raw" not in _FRAME:
        from player_model.props_harness import source_frame
        _FRAME["raw"] = source_frame(columns=GK_COLUMNS)
    return _FRAME["raw"]


def _c(check, status, value, rule, detail=""):
    return {"check": check, "status": status, "value": value, "rule": rule, "detail": detail}


def gk_conceded(raw: pd.DataFrame | None = None, fx: pd.DataFrame | None = None) -> dict:
    """Keeper goals_conceded vs the 90-minute score on the frame the props features read."""
    from player_model import history_refetch as HR
    label = "Data truth 6a: keeper goals_conceded = match score"
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
    label = "Data truth 6c: provider-null counts stay missing"
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
    label = "Data truth 6b: team features keyed to their side"
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
    for label, fn in (("Data truth 6a: keeper goals_conceded = match score", lambda: gk_conceded(raw)),
                      ("Data truth 6b: team features keyed to their side", side_keys),
                      ("Data truth 6c: provider-null counts stay missing", lambda: null_missing(raw))):
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
