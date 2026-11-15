"""Phase 2 - adversarial train/serve parity of the rolling-form features (P0-1, main 1ea46192).

For every rolling-form feature (src.feature_engineering._FORM_FEATURES, both sides) plus
has_ht_history, the value TRAINING assigns a match (build_features with that match appended as a
played row) must equal the value SERVING assigns the same match (build_upcoming_features with it
as an upcoming fixture) - within TOL, and NaN exactly where the other side is NaN.

A training frame that has no HT data at all does not create the HT columns; an absent column is
compared as NaN (serving writes NaN for a stat the history does not carry).

Histories are synthetic and deliberately hostile: alternating venues with venue-specific stats,
uneven fixture counts, one club name in two leagues with interleaved dates, sparse / absent xG,
shots and HT scores, postponed (unscored) rows, board names that differ from the history
spelling, a newly promoted club (incl. one whose name shares a word with a club already in the
league), two matches on one date, a season boundary, and an unsorted history.

    python -m pytest tests/test_p2_parity_adversarial.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import feature_engineering as FE

TOL = 1e-9
L1, L2 = "England Premier League", "England Championship"

FORM_COLS = [c for pair in FE._FORM_FEATURES.values() for c in pair] + list(FE.HT_HISTORY_FEATURES)


# ── synthetic history ─────────────────────────────────────────────────────────

def _row(date, league, home, away, hg, ag, *, hs=10.0, as_=8.0, hst=4.0, ast=3.0, hc=5.0, ac=4.0,
         hf=11.0, af=13.0, hxg=1.3, axg=0.9, hib=7.0, aib=5.0, hht=0.0, aht=0.0):
    tg = (hg + ag) if pd.notna(hg) and pd.notna(ag) else np.nan
    return {"date": pd.Timestamp(date), "league": league, "home_team": home, "away_team": away,
            "home_goals": hg, "away_goals": ag, "total_goals": tg,
            "over25": float(tg > 2.5) if pd.notna(tg) else np.nan,
            "home_shots": hs, "away_shots": as_, "home_sot": hst, "away_sot": ast,
            "home_corners": hc, "away_corners": ac, "home_fouls": hf, "away_fouls": af,
            "home_xg": hxg, "away_xg": axg, "home_insidebox": hib, "away_insidebox": aib,
            "ht_home_goals": hht, "ht_away_goals": aht,
            "odds_over25": 1.9, "odds_under25": 1.95}


def _venue_split(team, league, start, k, opp_prefix, step_days=7):
    """`team` alternates home / away; every stat differs by venue, so a window built from the
    wrong matches (e.g. the last n AWAY games) gives a different number."""
    rows, d0 = [], pd.Timestamp(start)
    for i in range(k):
        d = d0 + pd.Timedelta(days=step_days * i)
        if i % 2 == 0:
            rows.append(_row(d, league, team, f"{opp_prefix}{i}", 3, 1, hs=15, as_=6, hst=7, ast=2, hc=9, ac=2,
                             hf=8, af=14, hxg=2.4, axg=0.5, hib=11, aib=3, hht=2, aht=0))
        else:
            rows.append(_row(d, league, f"{opp_prefix}{i}", team, 2, 0, hs=13, as_=4, hst=6, ast=1, hc=7, ac=1,
                             hf=10, af=16, hxg=1.8, axg=0.2, hib=9, aib=2, hht=0, aht=0))
    return rows


def _steady(team, league, start, k, opp_prefix, hg=1, ag=1, step_days=7, **kw):
    d0 = pd.Timestamp(start)
    return [_row(d0 + pd.Timedelta(days=step_days * i), league, team, f"{opp_prefix}{i}", hg, ag, **kw)
            for i in range(k)]


# ── the comparison ────────────────────────────────────────────────────────────

def _training_row(hist: pd.DataFrame, nxt: dict) -> pd.Series:
    full = pd.concat([hist, pd.DataFrame([nxt])], ignore_index=True)
    out = FE.build_features(full)
    hit = out[(out["home_team"] == nxt["home_team"]) & (out["away_team"] == nxt["away_team"])
              & (out["date"] == nxt["date"]) & (out["league"] == nxt["league"])]
    assert len(hit) == 1
    return hit.iloc[0]


def _serving_row(hist: pd.DataFrame, nxt: dict, board_home=None, board_away=None) -> pd.Series:
    up = {k: nxt[k] for k in ("date", "league", "home_team", "away_team", "odds_over25", "odds_under25")}
    up["home_team"] = board_home or up["home_team"]
    up["away_team"] = board_away or up["away_team"]
    out = FE.build_upcoming_features(pd.DataFrame([up]), hist)
    assert len(out) == 1
    return out.iloc[0]


def _mismatches(tr: pd.Series, sv: pd.Series) -> dict:
    bad = {}
    for col in FORM_COLS:
        assert col in sv.index, f"serving does not write {col}"
        a = tr[col] if col in tr.index else np.nan
        b = sv[col]
        if pd.isna(a) and pd.isna(b):
            continue
        if pd.isna(a) != pd.isna(b) or abs(float(a) - float(b)) > TOL:
            bad[col] = (a, b)
    return bad


def _assert_parity(hist, nxt, **board):
    hist = pd.DataFrame(hist) if not isinstance(hist, pd.DataFrame) else hist
    tr, sv = _training_row(hist, nxt), _serving_row(hist, nxt, **board)
    bad = _mismatches(tr, sv)
    assert not bad, f"train/serve mismatch (training, serving): {bad}"
    return tr, sv


# ── cases ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("k", [1, 4, 5, 6, 11])
@pytest.mark.parametrize("side", ["home", "away"])
def test_alternating_venues_any_count(k, side):
    hist = _venue_split("A", L1, "2026-01-03", k, "X") + _steady("B", L1, "2026-01-04", 3, "Z")
    nxt = _row("2026-06-01", L1, *(("A", "B") if side == "home" else ("B", "A")), 0, 0)
    tr, sv = _assert_parity(hist, nxt)
    col = "home_scored_last5" if side == "home" else "away_scored_last5"
    # the chronological window, recomputed by hand from the venue pattern (home 3, away 0)
    last = [3 if i % 2 == 0 else 0 for i in range(k)][-5:]
    assert sv[col] == pytest.approx(np.mean(last))


def test_uneven_counts_and_a_side_with_no_history():
    hist = _venue_split("A", L1, "2026-01-03", 2, "X") + _venue_split("B", L1, "2025-08-02", 17, "Y")
    _assert_parity(hist, _row("2026-06-01", L1, "A", "B", 1, 1))
    tr, sv = _assert_parity(hist, _row("2026-06-01", L1, "NEWCLUB", "B", 1, 1))
    assert all(pd.isna(sv[c]) for c, _ in FE._FORM_FEATURES.values())
    assert sv["home_has_ht_history"] == 0.0 and tr["home_has_ht_history"] == 0.0


@pytest.mark.parametrize("league", [L1, L2])
def test_one_club_name_in_two_leagues_interleaved(league):
    # "Dual" plays in both leagues on interleaved dates with very different numbers; a window
    # that crosses leagues gives a different value.
    hist = (_steady("Dual", L1, "2026-01-03", 8, "P", hg=4, ag=0, hs=20, hxg=3.0, hht=2)
            + _steady("Dual", L2, "2026-01-06", 8, "Q", hg=0, ag=3, hs=5, hxg=0.3, hht=0)
            + _steady("Opp1", L1, "2026-01-04", 6, "R") + _steady("Opp2", L2, "2026-01-05", 6, "S"))
    opp = "Opp1" if league == L1 else "Opp2"
    tr, sv = _assert_parity(hist, _row("2026-06-01", league, "Dual", opp, 0, 0))
    assert sv["home_scored_last5"] == (4.0 if league == L1 else 0.0)


def test_sparse_xg_shots_and_ht():
    hist = _venue_split("A", L1, "2026-01-03", 9, "X") + _steady("B", L1, "2026-01-04", 9, "Z")
    df = pd.DataFrame(hist)
    a_rows = (df["home_team"] == "A") | (df["away_team"] == "A")
    idx = df.index[a_rows]
    df.loc[idx[-5:], ["home_xg", "away_xg"]] = np.nan                 # A: no xG in its whole window
    df.loc[idx[[-1, -3]], ["home_shots", "away_shots"]] = np.nan      # A: 2 of 5 shot counts missing
    df.loc[idx[[-2, -4]], ["ht_home_goals"]] = np.nan                 # A: half an HT score
    b_idx = df.index[df["home_team"] == "B"]
    df.loc[b_idx[-5:], ["ht_home_goals", "ht_away_goals"]] = np.nan   # B: no HT score in its window
    df.loc[b_idx[-5:-1], ["home_insidebox", "away_insidebox"]] = np.nan
    tr, sv = _assert_parity(df, _row("2026-06-01", L1, "A", "B", 0, 0))
    assert pd.isna(sv["home_xg_last5"])                       # missing stays missing, never 0
    assert sv["away_has_ht_history"] == 0.0 and pd.isna(sv["away_ht_scored_last5"])
    assert sv["home_has_ht_history"] == 1.0


def test_stat_columns_absent_from_history():
    hist = pd.DataFrame(_venue_split("A", L1, "2026-01-03", 7, "X") + _steady("B", L1, "2026-01-04", 7, "Z"))
    hist = hist.drop(columns=["home_xg", "away_xg", "home_insidebox", "away_insidebox",
                              "ht_home_goals", "ht_away_goals"])
    nxt = {k: v for k, v in _row("2026-06-01", L1, "A", "B", 0, 0).items()
           if k not in ("home_xg", "away_xg", "home_insidebox", "away_insidebox", "ht_home_goals", "ht_away_goals")}
    tr, sv = _assert_parity(hist, nxt)
    assert pd.isna(sv["home_xg_last5"]) and sv["home_has_ht_history"] == 0.0


def test_ht_known_only_outside_the_window():
    df = pd.DataFrame(_venue_split("A", L1, "2026-01-03", 10, "X") + _steady("B", L1, "2026-01-04", 10, "Z"))
    late = df["date"] >= "2026-02-05"                         # the last 5 weeks: no HT anywhere
    df.loc[late, ["ht_home_goals", "ht_away_goals"]] = np.nan
    tr, sv = _assert_parity(df, _row("2026-06-01", L1, "A", "B", 0, 0))
    assert sv["home_has_ht_history"] == 0.0 and pd.isna(sv["home_ht_over05_rate"])


def test_postponed_unscored_rows_in_the_past():
    df = pd.DataFrame(_venue_split("A", L1, "2026-01-03", 9, "X") + _steady("B", L1, "2026-01-04", 9, "Z"))
    pp = df.index[(df["home_team"] == "A") | (df["away_team"] == "A")][[-2, -4]]
    df.loc[pp, ["home_goals", "away_goals", "total_goals", "over25", "home_shots", "away_shots",
                "home_sot", "away_sot", "home_corners", "away_corners", "home_fouls", "away_fouls",
                "home_xg", "away_xg", "home_insidebox", "away_insidebox", "ht_home_goals", "ht_away_goals"]] = np.nan
    _assert_parity(df, _row("2026-06-01", L1, "A", "B", 0, 0))


@pytest.mark.parametrize("board,history", [("Wolverhampton Wanderers", "Wolves"),
                                           ("Queens Park Rangers", "QPR"),
                                           ("Cadiz CF", "Cadiz")])
def test_board_name_differs_from_history_spelling(board, history):
    hist = _venue_split(history, L1, "2026-01-03", 8, "X") + _steady("B", L1, "2026-01-04", 8, "Z")
    nxt = _row("2026-06-01", L1, history, "B", 0, 0)           # training knows the history spelling
    tr, sv = _assert_parity(hist, nxt, board_home=board)        # the board carries the OddsAPI one
    assert pd.notna(sv["home_scored_last5"])
    nxt = _row("2026-06-01", L1, "B", history, 0, 0)
    _assert_parity(hist, nxt, board_away=board)


@pytest.mark.parametrize("promoted,board", [("Burnley", "Burnley FC"),
                                            ("Sheffield Weds", "Sheffield Wednesday")])
def test_newly_promoted_club_first_match_in_new_league(promoted, board):
    # Last season in the Championship, first match in the Premier League. Training's (league, team)
    # window is empty; serving must not borrow the Championship rows nor a near-named PL club
    # ("Sheffield Wednesday" must never become "Sheffield United").
    hist = (_venue_split(promoted, L2, "2025-08-09", 12, "C")
            + _steady("Sheffield United", L1, "2025-08-10", 10, "P"))
    nxt = _row("2026-08-15", L1, promoted, "Sheffield United", 0, 0)
    tr, sv = _assert_parity(hist, nxt)
    assert pd.isna(sv["home_scored_last5"]) and sv["home_has_ht_history"] == 0.0
    tr, sv = _assert_parity(hist, nxt, board_home=board)
    assert pd.isna(sv["home_scored_last5"]) and sv["home_has_ht_history"] == 0.0


def test_two_matches_on_one_date_in_history():
    rows = _venue_split("A", L1, "2026-01-03", 6, "X") + _steady("B", L1, "2026-01-04", 6, "Z")
    rows.append(_row("2026-02-07", L1, "A", "Twin", 5, 5, hs=30, hht=3, aht=3))   # same date as A's 6th match
    _assert_parity(rows, _row("2026-06-01", L1, "A", "B", 0, 0))
    _assert_parity(rows, _row("2026-06-01", L1, "B", "A", 0, 0))


def test_first_match_of_a_season_window_spans_the_break():
    rows = (_venue_split("A", L1, "2025-03-01", 10, "X")       # ends 2025-05-03
            + _steady("B", L1, "2025-03-02", 10, "Z"))
    tr, sv = _assert_parity(rows, _row("2025-08-16", L1, "A", "B", 0, 0))
    assert pd.notna(sv["home_scored_last5"])                  # both paths carry form over the break


def test_unsorted_history_with_shared_dates():
    rows = (_venue_split("A", L1, "2026-01-03", 9, "X") + _venue_split("B", L1, "2026-01-03", 9, "Y")
            + _steady("C", L1, "2026-01-03", 9, "Z"))
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=7).reset_index(drop=True)
    _assert_parity(df, _row("2026-06-01", L1, "A", "B", 0, 0))
    _assert_parity(df, _row("2026-06-01", L1, "C", "A", 0, 0))


# ── known limits, pinned so a change is noticed ────────────────────────────────

def test_serving_window_ignores_the_fixture_date():
    """Serving takes each team's LAST n history rows, whatever their date; training takes the n
    rows before the fixture. They agree only while the history holds nothing dated on/after the
    fixture. A rescheduled (unscored) row dated after the fixture breaks parity - this pins that
    so it is never silently relied on. Committed history (output/fd_history.parquet,
    af_history.parquet at data-truth-100 e07a4591) holds no row dated after 2026-11-02."""
    df = pd.DataFrame(_venue_split("A", L1, "2026-01-03", 9, "X") + _steady("B", L1, "2026-01-04", 9, "Z"))
    resched = _row("2026-06-20", L1, "A", "Late", np.nan, np.nan, hs=np.nan, as_=np.nan, hst=np.nan,
                   ast=np.nan, hc=np.nan, ac=np.nan, hf=np.nan, af=np.nan, hxg=np.nan, axg=np.nan,
                   hib=np.nan, aib=np.nan, hht=np.nan, aht=np.nan)
    df = pd.concat([df, pd.DataFrame([resched])], ignore_index=True)
    tr = _training_row(df, _row("2026-06-01", L1, "A", "B", 0, 0))
    sv = _serving_row(df, _row("2026-06-01", L1, "A", "B", 0, 0))
    assert _mismatches(tr, sv), "serving now cuts the window at the fixture date - update this pin"


@pytest.mark.parametrize("hist_role,fx_role,diverges", [("away", "home", True), ("home", "away", False),
                                                         ("home", "home", False), ("away", "away", False)])
def test_history_match_on_the_fixture_day(hist_role, fx_role, diverges):
    """A club with a history match on the SAME day as the served fixture. Training orders a club's
    same-day rows home-role first, so a home fixture sorts BEFORE a same-day away history match and
    its window excludes it; serving always includes it. Pinned: only (history away, fixture home)
    diverges. The merged history at data-truth-100 e07a4591 holds 0 (league, club, day) doubles."""
    rows = _venue_split("A", L1, "2026-01-03", 6, "X") + _steady("B", L1, "2026-01-04", 6, "Z")
    rows.append(_row("2026-06-01", L1, *(("A", "Twin") if hist_role == "home" else ("Twin", "A")), 5, 5,
                     hs=30, as_=30))
    nxt = _row("2026-06-01", L1, *(("A", "B") if fx_role == "home" else ("B", "A")), 0, 0)
    df = pd.DataFrame(rows)
    assert bool(_mismatches(_training_row(df, nxt), _serving_row(df, nxt))) is diverges
