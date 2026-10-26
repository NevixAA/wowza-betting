"""player_history integrity (owner-approved 2026-10-26): frozen window, point-in-time injury
features, the push guard and the shared writer group. See player_model/history_integrity.py."""
import datetime as dt
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from player_model import history_integrity as HI
from player_model import pipeline as P

ROOT = Path(__file__).resolve().parents[1]
TODAY = pd.Timestamp("2026-10-26")


def _rows(dates, pid=1, start_fid=1, **cols):
    out = []
    for i, d in enumerate(dates):
        r = {"fixture_id": start_fid + i, "player_id": pid, "date": pd.Timestamp(d), "league": "Championship",
             "goals": 0, "minutes": 90, "opp_strength_index": 0.5,
             "chronic_injury_risk": 0.0, "days_since_last_injury": 100.0, "return_from_injury_flag": 0.0}
        r.update(cols)
        out.append(r)
    return pd.DataFrame(out)


def _by_fid(df):
    return df.set_index("fixture_id").sort_index()


# ── 1. frozen window ─────────────────────────────────────────────────────────────────────────

def test_constants_are_named_and_documented():
    assert HI.FROZEN_AFTER_DAYS == 14
    assert HI.RAW_FACTS_MUTABLE_AFTER_FREEZE == ()      # evidence: no raw fact changed after 14 days
    assert HI.INJURY_REFRESH_WINDOW_DAYS == 60
    assert set(HI.INJURY_COLUMNS) == {"chronic_injury_risk", "days_since_last_injury",
                                      "return_from_injury_flag"}


def test_frozen_rows_are_immutable_and_window_rows_refresh():
    existing = _rows(["2026-08-01", "2026-10-01", "2026-10-12", "2026-10-20"])
    fresh = existing.copy()
    fresh["opp_strength_index"] = 0.9          # derived feature re-derived by the collect
    fresh["goals"] = 2                         # and a raw fact "corrected"
    out, st = HI.merge_history(existing, fresh, today=TODAY)
    o = _by_fid(out)
    # 08-01 and 10-01 are > 14 days old: untouched
    assert (o.loc[[1, 2], "opp_strength_index"] == 0.5).all() and (o.loc[[1, 2], "goals"] == 0).all()
    # 10-12 (exactly 14 days) and 10-20 are inside the window: refreshed
    assert (o.loc[[3, 4], "opp_strength_index"] == 0.9).all() and (o.loc[[3, 4], "goals"] == 2).all()
    assert st["frozen"] == 2 and st["refreshed"] == 2 and st["added"] == 0
    assert len(out) == len(existing)


def test_frozen_row_with_unparseable_date_is_frozen():
    existing = _rows(["2026-10-20"])
    existing["date"] = "not a date"
    fresh = existing.copy()
    fresh["opp_strength_index"] = 0.9
    out, st = HI.merge_history(existing, fresh, today=TODAY)
    assert out["opp_strength_index"].iloc[0] == 0.5 and st["frozen"] == 1


def test_new_rows_are_added_whatever_their_age():
    existing = _rows(["2026-10-20"])
    fresh = pd.concat([_rows(["2026-10-20"]), _rows(["2023-03-01", "2026-10-25"], start_fid=10)],
                      ignore_index=True)
    out, st = HI.merge_history(existing, fresh, today=TODAY)
    assert sorted(out["fixture_id"]) == [1, 10, 11]
    assert st["added"] == 2
    # a collect never removes a row
    out2, _ = HI.merge_history(out, _rows(["2026-10-25"], start_fid=11), today=TODAY)
    assert sorted(out2["fixture_id"]) == [1, 10, 11]


def test_raw_fact_policy_only_named_raw_columns_can_change_after_freeze():
    existing = _rows(["2026-07-01"])
    fresh = existing.copy()
    fresh["goals"] = 3
    fresh["opp_strength_index"] = 0.9
    # the shipped policy: nothing changes
    out, st = HI.merge_history(existing, fresh, today=TODAY)
    assert out["goals"].iloc[0] == 0 and out["opp_strength_index"].iloc[0] == 0.5
    # the mechanism, if evidence ever justifies a raw-fact exception: ONLY that column moves
    out, st = HI.merge_history(existing, fresh, today=TODAY, mutable_after_freeze=("goals",))
    assert out["goals"].iloc[0] == 3 and out["opp_strength_index"].iloc[0] == 0.5
    assert st["frozen_raw_updates"] == 1
    # a null fresh raw value never erases a stored one
    fresh["goals"] = np.nan
    out, _ = HI.merge_history(existing, fresh, today=TODAY, mutable_after_freeze=("goals",))
    assert out["goals"].iloc[0] == 0


def test_collect_never_writes_injury_columns():
    existing = _rows(["2026-10-20", "2026-01-10"], days_since_last_injury=12.0, chronic_injury_risk=1.0)
    fresh = existing.copy()
    fresh[["chronic_injury_risk", "days_since_last_injury", "return_from_injury_flag"]] = [0.0, 365.0, 0.0]
    out, _ = HI.merge_history(existing, fresh, today=TODAY)
    assert (out["days_since_last_injury"] == 12.0).all() and (out["chronic_injury_risk"] == 1.0).all()
    # a brand-new row gets NaN, not the 365/0/0 default
    out, _ = HI.merge_history(existing, _rows(["2026-10-25"], start_fid=50), today=TODAY)
    assert out.loc[out.fixture_id == 50, "days_since_last_injury"].isna().all()


def test_mode_collect_applies_the_frozen_window(tmp_path, monkeypatch):
    path = tmp_path / "player_history.parquet"
    stored = _rows(["2026-06-01", "2026-10-20"], days_since_last_injury=40.0)
    P._save_history(stored, path)
    fetched = pd.concat([_rows(["2026-06-01", "2026-10-20"]), _rows(["2026-10-25"], start_fid=9)],
                        ignore_index=True)
    fetched["opp_strength_index"] = 0.9
    fetched[["chronic_injury_risk", "days_since_last_injury", "return_from_injury_flag"]] = [0.0, 365.0, 0.0]
    monkeypatch.setattr(P, "HISTORY_CACHE", path)
    monkeypatch.setattr(P, "collect_match_history", lambda *a, **k: [{"x": 1}])
    monkeypatch.setattr(P, "enrich_sp_events", lambda r: r)
    monkeypatch.setattr(P, "enrich_league_quality", lambda r: r)
    monkeypatch.setattr(P, "build_features", lambda r: fetched.copy())
    monkeypatch.setattr(HI, "_utc_today", lambda: TODAY)
    P.mode_collect()
    o = _by_fid(pd.read_parquet(path))
    assert o.loc[1, "opp_strength_index"] == pytest.approx(0.5)        # frozen
    assert o.loc[2, "opp_strength_index"] == pytest.approx(0.9)        # in window
    assert (o.loc[[1, 2], "days_since_last_injury"] == 40.0).all()     # injury kept
    assert np.isnan(o.loc[9, "days_since_last_injury"])                # new row: NaN, not 365
    assert len(o) == 3


# ── 2. point-in-time injury features ─────────────────────────────────────────────────────────

ENTRIES = [
    {"start": "2025-01-10", "end": "2025-02-01"},
    {"start": "2025-09-01", "end": "2025-09-20"},
    {"start": "2026-03-01", "end": "2026-04-15"},
    {"start": "2026-10-01", "end": ""},                # ongoing today
]


def test_injury_value_is_as_of_the_match_date_not_today():
    f = HI.injury_features_asof(ENTRIES, "2025-10-01")
    assert f["days_since_last_injury"] == 11.0                     # 2025-09-20 -> 10-01
    assert f["chronic_injury_risk"] == 2.0                         # starts 2025-01-10, 2025-09-01
    assert f["return_from_injury_flag"] == 1.0
    f = HI.injury_features_asof(ENTRIES, "2024-06-01")             # before any spell
    assert f == {"chronic_injury_risk": 0.0, "days_since_last_injury": 365.0,
                 "return_from_injury_flag": 0.0}
    f = HI.injury_features_asof(ENTRIES, "2026-05-01")
    assert f["days_since_last_injury"] == 16.0
    assert f["chronic_injury_risk"] == 2.0                         # 2025-09-01, 2026-03-01
    # today (2026-10-26) would say 194 days -- the old stamped value -- and must NOT appear
    assert HI.injury_features_asof(ENTRIES, "2025-10-01")["days_since_last_injury"] != 194.0


def test_spell_starting_on_or_after_the_match_is_invisible():
    f = HI.injury_features_asof([{"start": "2025-05-05", "end": "2025-05-20"}], "2025-05-05")
    assert f["chronic_injury_risk"] == 0.0 and f["days_since_last_injury"] == 365.0


def _random_entries(rng, n, lo=pd.Timestamp("2022-01-01"), span=1700):
    out = []
    for _ in range(n):
        s = lo + pd.Timedelta(days=int(rng.integers(0, span)))
        dur = int(rng.integers(1, 120))
        out.append({"start": str(s.date()), "end": str((s + pd.Timedelta(days=dur)).date())})
    return out


def test_no_injury_field_uses_an_event_from_after_the_match():
    rng = np.random.default_rng(11)
    for _ in range(400):
        entries = _random_entries(rng, int(rng.integers(0, 6)))
        m = pd.Timestamp("2022-06-01") + pd.Timedelta(days=int(rng.integers(0, 1500)))
        ev = HI.injury_evidence(entries, m)
        if ev["last_end"] is not None:
            assert ev["last_end"] < m.date()
        assert all(s < m.date() for s in ev["starts"])
        f = HI.injury_features_asof(entries, m)
        # brute force, from the definition
        ends = [dt.date.fromisoformat(e["end"]) for e in entries
                if dt.date.fromisoformat(e["start"]) < m.date() and dt.date.fromisoformat(e["end"]) < m.date()]
        want = float((m.date() - max(ends)).days) if ends else 365.0
        assert f["days_since_last_injury"] == want
        n = sum(1 for e in entries
                if m.date() - dt.timedelta(days=365) <= dt.date.fromisoformat(e["start"]) < m.date())
        assert f["chronic_injury_risk"] == float(n)


def test_apply_injury_pit_rows_and_failed_players(tmp_path):
    df = pd.concat([_rows(["2025-10-01", "2026-05-01"], pid=1),
                    _rows(["2026-05-01"], pid=2, start_fid=10)], ignore_index=True)
    df[list(HI.INJURY_COLUMNS)] = np.nan
    n = HI.apply_injury_pit(df, {1: ENTRIES})          # player 2's fetch failed: absent
    assert n == 2
    assert df.loc[0, "days_since_last_injury"] == 11.0 and df.loc[1, "days_since_last_injury"] == 16.0
    assert df.loc[2, list(HI.INJURY_COLUMNS)].isna().all()


def test_refresh_writes_only_the_recent_window_point_in_time(tmp_path, monkeypatch):
    path = tmp_path / "player_history.parquet"
    today = dt.date.today()
    old_d, new_d = today - dt.timedelta(days=400), today - dt.timedelta(days=10)
    df = pd.concat([_rows([old_d, new_d], pid=7), _rows([new_d], pid=8, start_fid=20)], ignore_index=True)
    df["days_since_last_injury"] = 999.0               # what a today-stamp would have left
    P._save_history(df, path)
    spell_end = today - dt.timedelta(days=30)
    sidelined = {7: [{"start": str(spell_end - dt.timedelta(days=20)), "end": str(spell_end)}]}
    import player_model.data_fetcher as DF
    monkeypatch.setattr(P, "HISTORY_CACHE", path)
    monkeypatch.setattr(P, "_match_day_gate_skips", lambda: False)
    monkeypatch.setattr(DF, "fetch_all_player_sidelined", lambda d, **k: sidelined)
    monkeypatch.setattr(P.config, "PROP_LEAGUES", {"Championship": 40})
    P.mode_enrich_sidelined_live()
    o = _by_fid(pd.read_parquet(path))
    assert o.loc[1, "days_since_last_injury"] == 999.0                       # 400 days old: untouched
    assert o.loc[2, "days_since_last_injury"] == float((new_d - spell_end).days)   # 20, as of match
    assert o.loc[2, "return_from_injury_flag"] == 1.0
    assert o.loc[20, "days_since_last_injury"] == 999.0                      # fetch failed: untouched


def test_failed_sidelined_fetch_is_absent_not_empty(monkeypatch):
    import player_model.data_fetcher as DF

    def fake(pid):
        if pid == 2:
            raise RuntimeError("API-Football 500")
        return [] if pid == 1 else [{"start": "2026-01-01", "end": "2026-02-01", "reason": "x"}]
    monkeypatch.setattr(DF, "fetch_player_sidelined", fake)
    out = DF.fetch_all_player_sidelined(pd.DataFrame({"player_id": [1, 2, 3]}), max_workers=1)
    assert set(out) == {1, 3} and out[1] == []


# ── one-off repair: reconstruction from a today-stamp ───────────────────────────────────────

def _old_live_stamp(entries, today):
    """The pre-2026-10-26 enrich-sidelined-live formula, verbatim in effect."""
    t = today.date()
    year_ago = t - dt.timedelta(days=365)
    recent = [e for e in entries if e.get("start") and dt.date.fromisoformat(e["start"][:10]) >= year_ago]
    past = sorted([e for e in entries if e.get("end") and dt.date.fromisoformat(e["end"][:10]) < t],
                  key=lambda x: x["end"], reverse=True)
    days = float((t - dt.date.fromisoformat(past[0]["end"][:10])).days) if past else 365.0
    return float(len(recent)), days


def test_reconstruction_from_a_today_stamp_equals_point_in_time():
    rng = np.random.default_rng(5)
    hits_d = hits_c = 0
    for _ in range(3000):
        D = pd.Timestamp("2026-10-24")
        entries = [e for e in _random_entries(rng, int(rng.integers(0, 5)), span=1750)
                   if pd.Timestamp(e["start"]) < D]                          # data as of D
        m = D - pd.Timedelta(days=int(rng.integers(0, 1500)))
        c, v = _old_live_stamp(entries, D)
        days, chronic = HI.reconstruct_from_stamp(m, D, c, v)
        truth = HI.injury_features_asof(entries, m)
        if v == 365.0 and any(pd.Timestamp(e["end"]) == D - pd.Timedelta(days=365) for e in entries):
            continue                                   # exact-365 collision with the sentinel
        if days is not None:
            hits_d += 1
            assert days == truth["days_since_last_injury"]
        if chronic is not None:
            hits_c += 1
            assert chronic == truth["chronic_injury_risk"]
    assert hits_d > 1000 and hits_c > 500       # it recovers a lot, and every recovery is exact


def test_reconstruction_refuses_stamps_older_than_the_match():
    assert HI.reconstruct_from_stamp("2026-10-01", "2026-09-01", 0, 365) == (None, None)


# ── growth replay ────────────────────────────────────────────────────────────────────────────

def test_growth_replay_daily_collect_and_weekly_refresh_touch_only_recent_parts(tmp_path):
    root = tmp_path / "player_history.parquet"
    rng = np.random.default_rng(3)
    days = pd.date_range("2025-11-01", "2026-10-10", freq="D")
    hist = pd.DataFrame({"fixture_id": np.arange(len(days) * 4), "player_id": np.tile(np.arange(4), len(days)),
                         "date": np.repeat(days, 4), "league": "Championship",
                         "goals": rng.integers(0, 3, len(days) * 4),
                         "opp_strength_index": rng.random(len(days) * 4),
                         "chronic_injury_risk": 0.0, "days_since_last_injury": 365.0,
                         "return_from_injury_flag": 0.0})
    P._save_history(hist, root)
    n_parts = len(list(root.glob("*.parquet")))
    state = hist
    for today in pd.date_range("2026-10-11", "2026-10-25", freq="D"):
        # a collect re-derives every row of the last ~120 days and adds today's matches
        recent = state[pd.to_datetime(state["date"]) >= today - pd.Timedelta(days=120)].copy()
        recent["opp_strength_index"] = rng.random(len(recent))
        new = pd.DataFrame({"fixture_id": state["fixture_id"].max() + 1 + np.arange(4), "player_id": np.arange(4),
                            "date": today, "league": "Championship", "goals": 1,
                            "opp_strength_index": 0.5})
        state, _ = HI.merge_history(state, pd.concat([recent, new], ignore_index=True), today=today)
        st = P._save_history(state, root)
        months = {f"month_{m}.parquet" for m in
                  {today.strftime("%Y-%m"), (today - pd.Timedelta(days=HI.FROZEN_AFTER_DAYS)).strftime("%Y-%m")}}
        assert set(st["written"]) <= months, (today, st["written"])
    # the weekly refresh: fills only the window
    mask = pd.to_datetime(state["date"]) >= pd.Timestamp("2026-10-25") - pd.Timedelta(days=HI.INJURY_REFRESH_WINDOW_DAYS)
    n = HI.apply_injury_pit(state, {p: [] for p in range(4)}, mask)
    st = P._save_history(state, root)
    assert n == int(mask.sum()) and len(st["written"]) <= 3 < n_parts


# ── 4. race: shared writer group, fresh checkout, push guard ────────────────────────────────

WRITERS = [ROOT / ".github/workflows/player_history_extend.yml", ROOT / ".github/workflows/injury_refresh.yml"]


@pytest.mark.parametrize("wf", WRITERS, ids=lambda p: p.name)
def test_writers_share_one_concurrency_group_and_check_out_main(wf):
    import yaml
    y = yaml.safe_load(wf.read_text(encoding="utf-8"))
    assert y["concurrency"]["group"] == "player-history-writer"
    assert y["concurrency"]["cancel-in-progress"] is False
    steps = next(iter(y["jobs"].values()))["steps"]
    co = [s for s in steps if str(s.get("uses", "")).startswith("actions/checkout")][0]
    assert co["with"]["ref"] == "main"


@pytest.mark.parametrize("wf", WRITERS, ids=lambda p: p.name)
def test_writers_stage_the_directory_with_A_and_guard_before_push(wf):
    text = wf.read_text(encoding="utf-8")
    assert "git add -A -f player_history.parquet" in text
    g = text.index("python -m player_model.history_integrity guard origin/main")
    p = text.index("git push", g)
    assert text.rindex("git pull --rebase", 0, g) < g < p


def test_lost_keys():
    a = pd.DataFrame({"fixture_id": [1, 2, 3], "player_id": [1, 1, 1]})
    assert len(HI.lost_keys(a, a)) == 0
    assert HI.lost_keys(a, a.iloc[:2]).to_dict("records") == [{"fixture_id": 3, "player_id": 1}]


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_guard_cli_refuses_a_push_that_drops_committed_rows(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    root = tmp_path / "player_history.parquet"
    base = _rows(["2026-10-20", "2026-10-21", "2026-10-22"])
    P._save_history(base, root)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "base")
    _git(tmp_path, "branch", "origin_main")
    monkeypatch.chdir(tmp_path)
    assert HI.guard_cli("origin_main") == 0                       # nothing changed
    P._save_history(pd.concat([base, _rows(["2026-10-23"], start_fid=9)], ignore_index=True), root)
    assert HI.guard_cli("origin_main") == 0                       # rows only added
    P._save_history(base.iloc[:2], root)
    assert HI.guard_cli("origin_main") == 1                       # a committed row would vanish


# ── the real, repaired history ───────────────────────────────────────────────────────────────

def test_real_history_injury_columns_are_internally_consistent():
    path = ROOT / "player_history.parquet"
    if not path.exists():
        pytest.skip("no player_history in this checkout")
    df = pd.read_parquet(path, columns=list(HI.INJURY_COLUMNS) + ["player_id"])
    d, f = df["days_since_last_injury"], df["return_from_injury_flag"]
    assert (d.isna() == f.isna()).all()
    assert ((f == 1) == ((d > 0) & (d <= HI.RETURN_WINDOW_DAYS))).all()
    # a today-stamp makes every row of a player identical; point-in-time values do not
    x = df[d.notna() & (d != HI.NO_INJURY_DAYS)]
    share = x.groupby("player_id")["days_since_last_injury"].transform(
        lambda s: s.value_counts().iloc[0]) / x.groupby("player_id")["days_since_last_injury"].transform("size")
    assert float(share.mean()) < 0.5        # was 0.9956 before the 2026-10-26 repair
