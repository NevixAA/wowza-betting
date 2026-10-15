"""Fantasy V3 port (2026-10-15): fallback rules, the projection identity and the live squad panel."""
import json

import numpy as np
import pandas as pd
import pytest

from player_model import fantasy_v3 as V


def _write(tmp_path, monkeypatch, age_hours):
    csv, meta = tmp_path / "in.csv", tmp_path / "meta.json"
    monkeypatch.setattr(V, "OUT_CSV", csv)
    monkeypatch.setattr(V, "OUT_META", meta)
    pd.DataFrame([{"player_id": 1, "p_goal_v3": 0.30, "p_assist_v3": 0.20,
                   "xm_start": 0.8, "xm_m60": 0.6}]).to_csv(csv, index=False)
    ts = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=age_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta.write_text(json.dumps({"generated_at": ts}))


def _pl():
    return pd.DataFrame({"player_id": [1, 2], "p_goal": [0.5, 0.4], "p_assist": [0.1, 0.1],
                         "appearance": [2.0, 1.0]})


def test_fresh_inputs_swap_in_and_keep_v9(tmp_path, monkeypatch):
    monkeypatch.delenv("FANTASY_MODEL", raising=False)
    _write(tmp_path, monkeypatch, age_hours=2)
    out = V.apply_attack_and_minutes(_pl())
    a, b = out.iloc[0], out.iloc[1]
    assert a["fantasy_model"] == "v3" and a["p_goal"] == 0.30 and a["p_goal_v9"] == 0.5
    # (2*0.6 + (0.8-0.6)) / 0.8 = 1.75 conditional appearance points, P(start) 0.8
    assert a["appearance"] == pytest.approx(1.75) and a["_v3_start"] == pytest.approx(0.8)
    assert b["fantasy_model"] == "v9" and b["p_goal"] == 0.4 and np.isnan(b["_v3_start"])
    # the identity the study validated: (appearance + attack) x P(start) = app_v3 + P(start) x attack
    att = 0.30 * 5 + 0.20 * 3
    assert (a["appearance"] + att) * 0.8 == pytest.approx((2 * 0.6 + 0.2) + 0.8 * att)


def test_stale_or_switched_off_falls_back(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, age_hours=40)
    assert V.load_inputs() is None
    _write(tmp_path, monkeypatch, age_hours=1)
    monkeypatch.setenv("FANTASY_MODEL", "v9")
    assert V.load_inputs() is None
    out = V.apply_attack_and_minutes(_pl())
    assert (out["fantasy_model"] == "v9").all() and out["p_goal"].tolist() == [0.5, 0.4]


def test_missing_meta_is_stale(tmp_path, monkeypatch):
    monkeypatch.delenv("FANTASY_MODEL", raising=False)
    monkeypatch.setattr(V, "OUT_META", tmp_path / "nope.json")
    assert V.load_inputs() is None


def test_live_panel_extends_current_club_and_adds_next_row():
    from player_model.v3 import minutes as M
    days = pd.date_range("2026-09-01", periods=10, freq="7D")
    rows = []
    for i, d in enumerate(days):
        for pid in (1, 2):
            if pid == 1 and i >= 6:              # player 1 stops appearing after match 6
                continue
            rows.append({"fixture_id": 100 + i, "player_id": pid, "date": d, "league": "L", "team": "A",
                         "opponent": "B", "is_home": 1, "minutes": 90, "started": 1, "position": "F", "age": 25})
    h = pd.DataFrame(rows)
    p = M.build_panel(h, asof=days[-1] + pd.Timedelta(days=4))
    p1 = p[p["player_id"] == 1]
    assert (p1["fixture_id"] < 0).sum() == 1                       # one synthetic next row
    ext = p1[(p1["fixture_id"] > 0) & ~p1["in_tenure"]]
    assert len(ext) == 4 and (ext["appeared"] == 0).all()           # 4 missed matches seen as absences
    assert p1.loc[p1["in_tenure"], "fixture_id"].max() == 105       # training rows = the study's tenure
    f = M.add_features(p)
    nxt = f[(f["player_id"] == 1) & (f["fixture_id"] < 0)].iloc[0]
    assert nxt["appeared_l3"] == 0                                 # absent in his club's last 3
