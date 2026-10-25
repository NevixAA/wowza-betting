"""Yesterday's props results show exactly the props the morning album showed (2026-10-25).

Haaland to score was #6 of 12 on the 2026-10-24 album at 2.14; the results picture re-ranked every
settled prop by edge at the final price (1.70) and he dropped out of the top 12.
"""
import pandas as pd

from telegram_bot import daily_pack as D


def _tips(n_others=15):
    rows = [{"date": "2026-10-24", "kickoff_utc": "2026-10-24 16:30", "league": "Premier League",
             "match": "Aston Villa vs Manchester City", "player_id": 1100, "player_name": "E. Haaland",
             "market": "goals", "model_prob": 0.679, "market_odds": 2.14}]
    for i in range(n_others):                  # 5 props with a bigger edge than Haaland, 10 smaller: he is #6
        rows.append({"date": "2026-10-24", "kickoff_utc": "2026-10-24 14:00", "league": "La Liga",
                     "match": f"Home{i} vs Away{i}", "player_id": 2000 + i, "player_name": f"P{i}",
                     "market": "goals", "model_prob": 0.6, "market_odds": 3.0 + i if i < 5 else 1.5})
    return pd.DataFrame(rows)


def _ledger():
    rows = [{"match_date": "2026-10-24", "league": "Premier League", "home_team": "Aston Villa",
             "away_team": "Manchester City", "player_name": "E. Haaland", "player_id": 1100.0,
             "market": "goals", "model_prob": 0.68, "market_odds": 1.70, "result": "WIN"}]
    for i in range(15):
        rows.append({"match_date": "2026-10-24", "league": "La Liga", "home_team": f"Home{i}",
                     "away_team": f"Away{i}", "player_name": f"P{i}", "player_id": float(2000 + i),
                     "market": "goals", "model_prob": 0.9, "market_odds": 10.0 + i, "result": "LOSS"})
    return pd.DataFrame(rows)


def _setup(monkeypatch, tmp_path):
    log = tmp_path / "props_shown_log.csv"
    monkeypatch.setattr(D, "PROPS_SHOWN_LOG", log)
    frames = {"player_tips.csv": _tips(), "player_ledger.csv": _ledger()}
    monkeypatch.setattr(D, "_read", lambda name: frames.get(name, pd.DataFrame()).copy())
    captured = {}
    monkeypatch.setattr(D, "render_table", lambda title, cols, rows, **k: captured.setdefault("rows", rows) and [b"png"])
    return log, captured


def test_freeze_logs_what_was_shown_once(monkeypatch, tmp_path):
    log, _ = _setup(monkeypatch, tmp_path)
    D.props_image("2026-10-24", freeze=False)
    assert not log.exists()                                    # a preview never writes
    D.props_image("2026-10-24", freeze=True)
    first = pd.read_csv(log)
    assert "E. Haaland" in set(first["player_name"])
    D.props_image("2026-10-24", freeze=True)                   # a second send the same day changes nothing
    assert pd.read_csv(log).equals(first)


def test_results_show_the_shown_props_at_the_shown_odds(monkeypatch, tmp_path):
    log, captured = _setup(monkeypatch, tmp_path)
    D.props_image("2026-10-24", freeze=True)
    shown = pd.read_csv(log)
    captured.clear()
    D.props_results_image("2026-10-24")
    cells = [r["cells"] for r in captured["rows"] if "cells" in r]
    names = [c[0] for c in cells]
    assert set(names) == set(shown["player_name"])             # exactly the album's props, nothing re-ranked
    h = cells[names.index("E. Haaland")]
    assert h[4] == "2.14" and h[5] == "WON"                    # the odds shown in the morning, not the close


def test_no_log_falls_back_to_the_old_ranking(monkeypatch, tmp_path):
    _, captured = _setup(monkeypatch, tmp_path)
    D.props_results_image("2026-10-24")
    assert any("cells" in r for r in captured["rows"])
