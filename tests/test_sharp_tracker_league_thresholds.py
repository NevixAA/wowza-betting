"""Owner 2026-11-15 item 3: the live sharp tracker judges a move against PER-LEAGUE thresholds learned from
strictly earlier fixtures; no global constant; too little history -> no signal."""
from collections import Counter
from datetime import datetime, timedelta
import inspect

from src import sharp_tracker as st

NOW = datetime(2026, 11, 20, 12, 0)


def _fx(league, ko, over_open, over_now, lead_now=6.0, n_books=8, extra=()):
    """A totals record: open at 72 h, then optional extra (lead, over) snapshots, then the current one."""
    def snap(lead, over):
        return {"at": (ko - timedelta(hours=lead)).strftime("%Y-%m-%d %H:%M"),
                "odds": {"odds_over": over, "odds_under": round(1 / (1.05 - 1 / over), 4)}, "n_books": n_books}
    snaps = [snap(72, over_open)] + [snap(l, o) for l, o in extra] + [snap(lead_now, over_now)]
    return {"league": league, "home": "H", "away": "A", "date": ko.strftime("%Y-%m-%d %H:%M"),
            "market": "totals", "opening": snaps[0]["odds"], "snapshots": snaps}


def _history(league, n_past, past_move_over, upcoming_now, start=datetime(2026, 6, 1, 18, 0)):
    h = {}
    for i in range(n_past):
        ko = start + timedelta(days=i % 150, hours=i // 150)
        # small, spread past moves: open 2.00 -> 2.00 - k*0.005
        h[f"p{i}_totals"] = _fx(league, ko, 2.0, 2.0 - past_move_over * ((i % 10) + 1) / 10, lead_now=6.0)
    ko = NOW + timedelta(hours=6)
    h["up_totals"] = _fx(league, ko, 2.0, upcoming_now, lead_now=6.0)
    return h


def test_no_global_constants_left():
    src = inspect.getsource(st)
    for name in ("DRIFT_SHARP_PCT =", "DRIFT_STRONG_PCT =", "STEAM_PCT ="):
        assert name not in src


def test_too_little_league_history_gives_no_signal():
    h = _history("L", st.MIN_HISTORY - 1, 0.02, 1.70)          # a huge move, but 99 earlier fixtures
    counts = Counter()
    tips = st._build_tips(h, NOW, counts)
    assert tips.empty and counts.get("no_league_history", 0) >= 1


def test_enough_history_gives_a_league_relative_signal():
    h = _history("L", st.MIN_HISTORY + 20, 0.02, 1.80)          # past moves small, this one large
    tips = st._build_tips(h, NOW, Counter())
    over = tips[tips["market"] == "O/U 2.5 OVER"]
    assert len(over) == 1 and over["signal"].iloc[0] in ("STRONG", "STEAM_STRONG")
    assert over["threshold_n"].iloc[0] >= st.MIN_HISTORY
    under = tips[tips["market"] == "O/U 2.5 UNDER"]
    assert under["signal"].iloc[0] == "FADING"


def test_same_move_is_a_signal_in_a_calm_league_and_not_in_a_volatile_one():
    calm = _history("Calm", 130, 0.01, 1.95)
    wild = _history("Wild", 130, 0.20, 1.95, start=datetime(2026, 6, 1, 20, 0))
    wild = {("w" + k): v for k, v in wild.items()}
    t = st._build_tips({**calm, **wild}, NOW, Counter())
    assert set(t[t["market"] == "O/U 2.5 OVER"]["league"]) == {"Calm"}


def test_thresholds_use_strictly_earlier_dates_and_kicked_off_fixtures_only():
    h = _history("L", st.MIN_HISTORY + 20, 0.02, 1.80)
    th = st.LeagueThresholds(h, NOW)
    key = [k for k in th.moves if k[0] == "L"][0]
    days = [d for d, _ in th.moves[key]]
    ko = NOW + timedelta(hours=6)
    assert all(d < ko.date() for d in days)                     # the upcoming fixture is not learned from
    assert th._q(th.moves[key], min(days), (0.75,)) is None     # nothing strictly before the first day


def test_outside_the_window_there_is_no_signal():
    h = _history("L", st.MIN_HISTORY + 20, 0.02, 1.80)
    rec = h["up_totals"]
    rec["snapshots"] = rec["snapshots"][-1:]                   # no open snapshot near 72 h
    rec["snapshots"].insert(0, {**rec["snapshots"][0], "at": (datetime.strptime(rec["date"], "%Y-%m-%d %H:%M")
                                                            - timedelta(hours=100)).strftime("%Y-%m-%d %H:%M")})
    counts = Counter()
    t = st._build_tips(h, NOW, counts)
    assert t.empty and counts.get("no_window") == 1
