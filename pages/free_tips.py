"""Today's free tips — the FREE tier's daily 2 (owner brief 2026-10-18).

THE TOP 2 OF "BEST BETS" (src/top_picks.py): most expected profit among tips with at least a 45% chance
to win, P(win) learned from past results - the same list the Telegram album and the Pro Tips page lead
with, frozen each morning and scored. Before the album freezes today's lists, the same list is shown
as a preview. If Best bets is empty, the earlier rule is the fallback: tips the approved bankroll
policy stakes, ranked by the track's evidence (dashboard_access.free_tip_pick). Never more than
FREE_TIPS_PER_DAY rows reach this page.

Visual redesign 2026-10-25 (stage 3): shared page header and one selection card per tip (the same card the
Tips page uses: match, pick, price, chance; the legacy tier last and neutral). Same rows, same values.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from dashboard_access import FREE_TIPS_PER_DAY, current_tier, free_tip_pick  # noqa: E402
import dashboard_ui as ui  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MARKET = {"OU25": "Goals 2.5", "OVER15": "Over 1.5 goals", "OVER35": "Over 3.5 goals", "BTTS": "Both teams to score"}

ui.page_header("Today's free tips", f"{FREE_TIPS_PER_DAY} tips a day: the top of our Best bets list — the most "
               "expected profit among tips with at least a 45% chance. Wowza Pro shows all three top-picks lists and "
               "every tip.", eyebrow_text="Today · free")


@st.cache_data(ttl=300)
def _best_bets(day: str) -> tuple[pd.DataFrame, int]:
    from src import top_picks as TP
    fz = TP.frozen(day)
    lists = fz if not fz.empty else TP.todays_lists(day)
    bb = lists[lists["list"] == "BEST_BETS"].sort_values("rank").head(FREE_TIPS_PER_DAY) if len(lists) else lists
    tips = TP.load_tips()
    return bb, int((tips["day"] == day).sum()) if len(tips) else 0


@st.cache_data(ttl=300)
def _fallback() -> tuple[pd.DataFrame, dict]:
    p = config.OUTPUT_DIR / "stake_plan.csv"
    plan = pd.read_csv(p) if p.exists() else pd.DataFrame()
    pol = json.loads((ROOT / "config" / "bankroll_policy.json").read_text(encoding="utf-8"))
    return plan, pol


today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
day_s = today.strftime("%Y-%m-%d")
bb, n_day = _best_bets(day_s)

if len(bb):
    ui.section_header("Today")
    for i, r in enumerate(bb.itertuples(index=False)):
        if i % 2 == 0:
            cols = st.columns(2)
        ui.selection_card(league=str(r.league), kickoff="", home=r.home_team, away=r.away_team, pick=str(r.pick),
                          odds=f"{float(r.odds):.2f}", tier=str(r.signal_tier), show_plan=False,
                          supporting=[("Odds", f"{float(r.odds):.2f}"),
                                      ("Chance", "–" if pd.isna(r.p_win) else f"{float(r.p_win):.0%}")],
                          note=f"{str(r.signal_tier).title()} signal", container=cols[i % 2])
    if current_tier() == "free" and n_day > len(bb):
        st.info(f"Wowza Pro has **{n_day} tips** today, three top-picks lists and bankroll sizing — see Account.")
else:
    plan, policy = _fallback()
    tips, day, n_fb = free_tip_pick(plan, policy, today)
    if tips.empty:
        ui.empty_state("NO_DATA", "No tips qualify yet for today or the coming days — check back after the next update.")
    else:
        ui.section_header("Today" if day == day_s else f"Next tips: {day}")
        for i, r in enumerate(tips.itertuples(index=False)):
            _, home, away = (str(r.fixture).split("|") + ["", "", ""])[:3]
            mk = MARKET.get(str(r.market), str(r.market))
            side = str(getattr(r, "side", "") or "")
            pick_txt = (f"{side.title()} 2.5 goals" if r.market == "OU25" else
                        f"{mk} — Yes" if r.market == "BTTS" else mk)
            if i % 2 == 0:
                cols = st.columns(2)
            ui.selection_card(league=str(r.league), kickoff="", home=home, away=away, pick=pick_txt,
                              odds=f"{float(r.odds):.2f}", tier=str(r.signal_tier), show_plan=False,
                              supporting=[("Odds", f"{float(r.odds):.2f}")],
                              note=f"{r.signal_tier.title()} signal", container=cols[i % 2])

st.caption("Odds are the best price seen when the tip was made and can move. Chance = our honest win "
           "estimate. Tips are information, not advice; bet only what you can afford to lose.")
from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()
