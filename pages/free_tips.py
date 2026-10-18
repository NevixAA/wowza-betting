"""Today's free tips — the FREE tier's daily 2 (owner brief 2026-10-18).

"TOP 2 BY EVIDENCE" is decided by the approved bankroll policy, not by raw edge: only tips the policy
would actually stake (output/stake_plan.csv, final_fraction > 0 - never a paper league, a PAPER or
BLOCKED track, or a price outside the track's caps), ranked by the track's evidence_score in
config/bankroll_policy.json, then by stake size. The free tier therefore sees the best of what the
system itself trusts, and never more than FREE_TIPS_PER_DAY rows reach this page.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from dashboard_access import FREE_TIPS_PER_DAY, current_tier, free_tip_pick  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MARKET = {"OU25": "Goals 2.5", "OVER15": "Over 1.5 goals", "OVER35": "Over 3.5 goals", "BTTS": "Both teams to score"}

st.title("🎁 Today's free tips")
st.caption(f"{FREE_TIPS_PER_DAY} tips a day, chosen by the evidence behind each market — not by the biggest "
           "edge. Wowza Pro shows every tip, every market.")


@st.cache_data(ttl=300)
def _load() -> tuple[pd.DataFrame, dict]:
    p = config.OUTPUT_DIR / "stake_plan.csv"
    plan = pd.read_csv(p) if p.exists() else pd.DataFrame()
    pol = json.loads((ROOT / "config" / "bankroll_policy.json").read_text(encoding="utf-8"))
    return plan, pol


plan, policy = _load()
today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
tips, day, n_day = free_tip_pick(plan, policy, today)

if tips.empty:
    st.info("No tips qualify yet for today or the coming days — check back after the next update.")
else:
    when = "Today" if day == today.strftime("%Y-%m-%d") else f"Next tips: {day}"
    st.subheader(when)
    for r in tips.itertuples(index=False):
        _, home, away = (str(r.fixture).split("|") + ["", "", ""])[:3]
        mk = MARKET.get(str(r.market), str(r.market))
        side = str(getattr(r, "side", "") or "")
        pick_txt = (f"{side.title()} 2.5 goals" if r.market == "OU25" else
                    f"{mk} — Yes" if r.market == "BTTS" else mk)
        with st.container(border=True):
            c1, c2, c3 = st.columns([4, 2, 1])
            c1.markdown(f"**{home} vs {away}**  \n{r.league}")
            c2.markdown(f"**{pick_txt}**  \n{r.signal_tier.title()} signal")
            c3.metric("Odds", f"{float(r.odds):.2f}")
    if current_tier() == "free" and n_day > len(tips):
        st.info(f"Wowza Pro has **{n_day} tips** on {day} across every market, plus bankroll sizing — see Account.")

st.caption("Odds are the best price seen when the tip was made and can move. Tips are information, not "
           "advice; bet only what you can afford to lose.")
from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()
