"""Shared betting disclaimer / terms of use — rendered on every dashboard page."""
import streamlit as st

_TERMS = """
**By using this dashboard you acknowledge and agree to the following:**

**Information Only** — All content, opinions, statistics, sharp-money alerts, market
movements and betting signals here are provided for **informational and entertainment
purposes only**. Nothing shown should be considered financial, betting, legal or
investment advice.

**No Guarantees** — Sports betting involves risk. Past performance, odds movement,
sharp-money indicators or historical results **do not guarantee** future outcomes. No
owner, administrator, developer or contributor can guarantee profits or winning bets.

**Bet Responsibly** — Only wager funds you can afford to lose. You are solely responsible
for your own betting decisions, bankroll management and financial outcomes.

**Limitation of Liability** — The owners, administrators, developers, contributors and
affiliates shall not be held liable for any losses, damages, debts, claims or consequences
arising directly or indirectly from use of information shown here.

**Independent Decision Making** — Every wager is placed at the sole discretion of the
individual. Do your own research before making any betting decision.

**No Professional Advisory Relationship** — Use of this dashboard does not create any
advisory, fiduciary, financial, investment or professional relationship.

**Compliance With Local Laws** — You are responsible for ensuring sports betting and
gambling are legal in your jurisdiction and that you comply with all applicable laws.

**18+ Only** — Intended only for individuals who meet the legal gambling age in their
jurisdiction.

_Gamble responsibly._ 🎯
"""

_ONE_LINER = ("⚠️ Informational & entertainment only — **not** financial/betting advice. "
              "No guarantees; past results don't predict future outcomes. "
              "18+ · bet responsibly · only wager what you can afford to lose.")


def render_disclaimer(expanded: bool = False) -> None:
    """Full terms inside a collapsible expander."""
    with st.expander("⚠️ Disclaimer & Terms of Use", expanded=expanded):
        st.markdown(_TERMS)


def disclaimer_footer() -> None:
    """One-line notice + collapsible full terms. Call at the bottom of every page."""
    st.divider()
    st.caption(_ONE_LINER)
    render_disclaimer(expanded=False)
