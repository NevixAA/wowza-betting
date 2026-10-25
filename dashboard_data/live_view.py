"""Presentation for pages/6_⚡_Live.py, visual redesign 2026-10-25 (stage 2b).

ONE shared live match card replaces the page's three hand-written HTML blocks (29 hex / inline-style hits:
near-black card backgrounds, white text, grey labels, one hex accent per signal). The card is native Streamlit: a bordered container, the
signal as an icon + word chip, the minute as st.progress with its text, and the figures as the design
system's stat line - so it follows the theme and never sets a text colour.

PRESENTATION ONLY. The scanner (src/live_scanner.py) decides every signal; the page reads its CSVs and
passes each row's values in unchanged. `signal_values` / `game_values` / `ht_values` only pick and format
fields of one row - the same expressions the old HTML used, so the displayed strings are identical.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import dashboard_ui as ui

#: signal -> (icon, tone, description). Tone is semantic: an unvalidated signal is a WARNING, never a "strong" red.
SIGNAL_META = {
    "UNDER_HOLD":     ("🔒", "INFO", "UNDER 2.5 — model prediction holding, time running out"),
    "SLEEPING_GAME":  ("😴", "INFO", "UNDER 2.5 — both teams low scoring, game going nowhere"),
    "UNDER_RECOVERY": ("📉", "INFO", "UNDER 2.5 — 2 goals scored, 1 more needed for over, time left"),
    "STRONG_STUCK":   ("💪", "WARNING", "OVER 2.5 — strong attack team can't score, will push harder ⚠️ unvalidated"),
    "COMEBACK":       ("🔥", "WARNING", "OVER 2.5 — losing team with strong attack pushing for goals ⚠️ unvalidated"),
    "HT_UNDER_0.5":   ("⏱", "INFO", "HT UNDER 0.5 — 0-0 late first half, barely any time left"),
    "HT_UNDER_1.5":   ("🧊", "INFO", "HT UNDER 1.5 — low scoring first half, fair price attractive"),
    "HT_OVER_0.5":    ("⚡", "INFO", "HT OVER 0.5 — strong attack teams, 0-0, time still left"),
}
FULL_TIME = 90


def chip(icon: str, word: str, tone: str = "INFO") -> str:
    """Icon + word as a theme-aware native badge (never colour alone)."""
    return f":{ui.TONE_COLOUR.get(tone, 'gray')}-badge[{ui._md_safe(icon)} {ui._md_safe(word)}]"


def signal_chip(signal_type: str) -> str:
    icon, tone, _ = SIGNAL_META.get(signal_type, ("📌", "NEUTRAL", signal_type))
    return chip(icon, signal_type, tone)


def live_match_card(*, match: str, league: str = "", minute=None, score: str = "", chips: str = "",
                    icon: str = "", stats=None, note: str = "", eyebrow_text: str = "", container=None) -> None:
    """THE live match card: match, league, minute (progress), score, signal chip(s), figures, one note.

    `minute=None` (a pre-match half-time card) drops the progress bar; nothing else changes."""
    with (container or st).container(border=True):
        top = " · ".join(str(x) for x in (eyebrow_text or league,) if x)
        if top:
            ui.eyebrow(top)
        st.markdown(f"<div class='wz-title'>{ui._html.escape(((icon + ' ') if icon else '') + str(match))}</div>",
                    unsafe_allow_html=True)
        if chips:
            st.markdown(chips)
        if minute is not None:
            m = int(minute)
            st.progress(round(min(m / FULL_TIME, 1.0), 2), text=f"⏱ {m}' · ⚽ Score {score}" if score != "" else f"⏱ {m}'")
        if stats:
            ui.stat_line(stats)
        if note:
            st.caption(note)


# ── row -> card values (the old HTML's expressions, unchanged) ──────────────────────────────────
def signal_values(row) -> dict:
    """One live_tips.csv row -> what the signal card shows."""
    icon, _tone, _desc = SIGNAL_META.get(row["signal_type"], ("📌", "NEUTRAL", row["signal_type"]))
    bet = row["bet"]
    is_under = "UNDER" in bet
    fair_odds = row["fair_under_odds"] if is_under else row["fair_over_odds"]
    live_p = row["live_p_under"] if is_under else row["live_p_over"]
    return {"icon": icon, "match": row["match"], "league": row["league"], "minute": int(row["elapsed_mins"]),
            "score": row["score"], "signal": row["signal_type"], "bet": bet,
            "stats": [("Bet", f"{bet}"), ("Fair price", f"{fair_odds}"),
                      (f"Live P({'UNDER' if is_under else 'OVER'})", f"{live_p * 100:.0f}%"),
                      ("Pre-match P(over)", f"{row['pre_p_over'] * 100:.0f}%")],
            "note": f"⚠️ Bet only if your bookmaker's live odds > {fair_odds}"}


def game_values(g, signal_matches: set) -> dict:
    """One live_games.csv row -> the compact monitored-game card."""
    has_signal = g["match"] in signal_matches
    has_pred = str(g.get("has_prediction", "")).lower() == "true"
    badge = (chip("🚨", "SIGNAL", "PRIMARY") if has_signal else
             chip("📊", "tracked", "INFO") if has_pred else chip("👁", "watching", "NEUTRAL"))
    p_under, fair = g.get("live_p_under"), g.get("fair_under_odds")
    stats = ([("P(under)", f"{p_under}%"), ("Fair UNDER", f"{fair}")]
             if pd.notna(p_under) and p_under else None)
    return {"match": g["match"], "league": g["league"], "minute": g["elapsed_mins"], "score": g["score"],
            "chips": badge, "stats": stats, "note": "" if stats else "No pre-match prediction"}


def ht_values(row) -> dict:
    """One predictions.csv row -> the half-time card (pre-match, so no minute)."""
    p05 = float(row["p_ht_over05"])
    p15 = float(row["p_ht_over15"]) if pd.notna(row.get("p_ht_over15")) else None
    fair_over05 = round(1 / max(p05, 0.01), 2)
    fair_under05 = round(1 / max(1 - p05, 0.01), 2)
    fair_over15 = round(1 / max(p15, 0.01), 2) if p15 else None
    sig = []
    if p05 >= 0.75:
        sig.append(chip("⚡", f"HT OVER 0.5  ({p05*100:.0f}%) — fair {fair_over05}", "INFO"))
    if p05 <= 0.30:
        sig.append(chip("🧊", f"HT UNDER 0.5 ({(1-p05)*100:.0f}%) — fair {fair_under05}", "INFO"))
    if p15 and p15 >= 0.60:
        sig.append(chip("🔥", f"HT OVER 1.5  ({p15*100:.0f}%) — fair {fair_over15}", "INFO"))
    if p15 and p15 <= 0.25:
        fair_under15 = round(1 / max(1 - p15, 0.01), 2)
        sig.append(chip("🔒", f"HT UNDER 1.5 ({(1-p15)*100:.0f}%) — fair {fair_under15}", "INFO"))
    stats = [("P(HT OVER 0.5)", f"{p05*100:.0f}%"), ("Fair OVER 0.5", f"{fair_over05}"),
             ("Fair UNDER 0.5", f"{fair_under05}")]
    if p15:
        stats += [("P(HT OVER 1.5)", f"{p15*100:.0f}%"), ("Fair OVER 1.5", f"{fair_over15}")]
    return {"match": f"{row['home_team']} vs {row['away_team']}",
            "eyebrow": f"{row.get('league', '')} · 📅 {str(row['date'])[:10]}",
            "chips": " ".join(sig) if sig else chip("–", "No strong signal", "NEUTRAL"), "stats": stats}


def last_scan(df: pd.DataFrame) -> str | None:
    """The scanner's own recorded time-of-day (updated_at) - a record inside the file, not an mtime."""
    if df is not None and not df.empty and "updated_at" in df.columns:
        v = df["updated_at"].dropna()
        return str(v.iloc[0]) if len(v) else None
    return None


def history_config() -> dict:
    return {"min": ui.num_col("min", fmt="%d"), "FT goals": ui.num_col("FT goals", fmt="%.0f"),
            "fair_U": ui.num_col("fair_U", fmt="%.2f"), "fair_O": ui.num_col("fair_O", fmt="%.2f"),
            "P(U)": ui.pct_col("P(U)"), "P(O)": ui.pct_col("P(O)"), "pre P(O)": ui.pct_col("pre P(O)")}


def raw_config() -> dict:
    return {"elapsed_mins": ui.num_col("elapsed_mins", fmt="%d"),
            "fair_under_odds": ui.num_col("fair_under_odds", fmt="%.2f"),
            "fair_over_odds": ui.num_col("fair_over_odds", fmt="%.2f"),
            "live_p_under": ui.pct_col("live_p_under"), "live_p_over": ui.pct_col("live_p_over"),
            "pre_p_over": ui.pct_col("pre_p_over"), "lam_remaining": ui.num_col("lam_remaining", fmt="%.2f")}
