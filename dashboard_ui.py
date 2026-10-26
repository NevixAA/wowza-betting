"""
Wowza dashboard design system — the ONE shared presentation module (visual redesign 2026-10-25).
==============================================================================================
Before this, two modules did the same job: `dashboard_ui.py` (tables, metric rows, page headers)
and `dashboard_data/ui.py` (charts, chips, banners, number formatting). They disagreed on colours,
both had a `table()` with different behaviour, and pages filled the gaps with their own hex codes
and <style> blocks. Everything now lives here; `dashboard_data/ui.py` is a thin re-export kept so
no page breaks. Full guide: docs/DASHBOARD_DESIGN_SYSTEM_2026_10.md.

PRESENTATION ONLY. Imports no model code, reads no output file, used by no workflow. Every value a
component shows is passed in by the caller from the same canonical source as before.

RULES THAT ARE NOT NEGOTIABLE HERE (carried over from dashboard_data/ui.py, still enforced):
  1. Colour is never the only cue. Every status carries its word and an icon.
  2. Never a dual axis. Two measures on different scales become two charts.
  3. n travels with every rate. `pct(x, n)` prints the denominator; `sample_marker` flags n < floor.
  4. Numbers stay numbers in tables. Format at render time with st.column_config so sorting works.
  5. Pages never invent hex colours. They use TOKENS / tone names; custom HTML text colour is always
     `inherit` (only accents take a token), so nothing turns unreadable if the viewer switches theme.

TOKENS mirror .streamlit/config.toml ([theme.dark] / [theme.light]); tests/test_design_system.py
fails if the two drift apart.
"""
from __future__ import annotations

import html as _html
import re

import pandas as pd
import streamlit as st

# ═════════════════════════════════════════════════════════════════════════════════════════════
# TOKENS
# ═════════════════════════════════════════════════════════════════════════════════════════════
# Semantic, not decorative: a page asks for POSITIVE or RESEARCH, never for "#2BB59B".
# Polarity is teal-green against vermilion rather than pure green/red: the pair stays apart under
# protanopia and deuteranopia (one carries blue, the other none), and the sign is always also in
# the text (`units()` prints + / −). Repo series colours are the CVD-validated set this dashboard
# already used (v9 blue vs Pro magenta separated where the obvious violet did not).
DARK: dict[str, str] = {
    "BACKGROUND": "#0B1016", "SURFACE": "#121922", "SURFACE_ELEVATED": "#18212C",
    "SURFACE_MUTED": "#131A23", "BORDER": "#243040", "BORDER_STRONG": "#3A4A5E",
    "TEXT_PRIMARY": "#E7EDF4", "TEXT_SECONDARY": "#A9B6C4", "TEXT_MUTED": "#7C8A9A",
    "PRIMARY": "#4C8DF6", "PRIMARY_SOFT": "#4C8DF626",
    "POSITIVE": "#2BB59B", "NEGATIVE": "#E86A3F", "WARNING": "#E2A93B", "INFO": "#5AA9E6",
    "LIVE": "#2BB59B", "PAPER": "#E2A93B", "RESEARCH": "#A47EF2", "BLOCKED": "#E86A3F",
    "NO_DATA": "#7C8A9A",
    "V9": "#3B82F6", "PRO": "#CB3FAE", "V11": "#B8820C", "FANTASY": "#0E9B72",
}
LIGHT: dict[str, str] = {
    "BACKGROUND": "#F6F8FB", "SURFACE": "#FFFFFF", "SURFACE_ELEVATED": "#FFFFFF",
    "SURFACE_MUTED": "#EEF2F7", "BORDER": "#DCE3EC", "BORDER_STRONG": "#B8C4D3",
    "TEXT_PRIMARY": "#0F172A", "TEXT_SECONDARY": "#3E4C5E", "TEXT_MUTED": "#64748B",
    "PRIMARY": "#1E6FD9", "PRIMARY_SOFT": "#1E6FD91F",
    "POSITIVE": "#0E8A74", "NEGATIVE": "#C8461C", "WARNING": "#B26B00", "INFO": "#1D74B8",
    "LIVE": "#0E8A74", "PAPER": "#B26B00", "RESEARCH": "#6D3FD1", "BLOCKED": "#C8461C",
    "NO_DATA": "#64748B",
    "V9": "#1E6FD9", "PRO": "#C026A3", "V11": "#E8A020", "FANTASY": "#12916A",
}
TOKEN_NAMES = tuple(DARK)

#: Sequential ramp for magnitude: ONE hue, monotone. Never a rainbow.
RAMP = ["#DBEAFE", "#93C5FD", "#60A5FA", "#3B82F6", "#2563EB", "#1D4ED8"]
#: Categorical series (also theme.chartCategoricalColors): calm, distinct, no rainbow.
CATEGORICAL = ["#4C8DF6", "#E2A93B", "#A47EF2", "#2BB59B", "#E86A3F", "#7C8A9A"]

#: Semantic tone -> Streamlit's named colour, whose value .streamlit/config.toml sets to the token.
#: Native badges/coloured text are theme-aware, so a chip is readable in light AND dark.
TONE_COLOUR = {"POSITIVE": "green", "LIVE": "green", "NEGATIVE": "red", "BLOCKED": "red",
               "WARNING": "orange", "PAPER": "orange", "RESEARCH": "violet", "INFO": "blue",
               "PRIMARY": "blue", "NO_DATA": "gray", "NEUTRAL": "gray"}

# ── backwards-compatible constants (were dashboard_data/ui.py) ───────────────────────────────────
REPO_COLOUR = {"v9": LIGHT["V9"], "pro": LIGHT["PRO"], "v11": LIGHT["V11"], "fantasy": LIGHT["FANTASY"]}
REPO_COLOUR_DARK = {"v9": DARK["V9"], "pro": DARK["PRO"], "v11": DARK["V11"], "fantasy": DARK["FANTASY"]}
POS, NEG, NEUTRAL = LIGHT["POSITIVE"], LIGHT["NEGATIVE"], "#9CA3AF"
POS_DARK, NEG_DARK = DARK["POSITIVE"], DARK["NEGATIVE"]
STATUS = {"CURRENT": LIGHT["POSITIVE"], "AGING": LIGHT["WARNING"], "STALE": LIGHT["NEGATIVE"],
          "FAILED": LIGHT["NEGATIVE"], "UNKNOWN": LIGHT["NO_DATA"]}
INK, INK_MUTED = LIGHT["TEXT_PRIMARY"], LIGHT["TEXT_MUTED"]


def is_dark() -> bool:
    """Whether the viewer is on the dark theme. Unknown -> the configured base (dark by default)."""
    try:
        t = getattr(st.context.theme, "type", None)
        if t:
            return str(t).lower() == "dark"
    except Exception:                                                 # noqa: BLE001
        pass
    try:
        return str(st.get_option("theme.base") or "dark").lower() != "light"
    except Exception:                                                 # noqa: BLE001
        return True


def tokens() -> dict[str, str]:
    """The token set for the viewer's current theme."""
    return DARK if is_dark() else LIGHT


def token(name: str) -> str:
    return tokens()[name]


def repo_colour(repo: str) -> str:
    return tokens().get(str(repo).upper(), tokens()["TEXT_MUTED"])


def polarity() -> tuple[str, str]:
    """(profit, loss) colours for the current theme."""
    t = tokens()
    return t["POSITIVE"], t["NEGATIVE"]


# ═════════════════════════════════════════════════════════════════════════════════════════════
# STATUS VOCABULARY — icon + word + tone, never colour alone
# ═════════════════════════════════════════════════════════════════════════════════════════════
# The icons are glyphs, not emoji, apart from the shield: PROTECTED is a governance state the owner
# asked to see as a shield, and it is deliberately styled as information, not as an error.
STATUSES: dict[str, tuple[str, str, str]] = {
    # production states (track_registry / stake_plan deployment_state)
    "ACTIVE":      ("●", "ACTIVE", "POSITIVE"),
    "ACTIVE_LOW":  ("◕", "ACTIVE · LOW", "POSITIVE"),
    "LIVE":        ("◉", "LIVE", "LIVE"),
    "PAPER":       ("◐", "PAPER", "PAPER"),
    "BLOCKED":     ("⊘", "BLOCKED", "BLOCKED"),
    # research regimes
    "PROTECTED":   ("🛡", "PROTECTED", "PRIMARY"),
    "REPAIR":      ("↻", "REPAIR", "WARNING"),
    "LEARNING":    ("◌", "LEARNING", "RESEARCH"),
    "RESEARCH":    ("◇", "RESEARCH", "RESEARCH"),
    "SHADOW":      ("◆", "SHADOW", "RESEARCH"),
    "QUALIFIED":   ("✓", "QUALIFIED", "POSITIVE"),
    "OBSERVE":     ("◎", "OBSERVE", "INFO"),
    # market-first engine decisions (shadow)
    "BET":         ("◆", "BET · SHADOW", "RESEARCH"),
    "NO_BET":      ("–", "NO BET", "NO_DATA"),
    "NO BET":      ("–", "NO BET", "NO_DATA"),
    # health
    "PASS":        ("✓", "PASS", "POSITIVE"),
    "WARN":        ("▲", "WARN", "WARNING"),
    "FAIL":        ("✕", "FAIL", "NEGATIVE"),
    "GREEN":       ("✓", "GREEN", "POSITIVE"),
    "AMBER":       ("▲", "AMBER", "WARNING"),
    "RED":         ("✕", "RED", "NEGATIVE"),
    "HEALTHY":     ("✓", "HEALTHY", "POSITIVE"),
    # data availability — four DIFFERENT claims, never merged
    "NO_DATA":     ("○", "NO DATA", "NO_DATA"),
    "NOT_CONNECTED": ("⌁", "FEED NOT CONNECTED", "NO_DATA"),
    "STALE":       ("◷", "STALE", "WARNING"),
    "ERROR":       ("✕", "ERROR", "NEGATIVE"),
    # freshness (dashboard_data.core.Freshness states + artifact manifest verdicts)
    "CURRENT":     ("●", "CURRENT", "POSITIVE"),
    "FRESH":       ("●", "FRESH", "POSITIVE"),
    "AGING":       ("◔", "AGING", "WARNING"),
    "FAILED":      ("✕", "FAILED", "NEGATIVE"),
    "UNKNOWN":     ("○", "UNKNOWN", "NO_DATA"),
    "HISTORICAL":  ("◷", "HISTORICAL", "NO_DATA"),
    "DATED_STUDY": ("◷", "DATED STUDY", "INFO"),
    # evidence accrual
    "ACCRUING":    ("◔", "ACCRUING", "RESEARCH"),
    # evidence type: a replay of the past can describe, never confirm; forward evidence is collected live
    "RETROSPECTIVE": ("◷", "RETROSPECTIVE · REPLAY, NOT FORWARD EVIDENCE", "WARNING"),
    "PROSPECTIVE": ("▸", "PROSPECTIVE · FORWARD", "RESEARCH"),
    "REGISTERED":  ("○", "REGISTERED", "RESEARCH"),
    "SMALL_SAMPLE": ("◌", "SMALL SAMPLE", "WARNING"),
    # legacy signal tiers: secondary metadata, deliberately NEUTRAL — a tier is the size of the
    # model-vs-price gap, not evidence, so it must never look like a strength rating.
    "SNIPER":      ("·", "SNIPER", "NEUTRAL"),
    "MARKSMAN":    ("·", "MARKSMAN", "NEUTRAL"),
    "VALUABLE":    ("·", "VALUABLE", "NEUTRAL"),
}


def _status_key(state) -> str:
    s = str(state or "").strip().upper()
    return s.replace(" ", "_") if s.replace(" ", "_") in STATUSES else s


def status_meta(state) -> tuple[str, str, str]:
    """(icon, label, tone) for a state; unknown states keep their own word, neutral tone."""
    k = _status_key(state)
    if k in STATUSES:
        return STATUSES[k]
    word = str(state or "UNKNOWN").strip() or "UNKNOWN"
    return ("○", word.upper(), "NEUTRAL")


def _md_safe(s) -> str:
    """Text safe inside a Streamlit markdown directive (`:green-badge[...]`)."""
    return re.sub(r"[\[\]\\]", "", str(s))


def status_chip(state, *, label: str | None = None, tone: str | None = None) -> str:
    """Markdown for a status chip: a theme-aware native badge with icon AND word.

    Returns a string so several chips can share one line: `st.markdown(chip("PAPER") + " " + ...)`.
    """
    icon, word, t = status_meta(state)
    colour = TONE_COLOUR.get(tone or t, "gray")
    return f":{colour}-badge[{_md_safe(icon)} {_md_safe(label or word)}]"


def show_status(*states, container=None) -> None:
    """Render one or more chips on a single line."""
    (container or st).markdown(" ".join(status_chip(s) for s in states if s))


def tone_text(text, tone: str) -> str:
    """Coloured inline markdown text for a semantic tone (theme-aware)."""
    return f":{TONE_COLOUR.get(tone, 'gray')}[{_md_safe(text)}]"


# ═════════════════════════════════════════════════════════════════════════════════════════════
# NUMBER FORMATS — one way to write each kind of number
# ═════════════════════════════════════════════════════════════════════════════════════════════
def _missing(x) -> bool:
    try:
        return x is None or (not isinstance(x, str) and pd.isna(x))
    except Exception:                                                 # noqa: BLE001
        return False


def _signed(v: float, dp: int) -> str:
    return f"{v:+.{dp}f}"


def fmt_roi(x, dp: int = 1) -> str:
    """ROI from a FRACTION: 0.084 -> '+8.4%'."""
    return "—" if _missing(x) else f"{_signed(float(x) * 100, dp)}%"


def fmt_units(x, dp: int = 2) -> str:
    """P/L in units at flat 1u: '+12.47u'."""
    return "—" if _missing(x) else f"{_signed(float(x), dp)}u"


def fmt_odds(x) -> str:
    return "—" if _missing(x) else f"{float(x):.2f}"


def fmt_prob(x, dp: int = 0) -> str:
    """A 0-1 probability: 0.63 -> '63%'."""
    return "—" if _missing(x) else f"{float(x) * 100:.{dp}f}%"


def fmt_clv(x, dp: int = 1) -> str:
    """CLV already in PERCENT units (as the ledgers store it): 1.2 -> '+1.2%'."""
    return "—" if _missing(x) else f"{_signed(float(x), dp)}%"


def fmt_xpts(x) -> str:
    return "—" if _missing(x) else f"{float(x):.2f}"


def fmt_money(x) -> str:
    """FPL price in £m: 8.5 -> '£8.5m'."""
    return "—" if _missing(x) else f"£{float(x):.1f}m"


def pct(x, n=None, dp: int = 1) -> str:
    """A rate, with its denominator. `pct(0.55, 20)` is a fact; `pct(0.55)` is only a number."""
    if _missing(x):
        return "—"
    s = f"{float(x) * 100:.{dp}f}%"
    return f"{s} (n={n:,})" if n is not None else s


def units(x, dp: int = 2) -> str:
    """P/L is in UNITS at flat 1u stakes, never in currency and never as an ROI without n."""
    return fmt_units(x, dp)


def points(x, dp: int = 2) -> str:
    """Fantasy is scored in FPL points. Never mix this scale with units."""
    return "—" if _missing(x) else f"{float(x):.{dp}f} pts"


SMALL_SAMPLE_FLOOR = 30


def small_sample(n, floor: int = SMALL_SAMPLE_FLOOR) -> bool:
    """Below this, a rate is noise. Flagged on the page, never silently dropped."""
    return n is not None and not _missing(n) and int(n) < floor


def sample_marker(n, floor: int = SMALL_SAMPLE_FLOOR) -> str:
    """'' when n is adequate, else a SMALL SAMPLE chip carrying n (markdown)."""
    if not small_sample(n, floor):
        return ""
    return status_chip("SMALL_SAMPLE", label=f"SMALL SAMPLE · n={int(n)}")


def evidence_badge(kind: str, n=None, *, span: str = "", state: str = "") -> str:
    """'TRUE FORWARD · n=45 · 6 GW · ACCRUING' as one research-toned chip (markdown)."""
    bits = [str(kind).replace("_", " ").upper()]
    if n is not None and not _missing(n):
        bits.append(f"n={int(n):,}")
    if span:
        bits.append(str(span))
    if state:
        bits.append(str(state).replace("_", " ").upper())
    return f":violet-badge[{_md_safe(' · '.join(bits))}]"


def rank_badge(i: int) -> str:
    return f":blue-badge[#{int(i)}]"


# ═════════════════════════════════════════════════════════════════════════════════════════════
# LAYOUT + TYPOGRAPHY
# ═════════════════════════════════════════════════════════════════════════════════════════════
_CSS = """<style>
.wz-eyebrow{font-size:.72rem;font-weight:600;letter-spacing:.09em;text-transform:uppercase;opacity:.7;
  margin:0 0 .1rem 0;color:inherit}
.wz-stats{display:flex;flex-wrap:wrap;gap:.45rem 1.25rem;margin:.1rem 0 .2rem}
.wz-stat{display:flex;flex-direction:column;min-width:5.6rem}
.wz-stat-l{font-size:.72rem;opacity:.68;color:inherit;line-height:1.25}
.wz-stat-v{font-size:.98rem;font-weight:600;font-variant-numeric:tabular-nums;color:inherit;line-height:1.35}
.wz-title{font-weight:650;font-size:1.02rem;line-height:1.3;margin:.05rem 0 .1rem;color:inherit}
.wz-sub{font-size:.8rem;opacity:.7;color:inherit}
[data-testid="stMetricValue"]{font-variant-numeric:tabular-nums}
</style>"""


def inject_css() -> None:
    """The ONLY stylesheet in the app. Text colour is always `inherit`, so it follows the theme."""
    st.markdown(_CSS, unsafe_allow_html=True)


def space(size: str = "medium") -> None:
    """Vertical rhythm instead of st.divider / '---'. small | medium | large."""
    try:
        st.space(size)
    except Exception:                                                 # noqa: BLE001 - older Streamlit
        st.write("")


def eyebrow(text: str, container=None) -> None:
    (container or st).markdown(f"<div class='wz-eyebrow'>{_html.escape(str(text))}</div>",
                               unsafe_allow_html=True)


def caption(text: str, container=None) -> None:
    (container or st).caption(text)


def page_header(title: str, subtitle: str = "", *, badge: str = "", eyebrow_text: str = "",
                state: str | None = None, freshness: str | None = None) -> None:
    """Eyebrow/status line + title + one-line subtitle + optional state/freshness chips.

    Backwards compatible with the old `page_header(title, subtitle, badge=...)`. `freshness` is a
    ready string (e.g. from `freshness_badge`) and is the ONE place a page states its data age.
    """
    inject_css()
    if eyebrow_text:
        eyebrow(eyebrow_text)
    st.title(title, anchor=False)
    line = subtitle
    if badge:
        line = f"{badge} &nbsp;·&nbsp; {subtitle}" if subtitle else badge
    if line:
        st.caption(line, unsafe_allow_html=True)
    chips = " ".join(x for x in ((status_chip(state) if state else ""), freshness or "") if x)
    if chips:
        st.markdown(chips)


def section_header(title: str, caption_text: str = "", *, container=None) -> None:
    """A section title with optional one-line caption, spaced above (no divider)."""
    c = container or st
    if container is None:
        space("small")
    c.subheader(title, anchor=False)
    if caption_text:
        c.caption(caption_text)


def freshness_badge(state: str, detail: str = "") -> str:
    """Freshness as a chip (markdown): '● FRESH · 0.4 h old'."""
    icon, word, tone = status_meta(state)
    lab = f"{word}{(' · ' + detail) if detail else ''}"
    return f":{TONE_COLOUR.get(tone, 'gray')}-badge[{_md_safe(icon)} {_md_safe(lab)}]"


# ═════════════════════════════════════════════════════════════════════════════════════════════
# BLOCKS
# ═════════════════════════════════════════════════════════════════════════════════════════════
#: Paragraph break inside one st.markdown: ends the HTML block so the markdown chip line after it renders.
_PARA = chr(10) * 2


def stat_html(items) -> str:
    """Supporting figures as one wrapping row: [(label, value), ...] or dicts with label/value/n."""
    out = []
    for it in items or []:
        if isinstance(it, dict):
            lab, val = it.get("label", ""), it.get("value", "")
        else:
            lab, val = it[0], it[1]
        out.append(f"<div class='wz-stat'><span class='wz-stat-l'>{_html.escape(str(lab))}</span>"
                   f"<span class='wz-stat-v'>{_html.escape(str(val))}</span></div>")
    return "<div class='wz-stats'>" + "".join(out) + "</div>"


def stat_line(items, container=None) -> None:
    if items:
        (container or st).markdown(stat_html(items), unsafe_allow_html=True)


def metric_card(label: str, value, *, title: str = "", supporting=None, status: str = "",
                evidence: str = "", n=None, floor: int = SMALL_SAMPLE_FLOOR, help: str | None = None,
                delta=None, delta_color: str = "normal", chart_data=None, container=None) -> None:
    """THE card: one main number, 3–5 supporting figures, one status/evidence line.

    `status` / `evidence` are markdown strings (status_chip / evidence_badge). A SMALL SAMPLE marker
    is added automatically when `n` is below `floor`.
    """
    with (container or st).container(border=True):
        if title:
            eyebrow(title)
        if chart_data is not None:
            st.metric(label, value, delta=delta, delta_color=delta_color, help=help, chart_data=chart_data)
        else:
            st.metric(label, value, delta=delta, delta_color=delta_color, help=help)
        line = " ".join(x for x in (status, evidence, sample_marker(n, floor) if n is not None else "") if x)
        body = _PARA.join(x for x in ((stat_html(supporting) if supporting else ""), line) if x)
        if body:                    # one element: the stats block, then the chip line as markdown
            st.markdown(body, unsafe_allow_html=True)


def metric_grid(items: list[dict], *, per_row: int = 3, border: bool = True) -> None:
    """Metrics in rows of at most `per_row` (3 by default — never 5–6 squeezed columns).

    Columns stack on narrow screens, so a phone shows one card per row.
    """
    if not items:
        return
    for i in range(0, len(items), per_row):
        chunk = items[i:i + per_row]
        for col, it in zip(st.columns(per_row), chunk):
            col.metric(it.get("label", ""), it.get("value", ""), delta=it.get("delta"),
                       delta_color=it.get("delta_color", "normal"), help=it.get("help"), border=border)


def metric_row(items: list[dict], *, per_row: int = 4) -> None:
    """Back-compat: metrics in rows of `per_row`. New code uses metric_grid."""
    if not items:
        return
    for i in range(0, len(items), per_row):
        chunk = items[i:i + per_row]
        for col, it in zip(st.columns(len(chunk)), chunk):
            col.metric(it.get("label", ""), it.get("value", ""), delta=it.get("delta"),
                       delta_color=it.get("delta_color", "normal"), help=it.get("help"))


def selection_card(*, league: str, kickoff: str, home: str, away: str, pick: str, odds,
                   production_state: str = "", decision_state: str = "", tier: str = "",
                   supporting=None, evidence: str = "", note: str = "", show_plan: bool = True,
                   container=None) -> None:
    """One selection: who plays, what the pick is, the price, and what production / the shadow
    engine each say about it. The legacy tier is shown last and neutral — metadata, not strength."""
    with (container or st).container(border=True):
        st.markdown(f"<div class='wz-eyebrow'>{_html.escape(' · '.join(str(x) for x in (league, kickoff) if x))}</div>"
                    f"<div class='wz-title'>{_html.escape(str(home))} v {_html.escape(str(away))}</div>",
                    unsafe_allow_html=True)
        st.metric(pick, fmt_odds(odds) if not isinstance(odds, str) else odds)
        chips = [status_chip(production_state) if production_state else
                 (status_chip("NO_DATA", label="NOT IN STAKE PLAN") if show_plan else ""),
                 status_chip(decision_state) if decision_state else "",
                 status_chip(tier) if tier else ""]
        st.markdown((stat_html(supporting) + _PARA if supporting else "") + " ".join(c for c in chips if c),
                    unsafe_allow_html=True)
        if evidence or note:
            st.caption(" · ".join(x for x in (evidence, note) if x))


def player_card(container, *, name: str, position: str, points: float,
                team: str = "", price: float | None = None, fixture: str = "",
                p_goal: float | None = None, p_assist: float | None = None,
                flag: str = "") -> None:
    """A captaincy/pick card with the useful numbers VISIBLE (not hidden in a tooltip)."""
    with container.container(border=True):
        st.markdown(f"**{flag}{name}**")
        meta = " · ".join(x for x in (position, team, fmt_money(price) if price is not None else "") if x)
        if meta:
            st.caption(meta)
        st.metric("Expected points", f"{points:.2f}")
        bits = []
        if p_goal is not None and p_goal == p_goal:
            bits.append(("P(goal)", fmt_prob(p_goal)))
        if p_assist is not None and p_assist == p_assist:
            bits.append(("P(assist)", fmt_prob(p_assist)))
        stat_line(bits)
        if fixture:
            st.caption(f"Next: {fixture}")


def league_card(*, league: str, value, label: str = "P/L", supporting=None, state: str = "",
                regime: str = "", n=None, container=None) -> None:
    metric_card(label, value, title=league, supporting=supporting,
                status=" ".join(status_chip(s) for s in (state, regime) if s), n=n, container=container)


def research_card(*, title: str, value, label: str, supporting=None, evidence: str = "",
                  state: str = "RESEARCH", n=None, container=None) -> None:
    metric_card(label, value, title=title, supporting=supporting, status=status_chip(state),
                evidence=evidence, n=n, container=container)


def health_card(*, title: str, status: str, value: str = "", rule: str = "", container=None) -> None:
    with (container or st).container(border=True):
        eyebrow(title)
        st.markdown(status_chip(status) + (f" &nbsp; **{_md_safe(value)}**" if value else ""))
        if rule:
            st.caption(rule)


_EMPTY_COPY = {
    "NO_DATA": "Nothing recorded yet.",
    "NOT_CONNECTED": "This feed is not connected here, so nothing was measured — not 'measured, found nothing'.",
    "STALE": "The data exists but is older than its limit; it is not valid for current decisions.",
    "ERROR": "The data could not be read.",
}


def empty_state(kind: str = "NO_DATA", detail: str = "", container=None) -> None:
    """Four different claims that must never look alike: NO DATA / FEED NOT CONNECTED / STALE / ERROR."""
    k = _status_key(kind) if _status_key(kind) in _EMPTY_COPY else "NO_DATA"
    c = container or st
    with c.container(border=True):
        st.markdown(status_chip(k))
        st.caption(detail or _EMPTY_COPY[k])


def needs_attention(items: list[dict], *, container=None, healthy_text: str = "Nothing needs attention.") -> None:
    """Only meaningful warnings, worst first. Each item: {status, title, detail, source}. Healthy
    items are never passed in; an empty list shows one quiet line instead of a wall of green."""
    c = container or st
    if not items:
        c.markdown(status_chip("HEALTHY") + " " + _md_safe(healthy_text))
        return
    order = {"RED": 0, "FAIL": 0, "ERROR": 0, "STALE": 1, "NO_DATA": 2, "AMBER": 3, "WARN": 3}
    for it in sorted(items, key=lambda x: order.get(str(x.get("status", "")).upper(), 5)):
        with c.container(border=True):
            st.markdown(status_chip(it.get("status", "WARN")) + f" &nbsp;**{_md_safe(it.get('title', ''))}**")
            sub = " · ".join(str(x) for x in (it.get("detail"), it.get("source")) if x)
            if sub:
                st.caption(sub)


def freshness_chip(f, label: str = "") -> None:
    """A feed's state (dashboard_data.core.Freshness) as icon + word + age."""
    if f is None:
        st.markdown(status_chip("UNKNOWN") + (f" :gray[{_md_safe(label)}]" if label else ""))
        return
    detail = " · ".join(x for x in (label or f.source, f.label) if x)
    st.markdown(freshness_badge(f.state) + (f" :gray[{_md_safe(detail)}]" if detail else ""))


_BANNER_TONE = {"LIVE": "LIVE", "PAPER": "PAPER", "SHADOW / RESEARCH ONLY": "RESEARCH",
                "RESEARCH": "RESEARCH", "SHADOW": "RESEARCH", "REVIEW": "INFO", "PROTECTED": "PRIMARY"}


def state_banner(state: str, detail: str = "") -> None:
    """The one thing a reader must not misread: is this LIVE, PAPER, or RESEARCH?"""
    tone = _BANNER_TONE.get(str(state).upper(), status_meta(state)[2])
    icon = status_meta(state)[0] if _status_key(state) in STATUSES else "◇" if tone == "RESEARCH" else "●"
    chip = f":{TONE_COLOUR.get(tone, 'gray')}-badge[{_md_safe(icon)} {_md_safe(str(state).upper())}]"
    st.markdown(chip + (f" &nbsp;{detail}" if detail else ""))


def absent(reason: str) -> None:
    """A repo/feed that is not checked out. Never an empty chart instead (that reads as 'measured,
    found nothing')."""
    st.info(f"**Not connected.** {reason}")


def kpi(label: str, value: str, sub: str = "", tone: str | None = None) -> None:
    """Back-compat KPI block. Text inherits the theme colour; `tone` (a hex from polarity()/tokens)
    only tints the value."""
    style = f"color:{tone};" if tone else ""
    st.markdown(
        f"<div style='padding:8px 2px'>"
        f"<div class='wz-stat-l' style='text-transform:uppercase;letter-spacing:.04em'>{_html.escape(str(label))}</div>"
        f"<div style='{style}font-size:1.7rem;font-weight:650;line-height:1.25;"
        f"font-variant-numeric:tabular-nums'>{_html.escape(str(value))}</div>"
        f"<div class='wz-stat-l'>{_html.escape(str(sub))}</div></div>",
        unsafe_allow_html=True)


def autorefresh(minutes: int = 2, key: str = "refresh") -> None:
    """Re-run the page every `minutes` (st_autorefresh); degrades to no refresh, never to an error."""
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=minutes * 60 * 1000, key=key)
    except Exception:                                                 # noqa: BLE001
        pass


# ═════════════════════════════════════════════════════════════════════════════════════════════
# TABLES — numbers stay numbers; format with column_config at render time
# ═════════════════════════════════════════════════════════════════════════════════════════════
def pct_col(label: str, *, help: str = "", width: str | None = None):
    """A 0-1 probability shown as a percentage, still sorting numerically."""
    return st.column_config.NumberColumn(label, help=help or None, format="percent", width=width)


def bar_col(label: str, *, max_value: float, help: str = "", fmt: str = "%.2f", min_value: float = 0):
    """A value drawn as a filled bar. Needs an explicit max — an auto-scaled bar changes meaning."""
    return st.column_config.ProgressColumn(label, help=help or None, format=fmt,
                                           min_value=min_value, max_value=max_value)


def num_col(label: str, *, fmt: str = "%.2f", help: str = "", width: str | None = None):
    return st.column_config.NumberColumn(label, help=help or None, format=fmt, width=width)


def money_col(label: str = "£m", *, help: str = ""):
    return st.column_config.NumberColumn(label, help=help or None, format="£%.1f")


def text_col(label: str, *, help: str = "", width: str | None = None):
    return st.column_config.TextColumn(label, help=help or None, width=width)


def spark_col(label: str, *, help: str = "", y_min=None, y_max=None):
    """A per-row trend drawn in the cell. Set y_min/y_max when rows are compared with each other."""
    return st.column_config.LineChartColumn(label, help=help or None, y_min=y_min, y_max=y_max)


# Percentage columns, split by the CONVENTION OF THEIR SOURCE (never detected from the data — a
# filter can move the max and 0.8% would render as 80%).
_PCT_FRACTION = ("Start%", "P(goal)", "P(assist)", "P(SOT2+)", "P(start)", "Hit %", "Win %")
_PCT_OF_100 = ("Owned %", "Own %", "League own %", "Selected %")
_PCT = _PCT_FRACTION + _PCT_OF_100
_MONEY = ("£m", "Price")
_POINTS = ("Exp pts", "Pts", "xPts", "Projected", "Total pts", "xpts")
_PLAIN2 = ("Pts/£", "Def", "CS", "Bon", "xPts·rot", "Avg FDR", "FDR", "MAE", "Bias",
           "err_ours", "err_fpl", "mae_ours", "mae_fpl")
_INT = ("#", "GW", "gw", "Pens (career)", "Set-piece goals", "n_dgw", "n_bgw", "players")


def auto_config(df, *, bars: tuple[str, ...] = (),
                sparks: dict[str, dict] | tuple[str, ...] = ()) -> dict:
    """A column_config for whatever recognised columns `df` has (see the maps above)."""
    _spark_opts = ({k: {} for k in sparks} if isinstance(sparks, (tuple, list)) else dict(sparks))
    cfg: dict = {}
    for c in df.columns:
        if c in _spark_opts:
            cfg[c] = spark_col(c, **_spark_opts[c])
        elif c in bars:
            try:
                mx = float(pd.to_numeric(df[c], errors="coerce").max())
            except Exception:                                         # noqa: BLE001
                mx = 1.0
            cfg[c] = bar_col(c, max_value=max(mx, 0.1))
        elif c in _PCT_OF_100:
            cfg[c] = num_col(c, fmt="%.1f%%")
        elif c in _PCT_FRACTION:
            cfg[c] = pct_col(c)
        elif c in _MONEY:
            cfg[c] = money_col(c)
        elif c in _POINTS or c in _PLAIN2:
            cfg[c] = num_col(c, fmt="%.2f")
        elif c in _INT:
            cfg[c] = st.column_config.NumberColumn(c, format="%d", width="small")
    return cfg


def _stringify_mixed(df: pd.DataFrame) -> pd.DataFrame:
    """Arrow cannot type a column holding both '2026-09-22T01:39:02Z' and 59859, and Streamlit's
    automatic repair silently coerces. Stringify ONLY genuinely mixed object columns."""
    d = df
    for c in df.columns:
        if df[c].dtype == "object":
            kinds = {type(v).__name__ for v in df[c].dropna().head(200)}
            if len(kinds) > 1:
                if d is df:
                    d = df.copy()
                d[c] = d[c].map(lambda v: "" if v is None else str(v))
    return d


def table(df, *, bars: tuple[str, ...] = (), sparks: dict[str, dict] | tuple[str, ...] = (),
          height: int | None = None, container=None, auto_format: bool = True,
          column_config: dict | None = None, **kw) -> None:
    """`st.dataframe` with the shared formatting. Drop-in for raw calls.

    `height=None` is omitted (Streamlit rejects it). Mixed-type object columns are stringified so
    Arrow cannot silently coerce a value. Explicit `column_config` entries win over the automatic map.
    """
    target = container if container is not None else st
    if height is not None:
        kw["height"] = height
    cfg = auto_config(df, bars=bars, sparks=sparks) if auto_format else {}
    if column_config:
        cfg.update(column_config)
    target.dataframe(_stringify_mixed(df), hide_index=True, width="stretch",
                     column_config=cfg or None, **kw)


def compact_table(df, column_config: dict | None = None, *, height: int | None = None, **kw) -> None:
    """A dense, sortable table with explicit column formats (the preferred form for new code)."""
    table(df, column_config=column_config, height=height, **kw)


# ═════════════════════════════════════════════════════════════════════════════════════════════
# CHARTS (altair imported lazily: it costs ~0.3 s and most pages never draw one)
# ═════════════════════════════════════════════════════════════════════════════════════════════
def _axes():
    import altair as alt
    t = tokens()
    common = dict(labelColor=t["TEXT_MUTED"], titleColor=t["TEXT_MUTED"], domainColor=t["BORDER"],
                  tickColor=t["BORDER"])
    return alt.Axis(grid=False, **common), alt.Axis(grid=True, gridColor=t["BORDER"], **common)


def _base(df: pd.DataFrame, h: int):
    import altair as alt
    return alt.Chart(df).properties(height=h)


def pnl_bars(df: pd.DataFrame, x: str, y: str, n_col: str | None = None, title: str = "", h: int = 320,
             *, axis_title: str = "P/L (units, flat 1u)", value_title: str = "P/L (u)", fmt: str = "+.2f",
             n_title: str = "bets"):
    """Signed magnitude around zero; n rides in the tooltip so 4 bets never look like 400.

    Defaults draw P/L in units; `axis_title` / `value_title` / `fmt` let any other SIGNED measure (an
    ROI %, a coefficient, an error in points) use the same grammar without being mislabelled as P/L."""
    import altair as alt
    ax, axg = _axes()
    _pos, _neg = polarity()
    tips = [alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{y}:Q", format=fmt, title=value_title)]
    if n_col and n_col in df.columns:
        tips.append(alt.Tooltip(f"{n_col}:Q", title=n_title, format=","))
    return _base(df, h).mark_bar(cornerRadiusEnd=4, size=18).encode(
        y=alt.Y(f"{x}:N", sort="-x", title=None, axis=ax),
        x=alt.X(f"{y}:Q", title=axis_title, axis=axg),
        color=alt.condition(alt.datum[y] >= 0, alt.value(_pos), alt.value(_neg)),
        tooltip=tips,
    ).properties(title=title)


def rate_bars(df: pd.DataFrame, x: str, y: str, n_col: str, ref: float | None = None,
              title: str = "", h: int = 320, fmt: str = ".1%"):
    """A rate per category, n in the tooltip, optional reference line (what the rate must beat)."""
    import altair as alt
    ax, _ = _axes()
    t = tokens()
    bars = _base(df, h).mark_bar(cornerRadiusEnd=4, size=18).encode(
        y=alt.Y(f"{x}:N", sort="-x", title=None, axis=ax),
        x=alt.X(f"{y}:Q", title=None, axis=alt.Axis(format=fmt, labelColor=t["TEXT_MUTED"],
                                                    grid=True, gridColor=t["BORDER"])),
        color=alt.Color(f"{n_col}:Q", scale=alt.Scale(range=RAMP), legend=alt.Legend(title="n")),
        tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{y}:Q", format=fmt),
                 alt.Tooltip(f"{n_col}:Q", title="n", format=",")],
    ).properties(title=title)
    if ref is None:
        return bars
    line = alt.Chart(pd.DataFrame({"r": [ref]})).mark_rule(
        color=t["TEXT_MUTED"], strokeDash=[4, 4], size=1).encode(x="r:Q")
    return (bars + line).resolve_scale(x="shared")


def cumulative_line(df: pd.DataFrame, x: str, y: str, colour: str | None = None,
                    title: str = "", h: int = 300):
    """A running total over time, with a zero rule so the sign is readable at a glance."""
    import altair as alt
    ax, axg = _axes()
    colour = colour or repo_colour("v9")
    zero = alt.Chart(pd.DataFrame({"z": [0.0]})).mark_rule(
        color=token("BORDER_STRONG"), size=1).encode(y="z:Q")
    line = _base(df, h).mark_line(size=2, color=colour, interpolate="monotone").encode(
        x=alt.X(f"{x}:T", title=None, axis=ax),
        y=alt.Y(f"{y}:Q", title="cumulative units", axis=axg),
        tooltip=[alt.Tooltip(f"{x}:T"), alt.Tooltip(f"{y}:Q", format="+.2f")],
    ).properties(title=title)
    return (zero + line).resolve_scale(y="shared")


def calibration(df: pd.DataFrame, pred: str, actual: str, n_col: str,
                title: str = "Claimed vs realised", h: int = 320):
    """Claimed probability against what happened, against the diagonal."""
    import altair as alt
    _, axg = _axes()
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        color=token("BORDER_STRONG"), strokeDash=[5, 5], size=1).encode(x="x:Q", y="y:Q")
    pts = _base(df, h).mark_circle(opacity=0.85).encode(
        x=alt.X(f"{pred}:Q", title="claimed probability", scale=alt.Scale(domain=[0, 1]), axis=axg),
        y=alt.Y(f"{actual}:Q", title="realised rate", scale=alt.Scale(domain=[0, 1]), axis=axg),
        size=alt.Size(f"{n_col}:Q", scale=alt.Scale(range=[40, 600]), legend=alt.Legend(title="n")),
        color=alt.value(repo_colour("v9")),
        tooltip=[alt.Tooltip(f"{pred}:Q", format=".3f"), alt.Tooltip(f"{actual}:Q", format=".3f"),
                 alt.Tooltip(f"{n_col}:Q", title="n", format=",")],
    ).properties(title=title)
    return (diag + pts).resolve_scale(x="shared", y="shared")


def compare_bars(df: pd.DataFrame, cat: str, value: str, series: str,
                 order: list[str] | None = None, title: str = "", h: int = 320,
                 fmt: str = ".3f", y_title: str = ""):
    """Two or more measures on ONE scale, side by side. Never a second y-axis."""
    import altair as alt
    ax, axg = _axes()
    sc = alt.Scale(domain=order, range=RAMP[1::2]) if order else alt.Scale(range=RAMP[1::2])
    return _base(df, h).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X(f"{cat}:N", title=None, axis=ax),
        y=alt.Y(f"{value}:Q", title=y_title or None, axis=axg),
        xOffset=f"{series}:N",
        color=alt.Color(f"{series}:N", scale=sc, legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip(f"{cat}:N"), alt.Tooltip(f"{series}:N"),
                 alt.Tooltip(f"{value}:Q", format=fmt)],
    ).properties(title=title)


def coverage_bars(df: pd.DataFrame, cat: str, value: str, title: str = "", h: int = 300):
    """Magnitude with a single-hue ramp — counts, rows, partitions, coverage."""
    import altair as alt
    ax, axg = _axes()
    return _base(df, h).mark_bar(cornerRadiusEnd=4, size=16).encode(
        y=alt.Y(f"{cat}:N", sort="-x", title=None, axis=ax),
        x=alt.X(f"{value}:Q", title=None, axis=axg),
        color=alt.Color(f"{value}:Q", scale=alt.Scale(range=RAMP), legend=None),
        tooltip=[alt.Tooltip(f"{cat}:N"), alt.Tooltip(f"{value}:Q", format=",")],
    ).properties(title=title)
