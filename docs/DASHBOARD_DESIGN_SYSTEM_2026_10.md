# Wowza dashboard design system (2026-10)

Owner brief 2026-10-25 (visual redesign). **Presentation only**: nothing in this system computes,
selects, stakes, settles or promotes. Every value a component shows is handed to it by the page from
the same canonical source the page used before.

- Code: `dashboard_ui.py` (the one module). `dashboard_data/ui.py` is a re-export kept so older pages
  keep working; it may only define `table` (enforced by `tests/test_design_system.py`).
- Theme: `.streamlit/config.toml` mirrors the tokens for dark (default) and light. The test fails if
  they drift.

## 1. Goal and tone

Modern football analytics + professional quant dashboard + premium sportsbook feel, **not** casino.
Dark, calm, dense but uncluttered. One main number per card. Colour carries meaning or nothing.

## 2. Palette — semantic tokens

| Token | Dark | Light | Use |
|---|---|---|---|
| BACKGROUND | `#0B1016` | `#F6F8FB` | page |
| SURFACE | `#121922` | `#FFFFFF` | cards |
| SURFACE_ELEVATED | `#18212C` | `#FFFFFF` | popovers, raised cards |
| SURFACE_MUTED | `#131A23` | `#EEF2F7` | inputs, table headers (theme `secondaryBackgroundColor`) |
| BORDER | `#243040` | `#DCE3EC` | card + table borders, chart grid |
| BORDER_STRONG | `#3A4A5E` | `#B8C4D3` | zero rules, diagonals |
| TEXT_PRIMARY | `#E7EDF4` | `#0F172A` | body |
| TEXT_SECONDARY | `#A9B6C4` | `#3E4C5E` | secondary copy |
| TEXT_MUTED | `#7C8A9A` | `#64748B` | axis labels, captions |
| PRIMARY | `#4C8DF6` | `#1E6FD9` | interaction, links, neutral emphasis |
| PRIMARY_SOFT | `#4C8DF626` | `#1E6FD91F` | selected / hover fills |
| POSITIVE | `#2BB59B` | `#0E8A74` | profit, healthy, deployed |
| NEGATIVE | `#E86A3F` | `#C8461C` | loss, failure, blocked |
| WARNING | `#E2A93B` | `#B26B00` | warning, paper |
| INFO | `#5AA9E6` | `#1D74B8` | informational |
| LIVE | = POSITIVE | = POSITIVE | live deployed |
| PAPER | = WARNING | = WARNING | tracked, not staked |
| RESEARCH | `#A47EF2` | `#6D3FD1` | research / shadow |
| BLOCKED | = NEGATIVE | = NEGATIVE | blocked track |
| NO_DATA | `#7C8A9A` | `#64748B` | insufficient / unavailable |
| V9 / PRO / V11 / FANTASY | `#3B82F6` `#CB3FAE` `#B8820C` `#0E9B72` | `#1E6FD9` `#C026A3` `#E8A020` `#12916A` | repo series colours (CVD-validated set kept from the earlier dashboard) |

Polarity is teal-green against vermilion rather than pure green/red: one carries blue and the other
none, so the pair stays apart under protanopia/deuteranopia. The sign is always also in the text.

**Pages never write a hex colour.** They pass a tone (`POSITIVE`, `RESEARCH`, …) or call
`token("NAME")`. Custom HTML never sets a text colour (always `inherit`), so a viewer switching theme
can never get light text on a light surface.

### How tokens reach native elements

`.streamlit/config.toml` sets Streamlit's named colours to the tokens (`greenColor = POSITIVE`,
`orangeColor = WARNING`, `violetColor = RESEARCH`, `redColor = NEGATIVE`, `blueColor = PRIMARY`,
`grayColor = NO_DATA`). Status chips are native badges (`:green-badge[...]`), so they follow the theme
without any CSS.

## 3. Colour semantics

| Colour | Means | Never means |
|---|---|---|
| Green | positive, healthy, live deployed | "a strong tier" |
| Amber | paper, warning | research |
| Violet | research, shadow | a bet |
| Red | negative, failure, blocked | "SNIPER" |
| Grey | insufficient, unavailable, legacy metadata | healthy |
| Blue | primary, neutral interaction, governance (PROTECTED) | an error |

Legacy tiers (SNIPER / MARKSMAN / VALUABLE) are **grey** chips: a tier is the size of the
model-vs-price gap, not evidence. On O/U 2.5 a bigger tier has not meant a better result.

## 4. Status vocabulary

Every status is **icon + word** (`status_chip(state)`); colour supports it, never carries it.

| State | Chip | Tone |
|---|---|---|
| ACTIVE / ACTIVE_LOW | ● ACTIVE / ◕ ACTIVE · LOW | green |
| LIVE | ◉ LIVE | green |
| PAPER | ◐ PAPER | amber |
| BLOCKED | ⊘ BLOCKED | red |
| PROTECTED | 🛡 PROTECTED | blue (governance, not an error) |
| REPAIR | ↻ REPAIR | amber |
| LEARNING | ◌ LEARNING | violet |
| RESEARCH / SHADOW | ◇ RESEARCH / ◆ SHADOW | violet |
| QUALIFIED | ✓ QUALIFIED | green |
| OBSERVE | ◎ OBSERVE | blue |
| BET (shadow engine) | ◆ BET · SHADOW | violet |
| NO BET | – NO BET | grey |
| PASS / WARN / FAIL, GREEN / AMBER / RED | ✓ / ▲ / ✕ | green / amber / red |

### Empty vs error — four different claims

`empty_state(kind)`: **NO DATA** (nothing recorded yet) · **FEED NOT CONNECTED** (not measured here,
not "measured, found nothing") · **STALE** (exists, older than its limit, not valid for decisions) ·
**ERROR** (could not be read). They never share copy or styling.

## 5. Typography

| Helper | Renders |
|---|---|
| `page_header(title, subtitle, eyebrow_text=, state=, freshness=)` | eyebrow line, `st.title`, one-line caption, state + freshness chips |
| `section_header(title, caption)` | spaced `st.subheader` + one caption (no divider) |
| `eyebrow(text)` | small uppercase letter-spaced label |
| `caption(text)` | `st.caption` |

One title per page. Section titles are plain words; at most one emoji in the page title (the
sidebar already carries the page icon).

## 6. Spacing

`space("small" | "medium" | "large")` (`st.space`) replaces `st.divider()` and `"---"`. Sections are
separated by space and a header, not by rules. Cards use `st.container(border=True)`.

## 7. Cards

**One main number, 3–5 supporting figures, one status/evidence line.**

| Component | For |
|---|---|
| `metric_card(label, value, title=, supporting=[(l, v)], status=, evidence=, n=)` | any KPI card |
| `metric_grid(items, per_row=3)` | bordered `st.metric` grid, never 5–6 columns |
| `selection_card(...)` | a tip: league · kickoff, teams, pick + odds, production + shadow chips, legacy tier last |
| `player_card(...)` | Fantasy / props player |
| `league_card(...)` / `research_card(...)` / `health_card(...)` | league, study, health check |
| `needs_attention(items)` | only meaningful warnings, worst first; healthy items are never listed |
| `freshness_badge(state, detail)` | the page's ONE freshness statement, beside the header |
| `evidence_badge(kind, n, span=, state=)` | `TRUE FORWARD · n=45 · 6 GW · ACCRUING` |
| `rank_badge(i)` | `#1` |
| `sample_marker(n, floor=30)` | `SMALL SAMPLE · n=12` beside any rate on a small n |

## 8. Number formats

| Kind | Helper | Example |
|---|---|---|
| ROI (fraction in) | `fmt_roi` | `+8.4%` |
| P/L (flat 1u) | `fmt_units` / `units` | `+12.47u` (two decimals — the basis every existing P/L uses) |
| Odds | `fmt_odds` | `1.94` |
| Probability (0–1 in) | `fmt_prob` | `63%` |
| CLV (percent in) | `fmt_clv` | `+1.2%` |
| Fantasy expected points | `fmt_xpts` | `6.42` |
| FPL price | `fmt_money` | `£8.5m` |
| Rate with n | `pct(x, n)` | `55.0% (n=20)` |

Missing values render as `—`, never `0`.

## 9. Charts

Altair via the shared helpers (`pnl_bars`, `rate_bars`, `cumulative_line`, `calibration`,
`compare_bars`, `coverage_bars`); altair is imported lazily. Rules: no dual axes; n in every
tooltip; a zero rule on signed charts; one-hue ramp for magnitude (`RAMP`); categorical series use
`CATEGORICAL` (also the theme's `chartCategoricalColors`) — never a rainbow; shared scales whenever
rows are compared.

## 10. Tables

- `compact_table(df, column_config)` / `table(df, bars=, sparks=)`.
- **Numbers stay numbers.** Never pre-format a number as a string in a dataframe; use
  `st.column_config` (`pct_col`, `num_col`, `bar_col`, `money_col`, `spark_col`) so sorting works.
- Progress bars always have an explicit max. Sparklines that are compared share `y_min`/`y_max`.
- A table is the accessible fallback for a chart and the place for "all rows" — it comes after the
  cards that summarise it, not instead of them.

## 11. Mobile

- At most 3 cards per row; Streamlit stacks columns on narrow screens, so a phone shows one card per
  row. Never 5–6 metric columns.
- No mandatory super-wide table: the cards carry the decision; the full table is optional detail.
- Primary filters visible (date, market, league, model, state, mode); secondary filters inside
  `st.expander("Advanced filters")`.
- Freshness is stated once, beside the page header.

## 12. Accessibility (kept from `dashboard_data/ui.py`)

Colour never the only cue; every state has text; status = icon + label; small samples show n;
colour-blind-safe pairs; no dual-axis charts; no misleading (truncated or auto-rescaled) scales.

## 13. Examples

```python
import dashboard_ui as ui

ui.page_header("Tips", "Every live tip, one side per fixture.", eyebrow_text="Today",
               freshness=ui.freshness_badge("FRESH", "newest tip 25/10 13:12 UTC"))

ui.metric_card("Staked P/L", ui.fmt_units(5.47), title="Current performance",
               supporting=[("ROI", ui.fmt_roi(0.0075)), ("Win rate", ui.pct(0.484, 725))],
               status=ui.status_chip("LIVE"), n=725)

st.markdown(ui.status_chip("PAPER") + " " + ui.evidence_badge("RETROSPECTIVE", 421))
ui.empty_state("NOT_CONNECTED", "Pro checkout not present on this host.")
```

## 14. Migration status (2026-10-25, stage 1)

Redesigned on the system: **Home**, **Tips**. Still on older styling (stages 2–3): see the build
report; they keep working unchanged through the compatibility names (`kpi`, `metric_row`,
`state_banner`, `freshness_chip`, `table`).

**Stage 2b (2026-10-25):** **League Repair Lab**, **Leagues**, **Overview**, **Live**. Page-specific
components live beside each page's data module, never in `dashboard_ui.py`:

| Page | View module | Signature pieces |
|---|---|---|
| League Repair Lab | `dashboard_data/league_research_view.py` | overview strip, league cards REPAIR → LEARNING → PROTECTED, profile card, TOP 1/2/3/5 strip, weekly rank heatmap (`heatmap_frame` = `weekly_grid` cell for cell), production vs market-first shadow (one measure per chart) |
| Leagues | `dashboard_data/leagues_view.py` | league cards with state/regime/track chips and a trend line; profile = three cards (record, context, prices) |
| Overview | `dashboard_data/overview_view.py` | overall RED / AMBER / GREEN, one health card per check (worst first), staked cards, one data-feed card per subsystem |
| Live | `dashboard_data/live_view.py` | ONE live match card (match, league, minute progress, score, signal chip, fair price, live / pre-match probability) |

`tests/test_design_system.py` now applies the no-hex / no-`<style>` / no-inline-style / no-divider /
no-plotly / `page_header` rules to all six redesigned pages and their view modules;
`tests/test_dashboard_stage2b.py` pins the displayed values to the canonical functions.
