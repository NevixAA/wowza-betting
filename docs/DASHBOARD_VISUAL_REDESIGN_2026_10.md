# Dashboard visual redesign — October 2026 (stages 1, 2a, 2b, 3)

Owner brief 2026-10-25. **Presentation and UX only.** No model, betting, research, policy, staking,
settlement, CLV, first-sight, V11, Bet Builder generation, props, bankroll-policy, workflow or Telegram
logic was changed in any stage. Every displayed value comes from the same source it came from before,
and every stage was checked by rendering the old and the new page at the same moment and comparing
what they show.

The design system itself (tokens, components, formats, rules) is documented in
`docs/DASHBOARD_DESIGN_SYSTEM_2026_10.md`. This file is the record of the migration.

**Screenshots.** None are included: every check in this document ran headlessly (Streamlit `AppTest`),
which renders the element tree but no pixels. Every claim below is about what the page displays, measured
from that tree.

---

## Stage overview

| Stage | Commits | Pages |
|---|---|---|
| 1 — one design system | `c32a9e6d` (tokens, components, theme), `5d95a64c` (Home), `a40ec124` (Tips) | Home, Tips |
| dark only | `50af39e4` | theme: one consistent dark theme |
| 2a — Fantasy | `fb505ad2` (helpers, tools split), `321a62ca` (page) | Fantasy |
| 2b | `9c5d9e91` (Repair Lab), `74bbe412` (Leagues), `218a0701` (Live), `749fd9e4` (Overview), `a4501047` (Repair Lab cache), `30d03670` (tests), merged `958a126e` | League Repair Lab, Leagues, Live, Overview |
| 3 | `fd0f8dbd` (3a), `8e52c5aa` (3b), `78f27f51` (3c), `5d0eaae3` (3d), docs commit | every remaining page |

---

## Stage 1 — one design system, Home, Tips

**Before.** Two helper sets (`dashboard_ui.py` and `dashboard_data/ui.py`) with overlapping names;
pages hand-wrote hex colours and their own `<style>` blocks; legacy tiers were coloured as if a tier
were a strength (SNIPER red); empty, not-connected, stale and error states looked the same; Home was a
wall of equal metrics plus a navigation table; Tips was a hand-built HTML board whose numbers were
strings (sorting was alphabetical).

**Changes.** `dashboard_ui.py` became the single design system: semantic tokens (dark + light sets),
status vocabulary (icon + word + tone; legacy tiers neutral grey; PROTECTED as blue governance), number
formats, typography, spacing (`space` instead of dividers), cards, empty states with four distinct claims,
freshness / evidence / rank / small-sample badges, compact tables and lazy-imported altair charts.
`dashboard_data/ui.py` became a re-export. `.streamlit/config.toml` mirrors the tokens.
Home: three hero cards (Today's action, Current performance, System state), Best opportunities, recent
performance per market, Needs attention (non-GREEN only). Tips: compact filter toolbar (secondary filters
under "Advanced filters"), Next-up selection cards, CURRENT vs SHADOW strip, sortable numeric table.

**Non-regression.** Home and Tips old vs new at the same moment: 30 checks (16 metrics, 13 tables incl. the
119-row board field by field and its order), 0 mismatches. `tests/test_home_tips_values.py` pins the board
to `tips_rows`, the settled/CLV metrics to a frozen copy of the old logic, and the production columns to
`stake_plan.csv`. Timing: Tips warm 0.57 → 0.53 s, peak working set 186 → 180 MB; Home cold 2.16 → 2.03 s.

## Stage 2a — Fantasy

**Before.** One long page mixing decisions, research and model diagnostics; the standalone projection
lost to FPL's own `ep_next` and that was not where a reader looked.

**Changes.** Organised around FPL tasks: header strip (gameweek, deadline, projection health, freshness),
then Overview / Players / Transfers / Captaincy / Fixtures / Squad / Model lab. Decision cards (SAFE /
BALANCED / HIGH UPSIDE captaincy, transfer OUT → IN cards with gains and risk, fixture swing), the
fixture heatmap with one easy = green / hard = red semantic in every lens, and all model detail moved to
Model lab so it never competes with a decision. Heavy tools stay behind buttons.

**Non-regression.** Old vs new on the same artifacts: all 13 tables, 7 chart datasets and every captaincy,
top-projected, planner and metric value (70 checks, 0 mismatches). `tests/test_fantasy_page_values.py`.
Warm rerun and peak memory equal or lower.

## Stage 2b — League Repair Lab, Leagues, Live, Overview

**Before.** Emoji markdown and success/error banners for health; `kpi()` blocks with inline colour
styles; Live drew three hand-written HTML cards (29 hex / inline-style hits) and a hex-coloured plotly
bar; Leagues showed ten equal metrics and string-formatted tables.

**Changes.** Repair Lab: overview strip, league cards REPAIR → LEARNING → PROTECTED, profile card, TOP
1/2/3/5 strip with SMALL SAMPLE markers, weekly rank heatmap, production vs market-first shadow. Leagues:
league cards with state / regime / track chips and a trend line, numeric overview table. Live: ONE live
match card. Overview: overall RED / AMBER / GREEN, one health card per check, staked cards, one data-feed
card per subsystem. Page components live in `dashboard_data/<page>_view.py`.

**Non-regression.** `tests/test_dashboard_stage2b.py`: heatmap == `league_weekly_eval.weekly_grid` cell for
cell; strip and engine frames are `league_table` rows; Repair Lab renders `league_table`'s values on a
synthetic panel and the health JSON on real data; Overview renders the adapter's staked figures offline;
the live card shows the old card's values. Overview cold 9.8 → 6.0 s (concurrent store listings), peak
177 → 167 MB; Repair Lab warm rerun 0.50 → 0.17 s.

---

## Stage 3 — every remaining page

### Before (problems found)

| Page | Problems |
|---|---|
| Portfolio | `## markdown` title, four equal metrics per tab, `st.line_chart`, secondary toggles in the main sidebar list |
| Bankroll | `st.title`, a REVIEW banner, percentages pre-formatted as strings, ROI shown without n in a wide table |
| Player Props | its own `<style>` block, 9 hex colours, 12 inline styles, a hand-built HTML table per day with hardcoded tier colours, a `---` divider |
| Bet Builder | two `st.divider`, metric row of four, info boxes for the key caveat |
| Success Rates | `st.title`, two plotly charts with 9 hex colours (green/red/grey), colour-only significance |
| Model Performance | `st.title`, three dividers, `kpi()` blocks, remote v11/Pro reads re-fetched on every interaction |
| Model Info | `## markdown` title, two dividers, two plotly charts with hardcoded dark backgrounds and a red/grey/green ramp, thresholds as strings |
| Research (Pro) | `st.title`, a plotly heatmap, emoji-colour evidence state, `kpi()` blocks, ~25–37 s cold over HTTP with every read made one after another |
| Market Lab (V11) | `st.title`, a divider, `kpi()` blocks, momentum coefficients drawn on a "P/L (units)" axis |
| League Scout | `st.title`, six equal KPI columns, colour-only emoji stage circles, a divider |
| Agent Analysis | 22 hex colours, 17 inline styles, 4 `---` dividers, verdict carried by colour |
| Account, Free tips | `st.title`, a divider |
| System Health, Model Lab, Market Intelligence, Production | `st.title`, 4–5 metric columns, freshness stated in a separate caption |

### Changes made (per page)

- **Portfolio** — header with `SIMULATED · NOTIONAL BANK` and the newest-settled-bet freshness; one card
  per market (final bankroll main; bets, win rate, weeks, max drawdown, best/worst week beside it; PAPER
  chips on props and side markets; SMALL SAMPLE marker); equity curve against a dashed starting-bankroll
  rule; numeric weekly / per-bet tables; data source / discount / post-fix under "Advanced filters".
  `simulate()` untouched. View: `dashboard_data/portfolio_view.py`.
- **Bankroll** — decision cards (production policy, Pro recommendation with "CHANGE RECOMMENDED · NO/YES",
  last review + walk-forward months); one card per track (state chip ACTIVE / ACTIVE · LOW / PAPER /
  BLOCKED, base stake main, ROI only beside n and match-days); stake-plan card and per-day bars against
  the daily cap; drawdown card + bankroll/peak chart; numeric tables. Pro review cached 5 min.
  View: `bankroll_view.py`.
- **Player Props** — the HTML board (stylesheet, hex tier colours) replaced by the SAME rows as sortable
  per-day dataframes; three "model furthest above the price" cards, each PAPER · NOT A BET; market filter
  visible, league / min % under "Advanced filters"; record card with n. View: `props_view.py`.
- **Bet Builder** — settled-record card (hit rate beside the model's expected rate, settled n, P/L at fair
  odds, void), three candidate cards (fair odds main; joint vs independent probability, naive odds,
  dependency ratio), numeric candidate table, signed dependence chart in percentage points; odds range and
  correlation filter under "Advanced filters". View: `builder_view.py`.
- **Success Rates** — header; three status cards built only from numbers the page already showed
  (observations in view, cells shown / hidden, intervals clearing zero); Period and Scope visible, cutoff
  and min-n under "Advanced filters"; plotly → altair: ROI with its 95% CI against a zero rule (decisive =
  filled diamond + legend word), achieved win % against its own break-even tick; thin groups get a SMALL
  SAMPLE chip. View: `success_view.py`.
- **Model Performance** — one card per finding with n and evidence kind (LIVE LEDGER, SHADOW LOG,
  WALK-FORWARD, GAMEWEEK BENCHMARK); section headers; remote reads cached 5 min.
- **Model Info** — header noting that thresholds are `config.py` defaults; feature-importance and backtest
  charts in the shared grammar; thresholds numeric. View: `model_info_view.py`.
- **Research (Pro)** — header with RESEARCH state and research-index freshness; evidence strip (forward
  tests, Price Trader, champions, 1X2 tips sent with n); one card per frozen forward test with a
  `TRUE FORWARD · n · matchdays · STATE` badge and the historical discovery marked NOT COUNTED; Price
  Trader state chip; VALUABLE heatmap in altair on the polarity tokens. All Pro reads concurrent + cached.
  View: `research_view.py`.
- **Market Lab (V11)** — SHADOW header with shadow-log freshness; finding cards with n and evidence kind;
  momentum coefficients labelled as coefficients; reads cached.
- **League Scout** — header with scout-report freshness; three summary cards; leagues-by-stage bar column;
  glyph + word stages; priority / country under "Advanced filters"; reads cached.
- **Agent Analysis** — shared selection card for the tip, icon + word verdict chip (`verdict()` returns a
  tone name instead of a hex; words and rules unchanged), three metric cards, scoreline bar table, numeric
  fair-odds table, primary / secondary buttons.
- **Account, Free tips** — shared header; free tips as selection cards.
- **System Health, Model Lab, Market Intelligence, Production** — page header with state chip and the one
  freshness statement (the long STALE banner still appears whenever an artifact is not fresh), section
  headers, metric rows of at most three or 2 × 2.
- **Shared** — `dashboard_ui.pnl_bars` gained optional axis / value titles and format (defaults unchanged)
  so a signed non-P/L measure is never labelled "P/L (units)"; `dashboard_data/freshness` gained
  `header_badge` / `ts_badge` / `recorded_badge`, which return the SAME verdicts as `banner()` /
  `recorded()` as one header chip. Fantasy's player-card name lost its inline style.

### Non-regression checks (stage 3)

Old and new pages were rendered back to back in separate processes (AppTest, absolute paths, admin tier;
Success Rates and free tips also as free), and everything each page displayed was extracted: metric labels,
values and deltas; every markdown, caption, header and alert; every dataframe's records. For each old value
the new page was searched; for each old dataframe the new page had to contain the same records on the same
columns (a pre-formatted string such as `1.25%` counts as equal to the number `1.25` only within the old
string's own displayed precision; a missing value `—` equals an empty cell; a status cell may change its
leading icon but never its word).

| Page | Old metric values found | Old tables reproduced | Old text numbers found |
|---|---|---|---|
| Portfolio | 16 / 16 | 8 / 8 | 11 / 11 |
| Bankroll | 11 / 11 | 6 / 6 | 34 / 34 |
| Player Props | 8 / 8 | 2 / 2 (+ the HTML board: see test) | 619 / 625 — the 6 misses are CSS values (`128,128,128,.15`, hex digits) |
| Bet Builder | 4 / 4 | 3 / 3 | 23 / 23 |
| Success Rates (admin and free) | — | 4 / 4 each | 13 / 13, 15 / 15 |
| Model Performance | — | 4 / 4 | 42 / 42 |
| Model Info | 3 / 3 | 3 / 3 | 48 / 48 |
| Research (Pro) | 11 / 11 | 35 / 35 (36 / 36 with the store listing reachable) | 194 / 194 (199 / 199) |
| Market Lab (V11) | — | 3 / 3 | 41 / 41 |
| League Scout | 5 / 5 | 3 / 3 | 22 / 22 |
| Agent Analysis | 13 / 13 | 1 / 1 | 38 / 44 — the 6 misses are CSS values (`#16213e`, `1.15em`, …) |
| Account | — | — | 4 / 4 |
| Free tips (admin and free tier) | 4 / 4 each | — | 2 / 2, 3 / 3 |
| System Health / Model Lab / Market Intelligence / Production | 5/5, 4/4, 9/9, 4/4 | 4/4, 6/6, 10/10, 4/4 | 34/34, 27/27, 13/13, 17/17 |

Totals: **97 of 97** metric values, **96 of 96** tables, **every** numeric text token except 12 CSS
values. `tests/test_dashboard_stage3.py` (31 tests) pins the key values permanently — Portfolio cards and
weekly tables against the page's own `simulate()`; Bankroll rows against the old `pct()` strings and the
stake-plan arithmetic; the props boards against the old HTML board row for row; Bet Builder settled
arithmetic and dependence pairs; the Success-rates CI rule; Model Info thresholds; Research `load_all`
returning the same call values; the VALUABLE heatmap against the old pivot; the Agent verdict words; and
every stage-3 page rendering offline with Pro / V11 unreachable.

### Timing and memory (AppTest, fresh process per render, peak working set via GetProcessMemoryInfo)

| Page | Before | After | Note |
|---|---|---|---|
| Research (Pro), cold remote cache, GitHub listing quota available | 30.2 s, 167 MB | 6.8 s, 163 MB | concurrent reads; canonical-store tab included; 36 / 36 tables and 199 / 199 text numbers equal |
| Research (Pro), cold, listing quota exhausted (both sides equally) | 16.5 s / 13.4 s, 166 MB | 5.2 s / 5.1 s, 162 MB | two alternating pairs |
| Research (Pro), warm | 3.3 s | 3.0 s | |
| Market Lab (V11) | 9.3–9.8 s, 213 MB | 2.7 s, 184 MB | reads cached |
| League Scout | 4.1 s, 133 MB | 2.0 s, 129 MB | reads cached; a bar column instead of a chart keeps altair off the page |
| Model Performance | 2.8–3.6 s, 163 MB | 2.4–2.6 s, 158 MB | |
| Bankroll | 2.4 s, 151–156 MB | 2.3–2.4 s, 152 MB | |
| Portfolio | 3.0 s, 189 MB | 2.9 s, 189 MB | |
| Player Props | 2.2 s, 165 MB | 2.2 s, 166 MB | |
| Bet Builder | 2.3 s, 223 MB | 2.5 s, 223 MB | |
| Agent Analysis, 19–22, Free tips | equal within ±0.05 s and ±1 MB | | |
| Success Rates | 2.3 s, 153 MB | 2.6 s, 173 MB | **altair import** replaces plotly |
| Model Info | 2.3 s, 135 MB | 2.3 s, 150 MB | **altair import** replaces plotly |
| Account | 0.6 s, 69 MB | 1.9 s, 121 MB | **pandas + dashboard_ui import** |

The three heavier rows are one-time imports measured in a fresh process. In the running app they are shared:
altair is already loaded by every page that draws a chart (Overview, Leagues, Live, Fantasy, Repair Lab,
Portfolio, Bankroll …) and pandas by every page, so after the first page of a session the incremental cost
is about zero. They are listed rather than hidden because the rule is "never heavier", and in a cold
process those three pages are.

---

## Answers to the brief's final questions

**A — Canonical tokens.** `dashboard_ui.DARK` / `LIGHT`: BACKGROUND, SURFACE, SURFACE_ELEVATED,
SURFACE_MUTED, BORDER, BORDER_STRONG, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, PRIMARY, PRIMARY_SOFT,
POSITIVE, NEGATIVE, WARNING, INFO, LIVE (= POSITIVE), PAPER (= WARNING), RESEARCH, BLOCKED (= NEGATIVE),
NO_DATA, and the repo series V9 / PRO / V11 / FANTASY; plus `RAMP` (one-hue magnitude) and `CATEGORICAL`.
The app runs dark only (`50af39e4`); `.streamlit/config.toml` mirrors the dark set and the test fails if
they drift. Values are in the design-system doc §2.

**B — Page-specific custom styles removed.** Across the four stages: Home's and Tips' own `<style>` blocks
and the HTML tips board; Live's three HTML cards (29 hex / inline-style hits) and hex plotly bar; Overview's
inline-coloured `kpi()` blocks; the Player Props stylesheet and HTML board (9 hex, 12 inline styles); Agent
Analysis's HTML panels (22 hex, 17 inline styles); every plotly figure with hardcoded colours (Success
Rates 2, Model Info 2, Research 1, Live 1); the last inline style in a view module (Fantasy card name).
Before stage 3 the remaining pages carried 52 hex colours, 1 `<style>`, 30 inline styles, 15 dividers,
12 `st.title` and 9 plotly references; after stage 3 all are 0. The only stylesheet in the app is
`dashboard_ui._CSS`, whose text colour is always `inherit`.

**C — Navigation grouping.** `dashboard_access.NAV`, grouped by user intent (tiers still decide visibility,
unchanged): **Today** — Home, Today's free tips, Tips, Live; **Performance** — Leagues, Portfolio, Bankroll,
Success Rates; **Tools** — Fantasy, Player Props, Bet Builder; **Research Lab** — League Repair Lab, Model
Lab, Model Performance, Market Intelligence, Research (Pro), Market Lab (V11), League Scout, Agent
Analysis; **System** — System Health, Production, Overview, Model Info, Account. File names never changed;
display labels "Pro research" → "Research" and "V11 Market" → "Market Lab".

**D — Home first viewport.** Before: equal metric tiles and a navigation table. After: three hero cards —
**Today's action** (active staked tips from the stake plan, next kickoff, paper / no-stake and shadow counts
kept apart, Best bets #1), **Current performance** (staked P/L with a cumulative sparkline, ROI, win rate,
settled n, drawdown, last 7 days; paper VALUABLE shown apart), **System state** (overall health,
freshness, settlement, CLV coverage, production / shadow engine, protected-track lock). Needs attention
lists only what is not GREEN.

**E — How Tips distinguishes the five kinds.** Production = the stake-plan `deployment_state` chip
(LIVE / ACTIVE green, PAPER amber, BLOCKED red, "NOT IN STAKE PLAN" grey); paper = the PAPER chip and the
stake label ("paper" for VALUABLE); research = the Research view toggle, which adds the shadow columns;
legacy = the SNIPER / MARKSMAN / VALUABLE tier, always last and always neutral grey with "tier ≠
confidence" on O/U 2.5; market-first = the violet `market_edge_v2` decision chip (BET · SHADOW / NO BET)
with its EV lower bound and CLV evidence badge. The CURRENT vs SHADOW strip states that a higher tier is not
stronger evidence.

**F — Fantasy around decisions.** Task tabs in the order a manager decides (Overview → Players → Transfers
→ Captaincy → Fixtures → Squad), each opening with decision cards (captain SAFE / BALANCED / HIGH UPSIDE,
OUT → IN transfer cards with next / 3 / 5-GW gain, hit-adjusted gain, fixture swing and risk; best value;
fixture swing) and the table second; every model and research diagnostic moved to Model lab.

**G — League Repair Lab visuals.** TOP 1/2/3/5 strip: per side × model type, one card per subset with
flat-1u ROI as the main number and CLV mean, "n · GW" and win rate beside it, an OK / NOT ENOUGH DATA chip
and a SMALL SAMPLE marker below the minimum n. Weekly history: an altair rank heatmap, gameweeks × ranks
#1–#5, every cell carrying WIN / LOSS / PENDING / NO ROW as a word (PROD marked) and `weekly_grid`'s own
text in the tooltip — colour is never the only cue. Current vs shadow: a PRODUCTION · legacy engine chip
beside a SHADOW · market-first chip with the overlap line ("production selected k of the shadow top-5's
rows"), then paired dots, one measure per chart on one scale — ROI, mean certified CLV, Brier, log loss —
with n in every tooltip and the numeric tables underneath.

**H — Proof nothing changed.** Stage 1: 30 old-vs-new checks, 0 mismatches + `test_home_tips_values.py`.
Stage 2a: 70 checks, 0 mismatches + `test_fantasy_page_values.py`. Stage 2b: `test_dashboard_stage2b.py`
(heatmap cell for cell, table rows, rendered values against the canonical functions). Stage 3: 97 / 97
metric values, 96 / 96 tables and every numeric text token (except 12 CSS values) found on the new pages,
plus the 31 tests in `test_dashboard_stage3.py`. No file under `src/`, `player_model/`, `telegram_bot/`,
`config/`, `.github/` or the predict / settle / stake path was edited in stage 3 (only `pages/`,
`dashboard_data/`, `dashboard_ui.py`, `tests/`, `docs/`). Full suite: 1000 passed, 24 skipped.

**I — Remaining violations.** Scan of `pages/*.py` and `dashboard_data/*_view.py` (hex colour, `<style>`,
inline style, `st.divider` / `"---"`, `st.title`, plotly): **0 hits in every file**; every page in `pages/`
uses `page_header`. `tests/test_design_system.py::test_every_page_is_on_the_design_system` keeps it that
way for any page added later.

---

## Remaining work

- **Player Props freshness.** `output/player_tips.csv` records no generation timestamp, so the page states
  its fixture / player count instead of an age. A `generated_at` column written by `player_props.yml` would
  let it show a real freshness chip (a workflow change — out of scope here).
- **Cold-process imports.** Success Rates, Model Info and Account pay a one-time altair / pandas import when
  they are the first page a fresh server process renders (table above).
- **Hardcoded research figures.** Some pages quote dated study numbers in prose (Portfolio's "+18.6u" on the
  side market, Research's VALUABLE edge bands). They were kept verbatim — changing them is a content change,
  not a visual one — and should be wired to their artifacts.
- **Research (Pro) canonical store** still needs one GitHub contents-API listing per store; under the
  anonymous 60/hour limit the tab can show "no season store". A token-free fix is a small Pro-side index
  file listing the partitions.
- **Screenshots** for the owner review need a browser run (`streamlit run app.py`); none could be taken
  headlessly.
