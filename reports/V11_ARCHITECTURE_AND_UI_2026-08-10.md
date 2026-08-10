# Wowza V11 — Market Intelligence Engine: Strategy Comparison & UI Report

*Prepared 2026-08-10 · read-only research + design · no code changed, no commits. All build work targets `v10/` staging under the active CODE FREEZE 2026/27.*

## Executive summary

- **The one architectural shift:** our **live** betting decision is still **model-first** — `edge = model_prob − (1/single_bookmaker_odds)` with **no de-vig, one price source, and fixed edge thresholds** (`src/betting.py:244–247`). The disciplined **market-first** engine you want (power de-vig → cross-book consensus → model as a small residual → uncertainty lower-bound → CLV gate) **already exists as code** in `src/edge_engine.py`, but **it is not wired into the live pipeline**. V11 is mostly an *integration + data-plumbing* job, not a from-scratch build.
- **Honest feasibility flag:** the sharpest anchor in your sketch — the **Betfair Exchange price** — is **geo-blocked from Israel** and our own research already treats it as unusable (`research/EDGE_RECOVERY_INVESTIGATION.md:36`). V11 must anchor on a **cross-book no-vig consensus** (achievable today via OddsAPI's multi-book median, already fetched in `sharp_tracker.py`), not the exchange.
- **Top UI win:** replace plain `st.dataframe` with **color-coded EV/edge tables** (green→red via a pandas Styler) plus **open→close CLV sparklines** (`st.column_config.LineChartColumn`) — both are low-effort, use data we already store, and instantly make the dashboard read like a pro odds screen.

---

## Part 1 — What Wowza does TODAY (grounded in the code)

### 1.1 The live core signal is model-first, single-book, no de-vig

The function that actually decides live tips is `betting.evaluate_value`, called from `predict.predict_upcoming` (`src/predict.py:311,318`). Its edge is a raw model-minus-implied difference:

```python
# src/betting.py:244-247
df["impl_prob_over"]  = 1.0 / ov_odds
df["impl_prob_under"] = 1.0 / un_odds
df["edge_over"]  = p_over  - df["impl_prob_over"]
df["edge_under"] = p_under - df["impl_prob_under"]
```

Three things to note, all load-bearing:
1. **No de-vig.** `impl_prob_over = 1/odds` and `impl_prob_under = 1/odds` are used raw. Their sum exceeds 1 (the bookmaker margin), so *both* implied probabilities are biased high and the "edge" is measured against a vigged price, not a fair price.
2. **Single price source.** `ov_odds`/`un_odds` come from the first O/U 2.5 quote OddsAPI returns per event (`src/predict.py:104–113`, `if not ov25: ov25 = pr` — first book wins, then `break`). No consensus, no best-price shopping.
3. **Fixed thresholds, model-driven tiering.** Tiers are assigned purely on the size of that model-minus-book edge (`src/betting.py:138–193`): SNIPER ≥ per-league threshold, MARKSMAN ≥ 0.14, VALUABLE ≥ 0.04 (`config.py:236–241`), with a per-league `EDGE_CEILING` = 0.19 that *downgrades* suspiciously large edges (`config.py:241`, `betting.py:175–179`) — a hand-patch that implicitly admits big model-vs-book edges are usually miscalibration, not value.

So the live answer: **we are model-first.** The only market-structure steps that touch the live tip are (a) the min-odds filter (`MIN_OVER/UNDER_ODDS = 1.75`, `config.py:343–344`) and (b) a drift *tier nudge* (below). There is **no de-vig, no consensus, no CLV gate** in the live decision.

### 1.2 A market-first engine exists — but is dormant

`src/edge_engine.py` is a complete, well-reasoned **market-first** redesign, written precisely because "model disagrees with book" produced the −57% props result (its own docstring, `edge_engine.py:1–11`). It does everything the V11 sketch asks:

- **Power de-vig** two-sided (`power_devig`, `edge_engine.py:64–82`) with proportional fallback (`:54–61`).
- **Consensus baseline** with the exact hierarchy *exchange > cross-book median > single book* (`market_baseline`, `edge_engine.py:86–94`).
- **Model as a small residual:** blend weight on the model is **capped at 0.30** and shrinks on long odds / thin samples / rare markets (`blend_weight`, `edge_engine.py:98–109`) — "market is the baseline."
- **Uncertainty lower bound** (`lower_bound_prob`, `:117–125`) → **EV lower bound** (`ev_lb`, `:128–129`).
- **CLV as a hard gate:** SNIPER/MARKSMAN require positive rolling CLV, else the pick is capped at paper-only VALUABLE (`classify`, `edge_engine.py:190–197`), and a **longshot hard-cap at odds 6.0** (`MAX_BET_ODDS`, `:32,159`).

**Critical finding (confirmed by grep):** nothing in the live path imports `edge_engine.classify`. Its only consumer is `src/clv_capture.py`, which borrows `power_devig`/`proportional_devig` for CLV math. **The market-first brain is built and parked.** V11 is largely "turn it on, feed it real consensus + CLV."

### 1.3 De-vig: how / where

- **Live decision:** none (§1.1).
- **CLV module:** `clv_capture.py:32` de-vigs two-sided prices via `power_devig` to compute a no-vig `clv_prob` — but this is measurement, not a bet gate.
- **Cross-book median exists but is siloed.** `sharp_tracker.py:120–126` already computes the **median across ALL bookmakers** for O/U 2.5 and 1X2, plus a book count (`n_books`). This is the raw material for a consensus probability — but it feeds only the *sharp-movement sidecar* (`output/sharp_tips.csv`), **never the tip's edge**. So we *have* multi-book data and simply don't use it in the decision.
- **Price feeds:** OddsAPI (`regions=eu`, O/U + h2h) is the live source; Bet365-via-API-Football supplies odds elsewhere (props, side markets). Neither Pinnacle nor a true sharp anchor is ingested (`research/audit_sharp_soft.md:91,115`).

### 1.4 CLV: measured, not gated

Two CLV modules exist — `src/clv_tracker.py` (simple `(close−bet)/bet`) and the richer `src/clv_capture.py` (`clv_pct` + no-vig `clv_prob`, `log_bet`→`capture_close`→`clv_report`). Both are **post-hoc**: they log a bet and later fill the closing line. `clv_capture`'s docstring states the intent — *"Positive, persistent CLV … is the go/no-go gate before any real stake"* (`clv_capture.py:16`) — but in live code CLV **gates nothing**. It is a report, not a decision input. (The designed gate lives only in the dormant `edge_engine.classify:190`.)

The nearest live use of market structure is the **drift nudge**: `drift.py` snapshots open→current O/U odds, labels movement Confirmed/Conflicted at ±0.03 (`config.DRIFT_CONFIRM/CONFLICT_THRESHOLD`, `drift.py:139–159`), and `betting._apply_drift_adjustment` (`betting.py:196–212`) shifts a tier up/down one notch. Useful, but it's a soft tier tweak on top of the model-first edge — not a CLV or consensus gate.

### 1.5 Why model-minus-book edge is dangerous on longshots — from *our* data

- **The edge inverts with odds.** In the props/over data, backing overs loses **monotonically from −33.5% (odds 1.0–1.5) to −70.2% (odds 10+)**, with a 1.9% win rate at 10+ (`research/audit_ultimate.md:15,45–46`). The Bundesliga 2 SNIPER-UNDER cell shows the same shape at −41.9% (`research/standard_bets_breakdown.csv:7`).
- **The mechanism is calibration leverage at long odds.** A 3-point probability over-estimate at odds 15 (truth 0.067) reads as EV +0.5 — "a huge phantom edge — yet a guaranteed loser" (`research/audit_calibration.md:24`). Because `edge = p − 1/odds`, any small miscalibration is *amplified* precisely where `1/odds` is tiny. Re-fitting Platt made it worse (−75%) by pushing 3× more candidates over the line into the −EV longshot zone (`audit_calibration.md:68`).
- **CLV+ does not save you there.** The decisive finding: longshot overs (odds 6+) drift *toward* the over (positive CLV, +0.48pp) **yet settle at −67% ROI** (`research/EDGE_RECOVERY_INVESTIGATION.md:18,62,73`). The movement is margin + noise, not information.
- **AUC_marginal ≈ 0.5.** Against the market price, the model has essentially zero incremental information; its disagreements with the price are anti-predictive (`edge_engine.py:7–8`, `audit_sharp_soft.md:4`).

**Takeaway:** a model-first edge is a *longshot machine* — it systematically finds its biggest "edges" exactly where miscalibration is most leveraged and the true price is sharpest. The V11 posture (de-vig first, model capped at ≤0.30 weight, uncertainty lower-bound, longshot hard-cap, CLV gate) is the direct antidote — and it's already coded in `edge_engine.py`.

---

## Part 2 — Head-to-head: Wowza vs the market-first tools

| Dimension | **Wowza (live, `betting.py`)** | **Wowza (dormant `edge_engine.py`)** | **OddsJam** | **RebelBetting / Trademate** | **BetBurger** | **Betfair Exchange** |
|---|---|---|---|---|---|---|
| **Probability source** | One bookmaker's O/U price (OddsAPI first quote) | Consensus: exchange > cross-book median > single | Many books → sharp/consensus prob | Sharp consensus / Pinnacle-anchored | Cross-book scan | The exchange price *is* the probability |
| **De-vig** | **None** (`1/odds` raw) | Power method, two-sided (`:64`) | Multiplicative/power/Shin | Yes (sharp-anchored) | Yes (per pair) | Not needed (near-zero vig, matched book) |
| **Consensus / dispersion** | None (single book) | Median consensus (dispersion not yet used) | Full multi-book consensus + outlier flags | Consensus + value vs sharp | Cross-book spread is the product | N/A — single true price |
| **Stale / outlier detection** | None | Implicit (median vs single) | **Core**: flags stale soft books | Value = soft book lagging sharp | **Core**: arbitrage on stale/mispriced | Market self-corrects instantly |
| **CLV role** | Measured post-hoc, gates nothing | **Hard gate** on SNIPER/MARKSMAN (`:190`) | Tracked as the north-star KPI | Tracked; edge validation | Not central (arb is instant) | Closing price = the CLV benchmark itself |
| **Model role** | **The driver** (edge = model − book) | Small **residual**, weight ≤ 0.30 | No model — market *is* the model | No model — sharp line is truth | No model — pure price arb | No model |
| **Staking / EV gating** | Fixed edge thresholds + Kelly option (off) | EV **lower bound** ≥ band floors + uncertainty penalty | +EV% threshold, Kelly | Filtered value % + staking plan | Guaranteed-profit filter | User-set |

### Concrete differences / what we're missing

1. **A fair price.** Everyone serious de-vigs *before* comparing. We compare the model to a *vigged single price* live. Biggest, cheapest gap — the de-vig code already exists.
2. **Consensus + dispersion.** OddsJam/RebelBetting treat the multi-book *consensus* as truth and the *dispersion* as a confidence signal. We fetch the median (`sharp_tracker.py:120`) but throw it away at decision time, and never compute dispersion.
3. **Stale/outlier detection.** The pros' actual edge is "a soft book hasn't moved yet." We have no notion of best-price vs consensus, so we can't detect or shop it.
4. **Model as anchor, not driver.** We invert the correct hierarchy: they start from the market and let a model nudge; we start from the model and let drift nudge. Our own AUC≈0.5 evidence says their ordering is right for us too.
5. **CLV as a gate, not a scoreboard.** We measure CLV after the fact; they let CLV decide whether an edge is real before staking. `edge_engine.classify` already implements exactly this — unused.

**Important nuance from our own audit:** the pure sharp-vs-soft line-shopping game (OddsJam/RebelBetting core) is **structurally unreachable for us** — the leagues where we're soft-beatable (2nd-div / new-format O/U) *have no prop markets and thin book coverage*, while the markets with many books are the efficient ones (`audit_sharp_soft.md:119–124`). So V11 should borrow their **discipline** (de-vig, consensus, CLV gate) rather than try to *become* a line-shopping tool. Our moat stays "obscure-league team O/U where we have an information edge," now filtered through market-first hygiene.

---

## Part 3 — The "Wowza V11 Market Intelligence Engine"

Target pipeline: `sharp market → power de-vig → consensus prob → cross-book dispersion → stale/outlier detection → our model as a RESIDUAL → uncertainty → expected CLV → EV lower bound → BET / NO BET`

| Stage | Have today (cite) | Missing | Build step (in `v10/`) |
|---|---|---|---|
| **1. Sharp market feed** | OddsAPI multi-book + Bet365-via-API-Football; `sharp_tracker` fetches all books + `n_books` (`sharp_tracker.py:91–126`) | No Pinnacle/exchange anchor | Ingest OddsAPI **per-event, all-books** into the decision path (not just the sidecar). Add Pinnacle later. |
| **2. Power de-vig** | `power_devig` / `proportional_devig` done (`edge_engine.py:54–82`) | Not called live; **1X2 needs 3-way** de-vig (`x12_deep_research.md:85`) | Call `power_devig` on every two-sided market live; extend to 3-way for 1X2. |
| **3. Consensus prob** | `market_baseline` (exchange>median>single) done (`edge_engine.py:86–94`); median in `sharp_tracker` | Consensus never reaches the tip | Feed per-book de-vigged probs into `market_baseline`; store as `p_market`. |
| **4. Cross-book dispersion** | Book count only (`n_books`) | No dispersion/variance metric | Add std/IQR of per-book fair probs → a confidence multiplier + "outlier book" flag. |
| **5. Stale/outlier book** | Drift open→close (`drift.py`); steam (`sharp_tracker.py:196–204`) | No best-price-vs-consensus gap | Compute `best_odds − consensus` and flag books lagging the median. |
| **6. Model as residual** | Blend weight ≤ 0.30, shrinks on long odds/thin n (`edge_engine.py:98–113`) | Live path uses model as driver | Replace `betting.evaluate_value` edge with `edge_engine.blend(p_model, p_market, w)`. |
| **7. Uncertainty** | `lower_bound_prob` (ECE + SE + penalties) done (`edge_engine.py:117–125`) | ECE not fed per-segment | Wire per-league/market calibration error (from `backtest.py`) into `ece`. |
| **8. Expected CLV** | `clv_capture` logs bet+close, `clv_report` aggregates | Not a rolling per-segment input | Compute rolling CLV per league×market; pass as `Candidate.rolling_clv`. |
| **9. EV lower bound** | `ev_lb` + odds-band floors (`edge_engine.py:128–129,39–43`) | — | Reuse as-is. |
| **10. BET / NO BET** | `classify` — NO_BET default, CLV-gated tiers, longshot cap (`edge_engine.py:150–198`) | Not invoked live | Route tips through `classify`; surface `tier` + `ev_lb` + `abs_edge` + reason. |

### The data question — honest feasibility

- **Betfair Exchange (timestamped): NOT feasible.** Exchanges are **geo-blocked from Israel**; our research treats the "lay overpriced longshots on Betfair" path — the most promising props inefficiency — as **blocked/dead for us** (`EDGE_RECOVERY_INVESTIGATION.md:36,102,190`). V11 should **drop the exchange tier** of `market_baseline` and anchor on the cross-book median. Consensus is our ceiling.
- **Multi-book odds: feasible today.** OddsAPI already returns all EU books per event; `sharp_tracker.py:91–126` proves we parse them. We just aren't routing them into the decision — cheapest high-value plumbing in the plan.
- **Pinnacle sharp anchor:** only via OddsAPI paid tier; **parked** (paid odds parked). Consensus-of-EU-books is the pragmatic V11 anchor.
- **Timestamped open→close:** already persisted for standard O/U (`drift.py:99–136`), new-format, and 1X2 h2h forward (commit `ed4b1bc`). Expected-CLV and stale-book signals are buildable from data we already collect — main constraint is calendar time to accumulate this season's curves.

### Governance

Design/roadmap only. Any build goes in **`v10/` staging** under **CODE FREEZE 2026/27**. V11 validated like `x12`: OOS per league×market, then **live CLV** (backtests inflate ~7×), real money only on segments +OOS **and** +CLV — consistent with the season-start staking plan and model-isolation rule.

---

## Part 4 — Dashboard UI upgrades (Streamlit, pro-odds-screen inspired)

**Current style** (`app.py`, `pages/1_📊_Dashboard.py`, `pages/6_⚡_Live.py`): dark theme, hand-rolled HTML gradient cards, plain `st.dataframe` for tables, `st_autorefresh`. Clean but static — tables are un-styled, no visual encoding of edge magnitude or line movement.

### Prioritized — highest impact, lowest effort

**1. Color-coded EV/edge tables (green +EV → red −EV).** ⭐ *Biggest win, ~1 hr.*
Replace plain `st.dataframe` (Dashboard MARKSMAN/VALUABLE/side markets, `1_📊_Dashboard.py:204,232,340`) with a pandas `Styler`:
```python
sty = df.style.background_gradient(cmap="RdYlGn", subset=["Edge","EV"], vmin=-0.1, vmax=0.1)\
             .format({"Edge":"{:+.1%}","EV":"{:+.1%}"})
st.dataframe(sty, use_container_width=True, hide_index=True)
```
Renders like an OddsJam +EV board. Fits Dashboard, Player Props, Portfolio.

**2. Open→close / CLV sparklines per tip.** ⭐ *~2 hrs, data already exists.*
Feed per-fixture snapshot lists into `st.column_config.LineChartColumn`:
```python
st.dataframe(df, column_config={
  "odds_curve": st.column_config.LineChartColumn("Open→Close", y_min=1.5, y_max=3.0)})
```
Green up-tick = line moved our way (positive CLV) at a glance. Fits Dashboard + a new Portfolio→CLV view.

**3. Edge as an inline progress bar.** *~30 min.*
`st.column_config.ProgressColumn("Edge", min_value=0, max_value=0.25, format="%.1f%%")` — magnitude becomes pre-attentive. Every tip table.

**4. Market-movement / steam board.** ⭐ *~1–2 hrs, data already computed.*
`sharp_tracker.py` already writes `output/sharp_tips.csv` (`opening_odds→current_odds`, `drift_pct`, `signal`, `consensus_pct`, `n_books`, `steam`) — **not surfaced on any page.** Add a "Steam Board" (Live Center or new tab): Styler table sorted by signal, `drift_pct` colored green(shortening)→red(drifting), 🔥 for steam, consensus% bar. Near-zero cost — CSV already exists.

**5. BET / NO BET decision chip with EV lower bound + uncertainty.** *~1–2 hrs (pairs with V11).*
Once `edge_engine.classify` runs, render its output as a chip per SNIPER card (BET · EV≥x% · p_lb) / red NO BET + the `reason` string (`edge_engine.py:153`). Turns the opaque tier into a transparent decision. Dashboard.

### Secondary (higher effort / lower urgency)

- **Multi-book comparison grid + best-price highlight + hold% column** — needs V11 multi-book ingest (per-book prices). Best delivered with V11. `streamlit-aggrid` for real density/pinned columns; start with native Styler.
- **Consensus-vs-best-book delta column** (substitute for the geo-blocked exchange-vs-book delta) once multi-book ingest lands.
- **Pro dark theme polish + monospace odds** (`font-variant-numeric: tabular-nums`) + small tier/edge badges (reuse `drift_badge`/`odds_badge`, `1_📊_Dashboard.py:44–54`).

---

## Prioritized roadmap

**Now (design/paper only, `v10/` under freeze):**
1. **UI #1 + #2 + #4** — color-coded tables, CLV sparklines, surface the already-computed steam board. Pure dashboard, no model risk, immediate "pro screen" feel.
2. **Wire multi-book ingest into a `v10` shadow decision** — route OddsAPI all-books median + `power_devig` into a shadow `p_market`, log beside the live model edge. Zero live impact; builds the consensus dataset.

**This season (collect + validate):**
3. **Turn on `edge_engine.classify` in shadow** — de-vig → consensus → capped-model blend → EV lower bound → CLV gate; compare BET/NO_BET vs live tips on paper. Feed per-segment ECE (`backtest.py`) + rolling CLV (`clv_capture`).
4. **UI #3 + #5** — edge bars + BET/NO BET chip surfacing `ev_lb`/`p_lb`/reason.

**After the season (promote what survives):**
5. Promote market-first engine to live **only** for segments +OOS **and** +CLV live, sized per the season-start staking plan. Drop the exchange tier (geo-blocked); anchor on cross-book consensus. Add multi-book grid + hold% + consensus-delta columns.

**One-line verdict:** the market-first brain is already written (`edge_engine.py`) and the multi-book data is already fetched (`sharp_tracker.py`) — V11 is 80% *connecting existing parts* and *letting CLV decide*, not new modeling. Fastest visible payoff: three dashboard changes using data we already store.
