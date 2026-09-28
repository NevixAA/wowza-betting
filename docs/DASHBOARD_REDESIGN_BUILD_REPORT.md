# Dashboard redesign — build report

Date: 2026-09-28. Written for: Nevo, as the record of what was changed, what was measured, what
was found, and what was deliberately left alone.

---

## Verdict

Done and validated. All **14 pages** render without exception under Streamlit's AppTest harness.
Rendering every page now changes **zero** predictive files — which was not true before this work,
and finding that was the most valuable thing in the mission.

---

## 1. What was found in the audit

| Finding | Measured |
|---|---|
| Pages reading outside v9 | **1 of 9** (Bet Builder only) |
| Pages reading v11 | **0** |
| Charts vs tables | **14 charts, 51 tables** — 3.6 : 1 |
| Predictive files mutated by opening the dashboard | **2** (see §4) |

---

## 2. What was built

**A read-only adapter layer, `dashboard_data/`** — 6 modules. Nothing in it trains, promotes,
stakes, writes a model input or touches a collector.

- `core.py` — repo discovery, safe reads, the freshness classifier
- `v9.py` — predictions, ledger, performance by track and tier, per-league, health, **calibration**
- `pro.py` — system contract, partitioned season store, health artifacts, walk-forward results
- `v11.py` — shadow log, residual, movement, momentum (with provenance)
- `fantasy.py` — tips, ledger, benchmark, settlement, health, availability, PL team history
- `ui.py` — the shared palette and chart grammar

**Four new pages:**

- 🏠 **Overview** — the whole estate on one screen, per repo, nothing summed across them
- 🔬 **Pro** — does retraining help, what evidence is stored, what governance enforces
- 🛰️ **V11 Market** — does the model add anything once the price is known
- 🎓 **Model Performance** — accuracy, measured separately from profit

**Chart density:** now **31 charts against 58 tables — 1.9 : 1**, from 3.6 : 1.

---

## 3. What the numbers say

### Overconfidence, computed fresh from the ledger

Wowza's edge is `model probability − 1/odds` and the ledger stores both, so the model's claim is
recoverable without loading a model file. Across **1,325 settled bets**:

- claimed **0.5306**, realised **0.4053** → overconfident by **+12.54 pp**, z = **9.29**

This reproduces the documented +13.64pp / n=792 finding on a larger sample, which is the right
kind of confirmation — an independent recomputation on more data landing in the same place.

**And the tier ladder is inverted.** SNIPER — the full-stake tier — is the *worst* calibrated:

| tier | n | claimed | realised | gap |
|---|---|---|---|---|
| VALUABLE | 777 | 0.5070 | 0.4041 | +10.29 pp |
| MARKSMAN | 260 | 0.5062 | 0.3923 | +11.39 pp |
| **SNIPER** | 288 | 0.6164 | 0.4201 | **+19.63 pp** |

If the tiers sorted signal the gaps would narrow going up. They widen. This is on the Model
Performance page at full size, not in a footnote.

### The other three headline results, all on the board

- **v11 residual:** market alone 0.2405 Brier, market + Wowza 0.2404, on 1,048 fixtures. Zero.
- **Pro walk-forward:** canonical 0.24697 beats frozen 0.25525 over 41 months. Retraining works.
- **Fantasy:** FPL's free `ep_next` beats our projection, 1.535 vs 2.164 MAE. 0 of 2 gameweeks won.

---

## 4. The defect this mission found: **the dashboard was writing to training data**

`git status` after rendering every page showed `output/af_history.parquet` and
`output/training_coverage.json` modified, both stamped at the exact minute of the render.

**Cause.** `pages/6_⚡_Live.py` called `src.data_loader.load_all_matches()` to build a
home/away/date → goals lookup. That function is the *training-data path*: it runs API-Football
enrichment and then banks the result by writing `af_history.parquet`, and it writes
`training_coverage.json` on the way through.

So opening the Live page ran an enrichment pass, potentially spent API quota, and changed two
files the models train on. Nothing about settling a live signal needs any of that.

**Fix.** The page now reads final scores straight from the committed parquets
(`fd_history`, `af_history`, `af_ht_history`). No enrichment, no API calls, no writes.

**Coverage was checked, not assumed.** `af_history` holds full-time goals on only **277 of its
89,531 rows** — it is a shots and xG store, and the scores in the old merged frame came from
`fd_history` regardless. The new lookup settles **62,617 matches from 2019-02-15 to yesterday**,
with 6,206 carrying half-time goals: the same set the old path could settle.
`fd_history.parquet` is refreshed and committed by `predict.yml` (every five minutes) and
`retrain.yml`, so reading it costs no freshness in production.

**Verified:** all 14 pages rendered again afterwards → no predictive file modified.

---

## 5. The second defect: the Fantasy board was ranking on the wrong column

`output/fantasy_tips.csv` on disk had **8 unplayable players in its top 20** — Ekitiké at rank 2
while injured, Romero at 3 while unavailable.

The ranking code was already correct: `rank_and_captains()` sorts by `xpts_uncond`
(points × start probability, hard zero for anyone ruled out). The **artifact was stale** — it
predated the fix and did not even contain the `xpts_uncond` column. Regenerated: **0 unplayable in
the top 20.**

That nearly caused a third bug. My first `dashboard_data.fantasy.board()` sorted by
`fixture_adj_pts` — the *conditional* number, what a player scores *if he plays* — which puts
Ekitiké straight back at rank 2. `board()` now sorts by `overall_rank` and the docstring says why,
because this is a mistake that comes back every time someone re-sorts on the number that looks
biggest.

The Fantasy page now publishes the startability split, so "nobody is unavailable" and "the
availability check returned nothing" can no longer look identical.

---

## 6. Two smaller corrections made along the way

**Settlement disagreed with the benchmark by 1.4 points of bias.** My first `settlement()` scored
the conditional projection; `fantasy_settle.py` scores the *unconditional* one (× P(start)),
because 46% of real gameweek scores are zero and a conditional number never claimed to predict
those. Two biases for the same gameweek on one dashboard is worse than one wrong number. Now it
reproduces the settler exactly — GW4 MAE 2.245 / bias 1.0495, matching 2.245 / 1.0495.

**Arrow was silently coercing mixed-type columns.** A table holding both `'2026-09-22T01:39:02Z'`
and `59859` triggered Streamlit's automatic type repair, which can change a value. `ui.table()`
now renders mixed object columns as text, so what was written is what is shown.

---

## 7. Palette: computed, not chosen

Every colour was run through the six checks — lightness band, chroma floor, colour-blind
separation on adjacent pairs, normal-vision separation, contrast — in **both** light and dark mode.

It caught a real failure on the first attempt. Blue for v9 against violet for Pro separates by
**ΔE 2.5 under protanopia** and only **12.0 for normal vision**, below the floor of 15. Those two
repos sit side by side on nearly every comparison here. Pro is **magenta** instead because the
measurement said so — no amount of looking at the two swatches would have revealed it.

Dark mode is selected, not flipped: its band is L 0.48–0.67 against 0.43–0.77 for light, so the
same hexes fail there. Both sets are stored and validated.

| set | light | dark |
|---|---|---|
| repos (v9 / Pro / v11 / Fantasy) | `#1E6FD9 #C026A3 #E8A020 #12916A` | `#3B82F6 #CB3FAE #B8820C #0E9B72` |
| polarity (profit / loss) | `#0894AF #C2410C` | `#12A0BC #D9632B` |

All PASS. Colour is never the only cue regardless — every state carries its word.

---

## 8. What was deliberately NOT changed

- **No model, threshold, stake, tier or selection rule.** This mission touched presentation and
  one page's data source. The SNIPER calibration finding is *reported*, not acted on — changing a
  live selection rule needs its own evidence and its own decision.
- **No training input.** The Premier League team history landed in its own file
  (`output/pl_team_history.parquet`, 4,660 rows / 2,330 fixtures / 7 seasons, 100% coverage on
  shots, corners and possession) precisely so Fantasy could use it without touching `af_history`,
  which feeds the team model's training. The betting universe is unchanged.
- **No workflow.** No cron, no cadence, no CI file.
- **The existing nine pages' behaviour**, apart from the Live page's read-only fix and two charts
  added to the Fantasy page.

---

## 9. Known limitations

- **The FPL feed is currently unreachable** from this machine — TLS inspection blocks
  `fantasy.premierleague.com`, so the API serves a cached bootstrap past its 12-hour TTL. The
  Fantasy page says so on screen rather than presenting stale data as current. CI is unaffected.
- **Fantasy settlement rests on 2 gameweeks.** The verdict against `ep_next` is consistent across
  both but the sample is small, and the page shows n.
- **Pro's `available` check is a directory test**, not a git check. A stale checkout reads as
  present; the contract's commit SHAs are shown so a reader can tell.

---

## 10. Files changed

**New:** `dashboard_data/` (7 files), `pages/0_🏠_Overview.py`, `pages/7_🔬_Pro.py`,
`pages/8_🛰️_V11_Market.py`, `pages/9_🎓_Model_Performance.py`,
`docs/DASHBOARD_THREE_REPO_REDESIGN.md`, this report, `output/pl_team_history.parquet`.

**Modified:** `pages/6_⚡_Live.py` (read-only results lookup), `pages/13_⚽_Fantasy.py`
(availability split + benchmark charts), `output/fantasy_tips.csv` and
`output/fantasy_projection_log.csv` (regenerated — the stale ranking).
