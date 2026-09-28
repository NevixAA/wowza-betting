# The Wowza dashboard — what it shows and why

Written for: whoever opens this dashboard next, including a future me who has forgotten the
reasoning. It explains what each page is for, what it deliberately refuses to do, and where the
numbers come from.

---

## The problem this solved

Wowza is three repositories. The dashboard was one.

Of nine pages, exactly one reached outside v9 — Bet Builder, reading a single CSV from a sibling
checkout. **Nothing had ever read v11.** So the only way to answer "is Wowza healthy" was to open
three repos in three terminals, and the only way to see what Pro had proven was to read the repo.

There was a second, quieter problem. The board carried **51 tables against 14 charts** — a ratio
of 3.6 to 1. A table is the right answer when someone needs the exact value. It is the wrong
answer when the question is "is this getting better or worse", because it makes the reader do
arithmetic the page should have done.

---

## What is now on the board

| Page | Repo | What it answers |
|---|---|---|
| 🏠 **Overview** | all three | Is the estate healthy? Where did the money go? What is currently known to be invalid? |
| 🔬 **Pro** | Pro | Does retraining help? What evidence is stored? What is governance actually enforcing? |
| 🛰️ **V11 Market** | v11 | Does the model add anything once the price is known? |
| 🎓 **Model Performance** | all three | Is the model accurate — separately from whether it made money? |
| ⚽ **Fantasy** | v9 | The FPL board, and how it scores against FPL's own published number. |
| (existing pages) | v9 | Tips, ledger, live, props, portfolio, Bet Builder — unchanged in behaviour. |

Chart-to-table ratio is now **1.9:1**, from 3.6:1.

---

## Four rules the pages follow

### 1. Nothing is summed across repos

There is no "Wowza score". v9 is measured in **units** of flat 1u stake, Pro in **log loss and
AUC**, v11 in **Brier**, Fantasy in **FPL points**. Adding those would be a fabrication with a
number on it, which is worse than no number.

Each repo also carries a **state banner** saying what it is: v9 is LIVE, Pro is RESEARCH, v11 is
SHADOW / RESEARCH ONLY. Without that, someone reads a v11 shadow log as a betting record.

### 2. Sample size travels with every rate

`ui.pct()` will not format a rate without its denominator. A 100% hit rate on 3 bets and on 300
bets are different facts and must not look alike.

Small samples are **flagged, never hidden**. A league dropped for thinness reads as "we don't bet
there"; the truth is "we have not measured it yet", and those call for opposite actions.

### 3. Freshness comes from content, never from file times

`git checkout` rewrites every file's modification time on every CI run. Anything that ages a file
by `stat()` therefore reads as roughly zero hours old and **never goes stale**.

This estate has been caught by that trap **three separate times**:

- `provenance._model_sha` hashed size and mtime, so it changed every run — 95 distinct values
  against 4 real model changes.
- `fpl_api._cached_fetch` served a six-week-old FPL snapshot, offering an injured player as a
  captaincy pick.
- The first version of the Fantasy health banner did it again.

So every age on this dashboard comes from the newest **record inside** a file, from a timestamp a
writer recorded, or from a date-partition name. An age that cannot be established is reported
**UNKNOWN and treated as old**, never as fresh — the FPL fix initially fell back to mtime when its
sidecar was missing and deadlocked itself, because the sidecar is only written by the fetch it was
suppressing.

### 4. Colour is never the only cue, and there is never a second y-axis

Every state carries its word (`🟢 CURRENT`, `⚠ thin`, `✔`). Every palette was **computed, not
chosen** — run through checks on lightness, chroma, colour-blind separation, normal-vision
separation and contrast, in both light and dark mode.

That measurement earned its keep immediately. The obvious first pick — blue for v9, violet for
Pro — separates by **ΔE 2.5 under protanopia** and only **12.0 for normal vision**, below the
floor of 15. Those two repos sit side by side on nearly every comparison here. Pro is magenta
instead because the measurement said the violet was unreadable, which no amount of looking at it
would have revealed.

Two measures on different scales get **two charts**, never two axes. A twin-axis plot can be tuned
to show whatever relationship the author wants and the reader has no way to tell.

---

## What the pages say that is uncomfortable

A dashboard that only shows what went well is not an instrument. Three of the headline results
here are negative, and they lead their sections rather than sitting in a footnote.

### The model is overconfident, and the tier ladder is inverted

Across **1,325 settled bets** the model claimed things would happen **53.1%** of the time. They
happened **40.5%** of the time — overconfident by **+12.5 percentage points**, z = 9.3. On a
hundred bets it expects about 53 wins and gets about 41.

Worse, the gap does not shrink as you climb the tier ladder. It grows:

| tier | n | claimed | realised | overconfident by |
|---|---|---|---|---|
| VALUABLE | 777 | 0.507 | 0.404 | +10.3 pp |
| MARKSMAN | 260 | 0.506 | 0.392 | +11.4 pp |
| **SNIPER** | 288 | 0.616 | 0.420 | **+19.6 pp** |

**SNIPER is the worst-calibrated tier and it takes the full stake.** If the tiers sorted real
signal from weak signal the gaps would narrow going up. A tier that does not separate outcomes is
a label, not a signal — and it is currently driving stake size.

This is computed straight from the ledger, which stores both the edge and the price: Wowza's edge
is defined as `model probability − 1/odds`, so the claim is recoverable without loading a model.

### The model adds nothing the market did not already know

v11's residual test takes the bookmakers' prices, strips the margin out, and asks whether adding
Wowza improves the forecast. Across **1,048 fixtures** the Brier score moves by **0.0001** on a
scale where 0 is perfect and 0.25 is a coin flip.

That is indistinguishable from nothing, and it is why v11 defaults to NO_BET rather than to a
tier. Standalone accuracy cannot tell "the model knows something" from "the model knows what the
price already knew"; measuring **after the price is known** can.

### Fantasy loses to a number FPL publishes for free

FPL's own `ep_next` is more accurate than our projection — **1.535 points** of mean error against
our **2.164**, on the same players in the same gameweeks, and our projection runs about **+1.0
points high**. It has won **0 of 2** settled gameweeks.

That comparison stays on the page. A benchmark shown only when you win is not a benchmark. What
Wowza adds that `ep_next` does not is an explicit start probability, a fixture-by-fixture split
and a stated bias — useful even while the headline number loses.

### The one clearly positive result

Retraining works. Walk-forward across **41 months**, a model retrained on the canonical store
beats one left frozen on log loss (0.24697 against 0.25525), and the wider run improved **78
league-cells with 0 degraded**. The effect is small and consistent, which is what a real effect
looks like.

The consequence is a rule: **never delete old data.** Every window, decay and filter experiment
that threw history away lost to simply keeping it.

---

## Three traps the pages warn about on screen

**Every `*_snapshots` table is a change-log, not a panel.** Writers store consecutive *distinct*
values only, so at any single instant the file holds only the entities that just moved. Grouping
by `(key, timestamp)` undercounts depth badly — measured 3 books per bet where the real figure is
8, a 2.7x understatement. Carry the last observation forward per entity before aggregating. This
one storage decision has already produced three separate wrong conclusions.

**The thresholds in `config.py` are not the ones production runs.** `predict.yml` overrides them
per run. Reading `config.py` and stopping there produced the widely repeated claim that 37 of 38
staked MARKSMAN bets were below their own threshold; measured against each bet's own effective
floor the figure is 25 of 38 — still a real defect, but a different one.

**Momentum numbers published before 2026-09-10 are void.** `pd.merge_asof` returns a fresh
RangeIndex in sorted-key order, so `sort_index()` was a no-op and assigning the result back landed
every value on the wrong row. The much-quoted figures (mean reversion 0.995, fixed anchor 0.753,
shuffled residual 0.711, all beating v9's 0.703) are **unmeasured, not disproven**. The V11 page
says so in red above the chart, and the Overview page lists every invalidated result so dead
numbers stop circulating.

---

## Where the data comes from

`dashboard_data/` is a read-only adapter layer. Nothing in it trains, promotes, stakes, writes a
model input or touches a collector.

| module | reads |
|---|---|
| `core.py` | repo locations, safe file reads, the freshness classifier |
| `v9.py` | `predictions.csv`, `bets_ledger.csv`, retrain log, calibration from the ledger |
| `pro.py` | the system contract, the date-partitioned season store, health artifacts, walk-forward results |
| `v11.py` | shadow log, residual, movement detail, momentum controls |
| `fantasy.py` | tips, projection log, ledger, performance, health, PL team history |
| `ui.py` | the shared palette and chart grammar |

Pro and v11 are read from **sibling checkouts if present**. If they are not there, the page says
so plainly rather than rendering an empty chart — an empty chart reads as "measured, found
nothing", which is a different claim from "not connected".

---

## Running it

```bash
cd v9
streamlit run app.py
```

Pro and v11 panels appear automatically if `../wowzaV9-Pro` (or `../v10`) and `../wowza-v11` are
checked out beside `v9/`.
