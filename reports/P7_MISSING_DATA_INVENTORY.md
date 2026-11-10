# P7 — Missing-data inventory (2026-11-09)

Owner, 2026-11-09: "u have 2 weeks to collect the missing data". The deadline is go-live, Sat 2026-11-21.
BIGGEST RULE: 100% real data, no fakes, no bugs, no defaults, no global values. Missing stays
missing unless a REAL source fills it.

Measured on v9 data-truth-100 `bdab0d9e` (live collection files read from main `f2dcf533`). Data truth
gate: rules 1-5 GREEN; **rule 3 (missing data on the served board) is GREEN**: the board is priced and
features are present on all 10 served tracks. The gaps below are history and evidence gaps (training,
bars, research), not gaps on the live board.

Classes: **COLLECTIBLE_NOW** (existing source and budget, no new terms), **NEEDS_OWNER** (new paid
source, credits, or a terms question), **UNREACHABLE** (no real source exists, with the reason),
**RESOLVED** (already closed by other work).

## Summary

| # | Gap | Size | Blocks | Real source | Cost | Class |
|---|---|---|---|---|---|---|
| 1 | **Unmatched history rows** (name spellings not resolved to the provider fixture) | 6,464 af_history rows (Greece 1,124, China 1,085, National League 697, Romania 637, Argentina 407, Turkey 394, La Liga 2 313, Japan 312, ...); HT fill 2,684 rows (Romania 1,013, China 1,011, ...) | stats/xG/inside-box fills, HT scores, player re-fetch proof for those rows | API-Football fixture lists ALREADY collected for every one of those league-seasons; dates all match; only names differ (e.g. "Dag and Red" vs "Dagenham & Redbridge", "Boston Utd" vs "Boston United") | **0 calls** (proven aliases through the resolver) then the cached `/fixtures/statistics` answers (0 calls) or about 1 call per newly matched fixture | COLLECTIBLE_NOW |
| 2 | **HT (half-time) prices, Standard**: wrongly listed as never captured. Two-sided HT O/U 0.5 and 1.5 ARE captured since 2026-08-21 (17 leagues, about 2,590 matches on main) by `scripts/capture_std_sidemarket_odds_forward.py` | the bars study never read them, so HT is NO_PRICE_DATA | HT Over 0.5 / 1.5 bars (Standard) | already in `output/standard_sidemarket_odds_history.csv` | **0 calls**: join into src/league_bars / bars_opt (IN_PERIOD_SPLIT; all inside the holdout) | COLLECTIBLE_NOW |
| 3 | **HT prices, New-Format** | none captured | HT bars (New-Format) | the SAME API-Football `/odds` answer the NF capture already fetches contains the first-half O/U market; `scripts/capture_nf_odds_forward.py` just doesn't parse it | **0 extra calls** (parser change). History before the change: UNREACHABLE (API-Football keeps pre-match odds only about a week) | COLLECTIBLE_NOW (forward) |
| 4 | **football-data entry prices (Standard O/U 2.5)**: Pro `fd_raw` has no `E2_2324.csv` (League One 2023-24); newest files stop about 2026-10-10 | 602 League One rows unpriced; about 1 month of entry prices for every Standard league | Standard O/U 2.5 bars, CLV | football-data.co.uk free CSVs (the loader already downloads them in CI). Unreachable from this PC (connection fails); reachable from GitHub Actions | free, about 20 files | COLLECTIBLE_NOW (via Actions) |
| 5 | **New-Format price history before 2025-06** | O/U 2.5 3,530 of 23,383 prediction rows priced; BTTS 3,423 | NF walk-forward bars rest on about 1 season of prices | football-data `/new/` files have 1X2 CLOSING only (no O/U, no BTTS); API-Football keeps no odds history; OddsAPI historical endpoint (paid credits) | OddsAPI historical: about 10 credits per market per snapshot; with credits already short (#13), not affordable | NEEDS_OWNER |
| 6 | **Side markets two-sided history** (BTTS before Aug 2024; O1.5/O3.5 before 2026-04; pre-Aug history stamped 00:00 with no bookmaker column) | O1.5/O3.5 bars only IN_PERIOD_SPLIT | side bars | forward capture continues (API-Football, free on the quota); history: OddsAPI historical (credits) | forward: 0 extra; history: credits | forward COLLECTIBLE_NOW (accrues); history NEEDS_OWNER |
| 7 | **xG** | the provider gives xG on about 29% of af_history matches (`af_fullstats_audit.non_null_share.home_xg` 0.2948) | xG form features (Standard excludes xG already on New-Format) | API-Football has no xG for older seasons / many leagues; Understat (see #8) | none | UNREACHABLE (provider does not have it) |
| 8 | **Understat set-piece / shots** | `[setpiece] no mapped Understat data` in the props run; `understat_status.json`: SOURCE_UNREACHABLE (timeout from GitHub runners, 2026-11-02) | props set-piece features (NaN) | Understat (no public API; scraping) | n/a | NEEDS_OWNER (terms: scraping a site with no API; "I don't want to get sued") |
| 9 | **Props quotes before 2026-10-14**; goals2 has no line; sot3/sot4 one-sided | rare-props bars impossible | rare props research bars, props CLV | forward capture (API-Football `/odds` props + OddsAPI props) accrues; history: OddsAPI historical props (credits) | forward: existing; history: credits | forward accrues; history NEEDS_OWNER |
| 10 | **Sharp-money data before 2026-06-14** | `sharp_ledger.csv` starts 2026-06-14 (2,300 rows) | sharp-agreement filter in the bars study | our own capture; history only via OddsAPI historical Pinnacle lines (credits) | credits | NEEDS_OWNER (history); accrues forward |
| 11 | **Player re-fetch: national-team rows** | 6,354 rows in 247 fixtures not in the fixture lists | props history for international matches | API-Football `/fixtures?league&season` for the national competitions (about 1 call per competition-season) then `/fixtures/players` (1 call per fixture) | about 300 calls | COLLECTIBLE_NOW |
| 12 | **Player re-fetch: refused rows** | 4,371 (side_unproven 3,564, player_not_in_answer 407, player_id_invalid 76, player_twice_in_answer 35) | props history | side_unproven: may be provable through team ids (0 calls, investigate); the rest are provider-side | 0 calls | side_unproven COLLECTIBLE_NOW (proof work); rest UNREACHABLE |
| 13 | **OddsAPI credits running out** (not a data gap; a collection risk) | 77,169 remaining at 2026-11-08 15:01Z; burn about 3,900/day; projected exhaustion in about 19.8 days (around 2026-11-28); `monthly_allowance_assumed: 100000` is an assumption in the file | predict's OddsAPI prices after go-live | the OddsAPI plan | — | NEEDS_OWNER (check the plan's reset date and allowance) |
| 14 | Pro's own store (about 6,417 matches since 2026-01-22) | — | Pro models | Pro now reads v9's clean history (1X2 bridge, specials) | — | RESOLVED |
| 15 | 1X2 prices K-League / Romania / Saudi; specials prices; 3 player ids covering two players | — | — | other agents | — | out of scope here |

## Measured on the clean training frame (committed data, the Phase 2 loader, network blocked)

58,473 feature rows; since 2025-07-01: 13,578 (New-Format 6,029, Standard 7,549). Missing share since
2025-07-01, features over 3%:

| Track | Feature | Missing | Worst leagues | Class |
|---|---|---|---|---|
| Standard | home/away_xg_last5 | 34.0% | Ligue 2, Scottish Championship/League One/League Two, National League 100%; Greece 50%; La Liga 2 25%; Serie B 23% | UNREACHABLE (the provider gives no xG there) |
| Standard | home/away_insidebox_last5 | 19.8% | (follows #1 unmatched rows and provider coverage) | partly COLLECTIBLE_NOW via #1 |
| Standard | season goals/conceded/clean-sheet rates, corners/fouls rolls, combined_sot_ratio | 7.8-8.2% | early-season rows (no prior matches) plus #1 | mostly structural (first matches of a season); the rest via #1 |
| New-Format | home/away_xg_last5 | 13.8-13.9% | K-League 55%, Ireland 32%, Argentina 31%, Finland 26%, Liga MX 21%, Saudi 19% | UNREACHABLE (provider) |
| New-Format | season goals/conceded/clean-sheet rates | 4.7% | early-season rows | structural |

Prices on the frame since 2025-07-01 (missing share):

| Price | New-Format | Standard |
|---|---|---|
| O/U 2.5 | **36.6%** | 0.4% |
| BTTS | 56.8% | 45.6% |
| Over 1.5 | 83.7% | 84.3% |
| Over 3.5 | 80.2% | 84.1% |

**New-Format O/U 2.5 (#16):** the gap is HISTORY, not today's capture. From 2026-10-01 to 11-08 the
forward capture priced 729 match entries against 660 finished fixtures in the 15 New-Format leagues (100%
or more everywhere, entries include reschedules), **except Japan J-League at 30 of 56 (54%)**. Earlier gaps
(Romania, Saudi and K-League only since Aug 2026; J-League 23% since Jul 2025) can't be refilled: API-Football
keeps pre-match odds only about a week, so the history needs OddsAPI historical credits (NEEDS_OWNER). The
J-League forward gap is COLLECTIBLE_NOW (find why half the fixtures go unpriced; 0 extra calls expected).

## Collection plan (Phase 2, COLLECTIBLE_NOW only)

1. **#1 Unmatched rows**: prove aliases from same-date provider fixtures (both teams, same date, same
   league, score agrees) through `src/team_names`; then re-run the stats, HT and player fills from the
   cached answers. 0 calls for the proof; at most about 1 call per newly matched fixture not cached.
2. **#2 Standard HT prices**: join the existing two-sided HT rows into the bars (a bars-study change; hand
   to the bars agent, no collection).
3. **#3 New-Format HT prices**: parse the first-half O/U market in `capture_nf_odds_forward.py` (0 extra
   calls); it accrues from the next capture run.
4. **#4 football-data**: download `E2_2324.csv` and refresh every current-season file into Pro
   `fd_raw` through a GitHub Actions step (free).
5. **#11 National-team fixtures**: about 300 API-Football calls through the collection workflow.
6. **#12 side_unproven**: prove sides by team ids where the answer carries them (0 calls).
7. **#16 J-League forward prices**: find why 26 of 56 recent fixtures went unpriced and fix the capture
   (0 extra calls expected).
Total API-Football calls for Phase 2: about 300 plus up to about 1 per newly matched uncached fixture
(well under the 5,000 interim limit; 0 OddsAPI credits).

Every collected value goes beside the existing data (sidecars, corrections) with provenance; no ledger is
rewritten in place.

## Phase 2 progress (2026-11-09)

| Item | Done | Commit (prep/missing-data unless noted) | Effect |
|---|---|---|---|
| 1 Unmatched rows | 74 league-scoped aliases proven (forced-opponent evidence, a known score must agree, >= 3 dates, one-to-one, never contradicted; a provider rename becomes date ranges, e.g. Regensburg); iterated to a fixpoint (3 rounds); read only by the collectors' matchers | `3ac9f507f` | insidebox-lacking rows unmatched **6,041 -> 1,008**, 0 ambiguous. Left: Argentina 407 (the owner-removed Copa de la Liga rows, expected), China 298, Greece 135, Bundesliga 2 40. The fills run in the collection job |
| 2 Standard HT prices | joined into src/league_bars + bars_opt (Over side, same snapshot rule) | branch `prep/ht-bars-join` `680f714fc` (on prep/bars-serving) | 528 / 567 bet-time HT pairs, 17 leagues, 2026-08-21..11-03; merge after prep/bars-serving, then re-run the study |
| 3 New-Format HT prices | parser in capture_nf_odds_forward.py, 0 extra calls | `fe1620192` | accrues once the parser is on main (the capture runs from main) |
| 4 football-data | scripts/collect_fd_raw.py + step 0 'fdraw' (free) | `3ac9f507f` | 85 files (17 leagues x 2022-23..2026-27) into output/collect/fd_raw with a manifest; runs in Actions (this PC cannot reach the site) |
| 5 National-team rows | scripts/collect_refetch_extra_fixtures.py + step 15a 'refetchlists' | `3ac9f507f` | 247 fixtures proven by /fixtures?ids (13 calls), then the re-fetch reads them (about 247 calls) |
| 6 side_unproven | is_home proven from the provider when the row's club resolves to the provider's side; --redo-refused + step 15b 'playerredo' | `b7cfaa481`, `f9b651edb` | 3,566 of 4,371 refused rows had a wrong history is_home (e.g. Bochum / Koln home matches stored as away); 915 calls to re-read. **The corrected is_home is stored beside the original (is_home_orig): props training should read it (props owner)** |
| 7 J-League prices | diagnosed: wrong API-Football season until the 2026-10-22 config fix (`33c643b5`); since 10-24 every J-League fixture is priced | — | the 35 fixtures of 10-09..10-21 are unrecoverable (no odds history at the provider) |
| Results sync | scripts/sync_results_from_main.py + step 0b 'syncmain'; files this branch never changed are taken from main (what go-live does anyway), af_history gains only the rows main added since the branch point (three-way, removed duplicates never return) | `42814ff2b` | fd_history to 2026-11-06, player_history months 2026-08..11 from main, af_history +28 rows (2026-11-07/08) with a sidecar; identity pipeline 70,743 -> 71,013 matches, 0 duplicates. **Rerun 'syncmain' right before the go-live merge** |

Calls for the collection run: about 13 + 247 + 915 + up to about 1 per newly matched uncached fixture
(insidebox; at most about 5,000) + the journaled steady-state steps; 0 OddsAPI credits. Full suite: 2,913
passed (the one failure, a reader-classification test for the two new scripts, fixed in `c1a44afa5`).

## OddsAPI: credits reset monthly (owner 2026-11-09)

Owner: OddsAPI "resets monthly, fine". The provider's docs: usage credits reset on the first of every
month. The projected exhaustion (#13, about 2026-11-28 at about 3,900 credits/day) is therefore no
action item. It still bounds any historical purchase below: the menu has to fit beside the live burn
(about 117,000 credits a month), or a plan change.

## Historical odds: costed menu (owner 2026-11-09: "cost it, then ask"). NOTHING SPENT.

Pricing used (The Odds API v4 docs and its historical-data page; the pricing page may be dated, so
confirm on the live site before buying):
* historical odds (FEATURED markets h2h / totals main line; one request returns a whole sport at one
  snapshot): **10 credits x markets x regions** per snapshot; empty answers are free; history from
  2020-06-06 (5-minute snapshots from 2022-09);
* historical EVENT odds (additional markets: btts, alternate_totals, player props; one event per
  request): **10 x markets RETURNED x regions** per event; history only from **2023-05-03 05:30 UTC**;
  the docs say additional markets are "currently limited to US sports and selected bookmakers", so
  soccer availability must be probed before buying;
* historical events (event ids for the event-odds endpoint): **1 credit per call**, free when empty;
* paid plans only. Terms of use: storing indefinitely and training statistical / ML models are listed
  as permitted; reselling or redistributing the data as a product is prohibited. Our use (training,
  bars, internal tools) is permitted; the data must not be published as a dataset.

Counts come from the bars-study price universes (output/bars_opt/universe_*.parquet on prep/bars-serving,
the clean-retrain prediction rows with their bet-time prices) and player_history; computed in code with a
second, independent sum (scratchpad cost.py); both agree exactly. Leagues with no OddsAPI sport key cannot
be filled at any price: National League, Portuguese Primeira Liga, Romanian Superliga, Scottish
Championship, Scottish League One, Scottish League Two.

| # | Gap | What the purchase is | Size | Credits (1 region = eu) | Unlocks |
|---|---|---|---|---|---|
| H1 | New-Format O/U 2.5 before 2026-08 | one sport-level `totals` snapshot per league-day, taken before that day's first kickoff (pre-kickoff for every match that day); 1 market x 1 region x 10 | 5,933 league-days, 14 leagues (Romania has no key) | **59,330** (uk+eu 118,660) | 17,604 of the 19,200 unpriced New-Format O/U 2.5 prediction rows (2021-07..2026-07; MLS 2,335, Argentina 2,086, Brazil 1,591, J-League 1,542, Liga MX 1,520, Saudi 1,379, K-League 1,150, China 1,053, ...): New-Format O/U 2.5 bars from the full walk-forward instead of about one season. Risk: the `totals` main line is not always 2.5; such rows stay unpriced but are still billed |
| H2 | Side markets two-sided (BTTS, O1.5, O3.5) before 2026-04, both tracks | per event: `btts` + `alternate_totals` (up to 2 markets) x 1 region x 10, plus 1 events lookup per league-day | 24,839 events, 42,126 market slots, 7,372 league-days, 2023-05-03..2026-03-31 | **up to 428,632** (billed only on markets returned) | 66,112 unpriced side rows (Standard BTTS 7,634 / O1.5 13,147 / O3.5 13,147; New-Format 9,653 / 11,361 / 11,170): O1.5 / O3.5 bars from walk-forward data instead of IN_PERIOD_SPLIT, BTTS walk-forward for the leagues priced only since 2024-08. Not reachable at any price: 32,652 rows before 2023-05-03 (no additional-market history) and 14,787 in leagues without a key |
| H3 | Props before 2026-10-14 | per event: anytime scorer, shots on target, assists, cards (4 markets) x 4 regions (uk, eu, us, us2: props come from US books) x 10, plus lookups | 26,543 fixtures in the 24 OddsAPI prop leagues, 2023-05-03..2026-10-13 | **upper bound 4,254,634**; realistic **about 1,185,182** (only the 7,393 fixtures of the top-5 leagues + MLS are likely to return props; empty answers are free) | props CLV on real two-sided prices (today 8 rows), the rare-props exceptional bars, the props bars. goals2 has no OddsAPI line at all |
| H4 | Sharp money before 2026-06-14 | `h2h` + `totals` (2 markets) x 1 region x 10 x 8 snapshots per league-day (T-72h, -48h, -24h, -12h, -6h, -2h, -1h, close) | 7,655 league-days, 25,709 matches, 2023-06-14..2026-06-13 | **1,224,800** | the sharp-agreement filter in the bars study (excluded today because its data starts 2026-06-14), backtested over 3 seasons |
| — | **All four (H3 realistic)** | | | **2,897,944** | |

Plan arithmetic (prices from the provider's pricing page, which may be dated): 20K = $30/mo, 100K =
$59/mo, 5M = $119/mo, 15M = $249/mo. The live burn is about 117,000 credits a month (the quota file assumes a
100K allowance), so on a 100K plan **none** of H1-H4 fits beside the live collection. Break-even: one
month on the 5M plan (about $60 more than 100K) covers H1 + H2 + H3 (realistic) + H4 (2.9M) with about
2M to spare; H1 alone (59k) fits only after a plan step up or in a month where the live burn is below
about 40k. Before any purchase: a probe of 1 event per league for btts / alternate_totals / props
(at most 20-160 credits per league, free where empty) to confirm the soccer history exists.

## Probe result and FINAL buy list (owner 2026-11-09: "probe first"; nothing bought beyond the probe)

Probe: run 42648635984, `output/collect/oddsapi_history_probe.json`. **1,590 credits spent** (116 calls,
0 calls of unknown cost), one event per league on our busiest match day 2024-03..2025-03, 31 leagues.

What exists in OddsAPI soccer history (confirmed on the probed event):
* featured `totals` + `h2h`: every probed league (Saudi Pro League: no event on its probe day, so
  unconfirmed). BUT the featured `totals` main line is 2.5 for only **56% of New-Format events (64/114)**
  and 49% overall: the main line is often 2.25 / 2.75. So O/U 2.5 history is best bought per event.
* event `alternate_totals` (lines 0.5..3.5 and more; 1.5 and 2.5 in every probed event, 3.5 seen in 17
  leagues and hidden in the other 13 only by the probe's own 12-line summary cap) and `btts`: every
  probed league, 1-5 books.
* player props: ONLY Premier League (4 markets, 160 credits/event), Ligue 1 (2 markets, 80) and MLS
  (3 markets, 120). Every 2nd-division and New-Format prop league returned nothing (free), and so did
  Bundesliga / La Liga / Serie A on their probe days.

| Buy | What | Size | Credits | Unlocks |
|---|---|---|---|---|
| B1 = H1 + H2 together | per event `btts` + `alternate_totals` (eu), one call gives BTTS, O1.5, O2.5, O3.5; 2023-05-03 onwards; leagues with a key, Saudi excluded until confirmed | 24,527 events, 7,298 league-days | **497,838** | 72,834 unpriced rows: New-Format O/U 2.5 9,365, BTTS 8,772, O1.5 10,480, O3.5 10,289; Standard BTTS 7,634, O1.5 13,147, O3.5 13,147. Walk-forward bars for O1.5 / O3.5 (today only IN_PERIOD_SPLIT) and for New-Format O/U 2.5 and BTTS |
| B2 = H1 before 2023-05-03 | featured `totals` snapshot per league-day before the first kickoff (the only history before 2023-05-03) | 2,277 league-days, 6,860 rows | **22,770** | about 3,851 rows priced (the 56% with a 2.5 main line); the rest stays unpriced and is billed anyway |
| B3 = H3 | player props per event, Premier League / Ligue 1 / MLS only | 1,246 + 1,029 + 1,575 fixtures | **471,730** | props CLV and rare-props bars in those 3 leagues only; nothing for the 2nd divisions or New-Format (no history exists) |
| B4 = H4 | `h2h` + `totals`, 8 snapshots per league-day | 7,349 league-days | **1,175,840** | the sharp-agreement filter over 3 seasons |
| | **B1 + B2 + B3** | | **992,338** | |
| | **all four** | | **2,168,178** | |

Fit (plan prices from the provider's page, which may be dated): the live burn is about 117,000 credits a
month. B1 + B2 (520,608) or all four need a month on the 5M plan; on 100K none fits. Recommended order, by
value per credit: **B1** (the bars' largest evidence gap, 2 markets per call), then **B2** (cheap), then B3
(3 leagues only), then B4 (largest cost for one filter). Nothing beyond the probe was bought.

## Historical collector: dry-run plan (2026-11-10, 5M plan; NOTHING SPENT)

Owner 2026-11-10: "use all the 5M credits this month". Code: `src/oddsapi_history.py`,
`scripts/oddsapi_history_collect.py --dry-run` (plan: `output/collect/oddsapi_history_plan.json`, counts only).
The bought data goes ONLY to the private repo `NevixAA/wowza-exec`, orphan branch `odds-history`
(terms: storing and training allowed, redistribution prohibited); `src/oddsapi_history_adapter.py` attaches
it to universe rows by fixture id, output local / private only.

Universe: every fixture of our banks (af_history and fd_history under the owner's corrections, plus
player_history for the props-only leagues), keyed leagues only, deduplicated across sources per league-day.
A provider event is bought only when the provider's own listing of that league-day proves it is one of our
fixtures (src.team_names.resolve, one league, one day, one-to-one), so every count is an exact upper bound.
Credits are MAXIMA: billed only on markets returned, empty answers free, unmatched fixtures not bought. A
later tier never re-buys a market an earlier tier requested for the same match / league-day, snapshot and
region (C3 close asks h2h only where B2 bought totals; C4 asks uk only; C5 skips T-48h and close).

| Tier | Kind | Markets | Regions | Snapshots | Window | League-days | Matches | Odds calls | Lookups | Max credits | Cumulative | Wall clock (0.5-1 s/call) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | event | btts,alternate_totals | eu | close | 2023-05-03..2026-11-09 | 9,188 | 39,863 | 39,863 | 9,188 | **806,448** | 806,448 | 6.8-13.6 h |
| B2 | featured | totals | eu | close | 2020-06-06..2023-05-02 | 4,033 | 0 | 4,033 | 4,033 | **44,363** | 850,811 | 1.1-2.2 h |
| B4 | featured | h2h,totals | eu | T-72h/T-48h/T-24h/T-12h/T-6h/T-2h/T-1h/close | 2023-06-14..2026-06-13 | 7,850 | 0 | 62,800 | 0 | **1,256,000** | 2,106,811 | 8.7-17.4 h |
| B3 | event | props | uk,eu,us,us2 | close | 2023-05-03..2026-11-09 | 1,141 | 4,564 | 4,564 | 741 | **557,261** | 2,664,072 | 0.7-1.5 h |
| S | event | correct_score | uk,eu | close | 2023-05-03..2026-11-09 | 9,188 | 39,863 | 39,863 | 0 | **797,260** | 3,461,332 | 5.5-11.1 h |
| C1 | event | btts,alternate_totals | eu | T-48h | 2023-05-03..2026-11-09 | 9,188 | 39,863 | 39,863 | 0 | **797,260** | 4,258,592 | 5.5-11.1 h |
| C3 | featured | h2h,totals | eu | T-48h/close | 2020-06-06..2023-06-13 | 7,933 | 0 | 15,866 | 3,596 | **280,586** | 4,539,178 | 2.7-5.4 h |
| C2 | event | props | uk,eu,us,us2 | T-48h | 2023-05-03..2026-11-09 | 1,141 | 4,564 | 4,564 | 0 | **556,520** | 5,095,698 | 0.6-1.3 h |
| C4 | event | btts,alternate_totals | uk | close | 2023-05-03..2026-11-09 | 9,188 | 39,863 | 39,863 | 0 | **797,260** | 5,892,958 | 5.5-11.1 h |
| C5 | featured | h2h,totals | eu | T-72h..close (6 new) | 2020-06-06..2023-06-13 | 7,933 | 0 | 47,598 | 0 | **951,960** | 6,844,918 | 6.6-13.2 h |

Usable: 5,000,000 - floor 260,000 = 4,740,000. B1..C3 (4,539,178 max) fit; C2, C4, C5 run only on what the
returned-markets billing leaves. Props markets per the probe: Premier League 4, Ligue 1 2, MLS 3. S is
`correct_score` (listed in the provider's soccer markets; its HISTORY is unconfirmed): it runs only for leagues
where `--probe-s` (one P7 probe event per league, at most 20 credits each, free when empty) sees it returned.
Saudi Pro League is included only where the provider lists the event (the P7 probe found none); its share is
at most 179,584 of the total.
