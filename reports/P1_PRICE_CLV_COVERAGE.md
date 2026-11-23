# P1 item 7 - Price collection and CLV reliability

Owner deep-repair brief 2026-11-22, P1 item 7. Repos: NevixAA/wowza-betting (this report, PR branch
`p1-price-clv-coverage`, base main @ 61e869877), NevixAA/wowzaV9-Pro (PR #19 merged as b6ed915a, base main @ ea772825). wowza_v11 and wowza-exec: read only, no change (below). Counts only - no price
appears anywhere in this report.

## 1. Stage A - the finding, reproduced

Source: `output/v91_shadow.csv` (prospective shadow, O/U 2.5 only) and `output/clv_collection_health.json`
(update_results shadow pass), main @ 61e869877, measured 2026-11-22 18:59 UTC.

| measure | brief | reproduced now |
|---|---|---|
| settled prospective shadow rows (WIN/LOSS) | 1,164 | 1,179 |
| settlement coverage (settled / rows >= 240 min past kickoff) | 97.7% | 1,179 / 1,205 = 97.8% (26 open: no result yet) |
| certified close (strictly pre-kickoff, <= 30 min, odds > 1.0) | 45.9% | 546 / 1,179 = 46.3% |

The brief's numbers are the same all-time aggregate one day earlier. **The all-time share mixes two
regimes**:

| kickoff week | settled | certified T-30m | share |
|---|---:|---:|---:|
| 2026-W40 | 8 | 1 | 12.5% |
| 2026-W41 | 180 | 14 | 7.8% |
| 2026-W42 | 219 | 17 | 7.8% |
| 2026-W43 | 221 | 121 | 54.8% |
| 2026-W44 | 209 | 141 | 67.5% |
| 2026-W45 | 186 | 122 | 65.6% |
| 2026-W46 | 33 | 23 | 69.7% |
| 2026-W47 | 123 | 107 | 87.0% |

W40-W42 (407 rows, 32 certified) predate the 2026-10-22 capture fix (heartbeat + keep-alive). Since
`measure_from` 2026-10-23: 510 / 715 = 71.3% certified; after the 2026-11-16 bridge fix (W47): 87.0%.
The 7-day collection view (every fixture-market the captures saw, not just shadow rows): 1,574 / 1,726 =
91.2% with an observation inside T-30m. The guard (`check_clv_collection_health.py`) is PASS.

Pro weekly audit FAIL ("main O/U (per book): near-kickoff", T-1h 30% / T-30m 19%) reproduced on Pro main
@ ea772825 (audit 2026-11-22 19:49). It measures something else: `book_odds_snapshots.csv`, the per-book
OddsAPI file written only by v9 **predict** (~6 delivered runs/day, 44 in 7 days), **all-time** (2,237
fixtures since August), as a change-log with no heartbeat. Its "standard side-markets T-30m 4%" line reads the
Bet365 capture change-log without the heartbeat, also all-time. Neither is the source v9 certifies closes
from. See section 5.

## 2. Trace (provider -> certified CLV)

| step | where | finding |
|---|---|---|
| provider | OddsAPI eu `totals` (predict, entry price); API-Football `/odds` bookmaker 8 = Bet365 (std/nf NEAR captures) | two providers, two books |
| quote capture | `scripts/capture_std_sidemarket_odds_forward.py`, `scripts/capture_nf_odds_forward.py` in `std_odds_capture.yml` / `nf_odds_capture.yml` NEAR loops (base 40, keep-alive 90, bridge, cap 220 min) | change-log + last-seen heartbeat `output/odds_heartbeat/<writer>/` |
| fixture identity | `shadow_settlement.closing_price`: league-scoped via `league_registry`, teams via `team_names.resolve_result`, same `match_date`; ambiguous fallback refused | not the bottleneck: of 205 stale shadow rows since 10-23, 200 have the fixture in the capture archive (last obs > 30 min, a timing gap); 5 have the league/date but no matched quote (Argentina 3, J-League 1, Liga MX 1 - name mapping or no quote, unresolved) |
| market & line | `MARKETS` maps ou25/ou15/ou35/btts x side to exactly one archive key | no cross-line or cross-side match (tested) |
| capture timestamp | `snapshot_ts` = fetch time (UTC); heartbeat keeps the newest pre-kickoff observation per key | |
| pre-kickoff validation | `ts < kickoff`, odds > 1.0, kickoff parseable; heartbeat refuses post-kickoff / unknown kickoff / suspended / > 6 h | no post-kickoff contamination path found (tested) |
| entry price | `pipeline._log_shadow`: `entry_odds` = predict's `odds_over25/under25` = FIRST listed OddsAPI book quoting the side; book name not recorded | entry book unknowable after the fact |
| closing price | first archive whose last pre-kickoff obs is <= 30 min (dense predict archives first, then Bet365 captures) | **427 of 510 certified closes since 10-23 are Bet365 captures against an OddsAPI entry = cross-book CLV**; 83 come from the entry's own feed (book still unrecorded) |
| certified CLV | `clv = entry / close - 1` on OK closes; stale -> `last_pre_kickoff_odds` only, CLV blank | never a stale quote relabelled (tested) |
| suspended 1.0 | `index_archives` and PR #72 guard refuse odds <= 1.0 | coverage measurement now refuses them too (was counting them) |
| dedup | change-logs keep consecutive-distinct prices; heartbeat supplies "still seen" proof | Pro's measurement ignores the heartbeat (fixed in Pro PR) |
| props CLV | `prematch_capture.yml` (AF props/lineups/injuries) - separate track, `reports/P1_PROPS_CLV_VALIDATION.md` | delivered 11 runs in 3.5 days of 8/day requested (gaps 3-7 h); not part of the O/U certified close |
| v11 | `wowza_v11` samples v9's committed predictions; its `closing_snapshot()` is labelled LAST_PRE_KICKOFF_SNAPSHOT, never a close | no change needed |
| wowza-exec | private price stores (`af-prices-*`, `odds-history`) receive dual-writes | no change needed; nothing here alters what is written |

## 3. Root cause of the remaining misses (7 days to 2026-11-22 18:59, 1,726 fixture-markets)

152 fixture-markets (8.8%) had no observation inside T-30m. Attributed against the Actions run history
(`gh run list`, startedAt/updatedAt of every std/nf capture run):

| cause | new-format | standard | all |
|---|---:|---:|---:|
| NO capture run alive inside T-30m (GitHub delivered no run; or the run hit its 220-min cap at T-19m) | 64 | 24 | **88** |
| run alive, provider (Bet365 via AF) quoted the fixture no more (e.g. Ligue 2 Nantes v Grenoble, Sochaux v Dunkerque 11-20) | 8 | 32 | 40 |
| run alive, one market pulled near kickoff, the fixture's other markets covered (mostly O/U 2.5) | 7 | 17 | 24 |

Cron reality (7 days): std_odds_capture 50 runs, 37 gaps > 60 min with no run alive (139 of 187 h
uncovered); nf_odds_capture 43 runs, 26 such gaps (128 h); Pro scout near 23 runs, 16 gaps (71 h). Delivered
runs start up to ~80 min after they are created (queued behind the previous run). Most gaps hold no
kickoff - the keep-alive/bridge handles those - but a run must end at its cap, and the next one then waits
for cron. Example: nf run capped 15:41 on 11-22 (16:00 kickoff at T-19m), next delivered 19:48: the 17:00,
18:15 and 18:30 closes were never observed.

API limits / credit floors: no budget stop in the inspected runs; AF quota ~58-62k remaining mid-afternoon,
far above MIN_QUOTA / FLOOR. Stale prices: certified closes median 1.0 min before kickoff.

## 4. What changed

### wowza-betting (v9)

| file | change |
|---|---|
| `src/capture_schedule.py` | `chain_due()` + CLI `chain-due`: dispatch the next NEAR run when the loop ends with a kickoff <= CAP_MIN (220) away; depth cap 8; refuses with no measured next kickoff |
| `.github/workflows/std_odds_capture.yml`, `nf_odds_capture.yml` | capture step `id: capture` writes the chain decision; new last step "Chain the next NEAR run" (`gh workflow run <self> -f max_hours=12`, NEAR mode only, `actions: write`, kill switch `vars.CAPTURE_CHAIN_DISABLED`) |
| `src/shadow_settlement.py` | `close_feed_identity()`: SAME_FEED_BOOK_UNRECORDED / CROSS_FEED_BET365_VS_ODDSAPI / NO_CERTIFIED_CLOSE. Labelling only, the certification rule is unchanged |
| `src/clv_collection_health.py` | collection view refuses odds <= 1.0; adds `ladder` and `ladder_by_league_market` (T-6h/3h/1h/30m/10m counts); shadow view adds `by_kickoff_week` and `close_feed_identity` counts |
| `tests/test_clv_pre_kickoff_validation.py` (new, 23) | post-kickoff / at-kickoff / suspended / no-kickoff / stale never certified; stale dense archive never hides a certified capture close; heartbeat refusals; ladder counts; chain decision + workflow wiring |
| `tests/test_same_book_market_identity.py` (new) | market/line/side key identity, league and date scoping, feed-identity labels, no label claims same-book, `same_book_clv` refuses another book's close |

`output/clv_collection_health.json` is not committed here; update_results regenerates it (new keys are additive;
the dashboard reads only existing keys).

### wowzaV9-Pro

| file | change |
|---|---|
| `src/monitoring/snapshot_coverage.py` | `since_days` window; v9 last-seen heartbeat folded into the two capture archives (`CLOSE_SOURCES`); new source `newformat_odds_history.csv`; strictly pre-kickoff (`> 0`, was `>= 0`); odds <= 1.0 refused; CLI `--days` |
| `src/monitoring/weekly_audit.py` | coverage measured on the audit window (7 d); the two v9 close sources gated with the existing MAIN_FLOORS; the per-book OddsAPI file stays gated and its message says what it is |
| `src/scout/collect.py` | `next_kickoff_min()`, `chain_due()` (330 min reach, depth 6, never after a quota stop); summary carries `chain`, `chain_reason`, `chain_depth_next` |
| `.github/workflows/pro_scout_near.yml` | "Chain the next near-loop run" after the commit, reads THIS run's near.txt summary (never the committed json), kill switch `vars.SCOUT_CHAIN_DISABLED`, `actions: write` |
| `tests/test_scout_near_chain.py`, `tests/test_snapshot_coverage_window.py` (new) | chain decision + wiring; heartbeat proof, window, suspended / at-kickoff refusal, close-source list |

wowza_v11, wowza-exec: no change. v11 never calls a snapshot a close (`closing_snapshot()` labelled
LAST_PRE_KICKOFF_SNAPSHOT; 19.1% inside 30 min) and reads v9's committed predictions; exec receives the same
dual-writes as before.

## 5. Before / after (counts; measured, not projected, unless marked replay)

| measure | before | after |
|---|---|---|
| v9 shadow certified T-30m, all time | 546 / 1,179 (46.3%) | unchanged rule; now reported beside its week timeline (W47: 107 / 123) |
| v9 shadow certified since 10-23 | 510 / 715 (71.3%) | same; feed identity: 427 cross-feed (Bet365 close vs OddsAPI entry), 83 same feed (book unrecorded), 205 no certified close |
| v9 7-day collection, obs inside T-30m | 1,574 / 1,726 | **replay with chaining**: capture alive inside T-30m 1,087 -> 1,111 / 1,111 (std), 551 -> 615 / 615 (nf); the 88 no-run misses become reachable, 64 provider misses remain |
| Pro weekly "main O/U (per book)" (all-time, change-log) | T-1h 30%, T-30m 19% FAIL | on the 7-day window it is worse and stays FAIL: 67 / 394 T-1h, 24 / 394 T-30m (per-book OddsAPI, predict cadence) |
| Pro weekly "standard side-markets" (all-time, no heartbeat) | T-30m 4% (INFO) | 7 d + heartbeat: 1,961 / 2,114 T-30m, 1,886 T-10m (gated) |
| Pro "new-format capture" (not measured) | - | 7 d + heartbeat: 1,012 / 1,184 T-30m, 923 T-10m (gated) |
| Pro scout near-loop alive time (7 d) | 110 of 181 h, 16 gaps > 60 min | chained when a priced kickoff is <= 330 min ahead (no replay; per-fixture cost bounded by close bands) |

### Coverage ladder per league x market (v9 captures, 7 days)

Window: kickoffs 2026-11-15 18:59 UTC .. 2026-11-22 18:59 UTC (7 days), change-log + last-seen heartbeat, strictly pre-kickoff, odds > 1.0.

| league | market | fixture-markets | T-6h | T-3h | T-1h | T-30m | T-10m |
|---|---|---:|---:|---:|---:|---:|---:|
| Argentina Primera Division | over25 | 3 | 3 | 3 | 2 | 2 | 2 |
| Argentina Primera Division | under25 | 3 | 3 | 3 | 2 | 2 | 2 |
| Argentina Primera Division | over15 | 3 | 3 | 3 | 2 | 2 | 2 |
| Argentina Primera Division | under15 | 3 | 3 | 3 | 2 | 2 | 2 |
| Argentina Primera Division | over35 | 3 | 3 | 3 | 2 | 2 | 2 |
| Argentina Primera Division | under35 | 3 | 3 | 3 | 2 | 2 | 2 |
| Argentina Primera Division | btts_yes | 3 | 3 | 3 | 2 | 2 | 2 |
| Argentina Primera Division | btts_no | 3 | 3 | 3 | 2 | 2 | 2 |
| Austrian Bundesliga | over25 | 5 | 4 | 3 | 3 | 3 | 3 |
| Austrian Bundesliga | under25 | 6 | 6 | 5 | 5 | 5 | 4 |
| Austrian Bundesliga | over15 | 6 | 6 | 5 | 5 | 5 | 4 |
| Austrian Bundesliga | under15 | 6 | 6 | 5 | 5 | 5 | 4 |
| Austrian Bundesliga | over35 | 6 | 6 | 5 | 5 | 5 | 4 |
| Austrian Bundesliga | under35 | 6 | 6 | 5 | 5 | 5 | 4 |
| Austrian Bundesliga | btts_yes | 6 | 6 | 5 | 5 | 5 | 4 |
| Austrian Bundesliga | btts_no | 6 | 6 | 5 | 5 | 5 | 4 |
| Belgian First Division A | over25 | 9 | 9 | 9 | 9 | 8 | 7 |
| Belgian First Division A | under25 | 9 | 9 | 9 | 9 | 8 | 7 |
| Belgian First Division A | over15 | 9 | 9 | 9 | 9 | 8 | 7 |
| Belgian First Division A | under15 | 9 | 9 | 9 | 9 | 8 | 7 |
| Belgian First Division A | over35 | 9 | 9 | 9 | 9 | 8 | 7 |
| Belgian First Division A | under35 | 9 | 9 | 9 | 9 | 8 | 7 |
| Belgian First Division A | btts_yes | 9 | 9 | 9 | 9 | 8 | 7 |
| Belgian First Division A | btts_no | 9 | 9 | 9 | 9 | 8 | 7 |
| Brazil Serie A | over25 | 12 | 12 | 12 | 12 | 12 | 12 |
| Brazil Serie A | under25 | 12 | 11 | 11 | 11 | 11 | 11 |
| Brazil Serie A | over15 | 12 | 12 | 12 | 12 | 12 | 12 |
| Brazil Serie A | under15 | 12 | 12 | 12 | 12 | 12 | 12 |
| Brazil Serie A | over35 | 12 | 12 | 12 | 12 | 12 | 12 |
| Brazil Serie A | under35 | 12 | 12 | 12 | 12 | 12 | 12 |
| Brazil Serie A | btts_yes | 12 | 12 | 12 | 12 | 12 | 12 |
| Brazil Serie A | btts_no | 12 | 12 | 12 | 12 | 12 | 12 |
| Bundesliga 2 | over25 | 9 | 8 | 7 | 7 | 7 | 6 |
| Bundesliga 2 | under25 | 9 | 8 | 7 | 7 | 7 | 6 |
| Bundesliga 2 | over15 | 9 | 9 | 7 | 7 | 7 | 6 |
| Bundesliga 2 | under15 | 9 | 9 | 7 | 7 | 7 | 6 |
| Bundesliga 2 | over35 | 9 | 9 | 7 | 7 | 7 | 6 |
| Bundesliga 2 | under35 | 9 | 9 | 7 | 7 | 7 | 6 |
| Bundesliga 2 | btts_yes | 9 | 9 | 7 | 7 | 7 | 6 |
| Bundesliga 2 | btts_no | 9 | 9 | 7 | 7 | 7 | 6 |
| Championship | over25 | 12 | 11 | 11 | 11 | 11 | 11 |
| Championship | under25 | 12 | 11 | 11 | 11 | 11 | 11 |
| Championship | over15 | 12 | 12 | 12 | 12 | 12 | 12 |
| Championship | under15 | 12 | 12 | 12 | 12 | 12 | 12 |
| Championship | over35 | 12 | 12 | 12 | 12 | 12 | 12 |
| Championship | under35 | 12 | 12 | 12 | 12 | 12 | 12 |
| Championship | btts_yes | 12 | 12 | 12 | 12 | 12 | 12 |
| Championship | btts_no | 12 | 12 | 12 | 12 | 12 | 12 |
| Denmark Superliga | over25 | 5 | 4 | 4 | 3 | 3 | 3 |
| Denmark Superliga | under25 | 5 | 5 | 5 | 4 | 3 | 3 |
| Denmark Superliga | over15 | 5 | 5 | 5 | 4 | 3 | 3 |
| Denmark Superliga | under15 | 5 | 5 | 5 | 4 | 3 | 3 |
| Denmark Superliga | over35 | 5 | 5 | 5 | 4 | 3 | 3 |
| Denmark Superliga | under35 | 5 | 5 | 5 | 4 | 3 | 3 |
| Denmark Superliga | btts_yes | 5 | 5 | 5 | 4 | 3 | 3 |
| Denmark Superliga | btts_no | 5 | 5 | 5 | 4 | 3 | 3 |
| Dutch Eredivisie | over25 | 9 | 9 | 9 | 8 | 8 | 8 |
| Dutch Eredivisie | under25 | 9 | 9 | 9 | 9 | 9 | 9 |
| Dutch Eredivisie | over15 | 9 | 9 | 9 | 9 | 9 | 9 |
| Dutch Eredivisie | under15 | 9 | 9 | 9 | 9 | 9 | 9 |
| Dutch Eredivisie | over35 | 9 | 9 | 9 | 9 | 9 | 9 |
| Dutch Eredivisie | under35 | 9 | 9 | 9 | 9 | 9 | 9 |
| Dutch Eredivisie | btts_yes | 9 | 9 | 9 | 9 | 9 | 9 |
| Dutch Eredivisie | btts_no | 9 | 9 | 9 | 9 | 9 | 9 |
| Greek Super League | over25 | 6 | 6 | 5 | 5 | 5 | 4 |
| Greek Super League | under25 | 6 | 6 | 5 | 5 | 5 | 4 |
| Greek Super League | over15 | 6 | 6 | 5 | 5 | 5 | 4 |
| Greek Super League | under15 | 6 | 6 | 5 | 5 | 5 | 4 |
| Greek Super League | over35 | 6 | 6 | 5 | 5 | 5 | 4 |
| Greek Super League | under35 | 6 | 6 | 5 | 5 | 5 | 4 |
| Greek Super League | btts_yes | 6 | 6 | 5 | 5 | 5 | 4 |
| Greek Super League | btts_no | 6 | 6 | 5 | 5 | 5 | 4 |
| Japan J-League | over25 | 10 | 10 | 10 | 10 | 10 | 10 |
| Japan J-League | under25 | 10 | 10 | 10 | 9 | 9 | 9 |
| Japan J-League | over15 | 10 | 10 | 10 | 10 | 10 | 10 |
| Japan J-League | under15 | 10 | 10 | 10 | 10 | 10 | 10 |
| Japan J-League | over35 | 10 | 10 | 10 | 10 | 10 | 10 |
| Japan J-League | under35 | 10 | 10 | 10 | 10 | 10 | 10 |
| Japan J-League | btts_yes | 10 | 10 | 10 | 10 | 10 | 10 |
| Japan J-League | btts_no | 10 | 10 | 10 | 10 | 10 | 10 |
| K-League 1 | over25 | 6 | 6 | 6 | 6 | 6 | 6 |
| K-League 1 | under25 | 6 | 6 | 6 | 6 | 6 | 6 |
| K-League 1 | over15 | 6 | 6 | 6 | 6 | 6 | 6 |
| K-League 1 | under15 | 6 | 6 | 6 | 6 | 6 | 6 |
| K-League 1 | over35 | 6 | 6 | 6 | 6 | 6 | 6 |
| K-League 1 | under35 | 6 | 6 | 6 | 6 | 6 | 6 |
| K-League 1 | btts_yes | 6 | 6 | 6 | 6 | 6 | 6 |
| K-League 1 | btts_no | 6 | 6 | 6 | 6 | 6 | 6 |
| La Liga 2 | over25 | 10 | 9 | 9 | 9 | 9 | 7 |
| La Liga 2 | under25 | 10 | 10 | 9 | 9 | 9 | 7 |
| La Liga 2 | over15 | 10 | 10 | 9 | 9 | 9 | 7 |
| La Liga 2 | under15 | 10 | 10 | 9 | 9 | 9 | 7 |
| La Liga 2 | over35 | 10 | 10 | 9 | 9 | 9 | 7 |
| La Liga 2 | under35 | 10 | 10 | 9 | 9 | 9 | 7 |
| La Liga 2 | btts_yes | 10 | 10 | 9 | 9 | 9 | 7 |
| La Liga 2 | btts_no | 10 | 10 | 9 | 9 | 9 | 7 |
| League One | over25 | 12 | 12 | 12 | 12 | 12 | 12 |
| League One | under25 | 12 | 11 | 10 | 9 | 9 | 9 |
| League One | over15 | 12 | 12 | 12 | 12 | 12 | 12 |
| League One | under15 | 12 | 12 | 12 | 12 | 12 | 12 |
| League One | over35 | 12 | 12 | 12 | 12 | 12 | 12 |
| League One | under35 | 12 | 12 | 12 | 12 | 12 | 12 |
| League One | btts_yes | 12 | 12 | 12 | 12 | 12 | 12 |
| League One | btts_no | 12 | 12 | 12 | 12 | 12 | 12 |
| League Two | over25 | 12 | 12 | 12 | 11 | 11 | 11 |
| League Two | under25 | 12 | 11 | 11 | 11 | 11 | 11 |
| League Two | over15 | 12 | 12 | 12 | 12 | 12 | 12 |
| League Two | under15 | 12 | 12 | 12 | 12 | 12 | 12 |
| League Two | over35 | 12 | 12 | 12 | 12 | 12 | 12 |
| League Two | under35 | 12 | 12 | 12 | 12 | 12 | 12 |
| League Two | btts_yes | 12 | 12 | 12 | 12 | 12 | 12 |
| League Two | btts_no | 12 | 12 | 12 | 12 | 12 | 12 |
| Ligue 2 | over25 | 9 | 7 | 7 | 7 | 7 | 7 |
| Ligue 2 | under25 | 9 | 7 | 7 | 7 | 7 | 7 |
| Ligue 2 | over15 | 9 | 7 | 7 | 7 | 7 | 7 |
| Ligue 2 | under15 | 9 | 7 | 7 | 7 | 7 | 7 |
| Ligue 2 | over35 | 9 | 7 | 7 | 7 | 7 | 7 |
| Ligue 2 | under35 | 9 | 7 | 7 | 7 | 7 | 7 |
| Ligue 2 | btts_yes | 9 | 7 | 7 | 7 | 7 | 7 |
| Ligue 2 | btts_no | 9 | 7 | 7 | 7 | 7 | 7 |
| Mexico Liga MX | over25 | 7 | 7 | 7 | 7 | 7 | 6 |
| Mexico Liga MX | under25 | 7 | 7 | 7 | 7 | 7 | 5 |
| Mexico Liga MX | over15 | 7 | 7 | 7 | 7 | 7 | 6 |
| Mexico Liga MX | under15 | 7 | 7 | 7 | 7 | 7 | 6 |
| Mexico Liga MX | over35 | 7 | 7 | 7 | 7 | 7 | 6 |
| Mexico Liga MX | under35 | 7 | 7 | 7 | 7 | 7 | 6 |
| Mexico Liga MX | btts_yes | 7 | 7 | 7 | 7 | 7 | 6 |
| Mexico Liga MX | btts_no | 7 | 7 | 7 | 7 | 7 | 6 |
| National League | over25 | 11 | 11 | 11 | 11 | 11 | 11 |
| National League | under25 | 12 | 11 | 11 | 9 | 9 | 9 |
| National League | over15 | 12 | 12 | 12 | 12 | 12 | 12 |
| National League | under15 | 12 | 12 | 12 | 12 | 12 | 12 |
| National League | over35 | 12 | 12 | 12 | 12 | 12 | 12 |
| National League | under35 | 12 | 12 | 12 | 12 | 12 | 12 |
| National League | btts_yes | 12 | 12 | 12 | 12 | 12 | 12 |
| National League | btts_no | 12 | 12 | 12 | 12 | 12 | 12 |
| Norway Eliteserien | over25 | 8 | 7 | 7 | 6 | 4 | 2 |
| Norway Eliteserien | under25 | 8 | 8 | 8 | 7 | 5 | 2 |
| Norway Eliteserien | over15 | 8 | 8 | 8 | 7 | 5 | 2 |
| Norway Eliteserien | under15 | 8 | 8 | 8 | 7 | 5 | 2 |
| Norway Eliteserien | over35 | 8 | 8 | 8 | 7 | 5 | 2 |
| Norway Eliteserien | under35 | 8 | 8 | 8 | 7 | 5 | 2 |
| Norway Eliteserien | btts_yes | 8 | 8 | 8 | 7 | 5 | 2 |
| Norway Eliteserien | btts_no | 8 | 8 | 8 | 7 | 5 | 2 |
| Romanian Superliga | over25 | 7 | 6 | 5 | 4 | 4 | 4 |
| Romanian Superliga | under25 | 7 | 7 | 5 | 4 | 4 | 4 |
| Romanian Superliga | over15 | 7 | 7 | 6 | 5 | 5 | 5 |
| Romanian Superliga | under15 | 7 | 7 | 6 | 5 | 5 | 5 |
| Romanian Superliga | over35 | 7 | 7 | 6 | 5 | 5 | 5 |
| Romanian Superliga | under35 | 7 | 7 | 6 | 5 | 5 | 5 |
| Romanian Superliga | btts_yes | 7 | 7 | 6 | 5 | 5 | 5 |
| Romanian Superliga | btts_no | 7 | 7 | 6 | 5 | 5 | 5 |
| Saudi Pro League | over25 | 1 | 1 | 1 | 1 | 1 | 0 |
| Saudi Pro League | under25 | 1 | 1 | 1 | 1 | 1 | 0 |
| Saudi Pro League | over15 | 1 | 1 | 1 | 1 | 1 | 0 |
| Saudi Pro League | under15 | 1 | 1 | 1 | 1 | 1 | 0 |
| Saudi Pro League | over35 | 1 | 1 | 1 | 1 | 1 | 0 |
| Saudi Pro League | under35 | 1 | 1 | 1 | 1 | 1 | 0 |
| Saudi Pro League | btts_yes | 1 | 1 | 1 | 1 | 1 | 0 |
| Saudi Pro League | btts_no | 1 | 1 | 1 | 1 | 1 | 0 |
| Scottish Championship | over25 | 5 | 5 | 5 | 5 | 5 | 5 |
| Scottish Championship | under25 | 5 | 5 | 5 | 5 | 5 | 5 |
| Scottish Championship | over15 | 5 | 5 | 5 | 5 | 5 | 5 |
| Scottish Championship | under15 | 5 | 5 | 5 | 5 | 5 | 5 |
| Scottish Championship | over35 | 5 | 5 | 5 | 5 | 5 | 5 |
| Scottish Championship | under35 | 5 | 5 | 5 | 5 | 5 | 5 |
| Scottish Championship | btts_yes | 5 | 5 | 5 | 5 | 5 | 5 |
| Scottish Championship | btts_no | 5 | 5 | 5 | 5 | 5 | 5 |
| Scottish League One | over25 | 5 | 5 | 5 | 5 | 5 | 5 |
| Scottish League One | under25 | 5 | 5 | 5 | 4 | 4 | 4 |
| Scottish League One | over15 | 5 | 5 | 5 | 5 | 5 | 5 |
| Scottish League One | under15 | 5 | 5 | 5 | 5 | 5 | 5 |
| Scottish League One | over35 | 5 | 5 | 5 | 5 | 5 | 5 |
| Scottish League One | under35 | 5 | 5 | 5 | 5 | 5 | 5 |
| Scottish League One | btts_yes | 5 | 5 | 5 | 5 | 5 | 5 |
| Scottish League One | btts_no | 5 | 5 | 5 | 5 | 5 | 5 |
| Scottish League Two | over25 | 5 | 4 | 4 | 4 | 4 | 4 |
| Scottish League Two | under25 | 5 | 5 | 5 | 4 | 4 | 4 |
| Scottish League Two | over15 | 5 | 5 | 5 | 5 | 5 | 5 |
| Scottish League Two | under15 | 5 | 5 | 5 | 5 | 5 | 5 |
| Scottish League Two | over35 | 5 | 5 | 5 | 5 | 5 | 5 |
| Scottish League Two | under35 | 5 | 5 | 5 | 5 | 5 | 5 |
| Scottish League Two | btts_yes | 5 | 5 | 5 | 5 | 5 | 5 |
| Scottish League Two | btts_no | 5 | 5 | 5 | 5 | 5 | 5 |
| Scottish Premiership | over25 | 6 | 6 | 5 | 5 | 5 | 5 |
| Scottish Premiership | under25 | 6 | 6 | 6 | 6 | 6 | 6 |
| Scottish Premiership | over15 | 6 | 6 | 6 | 6 | 6 | 6 |
| Scottish Premiership | under15 | 6 | 6 | 6 | 6 | 6 | 6 |
| Scottish Premiership | over35 | 6 | 6 | 6 | 6 | 6 | 6 |
| Scottish Premiership | under35 | 6 | 6 | 6 | 6 | 6 | 6 |
| Scottish Premiership | btts_yes | 6 | 6 | 6 | 6 | 6 | 6 |
| Scottish Premiership | btts_no | 6 | 6 | 6 | 6 | 6 | 6 |
| Serie B | over25 | 10 | 10 | 10 | 9 | 9 | 9 |
| Serie B | under25 | 10 | 9 | 9 | 9 | 9 | 9 |
| Serie B | over15 | 10 | 10 | 10 | 10 | 10 | 10 |
| Serie B | under15 | 10 | 10 | 10 | 10 | 10 | 10 |
| Serie B | over35 | 10 | 10 | 10 | 10 | 10 | 10 |
| Serie B | under35 | 10 | 10 | 10 | 10 | 10 | 10 |
| Serie B | btts_yes | 10 | 10 | 10 | 10 | 10 | 10 |
| Serie B | btts_no | 10 | 10 | 10 | 10 | 10 | 10 |
| Sweden Allsvenskan | over25 | 7 | 6 | 6 | 6 | 6 | 5 |
| Sweden Allsvenskan | under25 | 7 | 7 | 7 | 7 | 7 | 6 |
| Sweden Allsvenskan | over15 | 7 | 7 | 7 | 7 | 7 | 6 |
| Sweden Allsvenskan | under15 | 7 | 7 | 7 | 7 | 7 | 6 |
| Sweden Allsvenskan | over35 | 7 | 7 | 7 | 7 | 7 | 6 |
| Sweden Allsvenskan | under35 | 7 | 7 | 7 | 7 | 7 | 6 |
| Sweden Allsvenskan | btts_yes | 7 | 7 | 7 | 7 | 7 | 6 |
| Sweden Allsvenskan | btts_no | 7 | 7 | 7 | 7 | 7 | 6 |
| Turkish Super Lig | over25 | 8 | 8 | 8 | 8 | 8 | 8 |
| Turkish Super Lig | under25 | 8 | 8 | 8 | 8 | 8 | 8 |
| Turkish Super Lig | over15 | 8 | 8 | 8 | 8 | 8 | 8 |
| Turkish Super Lig | under15 | 8 | 8 | 8 | 8 | 8 | 8 |
| Turkish Super Lig | over35 | 8 | 8 | 8 | 8 | 8 | 8 |
| Turkish Super Lig | under35 | 8 | 8 | 8 | 8 | 8 | 8 |
| Turkish Super Lig | btts_yes | 8 | 8 | 8 | 8 | 8 | 8 |
| Turkish Super Lig | btts_no | 8 | 8 | 8 | 8 | 8 | 8 |
| USA MLS | over25 | 5 | 5 | 5 | 5 | 5 | 5 |
| USA MLS | under25 | 5 | 5 | 5 | 5 | 5 | 5 |
| USA MLS | over15 | 5 | 5 | 5 | 5 | 5 | 5 |
| USA MLS | under15 | 5 | 5 | 5 | 5 | 5 | 5 |
| USA MLS | over35 | 5 | 5 | 5 | 5 | 5 | 5 |
| USA MLS | under35 | 5 | 5 | 5 | 5 | 5 | 5 |
| USA MLS | btts_yes | 5 | 5 | 5 | 5 | 5 | 5 |
| USA MLS | btts_no | 5 | 5 | 5 | 5 | 5 | 5 |
| **all** | | **1726** | **1694** | **1646** | **1605** | **1574** | **1479** |

### Certified shadow closes since 2026-10-23 (O/U 2.5 only; the shadow logs no other market)

| league | settled | certified T-30m | certified T-10m | stale (> 30 min) |
|---|---:|---:|---:|---:|
| Argentina Primera Division | 55 | 28 | 26 | 27 |
| Austrian Bundesliga | 23 | 15 | 10 | 8 |
| Brazil Serie A | 44 | 34 | 29 | 10 |
| Bundesliga 2 | 36 | 27 | 25 | 9 |
| Championship | 60 | 45 | 45 | 15 |
| China Super League | 16 | 16 | 16 | 0 |
| Denmark Superliga | 21 | 12 | 10 | 9 |
| Finland Veikkausliiga | 13 | 5 | 3 | 8 |
| Ireland Premier Division | 10 | 8 | 7 | 2 |
| Japan J-League | 40 | 29 | 26 | 11 |
| La Liga 2 | 49 | 38 | 31 | 11 |
| League One | 49 | 42 | 42 | 7 |
| League Two | 48 | 38 | 38 | 10 |
| Ligue 2 | 34 | 19 | 19 | 15 |
| Mexico Liga MX | 38 | 21 | 21 | 17 |
| Norway Eliteserien | 28 | 17 | 16 | 11 |
| Serie B | 49 | 45 | 41 | 4 |
| Sweden Allsvenskan | 38 | 19 | 14 | 19 |
| USA MLS | 64 | 52 | 48 | 12 |
| **all** | **715** | **510** | **467** | **205** |

## 6. Tests and gates (local, before the first push)

wowza-betting (worktree on main @ 61e869877 + this change, then rebased onto current main; the files this PR touches did not change upstream):

| check | result |
|---|---|
| full suite `python -m pytest -q -p no:cacheprovider -x tests` | 4,127 passed, 49 skipped, 6 xfailed; tree clean afterwards |
| new `tests/test_clv_pre_kickoff_validation.py` + `tests/test_same_book_market_identity.py` | 48 passed |
| related `test_clv_capture`, `test_capture_wide`, `test_shadow_settlement`, `test_clv_types` | passed (157 with the new identity file) |
| `python -m src.integrity.data_truth --gate` | STOPLIGHT GREEN; Data truth 2: 0 new defaults (two flagged on the first run were fixed). Data truth 3 RED on the HT models is already there on main and does not block the gate |
| `python -m src.research_lock --check` | OK (protected_tracks_modified=False) |
| `python scripts/build_track_registry.py --check` | in sync (100 entries) |
| `python scripts/dashboard_manifest.py --check` | missing code [], missing data [], exit 0 (nothing new for a page to read, so --write was not needed) |
| actionlint (std/nf capture workflows) | clean |

wowzaV9-Pro: full suite 610 passed; `python -m src.pro_tests` all checks passed; actionlint clean; PR #19
CI green, merged as b6ed915a.

## 7. Cost

* Chaining (v9 std/nf): API-Football only, inside the existing daily plan. Replay: +198 (std) / +294 (nf)
  runner minutes a day; upper bound at the measured busy rate (~6 odds calls a minute: 1,392 calls over 217 min
  on Sat 11-21) = 1,187 / 1,765 AF calls a day. Quota observed mid-afternoon 58-62k remaining; MIN_QUOTA and the
  collect FLOOR still stop every pass. Actions minutes are free (public repos). No OddsAPI credit.
* Chaining (Pro scout near): API-Football; calls are per (fixture, close band), so a chained run buys only
  bands that would otherwise be lost.
* Not done (needs credits): per-book OddsAPI near-kickoff capture. Assumptions: OddsAPI `/odds` cost =
  markets x regions = 1 credit (totals, eu); one call per (league, kickoff slot) at T-30..10m and T-10..0m.
  Measured: 79 league x kickoff slots in 7 days (138 fixtures). Cost: 158 credits a week, ~677 per 30 days
  (cross-checked per day: 11.3 slots x 2 x 30 = 677), 0.26% of the 260k credit floor per month. This would
  lift the Pro per-book T-30m line from 24 / 394 and give a cross-book / sharp close. **Stopped for the owner.**

## 8. Blockers and limits

* Provider availability: 64 of 152 weekly misses are Bet365-via-AF not quoting a fixture (40) or pulling one
  market (24) near kickoff while a run was alive. No chaining fixes that; only a second close source
  (per-book OddsAPI above) would.
* Same-book CLV is not provable for the shadow today: the entry is the first-listed OddsAPI book and its name
  is not recorded; 84% of certified closes (427 / 510) are Bet365. Recording the entry book is free (it is in
  the same OddsAPI response) but changes `predictions.csv` and the frozen shadow schema -> owner decision.
* The chain only helps when GitHub delivers the dispatch; a cancelled/timed-out run does not chain (cron
  remains the fallback). Not yet observed in production: first evidence is the next live std/nf/scout run.

## 9. Production impact

* More capture runs alive near kickoff (std/nf/scout); same data files, same schemas, same commit paths.
* No change to certification (30-minute rule, pre-kickoff, odds > 1.0), staking, bankroll, promotions, bars,
  tips, Telegram or history files. `clv_collection_health.json` gains keys only; the guard's PASS/FAIL logic
  and floors are unchanged.
* Pro weekly audit: two new gated lines (currently PASS on the window); the per-book OddsAPI line stays FAIL and
  now says why.

## 10. Owner approval needed?

1. Not for the chaining (inside existing API-Football budget, kill switches `CAPTURE_CHAIN_DISABLED`,
   `SCOUT_CHAIN_DISABLED`).
2. **Yes**: per-book OddsAPI near-kickoff capture, ~677 credits / 30 days (section 7).
3. **Yes**: whether a cross-feed close (Bet365 capture vs OddsAPI entry) may count as certified CLV, and
   whether to record the entry bookmaker in predict + shadow (free; schema change).
