# Pre-kickoff capture: output contract

Owner approval: 2026-11-04. Status: **built and tested, SWITCHED OFF** until the big data collection
(`data_truth_collect.yml`) ends on 2026-11-08. Producer: `scripts/capture_prematch.py`, run by
`.github/workflows/prematch_capture.yml`. Tests: `tests/test_prematch_capture.py`.

This contract is written for the props matchup research in wowzaV9-Pro
(`research/props-matchup`, `reports/R_PROPS_MATCHUP_SPEC.md`, reader `src/props_matchup/inputs.py`).
v9 never writes to Pro. Pro reads these files read-only through `V9_LOCAL`.

## Why it exists

The Pro edge test needs data whose timing before kickoff is **proven**. Pro measured that 18.8% of a
fixture's final injury list is added after kickoff, so history downloaded after the match carries
hindsight. On 2026-11-04 Pro's readiness gate measured 0% lineups with player ids before kickoff,
and props prices with no fixture_id, player_id or kickoff time. This job captures all three at fixed
lead times with the real fetch time. API-Football `/odds` is pre-match only, so the job cannot be
backfilled: data starts on the day the job is switched on.

## Clock semantics (read this first)

| Column | Meaning |
|---|---|
| `kickoff_utc` | The provider's `fixture.timestamp` from the newest `/fixtures?date` list read **before** the capture (re-read at most every 60 minutes). ISO-8601 UTC, seconds. |
| `requested_at_utc` | Clock time just before the request was sent (in `slots` and `lists`). |
| `captured_at_utc` | Clock time just **after** the provider's answer arrived. This is the observation time: everything in the answer was known by then. Use it as Pro's `observed_at`. |
| `minutes_before_kickoff` | `(kickoff_utc - captured_at_utc)` in minutes, 2 decimals. Positive = before kickoff. |
| `kickoff_utc_response` | (`slots`, odds and injuries only) the fixture time inside the answer itself. If it differs from `kickoff_utc`, the kickoff moved. Missing when the answer carries none (lineups never do). |
| `odds_update_utc` | (`prop_odds`) the provider's own `update` time for that odds block. |

Pro's pit class for every table below except `prop_player_ids` is **`observed_pre_kickoff`** when
`minutes_before_kickoff > 0`. A kickoff can move after capture. Before treating a row as pre-kickoff,
re-check `captured_at_utc` against the fixture's **final** kickoff (for example the post-match fixture
list). A postponed fixture (status other than `NS` in a later list) is never captured again under the
same `(fixture_id, endpoint, slot)` key while the journal is read back (40 days).

## Slots

| `slot` | `endpoint` | Window (minutes before kickoff) | Content |
|---|---|---|---|
| `injuries_120` | `injuries` | 90 to 120 | `/injuries?fixture` |
| `odds_75` | `odds` | 55 to 75 | `/odds?fixture`, before the team news |
| `lineups_45` | `fixtures/lineups` | 25 to 45 | `/fixtures/lineups?fixture`: confirmed XI and substitutes with player ids |
| `odds_15` | `odds` | 2 to 15 | `/odds?fixture`, after the team news |

A pass every 5 minutes fetches a slot the first time it finds the fixture inside the window, so a
capture normally lands within about 5 minutes of the window's upper edge. A fixture first seen after
a window has closed is **missed** for that slot. No row is written and nothing is invented. Only
fixtures of the registry leagues (`src.league_registry`, `api_football_id`, 48 ids on 2026-11-04)
with provider status `NS` and a known kickoff are captured.

## Files

Everything is under `output/prematch/`:

```
output/prematch/<table>/<YYYY-MM>/<run_tag>.parquet      (+ <run_tag>_p2.parquet ... when a part passes 40 MB)
output/prematch/af_budget.json                             this job's API-Football calls per UTC day
```

* `<YYYY-MM>` is the month of `captured_at_utc` (`mapped_at_utc` for `prop_player_ids`).
* `<run_tag>` is `<YYYYMMDDTHHMM>Z-<GitHub run id>`, one per workflow run. Files are append-only per
  run, and an older run's file is never rewritten.
* Every file has the explicit schema below, so `pd.read_parquet("<table>/<YYYY-MM>")` reads a month
  as one dataset. Pro's `_read_parquet(dir)` (a recursive glob of `*.parquet`) works unchanged.
* **Read all parts and de-duplicate on the table key, keeping the first row.** The v9 reader is
  `scripts/capture_prematch.read_table(table, root)`.

Types: `int` = int64 (nullable), `float` = float64, `str` = string, `bool` = bool.
`COMMON` = `fixture_id` int, `league_id` int, `league` str (registry name), `season` str,
`kickoff_utc` str, `slot` str, `endpoint` str, `captured_at_utc` str, `minutes_before_kickoff` float.

| Table | Key | Columns (after COMMON where listed) |
|---|---|---|
| `slots` (the journal) | `fixture_id, endpoint, slot` | COMMON + `home_id, away_id` int, `requested_at_utc` str, `status` str (`ok` / `empty`), `n_items` int, `n_pages` int, `pages_total` int, `kickoff_utc_response` str, `run_tag` str |
| `lists` | `list_date, captured_at_utc` | `list_date` str (UTC day asked), `requested_at_utc`, `captured_at_utc` str, `n_fixtures_all` int (whole answer), `n_registry` int, `run_tag` str |
| `fixtures` | `fixture_id, captured_at_utc` | `list_date, captured_at_utc` str, `fixture_id, league_id` int, `league, season, kickoff_utc, status_short` str, `home_id` int, `home_name` str, `away_id` int, `away_name` str. One row per registry fixture per list read: the kickoff and status history. |
| `injuries` | `fixture_id, slot, row` | COMMON + `row` int (provider order), `team_id` int, `team_name` str, `player_id` int, `player_name` str, `type` str (for example `Missing Fixture` / `Questionable`), `reason` str |
| `lineup_teams` | `fixture_id, slot, team_id` | COMMON + `team_id` int, `team_name, formation` str, `coach_id` int, `coach_name` str, `n_start, n_subs` int |
| `lineup_players` | `fixture_id, slot, team_id, role, order` | COMMON + `team_id` int, `role` str (`start` / `sub`), `order` int (provider order), `player_id` int, `player_name` str, `number` int, `pos` str, `grid` str |
| `prop_odds` | `fixture_id, slot, row` | COMMON + `row, page` int, `odds_update_utc` str, `bookmaker_id` int, `bookmaker_name` str, `bet_id` int, `bet_name` str, `bet_prefix` str (`home` / `away` / missing), `market_v9` str, `value_raw` str, `odd_raw` str, `odd` float, `player_name` str, `line` float, `side_word` str |
| `odds_bets` | `fixture_id, slot, bet_id, bet_name` | COMMON + `bet_id` int, `bet_name` str, `is_player_prop` bool, `n_bookmakers` int, `bookmaker_ids` str (comma list), `n_values` int. Every bet the answer contained, player prop or not: the catalogue actually seen. |
| `prop_player_ids` | `fixture_id, player_name` | `fixture_id` int, `player_name` str (as in `prop_odds`), `norm_name` str, `player_id` int, `team_id` int, `status` str, `match_method` str, `roster_slot` str, `roster_captured_at_utc` str, `n_roster` int, `mapped_at_utc` str |
| `raw` | `endpoint, slot, fixture_id, page` | `fixture_id` int, `endpoint, slot` str, `page` int, `params_json, captured_at_utc, raw_scope, raw_json` str. `raw_scope = full` is the whole provider answer as compact JSON. `registry_items` (date lists only) keeps `get`, `parameters`, `results` and the registry-league items. |

### Missing stays missing

A field the provider does not send is null. `odd` is null when `odd_raw` does not parse. `player_name`
and `line` are null when `value_raw` does not parse under v9's own reader
(`player_model/prop_odds_af._clean_player` / `_parse_line`). `line` is on the Over scale: provider
`"Name - 1"` or `"Name - 1+"` = 1 or more = 0.5. `side_word` (`prop_odds_af.side_word`) is `over` /
`under` when the value text has that whole word exactly once, `yes` / `no` when the value or its last
`" - "` segment is exactly Yes / No, and null otherwise (a bare player name, the Yes side of a yes/no
player prop, names no side).

**Every side is kept** (owner decision 2026-11-05). Each provider value is its own row, so a book's
Yes and No, or Over and Under at the same `line`, are both stored with their own `odd`, nothing is
filtered by side. A real two-sided pair is: same `fixture_id, slot, bookmaker_id, bet_id, player_name,
line`, one row per side. v9 de-vigs such a pair for a props tip's fair probability
(`odds_fetcher.match_fair_to_tips`); where a book quotes one side only, nothing is paired. An `empty` slot (`n_items = 0`) is a real empty answer: for injuries, an observed empty
list (a real zero); for lineups, none published yet at capture.

### Player props: which bets

A bet is kept in `prop_odds` when v9's API-Football prop reader already maps it
(`prop_odds_af._BET_TO_MARKET`: `Anytime Goal Scorer`, `Player to be Booked`, `Player Assists`,
`Player to Score or Assist`, `Player Shots On Target Total`, `Player Shots Total`, also with a
`Home ` / `Away ` prefix), or when its name contains `player`, `scorer` or `goalscorer`. `market_v9` is
v9's vocabulary (`goals`, `cards`, `assists`, `goal_or_assist`, `sot`) and stays null for a prop
v9 does not model. For any other market, check `odds_bets` for names classified `False` before
concluding that it does not exist.

### Player ids (`prop_player_ids`)

The odds name players by text only. A name gets a provider `player_id` only from the **same
fixture's captured lineup** (`lineups_45`), with v9's rule (`odds_fetcher.attach_player_ids`):

* `exact`: the normalised name (accents, case and spaces folded) equals exactly one lineup player.
* `initial_surname`: first initial + surname equals exactly one lineup player AND exactly one quoted
  name of this fixture.

Any other case leaves `player_id` null, and `status` says why: `ambiguous`, `unmatched`,
`roster_entry_without_id`, or `no_roster` (lineup answer empty, or lineup slot missed). The mapping is
written only after the last odds window has closed, so it sees every quoted name. No squad list,
name store or other source is used. v9 has no player alias store. Pro can restrict to
`match_method == 'exact'` if it wants the strictest identity.

The mapping is **identity only, not timing**. The lineup it uses is captured at about T-45, after the
`odds_75` prices. Do not read "this player had a price at T-75" as "his place in the XI was known at
T-75". Use `lineup_players.captured_at_utc` for when the XI was known.

Join: `prop_odds` ⟕ `prop_player_ids` on `(fixture_id, player_name)`.

## Mapping to Pro's readiness checks

| Pro check | Source here |
|---|---|
| `injury_lists_pre_kickoff` | `injuries` + `slots` (`slot = injuries_120`, `status`): a fixture with a journal row has an observed list, possibly empty |
| `lineups_with_ids_pre_kickoff` | `lineup_players` (`player_id` not null, `minutes_before_kickoff > 0`) |
| `prices_pre_and_post_news` | `prop_odds` at `odds_75` (before) and `odds_15` (after) per `(fixture_id, player_id, market, bookmaker_id)` |
| `prop_odds_identity` | `prop_odds` ⟕ `prop_player_ids`: share with `player_id` not null |

## Budget and switch (operations)

* At most 1,000 API-Football calls per UTC day, journaled in `output/prematch/af_budget.json`, never in
  the big job's `output/collect/af_budget.json`. It stops when the account's remaining daily quota
  falls below `collect_common.FLOOR - 1,000` = 24,000. A stop is a clean exit.
* Dry run, 2026-11-04 (`python scripts/capture_prematch.py --dry-run --end 2026-11-04`, the stored
  registry fixture lists of 2026-10-07 to 2026-11-03): 62.4 fixtures/day on average, 185 at peak.
  That gives 4 calls per fixture + 27 list reads/day = 276 calls/day on average and 767 at peak.
* Switch on: `gh variable set PREMATCH_CAPTURE_ENABLED --body true --repo NevixAA/wowza-betting`.
  The workflow file must be on the default branch (`main`) for its schedule to fire.
  Switch off: `gh variable delete PREMATCH_CAPTURE_ENABLED --repo NevixAA/wowza-betting`.
