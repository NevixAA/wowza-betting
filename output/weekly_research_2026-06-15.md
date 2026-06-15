# Weekly Research Report — 2026-06-15

**System:** Wowza Betting ML v9.2 | **Analyst:** wowza-research-bot | **Date:** 2026-06-15

---

## 1. Dead / Weak Features

### 1a. Effectively-Constant Features (99.3% Null → Fallback Constants)

The six opponent-defender matchup features are populated in **only 83 of 11,360 rows** (0.7%). For 99.3% of predictions these columns fall back to hardcoded constants (0.50, 1.50, 0.25). They behave as dead features during training:

| Feature | Null % | Fallback Value |
|---|---|---|
| `opp_def_aerial_win_rate` | 99.3% | 0.50 |
| `opp_def_fouls_pg` | 99.3% | 1.50 |
| `opp_def_cards_pg` | 99.3% | 0.25 |
| `aerial_matchup_score` | 99.3% | computed from above |
| `foul_draw_matchup_score` | 99.3% | computed from above |
| `opp_def_aggression` | 99.3% | computed from above |

**Root cause:** `build_upcoming_features()` tries to look up opponent defenders in `history_df` by team name prefix match, but the parquet only holds 285 teams from club seasons — it rarely has entries for both sides of a WC fixture. These six features are effectively feeding the model constant noise. **Action: either fix the lookup or drop them until proper population is achieved.**

### 1b. High-Variance Features Needing Capping

These raw rolling values contain heavy outliers that bias gradient boosting splits:

| Feature | Mean | Max | CV | Recommended Cap |
|---|---|---|---|---|
| `goal_involvement_rate` | 0.258 | 9.0 per 90 | 2.61 | 2.0 |
| `sot_quality_score` | 0.186 | 4.0 | 2.19 | 1.5 |
| `goals_pg` | 0.099 | 4.0 | 2.76 | 1.5 |
| `assists_pg` | 0.072 | 3.0 | 3.15 | 1.0 |
| `card_exposure_index` | 0.076 | 1.78 | 2.82 | 1.0 (already capped elsewhere) |

The `goal_involvement_rate` max of 9.0/90 is physically impossible for a player with full minutes — it's an artifact of very short appearances (5–15 min substitute with a goal). A simple `clip(upper=2.5)` before training would help regularise the gradient boosting learner.

### 1c. `shooting_efficiency_index` Redundancy

`shooting_efficiency_index = goals_pg / sot_pg` is nearly identical to `sot_rate` in information content. Both roll the same ratio of goals-to-shots-on-target. Consider dropping one or confirming via SHAP that they contribute independently.

---

## 2. Model Performance

### 2a. Tier Distribution — Critical Issue

**All 627 rows in `player_tips.csv` are WATCH tier.** Zero SNIPER, MARKSMAN, or VALUABLE signals have been generated this run.

| Tier | Count |
|---|---|
| SNIPER | 0 |
| MARKSMAN | 0 |
| VALUABLE | 0 |
| WATCH | 627 |

### 2b. EV Distribution — Strongly Negative

252 of 627 rows have market odds available. EV statistics:

| Metric | Value |
|---|---|
| Mean EV | **-0.484** |
| Median EV | -0.494 |
| Max EV | +0.014 |
| Positive EV signals | 2 (both WATCH, EV < 0.02) |

Every single priced signal has near-zero or negative EV. Representative examples:
- Ronaldo SOT: model_prob=0.605, market_odds=1.12 → EV = **-0.323** (book implies 89% vs our 60%)
- Haaland SOT: model_prob=0.528, market_odds=NaN (no odds available)
- Messi SOT: model_prob=0.451, market_odds=1.20 → EV = **-0.459**

**Root cause:** The bookmakers' SOT odds for WC superstar players are extremely compressed (1.12–1.55 range), reflecting a "star premium" that our model does not capture. The market is implying 65–90% probability while our model sits at 45–60%. This is likely accurate bookmaker pricing — our model needs a WC context recalibration.

### 2c. Market Distribution — Cards and Assists Missing

| Market | Rows | % of output |
|---|---|---|
| sot | 596 | 95.1% |
| goals | 31 | 4.9% |
| cards | 0 | 0% |
| assists | 0 | 0% |

Cards and assists produce **zero signals.** With `target_cards` base rate of 10.4% and `target_assists` at 6.9%, a calibrated model should produce some non-negligible predictions. The `p_model < 0.15` filter in `predict.py:374` is silently dropping all card/assist predictions. This warrants investigation — either the models are poorly calibrated or the 0.15 floor is too high for these markets.

### 2d. Signal Rate by League

| League | Signals | Priced |
|---|---|---|
| World Cup | 489 (78%) | 252 |
| Ireland Premier | 88 (14%) | 0 |
| Finland Veikk | 50 (8%) | 0 |

Ireland Premier and Finland Veikk have **zero priced signals** despite generating 138 WATCH rows. The Odds API appears not to return player prop markets for these leagues.

---

## 3. Data Quality

### 3a. player_history.parquet

| Metric | Value |
|---|---|
| Total rows | 11,360 (not ~30k as noted in system overview) |
| Unique players | 4,291 |
| Unique teams | 285 |
| Date range | Requires inspection |

The 30k row target has not been reached. The parquet contains significant Friendlies data (3,571 rows, 31%) which may dilute quality — friendly matches have different tactical intensity and are likely to hurt calibration on competitive fixtures.

### 3b. WC 2026 Team Data Gap — Critical

**All 22 WC teams with history have only 1 fixture row each.** The remaining 26 WC teams (including Algeria, Congo DR, Iran, Iraq, New Zealand, Saudi Arabia, etc.) have zero rows in the parquet.

Teams with only 1 fixture row (selected):
Australia, Bosnia, Brazil, Canada, Czechia, Ecuador, Germany, Haiti, Ivory Coast, Japan, Mexico, Morocco, Netherlands, Paraguay, Qatar, Scotland, South Africa, South Korea, Switzerland, Türkiye, USA, Curaçao

With `MIN_GAMES_SIGNAL = 3`, players from the 26 zero-row teams are entirely suppressed from predictions. Players from the 22 one-row teams only narrowly qualify. The WC data collection (`mode_collect_wc`) has not gathered sufficient depth — the nightly retrain plan needs immediate execution.

### 3c. Duplicate Player Entries — Active Duplicates Found

7+ confirmed duplicate pairs in current `player_tips.csv`:

| Canonical Name | Duplicate Name | Match | Market |
|---|---|---|---|
| Vinícius Júnior | Vinicius Júnior | Brazil vs Haiti | sot, goals |
| Son Heung-min | Heung-min Son | Mexico vs South Korea | sot |
| Danilo | Danilo Oliveira / Danilo Santos | Brazil vs Haiti | sot |
| Jean-Philippe Mateta | Jean Philippe Mateta | France vs Senegal | sot, goals |
| Julián Álvarez | Julián Alvarez | Argentina vs Algeria | sot |
| Rubén Vargas | Ruben Vargas | Switzerland vs Bosnia | sot |
| Benji Michel | Michel Benjamin Stanley | HJK vs Inter Turku | sot |

These are duplicate prediction rows — each represents the same player predicted twice under two name spellings. The `drop_duplicates(subset=["match", "player_name", "market"])` in `predict.py:428` does not catch these because the names differ. **Each Telegram signal would go out twice for these players**, inflating stake exposure by 2×.

### 3d. League Data Composition (Quality Risk)

The parquet contains data from:
- Friendlies: 3,571 rows (31%) — **low quality, should be filtered or down-weighted**
- WC Qualifiers: 1,875 rows (16.5%)
- Club leagues: ~5,584 rows (49%)
- WC proper: 330 rows (2.9%)

Training on friendlies significantly pollutes the models because players play different roles (rotation squads, tactical experiments) in friendlies vs. competitive matches.

---

## 4. Formula Suggestions (Top 5, Prioritised)

### P1. Referee Strictness Feature (Cards Market)

**Status:** Code exists (`build_referee_profile` in `api_football.py:328`) but is not wired into `PLAYER_FEATURE_COLS`. The `predict.py` passes `referee_profiles` to `build_upcoming_features`, which accepts a `referee_profile` argument, but it is never used to set any feature in the row dict.

**Formula:**
```python
referee_strictness = (ref.get("yellows_per_game", 3.5) - 3.5) / 1.0  # z-score vs league avg
```
Add to `PLAYER_FEATURE_COLS` and to the row dict in `build_upcoming_features`. The cards AUC is 0.61 — referee strictness is the single highest-value unlockable feature for this market.

**Data needed:** `/fixtures/events` or `/fixtures/players` for referee's last 20 matches — already partially supported in `api_football.py`.

**Expected improvement:** +0.04–0.08 AUC on cards model (literature: referee variable explains ~8% of card variance in competitive football).

### P2. Tournament Context Multiplier (WC-specific)

**Problem:** Our model is trained on club football rolling stats. WC group stage has a fundamentally different tactical profile — conservative managers, underdog teams defending deep, star players playing fewer minutes to preserve fitness for knockouts.

**Formula:**
```python
is_group_stage = int(match_context.get("round", "") in ("Group Stage", "Group A-F"))
wc_conservative_factor = 0.85 if is_group_stage else 1.0
p_model *= wc_conservative_factor  # suppress all props in WC groups
```
Or alternatively, train a WC-specific calibration layer on the WC qualifier data (1,875 rows available).

**Expected improvement:** Reduces systematic over-estimation bias currently causing -0.484 mean EV on all WC priced signals.

### P3. Cap High-Variance Rolling Features

**Formula:** Apply `np.clip` before model training and prediction:
```python
df["goal_involvement_rate"] = df["goal_involvement_rate"].clip(upper=2.5)
df["sot_quality_score"]     = df["sot_quality_score"].clip(upper=1.5)
df["goals_pg"]              = df["goals_pg"].clip(upper=2.0)
df["assists_pg"]            = df["assists_pg"].clip(upper=1.5)
```
Also filter out Friendlies rows during training:
```python
df = df[df["league"] != "Friendlies"]  # in mode_train()
```

**Expected improvement:** Lower log-loss, better calibration curves, fewer false high-confidence signals.

### P4. Opponent xG-Conceded Feature (Replaces Null opp_def Cluster)

The opp_def cluster is 99.3% null — replace it with a team-level xG-conceded rolling metric using `/fixtures/statistics`. This returns shots allowed, saves, and possession per team, which can be aggregated to approximate defensive weakness.

**Formula:**
```python
# Collect via /fixtures/statistics for each completed WC fixture
opp_xg_proxy = opp_team_shots_on_target_conceded_pg / opp_team_goals_conceded_pg
# High ratio = keeper saving lots, defense leaking shots (high xG-against)
```
This would replace `opp_sot_conceded_pg` (which uses player-level aggregation, hence the sparse defender data) with a direct team-level stat.

**Data needed:** `/fixtures/statistics` — see Section 6c.

**Expected improvement:** Fills the 99.3% null gap in opp_def features; provides a real signal instead of fallback constants.

### P5. Player Name Normalisation (Deduplication Fix)

**Formula:** Apply Unicode NFKD normalisation + lowercase + remove hyphens before the `drop_duplicates` call, and also during the player history lookup:
```python
def _norm_name_strict(name: str) -> str:
    import unicodedata, re
    nfkd = unicodedata.normalize("NFKD", str(name).lower())
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[-\s]+", " ", stripped).strip()
```
Apply this normalisation to `player_name` before `drop_duplicates`. This alone would eliminate the 7 confirmed duplicate pairs (= 14+ duplicate Telegram signals per cycle).

---

## 5. Edge Patterns & Tier Suggestions

### 5a. Patterns in Priced Signals

From the 252 priced WC signals:

- **Negative EV is universal** — every single market odds available produces negative EV. The SOT market at short odds (1.12–1.55) is not a playable market given our model's calibration on WC matches.
- **Odds range sweet spot (not yet reachable):** The two signals with positive EV are at odds 6.0 and 6.5 (model_prob ~0.17). At these longer odds, our model occasionally finds genuine value. Suggestion: focus prop signal generation on odds > 4.0 only.
- **MINUTES flag (55 signals):** Players with a "MINUTES" lazy factor have low minutes_pg relative to their estimated minutes — these are likely squad rotation risks. The MINUTES flag correctly identifies them but they still output WATCH. This is working as intended.
- **SET_PIECE flag (32 signals):** All SET_PIECE flagged players are defenders/midfielders with aerial ability. None are generating actionable signals because odds aren't available for these markets in Ireland/Finland.

### 5b. Tier Threshold Adjustment Recommendations

The current SNIPER threshold (`EV ≥ 0.40`, `odds ≥ 5.0`) is appropriate — the problem is upstream (no positive EV at all, not that thresholds are wrong). However:

1. **Lower the `MIN_GAMES_SIGNAL` floor to 2 for WC matches** (currently 3): WC teams have 0–1 fixtures in history. Enforcing 3 games is over-restrictive for national team data — players have club form data too, which is more informative anyway.
2. **Add a WC-specific EV floor**: Do not output WC signals with odds < 2.0 regardless of model prob. Bookmakers price WC stars at extreme compression — these are statistically unplayable.
3. **Cards/Assists threshold audit**: Lower the `p_model < 0.15` filter to `< 0.08` temporarily to see if cards/assists signals appear. If they do, the 0.15 floor is the issue; if not, the models need retraining.

---

## 6. API-Football Credit Utilisation Plan

**Current usage:** ~822 calls/day (11% of 7,500 quota)  
**Available budget:** ~6,678 additional calls/day

---

### 6a. Lineups — Suppress DNP Signals Before Kick-off

**Research findings:**
- API-Football publishes confirmed lineups typically **60–75 minutes before kick-off** for top leagues. For WC matches, lineups are often published exactly 60 min before KO.
- The code already has `get_fixture_lineup()` in `api_football.py:247` and integrates it in `predict.py:307–321`.
- **Gap:** Lineups are currently only fetched at prediction time (once daily), but the system's 07:00 UTC daily digest runs before most WC lineups are published (WC matches begin at 15:00/18:00/21:00 UTC). The lineup check returns `lineup_available = False` and no suppression occurs.

**Proposed function (already exists, needs scheduling fix):**
```python
def get_fixture_lineup(fixture_id: int, cache_hours: int = 1) -> dict:
    """
    Get confirmed lineup for a fixture.
    Call with cache_hours=1 to refresh hourly near kick-off.
    Returns {team_name: [{"player_id", "player_name", "position", "started": bool}]}
    """
    data = _get("/fixtures/lineups", {"fixture": fixture_id}, cache_hours=cache_hours)
    if not data:
        return {}
    result = {}
    for team_data in data.get("response", []):
        team_name = team_data.get("team", {}).get("name", "")
        starters = [
            {"player_id": p["player"]["id"], "player_name": p["player"]["name"],
             "position": p.get("pos", ""), "started": True}
            for p in team_data.get("startXI", [])
        ]
        subs = [
            {"player_id": p["player"]["id"], "player_name": p["player"]["name"],
             "position": p.get("pos", ""), "started": False}
            for p in team_data.get("substitutes", [])
        ]
        result[team_name] = starters + subs
    return result
```

**Implementation plan:**
1. Add a second prediction run at 13:00 UTC (2h before first WC kick-off) specifically for lineup enrichment
2. Set `cache_hours=1` for lineup calls in the 13:00 run window
3. Add `starter_confirmed` as a lazy factor (+1 confidence tier signal) for confirmed starters

**Daily API cost:** 1 call per upcoming fixture × ~6 WC fixtures/day = **6 calls/day** (negligible; already cached 1h)

**Data availability risk:** Lineups occasionally not published until 30 min before KO, or delayed for injuries. The `lineup_available = False` fallback already handles this gracefully.

**ROI: HIGH** — Eliminating DNP signals for confirmed subs/non-starters directly reduces false positives. The MINUTES lazy factor currently handles part of this but not confirmed subs.  
**Implementation effort: 2 hours** (fix scheduling window; adjust cache_hours for near-KO runs)  
**Models benefiting:** All player prop markets (goals, sot, cards, assists)

---

### 6b. Injuries — Pre-match Void Filter

**Research findings:**
- API-Football `/injuries` returns reliable data for the **top 5 leagues** (Premier League, Bundesliga, La Liga, Serie A, Ligue 1) and Champions League. For WC 2026, the endpoint returns squad list changes (injuries, late withdrawals) as officially reported to FIFA.
- The code already has `get_injured_players()` in `api_football.py:147` and it's integrated in `predict.py:271–282`. **This is already working.**

**Proposed enhancement — pre-match void function:**
```python
def get_injury_status(player_id: int, league_id: int, season: str,
                      date_str: str, fixture_id: int | None = None) -> str:
    """
    Returns 'injured', 'suspended', 'doubtful', or 'available'.
    Uses /injuries endpoint with player-level resolution.
    Call before generating a signal; return 'VOID' for injured/suspended.
    Cache 4 hours — status stable within a day.
    """
    data = _get("/injuries", {
        "league": league_id, "season": season,
        "date": date_str,
        **({"fixture": fixture_id} if fixture_id else {}),
    }, cache_hours=4)
    if not data:
        return "available"
    for entry in data.get("response", []):
        if entry.get("player", {}).get("id") == player_id:
            reason = entry.get("player", {}).get("reason", "").lower()
            if "suspend" in reason:
                return "suspended"
            return "injured"
    return "available"
```

**Reliable leagues for injury data:** Premier League (39), Bundesliga (78), La Liga (140), Serie A (135), Ligue 1 (61), Champions League (2). World Cup (1) — limited but improving.  
**Unreliable:** Ireland Premier, Finland Veikkaaus — manual checks recommended.

**Daily API cost:** `/injuries` is 1 call per league/date (already implemented). No new calls needed — enhance the existing `injured_cache` to return per-player status rather than a set of names.

**Data availability risk:** Injury data lags official announcements by 2–12 hours. A player ruled out on match day at noon may not appear until 14:00.

**ROI: MEDIUM** — The void filter already exists and fires. The enhancement is moving from name-match to ID-match (more reliable), and adding status granularity (doubtful vs. confirmed out).  
**Implementation effort: 2 hours**  
**Models benefiting:** All markets (injury voids cascade to all)

---

### 6c. Team Match Statistics — Richer O/U Features for International Leagues

**Research: Which leagues lack shot/corner history from current sources?**

From `config.py`, `FBREF_LEAGUES` covers: Championship, League One, Bundesliga 2, Ligue 2, La Liga 2, Serie B. Our standard O/U model gets shots/corners/possession from `football-data.co.uk` for these. However:

- **WC 2026 (league_id=1)**: No shot or corner history from football-data.co.uk
- **Ireland Premier (357)**: No shot/corner data
- **Finland Veikkaaus (244)**: No shot/corner data
- **WC Qualifiers (various)**: No coverage

`/fixtures/statistics` fills this gap — it returns per-team: shots on target, shots total, corners, possession, saves, fouls per completed fixture.

**Proposed function:**
```python
def get_fixture_team_stats(fixture_id: int) -> dict[str, dict]:
    """
    Get team-level match statistics for a completed fixture.
    Returns {team_name: {shots_total, shots_on_target, corners, possession, saves, fouls}}.
    Cache 7 days for completed fixtures.
    """
    data = _get("/fixtures/statistics", {"fixture": fixture_id}, cache_hours=168)
    if not data:
        return {}
    result = {}
    for team_data in data.get("response", []):
        team_name = team_data.get("team", {}).get("name", "")
        stats = {s["type"].lower().replace(" ", "_"): s.get("value") or 0
                 for s in team_data.get("statistics", [])}
        result[team_name] = {
            "shots_total":      int(stats.get("total_shots", 0)),
            "shots_on_target":  int(stats.get("shots_on_goal", 0)),
            "corners":          int(stats.get("corner_kicks", 0)),
            "possession":       float(str(stats.get("ball_possession", "50%")).replace("%", "")),
            "saves":            int(stats.get("goalkeeper_saves", 0)),
            "fouls":            int(stats.get("fouls", 0)),
        }
    return result
```

**Implementation plan:**
1. **Historical backfill:** Fetch team stats for all WC 2026 fixtures already played. At tournament start (Jun 11) there will be ~30–40 completed fixtures. Cost: 30–40 one-time calls.
2. **Ongoing:** 1 call per completed fixture per day = ~6 calls/day during WC group stage.
3. Aggregate into rolling team-level features: `team_corners_per_match`, `team_shots_on_target_conceded`, `team_possession_avg`.
4. Use `opp_team_shots_on_target_conceded` as the new `opp_sot_conceded_pg` for international matches (replaces the player-level aggregation which requires defender data — currently 99.3% null).

**Daily API cost:** ~6 calls/day ongoing + 40 one-time backfill = **6 calls/day steady-state** (trivial).

**Data availability risk:** Available for all leagues where API-Football has data. Availability for WC 2026 confirmed — this is a flagship tournament for them.

**ROI: HIGH** — Directly fixes the opp_def cluster null problem; provides real team-level defensive data for WC/international matches; improves O/U model for new-format leagues.  
**Implementation effort: 4 hours** (collection + rolling aggregation + feature wiring)  
**Models benefiting:** O/U 2.5 ensemble, player props sot/goals (via opp_sot_conceded_pg fix)

---

### 6d. Referee Profiles — Card Model Improvement

**Background:** The cards model has AUC ~0.61 (the weakest market). Referee strictness is a known top-5 feature in academic card prediction models — strict referees give 40–80% more yellow cards than lenient ones in the same league.

**Existing infrastructure:** `build_referee_profile()` exists in `api_football.py:328` and is called in `pipeline.py:_build_referee_profiles()`. However, the strictness score is **never added to any feature row** — it's passed to `build_upcoming_features()` as `referee_profile` but the function ignores it (the `ref` variable is accepted but only used in a comment).

**Proposed function (enhancement of existing):**
```python
def get_fixture_events(fixture_id: int) -> list[dict]:
    """
    Get all in-match events (goals, cards, subs) with minute and player.
    Used to build referee profile: cards given per match.
    Cache 7 days for completed fixtures.
    """
    data = _get("/fixtures/events", {"fixture": fixture_id}, cache_hours=168)
    if not data:
        return []
    events = []
    for event in data.get("response", []):
        events.append({
            "fixture_id": fixture_id,
            "type":       event.get("type", ""),        # "Card", "Goal", "subst"
            "detail":     event.get("detail", ""),      # "Yellow Card", "Red Card"
            "minute":     event.get("time", {}).get("elapsed", 0),
            "player_id":  event.get("player", {}).get("id"),
            "player_name": event.get("player", {}).get("name", ""),
            "team_id":    event.get("team", {}).get("id"),
        })
    return events

def build_referee_profile_v2(referee_name: str, league_id: int, season: str,
                              last_n: int = 20) -> dict:
    """
    Build referee strictness profile using /fixtures/events for accurate card counts.
    Improvement over v1 which summed yellow_cards from /fixtures/players.
    """
    if not referee_name:
        return {"yellows_per_game": 3.5, "strictness_score": 0.0, "n_games": 0}

    fixtures = get_recent_fixtures(league_id, season, last_n=last_n)
    ref_fixtures = [
        f for f in fixtures
        if referee_name.lower() in (f.get("fixture", {}).get("referee", "") or "").lower()
    ]
    if len(ref_fixtures) < 5:
        return {"yellows_per_game": 3.5, "strictness_score": 0.0, "n_games": 0}

    total_yellows = 0
    for fix in ref_fixtures:
        fix_id = fix.get("fixture", {}).get("id")
        if fix_id:
            events = get_fixture_events(fix_id)
            total_yellows += sum(
                1 for e in events
                if e["type"] == "Card" and "Yellow" in e.get("detail", "")
            )
    n = len(ref_fixtures)
    avg_y = total_yellows / n
    strictness = (avg_y - 3.5) / 1.0  # z-score, league avg ≈ 3.5 yellows/match
    return {"yellows_per_game": round(avg_y, 2), "strictness_score": round(strictness, 2), "n_games": n}
```

**Wire into feature engineering** — add to `build_upcoming_features()`:
```python
ref = referee_profile or {}
referee_strictness = ref.get("strictness_score", 0.0)
# ... in the rows.append() dict:
"referee_strictness": round(referee_strictness, 3),
```
And add `"referee_strictness"` to `PLAYER_FEATURE_COLS`.

**Daily API cost:**
- 1 `get_recent_fixtures` call per league to find referee's fixtures (already cached)
- 1 `get_fixture_events` call per referee-matched fixture × ~20 fixtures = 20 calls (7-day cache, so steady-state ~3 new calls/day per league for new completed fixtures)
- For 15 PROP_LEAGUES × 3 = **45 calls/day** steady-state

**Data availability risk:** Referee data not always populated for WC qualifiers and lower leagues. Set `min_games=5` guard (already in code as `< 3`, should be raised to `5`).

**ROI: HIGH** — Cards AUC 0.61 is the weakest market. Referee strictness is the single most impactful unlockable feature. The infrastructure already exists — this is a wiring fix + `get_fixture_events` addition.  
**Implementation effort: 3 hours** (fix wiring, add get_fixture_events, raise min_games guard)  
**Models benefiting:** Cards (primary), can also improve fouls_drawn predictions for assists model

---

### Priority Matrix (ROI vs Effort)

| Endpoint / Feature | ROI | Effort (h) | Models | API Calls/Day | Risk |
|---|---|---|---|---|---|
| **6a. Lineups (scheduling fix)** | HIGH | 2 | All props | 6 | Low (graceful fallback exists) |
| **6d. Referee Profile (wiring fix)** | HIGH | 3 | Cards | 45 | Low (data rarely missing) |
| **6c. Team Stats (/fixtures/statistics)** | HIGH | 4 | O/U, sot/goals | 6 | Low (WC well-covered) |
| **P5. Name Normalisation** | HIGH | 1 | All | 0 | None |
| **P1. Referee Feature Wiring** | HIGH | 1 | Cards | 0 | None (code exists) |
| **6b. Injuries (ID-match upgrade)** | MEDIUM | 2 | All | 0 (already implemented) | Medium (lag) |
| **P3. Feature Capping + Friendlies Filter** | MEDIUM | 2 | All | 0 | Low |
| **P2. WC Context Multiplier** | MEDIUM | 3 | WC props | 0 | Medium (needs validation) |
| **P4. opp xG Proxy (replace opp_def cluster)** | MEDIUM | 4 | Goals, sot | 6 | Low |

**Total daily calls if all implemented:** ~63 additional/day → **885 total/day (still 11.8% of quota)**. Enormous remaining headroom.

---

## Summary — Top 3 Actions for Next Session

### Action 1: Fix Player Name Normalisation (1 hour, zero risk)
Apply `_norm_name_strict()` deduplication before `drop_duplicates` in `predict.py:428`. This immediately eliminates 14+ duplicate Telegram signals per cycle and prevents double-staking on the same player.

### Action 2: Wire Referee Strictness into Cards Model (3 hours, HIGH ROI)
The feature code (`build_referee_profile`) and the pipeline wiring (`_build_referee_profiles`) both exist. The only missing step is adding `"referee_strictness"` to the row dict in `build_upcoming_features()` and to `PLAYER_FEATURE_COLS`. Then retrain the cards model. This is the single highest-ROI improvement for the weakest market (AUC 0.61).

### Action 3: Collect /fixtures/statistics for WC Matches + Fix opp_def Cluster (4 hours, HIGH ROI)
Implement `get_fixture_team_stats()`, backfill the ~35 completed WC fixtures (35 API calls), and wire the team-level defensive stats into `opp_sot_conceded_pg` / `opp_goals_conceded_pg` for international matches. This replaces the 99.3% null opp_def cluster with real data and improves the O/U features for WC/Ireland/Finland leagues simultaneously.

---

*Report generated by wowza-research-bot | Codebase: NevixAA/wowza-betting | Analysis date: 2026-06-15*
