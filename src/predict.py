"""
Upcoming Fixture Predictor — v9
================================
v8 predict.py + three additions from v7:

1. Odds drift snapshot + enrichment
   Every run snapshots current odds to odds_history_v9.json.
   Drift columns (over_drift, under_drift, drift_signal) are added to
   predictions and used by betting.py to adjust signal_tier.

2. Postponement detection
   After fetching live fixtures, compare against the previous predictions.csv.
   Any fixture that was predicted but has disappeared from OddsAPI is flagged
   as POSTPONED in the output.

3. Both-losing guard
   Handled in betting.py via v7's best_params.json — predict.py just passes
   the league column through so betting.evaluate_value() can apply the guard.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

log = logging.getLogger(__name__)

# Populated by _fetch_odds_api each run; consumed by the outage health check.
_LAST_FETCH_STATS: dict = {}
# This run's per-book O/U quotes (every book, both sides, one snapshot). Read ONLY by the
# market_edge_v2 shadow (src/edge_engine_shadow.py) so its two-sided consensus is as fresh as the
# board. Recording only: nothing in the selection path reads it.
_LAST_BOOK_ROWS: list = []
# This run's model routing (R2-A fail-closed). Written into output/league_registry_health.json.
_LAST_ROUTING: dict = {}


def _record_routing(feat, std_mask, nf_mask, unknown_mask, nf_payload_missing: bool) -> dict:
    """Record which fixtures each O/U model scored and which were REFUSED. Never raises. Written
    to output/league_registry_health.json (`last_predict_routing`) by write_integrity_health()."""
    global _LAST_ROUTING
    try:
        def _fx(mask):
            sub = feat.loc[mask]
            return [{"league": str(r.get("league", "")), "home_team": str(r.get("home_team", "")),
                     "away_team": str(r.get("away_team", "")), "date": str(r.get("date", ""))[:10]}
                    for _, r in sub.iterrows()]
        _LAST_ROUTING = {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "fixtures": int(len(feat)), "standard": int(std_mask.sum()), "new_format": int(nf_mask.sum()),
            "unknown_league_fixtures": _fx(unknown_mask),
            "new_format_artifact_missing": bool(nf_payload_missing),
            "new_format_unscored_fixtures": int(nf_mask.sum()) if nf_payload_missing else 0,
            "policy": "unknown league -> model_type=unknown, no probability, no bet; missing NF artifact -> "
                      "NF fixtures unscored (no Standard fallback)",
            "status": ("FAIL" if (nf_payload_missing and bool(nf_mask.any()))
                       else "WARN" if bool(unknown_mask.any()) else "PASS"),
        }
    except Exception as e:                                             # noqa: BLE001
        _LAST_ROUTING = {"status": "FAIL", "error": f"{type(e).__name__}: {e}"}
    return _LAST_ROUTING


def write_integrity_health() -> None:
    """Write this run's routing / season / Standard-serving health files. Called by
    pipeline.mode_predict AFTER predict_upcoming (offline previews call predict_upcoming directly
    and therefore write nothing). Never raises."""
    try:
        from src import league_registry, season_resolver, standard_serving
        out = Path(config.OUTPUT_DIR)                  # honours a redirected OUTPUT_DIR (tests, previews)
        league_registry.write_health(routing=_LAST_ROUTING or None,
                                     path=out / league_registry.HEALTH_FILE.name)
        season_resolver.write_health(path=out / season_resolver.HEALTH_FILE.name)
        standard_serving.write_health(path=out / standard_serving.HEALTH_FILE.name)
    except Exception as e:                                             # noqa: BLE001
        log.warning(f"league/season/serving health not written: {e}")


# ── Fixture + odds fetching ───────────────────────────────────────────────────

def _fetch_odds_api(days_ahead: int = 7) -> pd.DataFrame:
    """Fetch upcoming fixtures + O/U 2.5 odds from OddsAPI for all enabled leagues."""
    try:
        import requests
    except ImportError:
        log.warning("requests not installed")
        return pd.DataFrame()

    rows = []
    n_queried = n_ok = n_err = 0   # health telemetry (see _LAST_FETCH_STATS)
    _failed_leagues: list[str] = []
    # Per-book prices already present in the OddsAPI response. Collected here and written to
    # output/book_odds_snapshots.csv below; see the note at the capture site.
    _book_rows: list[dict] = []
    _snap_ts = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    for league, sport_key in config.ODDS_API_SPORT_KEYS.items():
        if league not in config.ENABLED_LEAGUES:
            continue
        n_queried += 1
        try:
            r = requests.get(
                f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds",
                params={
                    "apiKey": config.ODDS_API_KEY,
                    "regions": "eu",
                    # BTTS is NOT supported on the /odds LIST endpoint — requesting
                    # "totals,btts" 422s the WHOLE call ("Markets not supported: btts"),
                    # which silently skipped every league and killed all tips since v9.2
                    # (2026-06-22). Totals only here; BTTS odds need the per-event endpoint.
                    "markets": "totals",
                    "oddsFormat": "decimal",
                },
                timeout=15,
            )
            if r.status_code != 200:
                n_err += 1
                # WARNING and named: at DEBUG a single league could drop off the board with no
                # record of which one (Data Truth audit finding 14).
                _failed_leagues.append(f"{league}: HTTP {r.status_code}")
                log.warning(f"OddsAPI {league}: HTTP {r.status_code} — league skipped this run")
                continue
            n_ok += 1

            for event in r.json():
                dt = pd.to_datetime(event.get("commence_time", ""), errors="coerce", utc=True)
                if pd.isna(dt):
                    continue
                dt = dt.tz_localize(None)
                now = datetime.utcnow()
                # PRE-MATCH ONLY: skip matches that have already kicked off (live/finished).
                # OddsAPI /odds returns IN-PLAY events with LIVE odds (e.g. Over 2.5 @ 6.0
                # when a match sits 0-0 late) — the pre-match model has no live-score input,
                # so it computes a huge FAKE edge -> false SNIPER (China SL bug 2026-07-05).
                # Live betting, if ever pursued, is a SEPARATE 'live alert' product.
                if dt <= now:
                    continue
                cutoff = now + timedelta(days=days_ahead)
                if dt > cutoff:
                    continue

                ov25, un25 = None, None
                ov15, ov35 = None, None
                btts_yes    = None
                for bm in event.get("bookmakers", []):
                    for mkt in bm.get("markets", []):
                        if mkt["key"] == "totals":
                            for outcome in mkt["outcomes"]:
                                pt = outcome.get("point")
                                nm = outcome.get("name")
                                pr = outcome.get("price")
                                # PER-BOOK CAPTURE (added 2026-08-19, additive only).
                                #
                                # OddsAPI returns EVERY bookmaker's price in this response and
                                # we already pay for it, but the `not ov25` guards below keep
                                # only the FIRST book and discard the rest. That discard is why
                                # multi-book consensus looked like it needed a new data source:
                                # v11 calls market_baseline({}, None) with an empty dict, so its
                                # "consensus" is one book's de-vig, and Prompt 3 section 7 calls
                                # that its highest-priority gap.
                                #
                                # Recording what is already in hand costs ZERO extra API calls
                                # and gives v11 and Pro a real cross-book median, dispersion and
                                # best-executable price. Nothing here alters ov25/un25/ov15/ov35
                                # or any model input — the selection logic below is untouched.
                                if pt in (1.5, 2.5, 3.5) and nm in ("Over", "Under") and pr:
                                    _book_rows.append({
                                        "snapshot_ts":  _snap_ts,
                                        "league":       league,
                                        "match_date":   dt.strftime("%Y-%m-%d"),
                                        "kickoff_utc":  dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                        "match":        f"{event.get('home_team','')} vs "
                                                        f"{event.get('away_team','')}",
                                        "bookmaker":    bm.get("key", ""),
                                        "market":       f"OU{str(pt).replace('.', '')}",
                                        "side":         nm.upper(),
                                        "odds":         float(pr),
                                    })
                                if pt == 2.5 and nm == "Over"  and not ov25:  ov25 = pr
                                if pt == 2.5 and nm == "Under" and not un25:  un25 = pr
                                if pt == 1.5 and nm == "Over"  and not ov15:  ov15 = pr
                                if pt == 3.5 and nm == "Over"  and not ov35:  ov35 = pr
                        elif mkt["key"] == "btts":
                            for outcome in mkt["outcomes"]:
                                if outcome.get("name") == "Yes" and not btts_yes:
                                    btts_yes = outcome.get("price")
                    # Stop only once EVERY price we want is in hand. This used to break as
                    # soon as Over 2.5 was found, so if that first bookmaker didn't quote
                    # 1.5/3.5 those prices were silently lost — odds_over15 and odds_over35
                    # were 100% blank for standard fixtures, which is why the side-market
                    # ledger has almost no rows. Bookmakers are already ordered by
                    # preference, so the 2.5 price still comes from the same book as before.
                    if ov25 and un25 and ov15 and ov35:
                        break

                # Best price per side across EVERY book (the loop above stops early). Feeds only
                # the new-format bookmaker_overround FEATURE, to match its best-of-books training
                # prices (src/feature_engineering.py); odds_over25/under25 are unchanged.
                best_ov25 = best_un25 = None
                for bm in event.get("bookmakers", []):
                    for mkt in bm.get("markets", []):
                        if mkt.get("key") != "totals":
                            continue
                        for outcome in mkt.get("outcomes", []):
                            pr = outcome.get("price")
                            if outcome.get("point") != 2.5 or not pr:
                                continue
                            # best seen price; None until a book quotes one (no typed 0.0 start)
                            if outcome.get("name") == "Over":
                                best_ov25 = float(pr) if best_ov25 is None else max(best_ov25, float(pr))
                            elif outcome.get("name") == "Under":
                                best_un25 = float(pr) if best_un25 is None else max(best_un25, float(pr))

                if ov25 and un25:
                    rows.append({
                        "league":       league,
                        "date":         dt.normalize(),   # UTC calendar day — JOIN KEY, do not change
                        "kickoff_utc":  dt,                # full UTC kickoff datetime — for local-tz DISPLAY only
                        "home_team":    event.get("home_team", ""),
                        "away_team":    event.get("away_team", ""),
                        "odds_over25":  float(ov25),
                        "odds_under25": float(un25),
                        "best_over25":  best_ov25,
                        "best_under25": best_un25,
                        "odds_over15":  float(ov15) if ov15 else None,
                        "odds_over35":  float(ov35) if ov35 else None,
                        "odds_btts":    float(btts_yes) if btts_yes else None,
                        # Kept so BTTS can be fetched per-event afterwards. Free: both come from
                        # the response we already paid for.
                        "_oa_event_id": event.get("id", ""),
                        "_oa_sport_key": sport_key,
                    })
        except Exception as e:
            n_err += 1
            _failed_leagues.append(f"{league}: {type(e).__name__}")
            log.warning(f"OddsAPI {league} exception: {e} — league skipped this run")

    # ── BTTS: per-event, whole board ─────────────────────────────────────────────────
    # BTTS is unavailable on the bulk /odds endpoint (asking for it 422s the ENTIRE call, which is
    # what silently wiped every tip in v9.2), so it needs one credit per event. See src/btts_odds.py
    # for the cost arithmetic and why the API-Football archives were rejected as the source.
    try:
        from src.btts_odds import fetch as _fetch_btts
        rows = _fetch_btts(rows)
    except Exception as e:
        log.warning(f"[btts] per-event fetch skipped ({e})")

    # Health telemetry: distinguishes "every league errored" (hard failure — the
    # btts-422 signature) from "leagues OK but no games". Read by the notifier.
    global _LAST_FETCH_STATS, _LAST_BOOK_ROWS
    _LAST_FETCH_STATS = {"queried": n_queried, "ok": n_ok, "err": n_err, "kept": len(rows),
                         "failed_leagues": _failed_leagues}
    _LAST_BOOK_ROWS = list(_book_rows)          # shadow-only copy; see the module-level note

    # Persist the per-book prices. Wrapped so it can NEVER affect predict: a timed-out or
    # crashed predict sends zero tips, which is far worse than a missing research file.
    # Stores price CHANGES only, like drift.py — an unchanged book rewrites nothing, so git
    # sees no diff and the file grows with real market movement rather than with run count.
    if _book_rows:
        try:
            _bp = config.OUTPUT_DIR / "book_odds_snapshots.csv"
            _bn = pd.DataFrame(_book_rows)
            if _bp.exists():
                _bn = pd.concat([pd.read_csv(_bp), _bn], ignore_index=True)
            _key = ["match_date", "match", "market", "side", "bookmaker"]
            _bn = _bn.sort_values(_key + ["snapshot_ts"])
            _prev = _bn.groupby(_key)["odds"].shift()
            _bn = _bn[_bn["odds"].ne(_prev)].sort_values("snapshot_ts")
            _bn.to_csv(_bp, index=False)
            log.info(f"[book_odds] {len(_book_rows)} quote(s) from "
                     f"{_bn['bookmaker'].nunique()} bookmaker(s) -> {_bp.name} "
                     f"(total {len(_bn):,})")
            # Rolling file (2026-10-22): finished match-dates move once into
            # book_odds_snapshots_archive/<match_date>.csv, so git stops re-storing the whole
            # history every run. Upcoming matches keep their full curve here. Never raises.
            from src.rolling_store import roll as _roll
            _roll(_bp)
        except Exception as e:
            log.warning(f"[book_odds] per-book capture skipped ({e})")

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["league", "home_team", "away_team"])


# ── Postponement detection ────────────────────────────────────────────────────

def _detect_postponed(current_df: pd.DataFrame) -> list[dict]:
    """
    Compare current live fixtures against the previous predictions.csv.
    Returns list of dicts for fixtures that were predicted but are no longer live.
    Only checks leagues covered by OddsAPI (reliable source-of-truth).
    """
    prev_file = config.OUTPUT_DIR / "predictions.csv"
    if not prev_file.exists():
        return []

    try:
        prev = pd.read_csv(prev_file, parse_dates=["date"])
    except Exception:
        return []

    # Only check upcoming (future) matches from previous run
    today = pd.Timestamp(datetime.utcnow().date())
    prev_upcoming = prev[prev["date"] >= today].copy()
    if prev_upcoming.empty:
        return []

    # Build set of current fixture keys
    current_keys = set(
        f"{r.home_team}|{r.away_team}|{str(r.date)[:10]}"
        for r in current_df.itertuples()
    )

    postponed = []
    for _, row in prev_upcoming.iterrows():
        k = f"{row['home_team']}|{row['away_team']}|{str(row['date'])[:10]}"
        if k not in current_keys:
            postponed.append({
                "date":       row["date"],
                "league":     row.get("league", ""),
                "home_team":  row["home_team"],
                "away_team":  row["away_team"],
                "status":     "POSTPONED",
            })

    return postponed


# ── Main predict function ─────────────────────────────────────────────────────

def predict_upcoming(
    historical:        pd.DataFrame,
    payload:           dict,
    payload_newformat: dict | None = None,
    side_payloads:     dict | None = None,
    days_ahead:        int = 7,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Full predict pipeline for upcoming fixtures.

    Parameters
    ----------
    historical : raw historical DataFrame from data_loader.load_all_matches()
    payload    : trained model payload from model.load_models()
    days_ahead : how far ahead to look for fixtures

    Returns
    -------
    (predictions_df, postponed_list)
      predictions_df : all fixtures with model score + signal_tier + drift columns
      postponed_list : fixtures that disappeared since last run
    """
    from src.feature_engineering import build_upcoming_features
    from src.model import predict_proba
    from src.betting import evaluate_value
    from src.drift import snapshot, enrich, purge_old
    from src import standard_serving as _std_serving

    # 0. Configuration gate (R2 finding 17): an invalid STANDARD_OU25_SERVING_RECIPE is a
    #    configuration error, raised BEFORE any OddsAPI credit is spent - never a silent legacy run.
    _std_serving.check_config()

    # 1. Fetch live fixtures
    log.info("Fetching upcoming fixtures from OddsAPI...")
    upcoming = _fetch_odds_api(days_ahead)
    log.info(f"  OddsAPI: {len(upcoming)} fixtures found")

    # Record fetch health so the notifier can alert on an outage (0 fixtures because
    # every league errored — the btts-422 signature that silently killed tips ~2wk).
    try:
        from src.health_check import record_fetch
        record_fetch(len(upcoming), _LAST_FETCH_STATS)
    except Exception as e:
        log.debug(f"health record skipped: {e}")

    if upcoming.empty:
        log.warning("No upcoming fixtures found")
        return pd.DataFrame(), []

    # 2. Detect postponements (before we snapshot — compare prev vs current)
    postponed = _detect_postponed(upcoming)
    if postponed:
        log.info(f"  Postponed/disappeared since last run: {len(postponed)}")
        for p in postponed:
            log.info(f"    [PST] {p['home_team']} vs {p['away_team']} ({str(p['date'])[:10]})")

    # 3. Snapshot current odds for drift tracking
    snapshot(upcoming)
    purge_old(cutoff_days=10)

    # 4. Build features
    log.info(f"Building features for {len(upcoming)} upcoming fixtures...")
    feat = build_upcoming_features(upcoming, historical)

    # 5. Model prediction — route each fixture to the correct model
    log.info("Running ensemble model(s)...")
    probas = np.full(len(feat), np.nan)

    std_mask = feat["league"].isin(config.STANDARD_FORMAT_LEAGUES)
    nf_mask  = feat["league"].isin(config.NEW_FORMAT_LEAGUES)

    if std_mask.any():
        p_std = predict_proba(feat[std_mask], payload=payload)
        probas[std_mask.values] = p_std.values
        log.info(f"  Standard model: {std_mask.sum()} fixtures")

    # FAIL CLOSED (R2 finding 8, 2026-11-01). Two fallbacks used to score a fixture with a model
    # that was never trained for it: a New Format fixture went to the STANDARD model when the NF
    # artifact was missing, and a league in neither list ("Fallback (unknown league)") always went
    # to the STANDARD model. Both now leave p_over25 EMPTY: no probability -> no edge -> AVOID ->
    # no tip, plus an ERROR log line and output/league_registry_health.json `last_predict_routing`.
    # Routing is by EXACT canonical name (config.model_type_for_league); a near-name such as
    # "Austria Bundesliga" or "championship" is unknown, never "close enough".
    nf_payload_missing = payload_newformat is None
    if nf_mask.any():
        if nf_payload_missing:
            log.error(f"  New-format model artifact NOT LOADED: {int(nf_mask.sum())} New Format "
                      f"fixture(s) left WITHOUT a probability (no Standard-model fallback; fail closed)")
        else:
            p_nf = predict_proba(feat[nf_mask], payload=payload_newformat)
            probas[nf_mask.values] = p_nf.values
            log.info(f"  New-format model: {nf_mask.sum()} fixtures")

    unknown_mask = ~std_mask & ~nf_mask
    if unknown_mask.any():
        _unk = feat.loc[unknown_mask, "league"].astype(str).value_counts().to_dict()
        log.error(f"  UNKNOWN LEAGUE: {int(unknown_mask.sum())} fixture(s) in league(s) {_unk} are in "
                  f"neither STANDARD_FORMAT_LEAGUES nor NEW_FORMAT_LEAGUES -> model_type=unknown, "
                  f"NO probability, NO bet (fail closed; no Standard-model fallback)")
    _record_routing(feat, std_mask, nf_mask, unknown_mask, nf_payload_missing)

    # Standard O/U 2.5 serving recipe (owner 2026-10-28, config.STANDARD_OU25_SERVING_RECIPE).
    # legacy_v9 (default): a no-op — nothing above or below changes. v9_repaired: the Standard rows'
    # over25 probability is replaced by the repaired payload's, scored on repaired as-of features
    # built from `historical` alone; `feat` itself (which the side/HT models read below) is not
    # modified except for the added serving_recipe / serving_no_form columns. Never raises.
    from src import standard_serving as _std_serving
    probas = _std_serving.apply(feat, std_mask, probas, historical)

    feat["p_over25"]   = probas
    feat["model_type"] = "unknown"
    feat.loc[std_mask, "model_type"] = "standard"
    feat.loc[nf_mask,  "model_type"] = "new_format"

    # NEW-FORMAT RECALIBRATION TO OUTCOMES (owner 2026-11-01, src/nf_calibration.py): NF overs were
    # under-rated ~10pp against results. A one-parameter shift fitted on settled fixtures predicted
    # by the FIXED pipeline; applied only once n >= 150 and its CI excludes zero, reported always.
    # p_over25_raw keeps the model's own number.
    feat["p_over25_raw"] = feat["p_over25"]
    try:
        from src import nf_calibration as _nfc
        _cal = _nfc.fit()
        if nf_mask.any():
            feat.loc[nf_mask, "p_over25"] = _nfc.apply(feat.loc[nf_mask, "p_over25"].astype(float), _cal)
        log.info(f"  NF calibration: n={_cal['n']} shift={_cal.get('shift')} ci={_cal.get('ci95')} "
                 f"-> {_cal.get('reason')}")
    except Exception as _e:                                            # noqa: BLE001
        log.warning(f"  NF calibration skipped ({type(_e).__name__}: {_e}) - raw probabilities served")

    # How much of each model's input actually arrived? Blank/constant features are imputed
    # silently and skew the calibrated logistic member — the new-format bug in one line.
    # Diagnostic only; never alters a prediction.
    try:
        from src.health_check import record_feature_health
        _cols = {"standard": payload.get("feature_cols", [])}
        if payload_newformat is not None:
            _cols["new_format"] = payload_newformat.get("feature_cols", [])
        record_feature_health(feat, _cols)
    except Exception as e:
        log.debug(f"feature health skipped: {e}")

    # Side-market predictions (BTTS / O1.5 / O3.5) — all known leagues.
    # TRACK ROUTING lives in src.model.predict_proba: a payload trained by a scheduled retrain since
    # 2026-11-02 is track-split (src.model.train_by_track) and scores every fixture with ITS OWN
    # track's model only (a track without a model -> NaN, no tip). A payload trained before that has
    # no `tracks` key: it is the pooled Standard+New Format model and keeps serving both, unchanged,
    # until the next retrain replaces it through the promotion gate.
    if side_payloads:
        side_mask = std_mask | nf_mask
        for target, sp in side_payloads.items():
            col = f"p_{target}"
            feat[col] = np.nan
            if side_mask.any():
                try:
                    feat.loc[side_mask, col] = predict_proba(feat[side_mask], payload=sp).values
                    log.info(f"  Side-market {target}: {side_mask.sum()} fixtures "
                             f"({std_mask.sum()} std + {nf_mask.sum()} nf)")
                except Exception as e:
                    log.warning(f"  Side-market {target} prediction failed: {e}")

    # 5b. HT model predictions (standard-format leagues only)
    from src.model import load_models as _load_models
    feat["p_ht_over05"] = np.nan
    feat["p_ht_over15"] = np.nan
    if config.HT_MODEL_FILE_05.exists() and std_mask.any():
        try:
            ht_payload_05 = _load_models(model_file=config.HT_MODEL_FILE_05)
            ht_payload_15 = _load_models(model_file=config.HT_MODEL_FILE_15) \
                            if config.HT_MODEL_FILE_15.exists() else None

            # THE HT MODELS NOW TRAIN ON THEIR OWN FEATURES, SO PREDICT MUST BUILD THEM TOO.
            # Training goes through feature_engineering.build_features; this path goes through
            # build_upcoming_features. A column present in one and missing in the other is the
            # most dangerous shape in this codebase: model._prep imputes an all-NaN column to
            # 0.0, the scaler maps that to roughly z=-37 and the logistic collapses. That is
            # what pipeline.py's _NF_DROP block records, where it crushed new-format P(over)
            # from ~0.51 to ~0.36.
            #
            # attach_ht_features rebuilds the accumulators over `historical` and emits the
            # upcoming rows only. Upcoming fixtures carry no half-time score, so they receive
            # features and update nothing — every one sees the full history, none sees another.
            _ht_ok = False          # score HT only on features attached to the right fixtures
            try:
                from src.ht_features import attach_ht_features, HT_FEATURE_COLS
                _ht = attach_ht_features(historical, feat)
                # Same fixture on every row, or nothing: a positional copy with a mere length
                # check is how every HT feature landed on the wrong fixture until 2026-11-01.
                _aligned = len(_ht) == len(feat) and all(
                    (_ht[k].astype(str).to_numpy() == feat[k].astype(str).to_numpy()).all()
                    for k in ("home_team", "away_team"))
                if not _aligned:
                    log.error("  HT features do not line up with the fixtures — HT not scored this run")
                elif len(_ht) == len(feat):
                    for _c in HT_FEATURE_COLS:
                        if _c in _ht.columns:
                            feat[_c] = _ht[_c].to_numpy()
                    _ht_ok = True
                    _cov = float(np.mean([feat[c].notna().mean() for c in HT_FEATURE_COLS
                                          if c in feat.columns]))
                    log.info(f"  HT features attached: {_cov:.0%} populated")
                    # AN ALL-EMPTY FEATURE SET IS WORSE THAN NONE. If the build produced
                    # nothing usable, say so loudly rather than scoring against imputed zeros.
                    if _cov < 0.5:
                        log.warning("  HT features are mostly empty — scores will be unreliable")
                else:
                    log.warning(f"  HT feature rows {len(_ht)} != fixtures {len(feat)} — skipped")
            except Exception as _e:                                   # noqa: BLE001
                log.error(f"  HT feature attach failed ({_e}) — HT not scored this run")

            # FAIL CLOSED: scoring on missing features imputes them (z ~ -37, see above) and emits
            # confident nonsense. No HT features, no HT probabilities.
            if _ht_ok:
                feat.loc[std_mask, "p_ht_over05"] = predict_proba(feat[std_mask], payload=ht_payload_05).values
                if ht_payload_15:
                    feat.loc[std_mask, "p_ht_over15"] = predict_proba(feat[std_mask], payload=ht_payload_15).values
                log.info(f"  HT model: {std_mask.sum()} fixtures scored")
        except Exception as e:
            log.warning(f"  HT model prediction failed: {e}")

    # 6. Value betting evaluation (includes both-losing guard)
    log.info("Applying value betting logic...")
    feat = evaluate_value(feat)

    # 7. Enrich with drift columns (uses history just written in step 3)
    log.info("Enriching with odds drift signal...")
    feat = enrich(feat)

    # 8. Re-run evaluate_value now that drift columns exist (single pass)
    feat = evaluate_value(feat)

    # 9. Fixture identity (issue #1, 2026-11-02): API-Football fixture_id where it resolves, and
    #    the OddsAPI event id in its own column. Identity only - no model input reads either.
    feat = attach_fixture_ids(feat)

    return feat, postponed


def _event_id(v) -> str | None:
    """The OddsAPI event id as sent, or None for a missing / blank one."""
    if v is None or (isinstance(v, float) and v != v):
        return None
    s = str(v).strip()
    return s if s and s not in ("nan", "None") else None


def attach_fixture_ids(df: pd.DataFrame, fixtures_for=None) -> pd.DataFrame:
    """Add `fixture_id` (API-Football fixture id, nullable Int64) and `oddsapi_event_id` (the
    OddsAPI event id, string) to a board of upcoming fixtures. Never raises.

    `fixture_id` is looked up per (league, UTC date): the league's API-Football id from the
    registry (src/league_registry), its season from src/season_resolver, that date's fixtures
    from API-Football, and both clubs resolved by src.team_names.resolve within that pool
    (player_model.api_football.match_fixture: ambiguity refused, the pair must be one fixture).
    A league without an id, an unknown season, no API data or an unresolved pair leaves
    fixture_id EMPTY (counted and logged); the OddsAPI id is never written into fixture_id.

    `fixtures_for(league_id, season, date_str) -> list[dict]` is injectable for tests; the
    default is player_model.api_football.fixtures_on_date_raw (cached 6 h)."""
    out = df.copy()
    out["oddsapi_event_id"] = (out["_oa_event_id"].map(_event_id) if "_oa_event_id" in out.columns
                               else pd.Series([None] * len(out), index=out.index, dtype=object))
    ids = pd.Series([pd.NA] * len(out), index=out.index, dtype="Int64")
    if out.empty or not {"league", "date", "home_team", "away_team"} <= set(out.columns):
        out["fixture_id"] = ids
        return out
    try:
        from src import league_registry as LR
        from src.season_resolver import is_known, resolve_api_season
        from player_model.api_football import fixtures_on_date_raw, match_fixture
        if fixtures_for is None:
            def fixtures_for(lid, season, day):
                return fixtures_on_date_raw(lid, season, day, cache_hours=6)
        why: dict[str, int] = {}
        days = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for (lg, day), g in out.groupby([out["league"].astype(str), days], sort=False):
            lid = (LR.entry(lg) or {}).get("api_football_id")
            if not lid:
                why["league has no API-Football id"] = why.get("league has no API-Football id", 0) + len(g)
                continue
            season = resolve_api_season(lg, day)
            if not is_known(season):
                why["season unknown"] = why.get("season unknown", 0) + len(g)
                continue
            fixtures = fixtures_for(int(lid), str(season), day)
            for i, r in g.iterrows():
                fx = match_fixture(fixtures, r["home_team"], r["away_team"])
                fid = ((fx or {}).get("fixture") or {}).get("id")
                if fid:
                    ids.at[i] = int(fid)
                else:
                    why["clubs not resolved to one API-Football fixture"] = \
                        why.get("clubs not resolved to one API-Football fixture", 0) + 1
        if why:
            log.info(f"[fixture_id] {int(ids.notna().sum())} of {len(out)} fixture(s) carry an "
                     f"API-Football fixture_id; without one: {why}")
    except Exception as e:                                            # noqa: BLE001
        log.warning(f"[fixture_id] API-Football fixture ids not attached ({type(e).__name__}: {e})")
    out["fixture_id"] = ids
    return out
