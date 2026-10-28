"""
Standard serving-feature audit (P0 bugs 2 + 3, Standard forensic repair 2026-10-28). READ-ONLY.
==============================================================================================

For EVERY feature in the exact serving payload (default models/model_v9_standard.pkl — not
FEATURE_COLS) this reports where it is generated in training and live (file:line, located from
the source at run time), and REPLAYS past fixtures: each fixture's TRAINING value (the historical
as-of row) against what the UPCOMING builder produces for it with history cut at kickoff day.
It does this for both recipes:

    legacy_v9    build_features            vs  build_upcoming_features            (what serves)
    v9_repaired  build_features_repaired   vs  build_upcoming_features_repaired   (one function)

Per feature x recipe: training/serving non-null rate, mean, std, median, mean/median/p95 absolute
difference, share exactly equal, and PASS / WARN / FAIL. A second block compares the committed
live board (output/predictions.csv, Standard rows) with the training distribution — that is where
a feature that is CONSTANT live (home_advantage = 1.0) shows up even though replay cannot see it.

    python scripts/standard_feature_audit.py --out <csv> [--json-out <json>] [--days 56]
           [--leagues "Championship,League One"] [--payload models/model_v9_standard.pkl]
           [--frame raw.pkl] [--allow-network] [--fail-on-required]

READ-ONLY: HTTP is stubbed unless --allow-network (so the loader serves the committed cache and
writes nothing, and the legacy builder's API-Football enrichment returns nothing — which is
exactly what CI's predict sees when a fetch fails). The only file written is --out (and --json-out).
Exit code 1 with --fail-on-required when a REQUIRED feature of the repaired recipe FAILS.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger("standard_feature_audit")

REQUIRED = ("home_scored_last5", "away_scored_last5", "home_conceded_last5",
            "away_conceded_last5", "home_advantage", "away_home_adv_factor")
#: Known semantic facts the replay cannot measure (it feeds the same price to both sides).
SEMANTIC = {
    "bookmaker_overround": {
        "price_derived": True,
        "meaning_match_legacy": "NO: training = football-data CLOSING average (AvgC>2.5 first); "
                                "live = first OddsAPI book's CURRENT price",
        "asof_available_legacy": "NO (closing is not known at scoring time)"},
    "home_season_goals_h": {"source_dependent": "live overwritten by API-Football /teams/statistics"},
    "away_season_goals_a": {"source_dependent": "live overwritten by API-Football /teams/statistics"},
    "home_season_conceded_h": {"source_dependent": "live overwritten by API-Football /teams/statistics"},
    "away_season_conceded_a": {"source_dependent": "live overwritten by API-Football /teams/statistics"},
    "home_cs_rate_h": {"source_dependent": "live overwritten by API-Football /teams/statistics"},
    "away_cs_rate_a": {"source_dependent": "live overwritten by API-Football /teams/statistics"},
    "home_xg_last5": {"source_dependent": "API-Football shot enrichment cache"},
    "away_xg_last5": {"source_dependent": "API-Football shot enrichment cache"},
    "home_insidebox_last5": {"source_dependent": "API-Football shot enrichment cache"},
    "away_insidebox_last5": {"source_dependent": "API-Football shot enrichment cache"},
}


def _stub_network() -> None:
    import requests

    def _boom(*a, **k):
        raise requests.exceptions.ConnectionError("network stubbed by standard_feature_audit")
    requests.get = requests.post = requests.request = _boom
    requests.Session.request = lambda self, *a, **k: _boom()
    for k in ("APIFOOTBALL_KEY", "API_FOOTBALL_KEY", "ODDS_API_KEY", "RAPIDAPI_KEY"):
        os.environ.pop(k, None)
    try:
        import src.api_football_ou as AF
        for fn in ("get_league_injury_context", "fetch_upcoming_team_stats",
                   "resolve_upcoming_fixture_ids", "fetch_lineup_features", "fetch_h2h_features",
                   "fetch_prematch_odds_features", "fetch_season_round_features",
                   "fetch_coach_features"):
            if hasattr(AF, fn):
                setattr(AF, fn, lambda *a, **k: {})
    except Exception:                                                  # noqa: BLE001
        pass
    try:
        import src.api_football as AF2
        AF2.build_standings_map = lambda *a, **k: {}
    except Exception:                                                  # noqa: BLE001
        pass


# ── source locations ────────────────────────────────────────────────────────────────────────

def _region(text: str, start_pat: str, end_pat: str | None) -> tuple[int, int]:
    s = re.search(start_pat, text, re.M)
    if not s:
        return 0, len(text)
    e = re.search(end_pat, text[s.end():], re.M) if end_pat else None
    return s.start(), (s.end() + e.start()) if e else len(text)


def _locate(path: Path, feat: str, start_pat: str, end_pat: str | None) -> str:
    text = path.read_text(encoding="utf-8")
    a, b = _region(text, start_pat, end_pat)
    body = text[a:b]
    pats = [rf'\["{re.escape(feat)}"\]\s*=', rf'"{re.escape(feat)}"']
    suffix = re.sub(r"^(home|away)_", "", feat)
    if suffix != feat:
        pats.append(rf'"{re.escape(suffix)}"')
    for p in pats:
        m = re.search(p, body)
        if m:
            line = text[:a + m.start()].count("\n") + 1
            return f"{path.relative_to(ROOT).as_posix()}:{line}"
    return "NOT_FOUND"


def locations(feat: str) -> dict:
    fe = ROOT / "src" / "feature_engineering.py"
    fr = ROOT / "src" / "features_repaired.py"
    out = {
        "legacy_training_src": _locate(fe, feat, r"^def build_features\(", r"^def build_upcoming_features\("),
        "legacy_live_src": _locate(fe, feat, r"^def build_upcoming_features\(", None),
        "repaired_src": _locate(fr, feat, r"^def build_features_repaired\(", r"^def _resolve_names\("),
    }
    if feat in ("home_advantage", "away_home_adv_factor"):
        out["legacy_training_src"] = _locate(fe, feat, r"^def _add_home_advantage\(", r"^def _add_rest_days\(")
    if feat in ("home_rest_days", "away_rest_days"):
        out["legacy_training_src"] = _locate(fe, feat, r"^def _add_rest_days\(", r"^def _add_season_venue_stats\(")
    if feat.endswith(("_season_goals_h", "_season_goals_a", "_season_conceded_h",
                      "_season_conceded_a", "_cs_rate_h", "_cs_rate_a")):
        out["legacy_training_src"] = _locate(fe, feat, r"^def _add_season_venue_stats\(", r"^def _compute_referee_stats\(")
    return out


# ── statistics ──────────────────────────────────────────────────────────────────────────────

def _stats(tr: pd.Series, sv: pd.Series) -> dict:
    tr = pd.to_numeric(tr, errors="coerce").astype(float)
    sv = pd.to_numeric(sv, errors="coerce").astype(float)
    both = tr.notna() & sv.notna()
    d = (tr[both] - sv[both]).abs()
    def _r(x):
        return None if x is None or not np.isfinite(x) else round(float(x), 6)
    return {"n": int(len(tr)), "train_nonnull": _r(tr.notna().mean()), "serve_nonnull": _r(sv.notna().mean()),
            "train_mean": _r(tr.mean()), "serve_mean": _r(sv.mean()),
            "train_std": _r(tr.std(ddof=0)), "serve_std": _r(sv.std(ddof=0)),
            "train_median": _r(tr.median()), "serve_median": _r(sv.median()),
            "mean_abs_diff": _r(d.mean()) if len(d) else None,
            "median_abs_diff": _r(d.median()) if len(d) else None,
            "p95_abs_diff": _r(d.quantile(0.95)) if len(d) else None,
            "share_exact": _r((d < 1e-9).mean()) if len(d) else None,
            "serve_constant": bool(sv.dropna().nunique() <= 1 and tr.dropna().nunique() > 1)}


def _status(st: dict) -> str:
    if st["n"] == 0:
        return "NO_DATA"
    if st["serve_constant"]:
        return "FAIL"
    nn = abs((st["train_nonnull"] or 0) - (st["serve_nonnull"] or 0))
    sd = st["train_std"] or 0.0
    p95 = st["p95_abs_diff"]
    if p95 is None:
        return "FAIL" if nn > 0.1 else "WARN"
    rel = p95 / sd if sd > 0 else (0.0 if p95 < 1e-9 else np.inf)
    if rel <= 0.01 and nn <= 0.01:
        return "PASS"
    if rel <= 0.25 and nn <= 0.10:
        return "WARN"
    return "FAIL"


# ── replay ──────────────────────────────────────────────────────────────────────────────────

def replay(raw: pd.DataFrame, train_raw: pd.DataFrame, leagues: list, days: int,
           max_dates: int | None, price_mode: str) -> dict:
    """Return {recipe: (train_values_frame, serve_values_frame)} aligned row-for-row."""
    from src.feature_engineering import build_features, build_upcoming_features
    from src.features_repaired import build_features_repaired, build_upcoming_features_repaired
    K = ["date", "league", "home_team", "away_team"]
    played = raw[raw["home_goals"].notna() & raw["away_goals"].notna()].copy()
    played["date"] = pd.to_datetime(played["date"]).dt.normalize()
    end = played["date"].max()
    fx = played[(played["league"].isin(leagues)) & (played["date"] > end - pd.Timedelta(days=days))]
    dates = sorted(fx["date"].unique())
    if max_dates:
        dates = dates[-max_dates:]
    fx = fx[fx["date"].isin(dates)]
    log.info(f"replaying {len(fx)} fixtures on {len(dates)} dates ({dates[0]} .. {dates[-1]})")
    out = {}

    # legacy: training = build_features over the COVID-excluded training raw (mode_train);
    # serve = build_upcoming_features with the predict-time history (no COVID filter) cut at D.
    tr_leg = build_features(train_raw)
    tr_leg["date"] = pd.to_datetime(tr_leg["date"]).dt.normalize()
    sv_rows = []
    for D in dates:
        up = fx[fx["date"] == D][K + ["odds_over25", "odds_under25"]].copy()
        hist = raw[pd.to_datetime(raw["date"]).dt.normalize() < D]
        f = build_upcoming_features(up, hist)
        f[K] = up[K].to_numpy()
        sv_rows.append(f)
    sv_leg = pd.concat(sv_rows, ignore_index=True)
    out["legacy_v9"] = (fx[K].merge(tr_leg, on=K, how="left"), fx[K].merge(
        sv_leg.drop_duplicates(K), on=K, how="left"))

    # repaired: one function both ways
    tr_rep = build_features_repaired(train_raw, price_mode=price_mode)
    sv_rows = []
    for D in dates:
        up = fx[fx["date"] == D][K + ["odds_over25", "odds_under25"]].copy()
        f = build_upcoming_features_repaired(up, raw, price_mode=price_mode, cutoff=D)
        f[K] = up[K].to_numpy()
        sv_rows.append(f)
    sv_rep = pd.concat(sv_rows, ignore_index=True)
    out["v9_repaired"] = (fx[K].merge(tr_rep, on=K, how="left"), fx[K].merge(
        sv_rep.drop_duplicates(K), on=K, how="left"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="CSV path for the per-feature parity table")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--days", type=int, default=56)
    ap.add_argument("--max-dates", type=int, default=None)
    ap.add_argument("--leagues", default=None, help="comma list; default = enabled Standard leagues")
    ap.add_argument("--payload", default=str(ROOT / "models" / "model_v9_standard.pkl"))
    ap.add_argument("--frame", default=None, help="pickled/parquet raw history (else the loader)")
    ap.add_argument("--predictions", default=str(ROOT / "output" / "predictions.csv"))
    ap.add_argument("--price-mode", default="none", choices=("none", "asof", "closing"))
    ap.add_argument("--allow-network", action="store_true")
    ap.add_argument("--fail-on-required", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("src.data_loader", "src.feature_engineering", "src.api_football_ou"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    if not a.allow_network:
        _stub_network()
    import config
    from src.model import load_models
    from src.features_repaired import feature_cols_for

    payload = load_models(model_file=Path(a.payload))
    serving = list(payload["feature_cols"])
    leagues = ([x.strip() for x in a.leagues.split(",")] if a.leagues else
               sorted(set(config.ENABLED_LEAGUES) & set(config.STANDARD_FORMAT_LEAGUES)))
    if a.frame:
        raw = (pd.read_parquet(a.frame) if a.frame.endswith(".parquet") else pd.read_pickle(a.frame))
    else:
        from src.data_loader import load_all_matches
        raw = load_all_matches()
    train_raw = raw[~raw["season"].isin(config.COVID_SEASONS)] if config.EXCLUDE_COVID_SEASONS else raw
    rep_cols = set(feature_cols_for(a.price_mode, serving))

    rp = replay(raw, train_raw, leagues, a.days, a.max_dates, a.price_mode)
    rows = []
    for recipe, (tr, sv) in rp.items():
        for feat in serving:
            st = _stats(tr.get(feat, pd.Series(np.nan, index=tr.index)),
                        sv.get(feat, pd.Series(np.nan, index=sv.index)))
            used = recipe == "legacy_v9" or feat in rep_cols
            status = _status(st) if used else "NOT_USED"
            sem = SEMANTIC.get(feat, {})
            if recipe == "legacy_v9" and sem.get("price_derived"):
                status = "FAIL"           # meaning mismatch the replay cannot see
            rows.append({"kind": "replay_asof_vs_upcoming", "recipe": recipe, "feature": feat,
                         "required": feat in REQUIRED, "status": status, **st,
                         **locations(feat),
                         "price_derived": bool(sem.get("price_derived", False)),
                         "source_dependent": sem.get("source_dependent", ""),
                         "meaning_note": sem.get("meaning_match_legacy", "") if recipe == "legacy_v9" else "",
                         "batch_dependent_imputation": ("YES (model._prep batch median)"
                                                        if recipe == "legacy_v9"
                                                        else "NO (persisted train-only imputer)"),
                         "leagues": ",".join(leagues), "days": a.days})

    # live board vs training distribution (legacy what serves): constant-live features show here
    try:
        board = pd.read_csv(a.predictions)
        board = board[board["league"].isin(config.STANDARD_FORMAT_LEAGUES)]
        tr_leg = rp["legacy_v9"][0]
        for feat in serving:
            st = _stats(tr_leg.get(feat, pd.Series(dtype=float)),
                        board.get(feat, pd.Series(np.nan, index=board.index)))
            # distributions only: row diffs are meaningless across different fixtures
            for k in ("mean_abs_diff", "median_abs_diff", "p95_abs_diff", "share_exact"):
                st[k] = None
            st["n"] = int(len(board))
            nn = abs((st["train_nonnull"] or 0) - (st["serve_nonnull"] or 0))
            status = ("FAIL" if st["serve_constant"] or nn > 0.25 else
                      "WARN" if nn > 0.05 else "PASS")
            rows.append({"kind": "live_board_vs_training_distribution", "recipe": "legacy_v9",
                         "feature": feat, "required": feat in REQUIRED, "status": status, **st,
                         **locations(feat), "leagues": "board", "days": None})
    except Exception as e:                                             # noqa: BLE001
        log.warning(f"live-board comparison skipped: {e}")

    tab = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(a.out, index=False)
    summ = (tab.groupby(["kind", "recipe", "status"]).size().unstack(fill_value=0)
               .reset_index().to_dict("records"))
    req_fail = tab[(tab["kind"] == "replay_asof_vs_upcoming") & (tab["recipe"] == "v9_repaired")
                   & tab["required"] & (tab["status"] == "FAIL")]["feature"].tolist()
    report = {"serving_payload": str(a.payload), "n_serving_features": len(serving),
              "summary": summ, "repaired_required_failures": req_fail,
              "legacy_fail": tab[(tab["recipe"] == "legacy_v9") & (tab["kind"] == "replay_asof_vs_upcoming")
                                 & (tab["status"] == "FAIL")]["feature"].tolist()}
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 1 if (a.fail_on_required and req_fail) else 0


if __name__ == "__main__":
    sys.exit(main())
