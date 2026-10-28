"""BEFORE/AFTER of the Standard O/U 2.5 serving switch on the committed board. READ-ONLY.

    python scripts/standard_switch_preview.py --out _preview [--payload P.pkl] [--history-network]

Runs src.predict.predict_upcoming twice on the fixtures of the committed output/predictions.csv —
STANDARD_OU25_SERVING_RECIPE = legacy_v9, then = v9_repaired — under the env predict.yml deploys
(thresholds etc.; CLAUDE.md trap 1), then the same tip pipeline pipeline.mode_predict runs
(generate_bets, _generate_side_bets, the direction guard, paper-league strip, flip guard), in memory.

  * OddsAPI is stubbed: the board's own fixtures and prices are fed in.
  * Every writer predict_upcoming calls (drift snapshot/purge, health records) is stubbed, and the
    network is blocked for the predict passes. History is loaded once (v9's own loader) before the
    block; pass --history-network to let it refresh the current seasons as predict would (CI).
  * Nothing under output/, models/ or telegram_bot/ is written. Results go to --out only.

Outputs (in --out):
  preview_fixtures.csv   per fixture: league, match, kickoff, odds, legacy/repaired p_over25, legacy
                         and repaired bet/side/tier/edge, change class, serving_recipe, model_version
  preview_summary.json   tips added/removed/flipped/tier-changed per league + the invariance proofs
  preview_report.md      the same, readable
Exit code 1 if any invariance proof fails (New Format rows / side models / HT / NF send rows).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

#: columns the switch adds or is ALLOWED to change on Standard rows (everything else must match)
OU_COLS = ("p_over25", "impl_prob_over", "impl_prob_under", "edge_over", "edge_under", "kelly_over",
           "kelly_under", "ev_over", "ev_under", "stake_over", "stake_under", "tier_over", "tier_under",
           "min_odds_blocked", "max_odds_blocked", "bet", "bet_stake", "signal_tier", "drift_shadow_tier", "best_side",
           "best_edge", "best_odds", "no_form_data", "both_losing", "serving_recipe", "serving_no_form",
           "model_version", "retrain_batch_id", "training_cutoff", "prediction_role")
#: proofs reported but not required (Standard non-O/U columns: an unlisted derived column would differ)
INFORMATIONAL = ("standard_rows_non_ou_columns",)
SIDE_P = ("p_btts", "p_over15", "p_over35", "p_ht_over05", "p_ht_over15")


def deployed_env() -> dict:
    """Non-secret env vars predict.yml sets that config.py reads (same rule as src.research_lock,
    duplicated so config is not imported before the env is in place)."""
    cfg = (ROOT / "config.py").read_text(encoding="utf-8")
    keys = set(re.findall(r'os\.getenv\(\s*"([A-Z0-9_]+)"', cfg))
    text = (ROOT / ".github" / "workflows" / "predict.yml").read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r'^\s+([A-Z0-9_]+):\s*(.*?)\s*$', text, flags=re.M):
        k, v = m.group(1), m.group(2)
        if k in keys and "${{" not in v:
            out[k] = v.split(" #")[0].strip().strip('"').strip("'")
    return out


def _hash(df) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()[:16]


def _block_network():
    import socket

    def _no(*a, **k):
        raise OSError("network blocked by standard_switch_preview")
    socket.socket.connect = _no                                   # type: ignore[assignment]
    socket.create_connection = _no                                # type: ignore[assignment]


def board_fixtures(pred_csv: Path):
    import pandas as pd
    b = pd.read_csv(pred_csv)
    keep = ["league", "date", "kickoff_utc", "home_team", "away_team", "odds_over25", "odds_under25",
            "odds_over15", "odds_over35", "odds_btts", "_oa_event_id", "_oa_sport_key"]
    up = b[[c for c in keep if c in b.columns]].copy()
    up["date"] = pd.to_datetime(up["date"], errors="coerce").dt.normalize()
    if "kickoff_utc" in up.columns:
        k = pd.to_datetime(up["kickoff_utc"], errors="coerce", utc=True)
        up["kickoff_utc"] = k.dt.tz_localize(None)
    return up.drop_duplicates(subset=["league", "home_team", "away_team"]).reset_index(drop=True), b


def run_mode(recipe: str, upcoming, historical, payloads: dict):
    """One predict pass + the in-memory tip pipeline. Returns dict of frames."""
    import config
    import pipeline
    from src import predict as P
    from src import direction_guard as DG
    from src.betting import generate_bets
    from src.ledger import drop_flipped
    from src.provenance import stamp_versions
    from src import model_registry as reg

    config.STANDARD_OU25_SERVING_RECIPE = recipe
    reg._version_cache.clear()
    P._fetch_odds_api = lambda days_ahead=7: upcoming.copy()
    t0 = time.perf_counter()
    preds, _ = P.predict_upcoming(historical, payloads["std"], payload_newformat=payloads["nf"],
                                  side_payloads=payloads["side"], days_ahead=7)
    secs = time.perf_counter() - t0
    preds = stamp_versions(preds)
    bets = generate_bets(preds)
    side = pipeline._generate_side_bets(preds, config.SIDE_MARKETS)
    g_bets, g_side, conflicts = DG.apply(bets, side)
    send_bets = drop_flipped(config.drop_paper_leagues(g_bets)) if not g_bets.empty else g_bets
    send_side = config.drop_paper_leagues(g_side) if not g_side.empty else g_side
    return {"preds": preds, "bets_pre": bets, "side_pre": side, "bets": g_bets, "side": g_side,
            "send_bets": send_bets, "send_side": send_side, "conflicts": conflicts, "secs": secs}


def _nf(df):
    import config
    if df is None or df.empty or "league" not in df.columns:
        return df
    return df[df["league"].isin(config.NEW_FORMAT_LEAGUES)].reset_index(drop=True)


def _std(df):
    import config
    if df is None or df.empty or "league" not in df.columns:
        return df
    return df[df["league"].isin(config.STANDARD_FORMAT_LEAGUES)].reset_index(drop=True)


def invariance(L: dict, R: dict) -> dict:
    """Byte-identity proofs. Each entry: {ok, legacy_hash, repaired_hash, rows}."""
    import pandas as pd
    out = {}
    added = ("serving_recipe", "serving_no_form")

    def _cmp(name, a, b):
        # serving_recipe / serving_no_form are the switch's own provenance columns (absent in legacy
        # mode by design); every OTHER byte must match.
        a = (a if a is not None else pd.DataFrame()).drop(columns=list(added), errors="ignore")
        b = (b if b is not None else pd.DataFrame()).drop(columns=list(added), errors="ignore")
        ha, hb = _hash(a), _hash(b)
        out[name] = {"ok": ha == hb, "legacy_hash": ha, "repaired_hash": hb, "rows": [len(a), len(b)]}

    pl, pr = L["preds"], R["preds"]
    drop = [c for c in ("serving_recipe", "serving_no_form") if c in pr.columns]
    _cmp("new_format_prediction_rows_all_columns", _nf(pl), _nf(pr.drop(columns=drop)))
    _cmp("side_and_ht_probabilities_all_rows",
         pl[[c for c in SIDE_P if c in pl.columns]], pr[[c for c in SIDE_P if c in pr.columns]])
    non_ou = [c for c in pl.columns if c not in OU_COLS]
    _cmp("standard_rows_non_ou_columns", _std(pl)[non_ou], _std(pr)[non_ou])
    _cmp("side_bets_before_direction_guard", L["side_pre"], R["side_pre"])
    _cmp("new_format_bets", _nf(L["bets"]), _nf(R["bets"]))
    _cmp("new_format_side_bets", _nf(L["side"]), _nf(R["side"]))
    _cmp("new_format_telegram_send_rows_bets_csv", _nf(L["send_bets"]), _nf(R["send_bets"]))
    _cmp("new_format_telegram_send_rows_side_bets_csv", _nf(L["send_side"]), _nf(R["send_side"]))
    return out


def fixture_table(L: dict, R: dict):
    import numpy as np
    import pandas as pd
    key = ["league", "date", "home_team", "away_team"]
    cols = ["p_over25", "bet", "best_side", "signal_tier", "best_edge", "no_form_data", "model_version"]
    a = L["preds"][key + ["kickoff_utc", "model_type", "odds_over25", "odds_under25"]
                   + [c for c in cols if c in L["preds"].columns]].copy()
    b = R["preds"][key + [c for c in cols + ["serving_recipe", "serving_no_form"] if c in R["preds"].columns]]
    t = a.merge(b, on=key, suffixes=("_legacy", "_repaired"), how="outer")
    t["match"] = t["home_team"].astype(str) + " v " + t["away_team"].astype(str)
    t["p_delta"] = t["p_over25_repaired"] - t["p_over25_legacy"]

    def _tip(bet, tier):
        return (str(bet) in ("OVER", "UNDER")) and str(tier) in ("SNIPER", "MARKSMAN", "VALUABLE")
    tl = [_tip(x, y) for x, y in zip(t["bet_legacy"], t["signal_tier_legacy"])]
    tr = [_tip(x, y) for x, y in zip(t["bet_repaired"], t["signal_tier_repaired"])]
    cls = []
    for i in range(len(t)):
        if tl[i] and not tr[i]:
            cls.append("REMOVED")
        elif tr[i] and not tl[i]:
            cls.append("ADDED")
        elif tl[i] and tr[i] and t["bet_legacy"].iat[i] != t["bet_repaired"].iat[i]:
            cls.append("FLIPPED")
        elif tl[i] and tr[i] and t["signal_tier_legacy"].iat[i] != t["signal_tier_repaired"].iat[i]:
            cls.append("TIER_CHANGED")
        elif tl[i] and tr[i]:
            cls.append("SAME_TIP")
        else:
            cls.append("NO_TIP_BOTH")
    t["change"] = cls
    t["track"] = np.where(t["model_type"].astype(str) == "standard", "standard",
                          np.where(t["model_type"].astype(str) == "new_format", "newformat", "unknown"))
    order = ["track", "league", "date", "kickoff_utc", "match", "odds_over25", "odds_under25",
             "p_over25_legacy", "p_over25_repaired", "p_delta",
             "bet_legacy", "best_side_legacy", "signal_tier_legacy", "best_edge_legacy",
             "bet_repaired", "best_side_repaired", "signal_tier_repaired", "best_edge_repaired",
             "change", "no_form_data_legacy", "no_form_data_repaired", "serving_recipe",
             "serving_no_form", "model_version_legacy", "model_version_repaired"]
    return t[[c for c in order if c in t.columns]].sort_values(["track", "league", "date", "match"])


def summarise(t, inv: dict, L: dict, R: dict, meta: dict) -> dict:
    import pandas as pd
    s = t[t["track"] == "standard"]
    per = (s.groupby(["league", "change"]).size().unstack(fill_value=0)
           if not s.empty else pd.DataFrame())
    d = pd.to_numeric(s["p_delta"], errors="coerce").dropna()
    return {
        **meta,
        "fixtures": {"total": int(len(t)), "standard": int(len(s)),
                     "newformat": int((t["track"] == "newformat").sum())},
        "standard_changes": s["change"].value_counts().to_dict(),
        "standard_changes_by_league": {lg: {k: int(v) for k, v in r.items() if v}
                                       for lg, r in per.iterrows()} if not per.empty else {},
        "standard_p_delta": ({"n": int(len(d)), "mean": round(float(d.mean()), 4),
                              "mean_abs": round(float(d.abs().mean()), 4),
                              "max_abs": round(float(d.abs().max()), 4)} if len(d) else {"n": 0}),
        "serving_recipe_counts": (R["preds"]["serving_recipe"].value_counts().to_dict()
                                  if "serving_recipe" in R["preds"].columns else {}),
        "standard_side_tips_suppressed_by_direction_guard": {
            "legacy": int(len(_std(L["side_pre"])) - len(_std(L["side"]))) if not L["side_pre"].empty else 0,
            "repaired": int(len(_std(R["side_pre"])) - len(_std(R["side"]))) if not R["side_pre"].empty else 0},
        "send_rows": {"legacy": {"bets": int(len(L["send_bets"])), "side": int(len(L["send_side"]))},
                      "repaired": {"bets": int(len(R["send_bets"])), "side": int(len(R["send_side"]))}},
        "predict_seconds": {"legacy": round(L["secs"], 1), "repaired": round(R["secs"], 1)},
        "invariance": inv,
        "invariance_all_ok": all(v["ok"] for k, v in inv.items() if k not in INFORMATIONAL),
    }


def report_md(summary: dict, t) -> str:
    s = t[t["track"] == "standard"]
    lines = [f"# Standard O/U 2.5 switch preview — board generated {summary.get('board_generated_at')}",
             "", f"Repaired model: `{summary.get('repaired_version')}` (training cutoff "
             f"{summary.get('repaired_training_cutoff')}). Env: predict.yml deployed values.", "",
             f"Invariance (New Format / side / HT / NF send rows byte-identical): "
             f"**{'PASS' if summary['invariance_all_ok'] else 'FAIL'}**", ""]
    for k, v in summary["invariance"].items():
        lines.append(f"- {k}: {'OK' if v['ok'] else 'DIFFERS'} ({v['legacy_hash']} / {v['repaired_hash']}, rows {v['rows']})")
    lines += ["", f"Standard fixtures: {summary['fixtures']['standard']} · changes {summary['standard_changes']}",
              f"p_over25 repaired - legacy: {summary['standard_p_delta']}",
              f"predict seconds legacy/repaired: {summary['predict_seconds']}", "",
              "| league | match | odds O/U | p legacy | p repaired | legacy tip | repaired tip | change |",
              "|---|---|---|---|---|---|---|---|"]
    def _f(x, fmt):
        try:
            return format(float(x), fmt)
        except (TypeError, ValueError):
            return str(x)

    for r in s.itertuples(index=False):
        lt = (f"{r.bet_legacy} {r.signal_tier_legacy} {_f(r.best_edge_legacy, '+.3f')}"
              if str(r.bet_legacy) in ("OVER", "UNDER") else "-")
        rt = (f"{r.bet_repaired} {r.signal_tier_repaired} {_f(r.best_edge_repaired, '+.3f')}"
              if str(r.bet_repaired) in ("OVER", "UNDER") else "-")
        lines.append(f"| {r.league} | {r.match} | {r.odds_over25}/{r.odds_under25} | "
                     f"{_f(r.p_over25_legacy, '.3f')} | {_f(r.p_over25_repaired, '.3f')} | {lt} | {rt} | "
                     f"{r.change} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--payload", default=None, help="repaired payload (default: the installed serving file)")
    ap.add_argument("--predictions", default=str(ROOT / "output" / "predictions.csv"))
    ap.add_argument("--history-network", action="store_true",
                    help="let v9's loader refresh current seasons (as predict does); predict passes stay offline")
    a = ap.parse_args(argv)

    env = deployed_env()
    for k, v in env.items():
        os.environ[k] = v
    os.environ.pop("APIFOOTBALL_KEY", None)                    # predict.yml does not set it
    os.environ.pop("ODDS_API_KEY", None)
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import config
    from src import model_registry as reg
    from src import standard_serving as SS
    from src.model import load_models

    payload_path = Path(a.payload) if a.payload else config.MODEL_FILE_STANDARD_REPAIRED
    config.MODEL_FILE_STANDARD_REPAIRED = payload_path
    reg.MODEL_FILES["standard_repaired"] = payload_path
    rep_payload = SS.load_payload(payload_path)                # validates

    # read-only: stub every writer predict_upcoming calls
    import src.drift as D
    import src.health_check as H
    D.snapshot = lambda *a_, **k_: None
    D.purge_old = lambda *a_, **k_: None
    H.record_fetch = lambda *a_, **k_: None
    H.record_feature_health = lambda *a_, **k_: None

    from src.data_loader import load_all_matches
    if not a.history_network:
        _block_network()
    historical = load_all_matches()
    _block_network()

    upcoming, board = board_fixtures(Path(a.predictions))
    payloads = {"std": load_models(model_file=config.MODEL_FILE_STANDARD),
                "nf": load_models(model_file=config.MODEL_FILE_NEWFORMAT) if config.MODEL_FILE_NEWFORMAT.exists() else None,
                "side": {t: load_models(model_file=f) for t, f in config.SIDE_MARKETS.items() if f.exists()}}
    L = run_mode("legacy_v9", upcoming, historical, payloads)
    R = run_mode("v9_repaired", upcoming, historical, payloads)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = fixture_table(L, R)
    t.to_csv(out / "preview_fixtures.csv", index=False)
    inv = invariance(L, R)
    rs = rep_payload.get("research") or {}
    meta = {"board_generated_at": str(board["generated_at"].iloc[0]) if "generated_at" in board.columns and len(board) else "",
            "board_rows": int(len(board)), "deployed_env": env,
            "repaired_payload": str(payload_path), "repaired_sha256": reg.sha256_file(payload_path),
            "repaired_version": reg.stamp_for("standard_repaired", payload_path)["model_version"],
            "repaired_training_cutoff": rs.get("training_cutoff"),
            "repaired_served_holdout_metrics": rep_payload.get("served_holdout_metrics"),
            "legacy_version": reg.stamp_for("standard")["model_version"],
            "history_network": bool(a.history_network)}
    summary = summarise(t, inv, L, R, meta)
    (out / "preview_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "preview_report.md").write_text(report_md(summary, t), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("fixtures", "standard_changes", "standard_p_delta",
                                              "invariance_all_ok", "predict_seconds")}, indent=2, default=str))
    return 0 if summary["invariance_all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
