"""Train + evaluate the side count models (offline; the serving artifact is committed).
====================================================================================

    python -m src.side_models.train            # holdout evaluation + final fit -> models/side_models/

1. COVERAGE. Every tracked league (config/leagues.json production_state ENABLED /
   ENABLED_PROVIDER_UNSUPPORTED / PAPER). A league without a football-data code has ONE source for
   corners and cards (API-Football) and is not covered. A league with one is covered for a family when
   the two-source overlap and agreement pass config/side_models.json `verification`.
2. HOLDOUT. Fit on verified fixtures before `holdout_split_date`, score the fixtures on/after it, and
   compare with the league's own training base rate per line (log loss, Brier, calibration).
3. FINAL FIT on every verified fixture -> models/side_models/side_models.json (+ holdout metrics CSV).
No price is read. No API call is made.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.side_models import data as D  # noqa: E402
from src.side_models import features as F  # noqa: E402
from src.side_models import model as M  # noqa: E402

CONFIG = ROOT / "config" / "side_models.json"
LEAGUES = ROOT / "config" / "leagues.json"
TRACKED_STATES = ("ENABLED", "ENABLED_PROVIDER_UNSUPPORTED", "PAPER")
EPS = 1e-15


def load_config(path: Path | None = None) -> dict:
    return json.loads(Path(path or CONFIG).read_text(encoding="utf-8"))


def tracked_leagues(path: Path | None = None) -> list:
    reg = json.loads(Path(path or LEAGUES).read_text(encoding="utf-8"))
    return [e for e in reg["leagues"] if e.get("production_state") in TRACKED_STATES]


def has_fd_stats(e: dict) -> bool:
    """football-data publishes match statistics (HC/AC/HY/AY/HR/AR) only in its mmz4281 season files;
    the 'new' extra-league files carry results and prices only."""
    return bool(e.get("football_data_code")) and e.get("football_data_format") == "mmz4281"


def single_source_leagues(cfg: dict, tracked: list) -> dict:
    """{league: window start} for tracked leagues without football-data match statistics (owner
    2026-11-18: API-Football is their single certified source)."""
    ss = cfg["single_source"]
    return {e["name"]: ss["window_from"] for e in tracked
            if not has_fd_stats(e) and e.get("api_football_id") is not None}


def coverage(cfg: dict, tracked: list, audit: dict) -> dict:
    """{family: {league: {"covered": bool, "basis", "reason", evidence...}}} for every tracked league.
    two_source:    overlap >= min_overlap_pct and agreement >= min_agree_pct (`verification`).
    single_source: API-Football stats on >= single_source.min_stats_pct of the window's fixtures, and
                   sane on >= single_source.min_sane_pct of those (`single_source`)."""
    v, ss = cfg["verification"], cfg["single_source"]
    out = {fam: {} for fam in cfg["families"]}
    for e in tracked:
        lg = e["name"]
        a = audit.get(lg)
        for fam in cfg["families"]:
            if a is None:
                out[fam][lg] = {"covered": False, "basis": None,
                                "reason": "no API-Football league id: no statistics source"}
                continue
            n = a["af_fixtures_in_window"]
            both = a[f"{fam}_both"]
            agree = a[f"{fam}_verified"]
            ov = 100.0 * both / n if n else 0.0
            ag = 100.0 * agree / both if both else 0.0
            if a["basis"] == "two_source":
                ok = ov >= v["min_overlap_pct"] and ag >= v["min_agree_pct"]
                why = (f"two sources: overlap {ov:.1f}% (needs {v['min_overlap_pct']}) / agreement {ag:.1f}% "
                       f"(needs {v['min_agree_pct']})")
                out[fam][lg] = {"covered": bool(ok), "basis": "two_source", "overlap_pct": round(ov, 2),
                                "agree_pct": round(ag, 2), "fixtures_in_window": int(n),
                                "both_sources": int(both), "verified": int(agree),
                                "window_from": a["fd_window_from"], "reason": "" if ok else why}
            else:
                ok = ov >= ss["min_stats_pct"] and ag >= ss["min_sane_pct"]
                why = (f"single source (API-Football): stats on {ov:.1f}% of window fixtures (needs "
                       f"{ss['min_stats_pct']}), sane {ag:.1f}% (needs {ss['min_sane_pct']}); by year "
                       f"{a.get('with_stats_pct_by_year')}")
                out[fam][lg] = {"covered": bool(ok), "basis": "single_source", "stats_pct": round(ov, 2),
                                "sane_pct": round(ag, 2), "fixtures_in_window": int(n),
                                "with_stats": int(both), "verified": int(agree),
                                "with_stats_pct_by_year": a.get("with_stats_pct_by_year"),
                                "window_from": a["fd_window_from"], "reason": "" if ok else why}
    return out


def training_frame(table: pd.DataFrame, league: str, family: str, fcfg: dict, since: str) -> pd.DataFrame:
    """Verified fixtures of one league x family (date >= since) with their as-of features, scored only."""
    t = table[(table["league"] == league) & (table["date"] >= since) & table[f"{family}_verified"]]
    if t.empty:
        return t.assign(home_for=[], home_against=[], away_for=[], away_against=[])
    f = F.build(table[table["league"] == league], t, family, **fcfg)
    return f[f["not_scored_reason"] == ""].reset_index(drop=True)


def _ll(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def _brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def _ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def _targets(fr: pd.DataFrame, family: str) -> dict:
    return {"home": fr[f"home_{family}"].to_numpy(float), "away": fr[f"away_{family}"].to_numpy(float),
            "total": (fr[f"home_{family}"] + fr[f"away_{family}"]).to_numpy(float)}


def fit_league(fr: pd.DataFrame, family: str) -> dict:
    y = _targets(fr, family)
    return {part: M.fit_part(fr, y[part], part) for part in M.PARTS}


def evaluate(train: pd.DataFrame, test: pd.DataFrame, family: str, markets: dict, league: str) -> list:
    params = fit_league(train, family)
    ytr, yte = _targets(train, family), _targets(test, family)
    rows = []
    for market, mc in markets.items():
        parts = ["total"] if mc["part"] == "total" else ["home", "away"]
        for part in parts:
            mu = M.mu_of(params[part], test, part)
            for line in mc["lines"]:
                p = M.p_over(mu, params[part]["alpha"], line)
                y = (yte[part] > line).astype(float)
                base = float((ytr[part] > line).mean())
                pb = np.full(len(y), base)
                rows.append({"league": league, "family": family, "market": market, "side": part, "line": line,
                             "n_train": int(len(train)), "n_test": int(len(test)),
                             "base_rate_train": round(base, 4), "over_rate_test": round(float(y.mean()), 4),
                             "mean_p_model": round(float(p.mean()), 4),
                             "logloss_model": round(_ll(p, y), 5), "logloss_base": round(_ll(pb, y), 5),
                             "brier_model": round(_brier(p, y), 5), "brier_base": round(_brier(pb, y), 5),
                             "ece_model": round(_ece(p, y), 4), "ece_base": round(_ece(pb, y), 4)})
    return rows


def run(*, write: bool = True, out_dir: Path | None = None) -> dict:
    cfg = load_config()
    fcfg = cfg["features"]
    tracked = tracked_leagues()
    fd_leagues = [e["name"] for e in tracked if has_fd_stats(e)]
    table, audit = D.build(fd_leagues, single_source=single_source_leagues(cfg, tracked))
    cov = coverage(cfg, tracked, audit)
    split = cfg["holdout_split_date"]
    metrics, models = [], {}
    for fam, fc in cfg["families"].items():
        models[fam] = {}
        for lg, c in cov[fam].items():
            if not c["covered"]:
                continue
            fr = training_frame(table, lg, fam, fcfg, c["window_from"])
            tr, te = fr[fr["date"] < split], fr[fr["date"] >= split]
            if len(tr) and len(te):
                metrics += evaluate(tr, te, fam, fc["markets"], lg)
            params = fit_league(fr, fam)
            models[fam][lg] = {"params": params, "n_fit": int(len(fr)),
                               "fit_from": str(fr["date"].min()), "fit_to": str(fr["date"].max()),
                               "n_holdout_train": int(len(tr)), "n_holdout_test": int(len(te))}
    data_end = str(table.loc[table["corners_verified"] | table["cards_verified"], "date"].max())
    body = {"config": {k: cfg[k] for k in ("families", "features", "verification", "single_source",
                                           "holdout_split_date")},
            "data_end": data_end, "coverage": cov,
            "data_audit": {lg: {k: v for k, v in a.items() if k != "crosswalk"} for lg, a in audit.items()},
            "crosswalk": {lg: a["crosswalk"] for lg, a in audit.items() if "crosswalk" in a}, "models": models}
    sha = M.params_sha(body)
    art = {"schema_version": "side/1.0", "model_version": f"side_nb_v1_{data_end}_{sha}",
           "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "sources": {"api_football": str(D.FILL.relative_to(ROOT)).replace("\\", "/"),
                       "fixtures": str(D.FIXTURES.relative_to(ROOT)).replace("\\", "/"),
                       "football_data": str(D.FD_RAW.relative_to(ROOT)).replace("\\", "/") + "/manifest.json"},
           **body}
    mdf = pd.DataFrame(metrics)
    if write:
        out_dir = Path(out_dir or (ROOT / "models" / "side_models"))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "side_models.json").write_text(json.dumps(art, indent=1) + "\n", encoding="utf-8")
        mdf.to_csv(out_dir / "holdout_metrics.csv", index=False, lineterminator="\n")
    return {"artifact": art, "metrics": mdf, "table": table}


def main(argv=None) -> int:
    res = run()
    art, mdf = res["artifact"], res["metrics"]
    print(f"model_version {art['model_version']}  data_end {art['data_end']}")
    for fam, cov in art["coverage"].items():
        on = [lg for lg, c in cov.items() if c["covered"]]
        print(f"{fam}: covered {len(on)}: {on}")
    if len(mdf):
        g = mdf.groupby(["family", "market"])[["logloss_model", "logloss_base", "brier_model", "brier_base"]].mean()
        print(g.round(5).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
