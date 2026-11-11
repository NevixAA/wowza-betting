"""
COMBINED-MODELS research (v9). Research only: nothing here is read by serving.

Owner idea 2026-11-11: "combine all the models together to find an edge in matches. Maybe in a game
with a higher % of many goals, it's more realistic to bet on SoT?"

Stages (each writes to --work, outside the repo; only counts/estimates go to reports/):

  props-wf   Walk-forward out-of-sample props probabilities with the SERVED recipe
             (player_model.props_harness HGBNaNNative = `hgb_nan_native`, the candidate promoted from
             run 42855107089), refitted at every calendar month M on every eligible row dated before
             M, for the development months only (before the team holdout start 2026-04-01).
  crosswalk  League-by-league team-name crosswalk v9 match frame <-> API-Football player fixtures,
             learned from exact normalised names and from score-proven same-date fixtures.
  join       One row per (player, fixture, market) with the props OOS probability, the team model's
             walk-forward OOS probabilities (champion_recipe), Pro's Dixon-Coles lambdas and 6+ flag.
  analyse    Q1 conditioning residuals, Q2 coherence, Q3 contradiction patterns, inventory.

Rules (owner): per league only (pooled is printed as context, never a decision); leak-free (every
input dated before kickoff); every configuration searched counted in ONE Holm family; strict PROVEN
(>= 150 bets and CI > 0 after Holm); no prices are read or written by this module except row COUNTS
of the props price files; spent windows (2026-04-01.. for the team model, 2026-09-01.. for props)
are never scored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_PATH = ROOT / "config" / "experiments" / "combined_models" / "protocol.json"
REPORT_DATA = ROOT / "reports" / "combined_models_data"

#: the team model's untouched holdout (phase2-retrain-1.0) starts here: development stops the day before
TEAM_HOLDOUT_START = pd.Timestamp("2026-04-01")
#: the props holdout / spent window starts here
PROPS_SPENT_START = pd.Timestamp("2026-09-01")
DEV_START = pd.Timestamp("2025-01-01")
MARKETS = ("sot", "sot2", "goals")
#: the props features that already carry team / opponent strength (double-counting check, Q1)
TEAM_CONTEXT_FEATURES = ("is_home", "team_goals_pg_roll", "opp_goals_conceded_pg", "opp_sot_conceded_pg",
                         "opp_strength_index", "opp_goals_conceded_vs_fwd_pg", "opp_sot_conceded_vs_fwd_pg",
                         "opp_goals_conceded_vs_mid_pg", "opp_sot_conceded_vs_mid_pg",
                         "prev_season_team_goals_pg", "prev_season_team_conceded_pg",
                         "prev_season_opp_goals_pg", "prev_season_opp_conceded_pg", "team_corners_pg",
                         "box_threat_vs_leaky_defense", "efficiency_vs_leaky_keeper",
                         "team_momentum_forward_matchup", "quality_adj_goals_pg", "quality_adj_sot_pg")


# ── generic statistics ───────────────────────────────────────────────────────────────────────────

def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1). NaN enters as 1."""
    p = np.asarray([1.0 if (v is None or not np.isfinite(v)) else float(v) for v in pvals], dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def cluster_bootstrap_diff(values, groups, clusters, reps: int = 2000, seed: int = 7) -> dict:
    """Mean(values | group==1) - mean(values | group==0), resampling CLUSTERS (fixtures) with
    replacement. Returns estimate, 95% CI, one-sided p (effect <= 0) and two-sided p."""
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups, dtype=int)
    c = pd.factorize(np.asarray(clusters))[0]
    k = c.max() + 1 if len(c) else 0
    s1 = np.bincount(c, weights=v * (g == 1), minlength=k)
    n1 = np.bincount(c, weights=(g == 1).astype(float), minlength=k)
    s0 = np.bincount(c, weights=v * (g == 0), minlength=k)
    n0 = np.bincount(c, weights=(g == 0).astype(float), minlength=k)
    if n1.sum() == 0 or n0.sum() == 0:
        return {"est": np.nan, "lo": np.nan, "hi": np.nan, "p_one": np.nan, "p_two": np.nan}
    est = s1.sum() / n1.sum() - s0.sum() / n0.sum()
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for b in range(reps):
        w = np.bincount(rng.integers(0, k, k), minlength=k).astype(float)
        a1, a0 = (w * n1).sum(), (w * n0).sum()
        draws[b] = ((w * s1).sum() / a1 - (w * s0).sum() / a0) if a1 > 0 and a0 > 0 else np.nan
    draws = draws[np.isfinite(draws)]
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"est": float(est), "lo": float(lo), "hi": float(hi), **_normal_p(est, draws)}


def _normal_p(est: float, draws: np.ndarray) -> dict:
    """p-values from the bootstrap standard error (normal approximation). The empirical bootstrap
    p-value cannot go below 1/(reps+1) (0.0005 at 2,000 draws), so with a Holm family of m cells
    no cell could ever pass (m x 0.0005 > 0.05 once m > 100): the SE form is used for the test,
    and the empirical share is kept beside it (p_emp_one) for reference."""
    from scipy.stats import norm
    se = float(np.std(draws, ddof=1)) if len(draws) > 1 else np.nan
    z = est / se if se and np.isfinite(se) and se > 0 else np.nan
    p_one = float(norm.sf(z)) if np.isfinite(z) else np.nan
    p_two = float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
    p_emp = float((1 + np.sum(draws <= 0)) / (1 + len(draws)))
    return {"se": se, "p_one": p_one, "p_two": p_two, "p_emp_one": p_emp}


def cluster_bootstrap_mean(values, clusters, reps: int = 2000, seed: int = 7) -> dict:
    """Mean of values with a cluster bootstrap CI and a one-sided p (mean <= 0)."""
    v = np.asarray(values, dtype=float)
    c = pd.factorize(np.asarray(clusters))[0]
    if len(v) == 0:
        return {"est": np.nan, "lo": np.nan, "hi": np.nan, "p_one": np.nan, "p_two": np.nan}
    k = c.max() + 1
    s = np.bincount(c, weights=v, minlength=k)
    n = np.bincount(c, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for b in range(reps):
        w = np.bincount(rng.integers(0, k, k), minlength=k).astype(float)
        draws[b] = (w * s).sum() / (w * n).sum()
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"est": float(v.mean()), "lo": float(lo), "hi": float(hi), **_normal_p(float(v.mean()), draws)}


def prior_cutoff(values: pd.Series, league: pd.Series, month: pd.Series, q: float,
                 min_prior: int) -> np.ndarray:
    """Per league, the q-quantile of the values of STRICTLY EARLIER months (NaN when fewer than
    min_prior earlier values). Never uses the row's own month: the cutoff is known before kickoff."""
    v = pd.to_numeric(values, errors="coerce").to_numpy(float)
    lg = np.asarray(league)
    mo = np.asarray(month)
    out = np.full(len(v), np.nan)
    for L in pd.unique(lg):
        li = np.flatnonzero(lg == L)
        lm, lv = mo[li], v[li]
        for m in sorted(pd.unique(lm)):
            prior = lv[(lm < m) & np.isfinite(lv)]
            if len(prior) >= min_prior:
                out[li[lm == m]] = float(np.quantile(prior, q))
    return out


def poisson_total_from_over(p15: float, p25: float, p35: float) -> float:
    """The Poisson mean whose P(>1.5), P(>2.5), P(>3.5) are closest (least squares) to the inputs.
    NaN when any input is missing. Used only as a descriptive 'implied total' for Q3."""
    ps = np.array([p15, p25, p35], dtype=float)
    if not np.all(np.isfinite(ps)):
        return np.nan
    from scipy.stats import poisson
    grid = np.linspace(0.3, 6.0, 571)
    pred = np.stack([poisson.sf(k, grid) for k in (1, 2, 3)], axis=1)
    return float(grid[np.argmin(((pred - ps) ** 2).sum(axis=1))])


# ── crosswalk ────────────────────────────────────────────────────────────────────────────────────

_DROP_TOKENS = {"fc", "cf", "sc", "ac", "afc", "cd", "ud", "sd", "rc", "club", "de", "fk", "if", "bk", "sk",
                "ff", "1", "the", "calcio", "sv", "vfb", "vfl", "tsg", "ssc", "as", "us", "ss", "cfr", "sporting"}


_LIGATURES = str.maketrans({"æ": "ae", "Æ": "Ae", "ø": "o", "Ø": "O", "ß": "ss", "ð": "d", "þ": "th",
                             "ł": "l", "Ł": "L", "œ": "oe", "ı": "i"})


def norm_name(s: str) -> str:
    """Lower case, accents stripped, punctuation removed, generic club tokens dropped."""
    if not isinstance(s, str):
        return ""
    s = s.translate(_LIGATURES)
    t = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    toks = [x for x in t.split() if x not in _DROP_TOKENS]
    return " ".join(toks)


def learn_crosswalk(v9: pd.DataFrame, af: pd.DataFrame, min_votes: int = 3, min_share: float = 0.9) -> pd.DataFrame:
    """Per league, map every v9 team name to one API-Football name.

    v9: league, date, home_team, away_team, home_goals, away_goals
    af: league, date, home_team, away_team, home_goals, away_goals (goals may be NaN), fixture_id

    Votes come from (1) identical normalised names seen in the same league (an exact-name vote per
    shared date), (2) same league, same date, same score, and one side already voted -> the other
    side votes, (3) the only fixture of a league-date in both sources with the same score. A pair is
    accepted when it has >= min_votes votes and >= min_share of that v9 name's votes, and the AF name
    is not claimed by another v9 name in the league. Returns league, v9_name, af_name, votes, share."""
    rows = []
    for lg in sorted(set(v9["league"]) & set(af["league"])):
        a = v9[v9["league"] == lg]
        b = af[af["league"] == lg]
        votes: dict = {}

        def vote(x, y, n=1):
            votes.setdefault(x, {}).setdefault(y, 0)
            votes[x][y] += n

        bn = {}
        for nm in pd.unique(pd.concat([b["home_team"], b["away_team"]])):
            bn.setdefault(norm_name(nm), set()).add(nm)
        exact = {}
        for nm in pd.unique(pd.concat([a["home_team"], a["away_team"]])):
            cand = bn.get(norm_name(nm), set())
            if len(cand) == 1:
                exact[nm] = next(iter(cand))
        bd = {d: g for d, g in b.groupby("date")}
        for d, g in a.groupby("date"):
            gb = bd.get(d)
            if gb is None:
                continue
            for h, aw, hg, ag in zip(g["home_team"], g["away_team"], g["home_goals"], g["away_goals"]):
                for bh, ba, bhg, bag in zip(gb["home_team"], gb["away_team"], gb["home_goals"], gb["away_goals"]):
                    score_ok = (np.isfinite(bhg) and np.isfinite(bag) and int(bhg) == int(hg) and int(bag) == int(ag))
                    if exact.get(h) == bh and exact.get(aw) == ba:
                        vote(h, bh, 2); vote(aw, ba, 2)
                    elif score_ok and exact.get(h) == bh:
                        vote(aw, ba)
                        vote(h, bh)
                    elif score_ok and exact.get(aw) == ba:
                        vote(h, bh)
                        vote(aw, ba)
            if len(g) == 1 and len(gb) == 1:
                r, s = g.iloc[0], gb.iloc[0]
                if np.isfinite(s["home_goals"]) and np.isfinite(s["away_goals"]) \
                        and int(s["home_goals"]) == int(r["home_goals"]) and int(s["away_goals"]) == int(r["away_goals"]):
                    vote(r["home_team"], s["home_team"]); vote(r["away_team"], s["away_team"])
        # second round: names accepted in round one act as anchors for score votes
        acc = {}
        for x, d in votes.items():
            y, n = max(d.items(), key=lambda kv: kv[1])
            if n >= min_votes and n / sum(d.values()) >= min_share:
                acc[x] = y
        for d, g in a.groupby("date"):
            gb = bd.get(d)
            if gb is None:
                continue
            for h, aw, hg, ag in zip(g["home_team"], g["away_team"], g["home_goals"], g["away_goals"]):
                for bh, ba, bhg, bag in zip(gb["home_team"], gb["away_team"], gb["home_goals"], gb["away_goals"]):
                    if not (np.isfinite(bhg) and np.isfinite(bag) and int(bhg) == int(hg) and int(bag) == int(ag)):
                        continue
                    if acc.get(h) == bh and h not in exact and aw not in acc:
                        vote(aw, ba)
                    elif acc.get(aw) == ba and aw not in exact and h not in acc:
                        vote(h, bh)
        # an AF club can carry two spellings in different seasons (e.g. 'FC Nurnberg' 2023/24, then
        # '1. FC Nürnberg'): several AF names are accepted for one v9 name only when each has
        # >= min_votes, together they hold >= min_share, and their date ranges never overlap
        span = {}
        for nm, dd in pd.concat([b[["home_team", "date"]].rename(columns={"home_team": "t"}),
                                 b[["away_team", "date"]].rename(columns={"away_team": "t"})]).groupby("t")["date"]:
            span[nm] = (dd.min(), dd.max())
        claimed: dict = {}
        best = {}
        for x, d in votes.items():
            tot = sum(d.values())
            ys = sorted([(y, n) for y, n in d.items() if n >= min_votes], key=lambda kv: -kv[1])
            if not ys:
                continue
            keep = [ys[0]]
            for y, n in ys[1:]:
                if all(span[y][1] < span[k][0] or span[y][0] > span[k][1] for k, _ in keep):
                    keep.append((y, n))
            share = sum(n for _, n in keep) / tot
            if share >= min_share:
                best[x] = (keep, share)
                for y, _ in keep:
                    claimed.setdefault(y, []).append(x)
        for x, (keep, share) in best.items():
            for y, n in keep:
                if len(claimed[y]) > 1:
                    continue                               # two v9 names -> one AF name: refused
                rows.append({"league": lg, "v9_name": x, "af_name": y, "votes": int(n), "share": float(share)})
    return pd.DataFrame(rows, columns=["league", "v9_name", "af_name", "votes", "share"])


def map_fixtures(v9: pd.DataFrame, af: pd.DataFrame, xw: pd.DataFrame) -> pd.DataFrame:
    """v9 fixture rows -> AF fixture_id via the crosswalk, same league and same date (then +-1 day
    when the exact date finds nothing and the shifted match is unique). A known score that disagrees
    drops the pair (counted in `status`)."""
    m: dict = {}
    for r in xw.itertuples():
        m.setdefault((r.league, r.v9_name), []).append(r.af_name)
    v = v9.copy()
    v["af_home"] = [m.get((l, t)) for l, t in zip(v["league"], v["home_team"])]
    v["af_away"] = [m.get((l, t)) for l, t in zip(v["league"], v["away_team"])]
    look = {}
    php = af["home_player_goals"] if "home_player_goals" in af.columns else pd.Series(np.nan, index=af.index)
    pap = af["away_player_goals"] if "away_player_goals" in af.columns else pd.Series(np.nan, index=af.index)
    for r, a1, a2 in zip(af.itertuples(), php, pap):
        look.setdefault((r.league, r.home_team, r.away_team), []).append(
            (r.date, r.fixture_id, r.home_goals, r.away_goals, a1, a2))
    fid, status = [], []
    for r in v.itertuples():
        if r.af_home is None or r.af_away is None:
            fid.append(np.nan); status.append("no_alias"); continue
        cands = [c for h in r.af_home for a in r.af_away for c in look.get((r.league, h, a), [])]
        hit = [c for c in cands if c[0] == r.date]
        if not hit:
            hit = [c for c in cands if abs((c[0] - r.date).days) == 1]
        if len(hit) != 1:
            fid.append(np.nan); status.append("no_fixture" if not hit else "ambiguous"); continue
        d, f, hg, ag, phg, pag = hit[0]
        if np.isfinite(hg) and np.isfinite(ag) and (int(hg) != int(r.home_goals) or int(ag) != int(r.away_goals)):
            # the GK goals_conceded score disagrees. It is a known player_history defect (report Q3):
            # keep the pair only when the players' own goals fit inside the v9 score on both sides
            # (own goals are the only legitimate gap); otherwise the pair is dropped
            if np.isfinite(phg) and np.isfinite(pag) and phg <= r.home_goals and pag <= r.away_goals:
                fid.append(f); status.append("ok_gk_score_defect"); continue
            fid.append(np.nan); status.append("score_mismatch"); continue
        if np.isfinite(phg) and np.isfinite(pag) and (phg > r.home_goals or pag > r.away_goals):
            fid.append(np.nan); status.append("score_mismatch"); continue
        fid.append(f); status.append("ok")
    v["fixture_id"] = fid
    v["map_status"] = status
    return v


# ── props walk-forward ───────────────────────────────────────────────────────────────────────────

def props_walk_forward(work: Path, months: list[pd.Timestamp], markets=MARKETS, log=print) -> pd.DataFrame:
    """Served recipe (hgb_nan_native), refitted at the start of every month on rows dated before it."""
    from player_model import props_harness as PH
    from player_model import config as PC
    plan = PH.load_plan()
    dec = PH.owner_decisions(plan)
    raw = PH.source_frame()
    frame = PH.build_frame(raw)
    ok = PH.eligible(frame, dec["feature_rule"]).to_numpy()
    elig = frame[ok].copy()
    features = PH.harness_features(elig)
    log(f"[props-wf] eligible rows {len(elig)}, features {len(features)}, rule {dec['feature_rule']}")
    d = pd.to_datetime(elig["date"]).dt.normalize()
    parts = []
    for market in markets:
        y_col = PH.TARGETS[market]
        for m0 in months:
            m1 = m0 + pd.offsets.MonthBegin(1)
            if m1 > TEAM_HOLDOUT_START:
                raise ValueError("props-wf month reaches the team holdout")
            tr = elig[(d < m0) & elig[y_col].notna()]
            te = elig[(d >= m0) & (d < m1) & elig[y_col].notna()]
            if te.empty:
                continue
            fitted = PH.fit_window(tr, features, y_col, names=("hgb_nan_native",))
            p = fitted["hgb_nan_native"].predict(te[features])
            keep = [c for c in ("date", "fixture_id", "player_id", "player_name", "league", "home_team",
                                "away_team", "team", "is_home", "position", "started", "minutes") if c in te.columns]
            out = te[keep].copy()
            out["market"] = market
            out["y"] = te[y_col].to_numpy(float)
            out["p_props"] = p
            out["fit_through"] = str((m0 - pd.Timedelta(days=1)).date())
            out["feat_nan_share"] = te[features].isna().mean(axis=1).to_numpy()
            for c in TEAM_CONTEXT_FEATURES:
                if c in te.columns:
                    out[c] = pd.to_numeric(te[c], errors="coerce").to_numpy()
            parts.append(out)
            log(f"[props-wf] {market} {m0.date()}: train {len(tr)} test {len(te)}")
    res = pd.concat(parts, ignore_index=True)
    res.to_parquet(work / "props_wf.parquet")
    (work / "props_wf_meta.json").write_text(json.dumps({
        "recipe": "player_model.props_harness.HGBNaNNative (served candidate hgb_nan_native)",
        "feature_rule": dec["feature_rule"], "features": features,
        "features_sha256": hashlib.sha256(json.dumps(features).encode()).hexdigest(),
        "months": [str(m.date()) for m in months], "markets": list(markets)}, indent=1))
    return res


# ── team predictions ─────────────────────────────────────────────────────────────────────────────

TEAM_MARKETS = ("over15", "over25", "over35", "btts")


def load_team(paths: list[Path]) -> pd.DataFrame:
    """champion_recipe walk-forward OOS rows only (never the holdout stage), one row per match with
    p_over15/p_over25/p_over35/p_btts. A match scored by two folds keeps the EARLIER fold's
    prediction (less training data, still out of sample) - counted."""
    parts = []
    for p in paths:
        d = pd.read_parquet(p)
        d = d[(d["stage"] == "walk_forward") & (d["candidate"] == "champion_recipe")
              & d["market"].isin(TEAM_MARKETS)]
        parts.append(d)
    t = pd.concat(parts, ignore_index=True)
    t = t.sort_values(["row_id", "market", "fold"]).drop_duplicates(["row_id", "market"], keep="first")
    w = t.pivot_table(index=["row_id", "date", "league", "track"], columns="market", values="p").reset_index()
    w.columns = [c if c in ("row_id", "date", "league", "track") else f"p_{c}" for c in w.columns]
    y = t[t["market"] == "over25"][["row_id", "y"]].rename(columns={"y": "y_over25"})
    w = w.merge(y, on="row_id", how="left")
    sp = w["row_id"].str.split("|", expand=True)
    w["home_team"], w["away_team"] = sp[2], sp[3]
    w["date"] = pd.to_datetime(w["date"]).dt.normalize()
    if (w["date"] >= TEAM_HOLDOUT_START).any():
        raise ValueError("team walk-forward rows inside the holdout")
    return w


# ── join ─────────────────────────────────────────────────────────────────────────────────────────

def af_fixtures(history: pd.DataFrame) -> pd.DataFrame:
    """One row per API-Football fixture in player_history: names, date and the 90-minute score
    read from the goalkeepers' goals_conceded (home goals = what the away GKs conceded), plus each
    side's summed player goals. A side without a GK row leaves the score missing (never filled)."""
    h = history.copy()
    h["date"] = pd.to_datetime(h["date"]).dt.normalize()
    fx = h.drop_duplicates("fixture_id")[["fixture_id", "date", "league", "home_team", "away_team"]]
    gk = h[h["position"] == "G"].groupby(["fixture_id", "team"])["goals_conceded"].sum().rename("conc").reset_index()
    fx = fx.merge(gk.rename(columns={"team": "away_team", "conc": "home_goals"}), on=["fixture_id", "away_team"], how="left")
    fx = fx.merge(gk.rename(columns={"team": "home_team", "conc": "away_goals"}), on=["fixture_id", "home_team"], how="left")
    pg = h.groupby(["fixture_id", "team"])["goals"].sum().rename("pg").reset_index()
    fx = fx.merge(pg.rename(columns={"team": "home_team", "pg": "home_player_goals"}), on=["fixture_id", "home_team"], how="left")
    fx = fx.merge(pg.rename(columns={"team": "away_team", "pg": "away_player_goals"}), on=["fixture_id", "away_team"], how="left")
    return fx


def with_prior_terciles(df: pd.DataFrame, col: str, min_prior: int = 200) -> pd.DataFrame:
    d = df.copy()
    d[f"{col}_lo"] = prior_cutoff(d[col], d["league"], d["month"], 1 / 3, min_prior)
    d[f"{col}_hi"] = prior_cutoff(d[col], d["league"], d["month"], 2 / 3, min_prior)
    v = d[col].to_numpy(float)
    grp = np.where(v >= d[f"{col}_hi"].to_numpy(), 1, np.where(v <= d[f"{col}_lo"].to_numpy(), 0, -1)).astype(float)
    grp[~np.isfinite(d[f"{col}_lo"].to_numpy()) | ~np.isfinite(v)] = np.nan
    d[f"{col}_grp"] = grp
    return d


def side_lambda_terciles(dc: pd.DataFrame, min_prior: int = 200) -> pd.DataFrame:
    """Earlier-month cutoffs of SIDE lambdas (home and away sides pooled within the league)."""
    s = pd.concat([dc[["league", "month"]].assign(lam=dc["lam_home"].to_numpy()),
                   dc[["league", "month"]].assign(lam=dc["lam_away"].to_numpy())], ignore_index=True)
    out = []
    for lg, g in s.groupby("league"):
        for m in sorted(g["month"].unique()):
            prior = g.loc[g["month"] < m, "lam"].dropna()
            if len(prior) >= min_prior:
                out.append({"league": lg, "month": m, "lam_side_lo": float(np.quantile(prior, 1 / 3)),
                            "lam_side_hi": float(np.quantile(prior, 2 / 3))})
    return pd.DataFrame(out)


def build_join(work: Path, team_paths: list[Path], log=print) -> dict:
    v9 = pd.read_parquet(work / "v9_history.parquet")
    v9["date"] = pd.to_datetime(v9["date"]).dt.normalize()
    dc = pd.read_parquet(work / "dc_wf.parquet")
    dc["date"] = pd.to_datetime(dc["date"]).dt.normalize()
    hist = pd.read_parquet(ROOT / "player_history.parquet",
                           columns=["fixture_id", "date", "league", "home_team", "away_team", "team", "position",
                                    "goals_conceded", "goals"])
    af = af_fixtures(hist)
    xw = learn_crosswalk(v9, af)
    xw.to_csv(work / "crosswalk.csv", index=False)
    mp = map_fixtures(v9, af, xw)
    mp.to_parquet(work / "fixture_map.parquet")
    team = load_team(team_paths)
    team["month"] = team["date"].dt.to_period("M").astype(str)
    team = with_prior_terciles(team, "p_over25")
    dc["month"] = dc["month"].astype(str)
    dc["dc_lam_total"] = dc["lam_home"] + dc["lam_away"]
    dc = with_prior_terciles(dc, "dc_lam_total")
    st = side_lambda_terciles(dc)
    keys = ["league", "date", "home_team", "away_team"]
    fx = mp[keys + ["home_goals", "away_goals", "fixture_id", "map_status"]].merge(
        team[keys + ["track", "p_over15", "p_over25", "p_over35", "p_btts", "y_over25", "p_over25_grp"]],
        on=keys, how="left").merge(
        dc[keys + ["lam_home", "lam_away", "dc_lam_total", "dc_lam_total_grp", "p_ge6", "cand_ge6", "n_train"]],
        on=keys, how="left")
    fx["month"] = fx["date"].dt.to_period("M").astype(str)
    fx = fx.merge(st, on=["league", "month"], how="left")
    fx.to_parquet(work / "fixtures_joined.parquet")
    props = pd.read_parquet(work / "props_wf.parquet")
    props["date"] = pd.to_datetime(props["date"]).dt.normalize()
    props["fixture_id"] = pd.to_numeric(props["fixture_id"], errors="coerce")
    f2 = fx[fx["fixture_id"].notna()].drop_duplicates("fixture_id").drop(columns=["date", "league", "home_team", "away_team"])
    f2 = f2.rename(columns={"month": "fx_month"})
    j = props.merge(f2, on="fixture_id", how="left")
    j["month"] = j["date"].dt.to_period("M").astype(str)
    home = pd.to_numeric(j["is_home"], errors="coerce") == 1
    j["dc_lam_side"] = np.where(home, j["lam_home"], j["lam_away"])
    j["dc_lam_opp"] = np.where(home, j["lam_away"], j["lam_home"])
    sg = np.where(j["dc_lam_side"] >= j["lam_side_hi"], 1.0, np.where(j["dc_lam_side"] <= j["lam_side_lo"], 0.0, -1.0))
    sg[~np.isfinite(j["lam_side_lo"].to_numpy(float)) | ~np.isfinite(j["dc_lam_side"].to_numpy(float))] = np.nan
    j["dc_lam_side_grp"] = sg
    # props volume V per fixture from sot starters, earlier-month cutoffs
    s = j[(j["market"] == "sot") & (pd.to_numeric(j["started"], errors="coerce") == 1)]
    V = s.groupby("fixture_id").agg(V=("p_props", "mean"), n_starters=("p_props", "size"),
                                    league=("league", "first"), month=("month", "first")).reset_index()
    V.loc[V["n_starters"] < 18, "V"] = np.nan
    V = V.sort_values("month")
    V["V_lo"] = prior_cutoff(V["V"], V["league"], V["month"], 1 / 3, 100)
    V["V_hi"] = prior_cutoff(V["V"], V["league"], V["month"], 2 / 3, 100)
    vg = np.where(V["V"] >= V["V_hi"], 1.0, np.where(V["V"] <= V["V_lo"], 0.0, -1.0))
    vg[~np.isfinite(V["V_lo"].to_numpy(float)) | ~np.isfinite(V["V"].to_numpy(float))] = np.nan
    V["V_grp"] = vg
    j = j.merge(V[["fixture_id", "V", "n_starters", "V_grp"]], on="fixture_id", how="left")
    # props tips: starters above the league's earlier-month 0.8 quantile of starters' p_props
    j["tip"] = np.nan
    for market in MARKETS:
        m = (j["market"] == market) & (pd.to_numeric(j["started"], errors="coerce") == 1)
        cut = prior_cutoff(j.loc[m, "p_props"], j.loc[m, "league"], j.loc[m, "month"], 0.8, 200)
        j.loc[m, "tip"] = np.where(np.isfinite(cut), (j.loc[m, "p_props"].to_numpy() >= cut).astype(float), np.nan)
    j.to_parquet(work / "joined.parquet")
    V.to_parquet(work / "props_volume.parquet")
    info = {"crosswalk_pairs": int(len(xw)), "v9_fixtures": int(len(mp)),
            "map_status": mp["map_status"].value_counts().to_dict(),
            "props_rows": int(len(props)),
            "props_rows_with_fixture_join": int(j["map_status"].isin(["ok", "ok_gk_score_defect"]).sum()),
            "props_rows_with_team": int(j["p_over25"].notna().sum()), "props_rows_with_dc": int(j["lam_home"].notna().sum())}
    log(json.dumps(info))
    return info


# ── analysis ─────────────────────────────────────────────────────────────────────────────────────

MIN_FIX_READ = 30            # a cell is read only with >= this many fixtures in EACH group


def _cell(h: pd.DataFrame, value: str, grp: str, reps: int, seed: int, one_sided: bool) -> dict:
    n1 = h.loc[h[grp] == 1, "fixture_id"].nunique()
    n0 = h.loc[h[grp] == 0, "fixture_id"].nunique()
    rec = {"fix_1": int(n1), "fix_0": int(n0), "rows_1": int((h[grp] == 1).sum()), "rows_0": int((h[grp] == 0).sum()),
           "mean_1": float(h.loc[h[grp] == 1, value].mean()) if n1 else np.nan,
           "mean_0": float(h.loc[h[grp] == 0, value].mean()) if n0 else np.nan,
           "readable": bool(n1 >= MIN_FIX_READ and n0 >= MIN_FIX_READ)}
    if rec["readable"]:
        b = cluster_bootstrap_diff(h[value], h[grp], h["fixture_id"], reps, seed)
        rec.update(effect=b["est"], ci_lo=b["lo"], ci_hi=b["hi"], se=b["se"],
                   p=b["p_one"] if one_sided else b["p_two"], p_emp_one=b["p_emp_one"])
    else:
        rec.update(effect=np.nan, ci_lo=np.nan, ci_hi=np.nan, se=np.nan, p=np.nan, p_emp_one=np.nan)
    return rec


def run_tests(j: pd.DataFrame, fx: pd.DataFrame, reps: int = 2000, seed: int = 7) -> pd.DataFrame:
    """Every Q1 and Q2 cell of the protocol, per league plus a pooled context row."""
    out = []
    j = j[j["date"] < TEAM_HOLDOUT_START].copy()
    j["res"] = j["y"] - j["p_props"]
    leagues = sorted(j.loc[j["p_over25"].notna() | j["lam_home"].notna(), "league"].unique())

    def add(test, market, cond, data, value, grp, one_sided):
        d = data[data[grp].isin([0, 1])]
        for lg in leagues + ["ALL (context)"]:
            h = d if lg == "ALL (context)" else d[d["league"] == lg]
            rec = {"question": test.split("_")[0], "test": test, "market": market, "conditioner": cond, "league": lg,
                   "in_family": lg != "ALL (context)", "one_sided": one_sided}
            rec.update(_cell(h, value, grp, reps, seed, one_sided))
            out.append(rec)

    for market in MARKETS:
        m = j[j["market"] == market]
        add("Q1_team_p_over25", market, "team_p_over25", m, "res", "p_over25_grp", False)
        add("Q1_dc_lam_total", market, "dc_lam_total", m, "res", "dc_lam_total_grp", False)
        add("Q1_dc_lam_side", market, "dc_lam_side", m, "res", "dc_lam_side_grp", False)
    # Q2 team side: fixtures, one row each
    f = fx[(fx["date"] < TEAM_HOLDOUT_START) & fx["fixture_id"].notna()].drop_duplicates("fixture_id").copy()
    V = j.drop_duplicates("fixture_id")[["fixture_id", "V_grp"]]
    f = f.merge(V, on="fixture_id", how="inner")
    f = f[(f["p_over25_grp"] == 1) & f["y_over25"].notna()].copy()
    f["res_team"] = f["y_over25"] - f["p_over25"]
    f["dc_grp"] = f["cand_ge6"]
    add("Q2_T2a_team_tips_props_agree", "over25", "props_high_vs_low", f, "res_team", "V_grp", True)
    add("Q2_T2c_team_tips_dc_flag", "over25", "dc_ge6_flag", f, "res_team", "dc_grp", True)
    for market in MARKETS:
        t = j[(j["market"] == market) & (j["tip"] == 1)].copy()
        t["team_grp"] = t["p_over25_grp"]
        t["dc_grp"] = t["cand_ge6"]
        add("Q2_T2b_props_tips_team_agree", market, "team_high_vs_low", t, "res", "team_grp", True)
        add("Q2_T2d_props_tips_dc_flag", market, "dc_ge6_flag", t, "res", "dc_grp", True)
    res = pd.DataFrame(out)
    fam = res["in_family"].to_numpy()
    p = res.loc[fam, "p"].to_numpy(float)
    p = np.where(res.loc[fam, "readable"].to_numpy() & np.isfinite(p), p, 1.0)
    res["p_holm"] = np.nan
    res.loc[fam, "p_holm"] = holm(p)
    res["signal"] = res["in_family"] & res["readable"] & (res["p_holm"] < 0.05)
    return res


def logistic_slope(df: pd.DataFrame, cond: str) -> dict:
    """Context only: logit P(y) = logit(p_props) + a + b * z(cond), z standardised within league,
    fitted by Newton-Raphson; fixture-clustered sandwich SE. b > 0: the conditioner carries signal
    the props probability misses; a: overall calibration shift."""
    from scipy.stats import norm
    g = df[np.isfinite(df[cond]) & np.isfinite(df["p_props"])].copy()
    p = g["p_props"].clip(1e-6, 1 - 1e-6).to_numpy()
    off = np.log(p / (1 - p))
    z = g.groupby("league")[cond].transform(lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1.0)).to_numpy()
    X = np.column_stack([np.ones(len(z)), z])
    y = g["y"].to_numpy(float)
    beta = np.zeros(2)
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(off + X @ beta)))
        W = mu * (1 - mu)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    mu = 1 / (1 + np.exp(-(off + X @ beta)))
    bread = np.linalg.inv(X.T @ (X * (mu * (1 - mu))[:, None]))
    sc = X * (y - mu)[:, None]
    cl = pd.factorize(g["fixture_id"])[0]
    S = np.zeros((cl.max() + 1, 2))
    np.add.at(S, cl, sc)
    V = bread @ (S.T @ S) @ bread
    se = float(np.sqrt(V[1, 1]))
    return {"a": float(beta[0]), "b": float(beta[1]), "se": se, "p_two": float(2 * norm.sf(abs(beta[1]) / se)),
            "n": int(len(g))}


def context_checks(j: pd.DataFrame, reps: int = 1000, seed: int = 7) -> dict:
    """Context only (never a decision, not in the Holm family): logistic slopes, the home/away
    confound of the side-lambda effect, position splits, and the double-counting check (how much of
    the Dixon-Coles side expectation the props team features already carry)."""
    from scipy.stats import spearmanr
    j = j[j["date"] < TEAM_HOLDOUT_START].copy()
    j["res"] = j["y"] - j["p_props"]
    out = {"logistic_slope": {}, "side_effect_by_venue": {}, "side_effect_by_position": {},
           "residual_by_venue": {}, "spearman_dc_side_vs_props_features": {}, "calibration_by_league": {}}
    for market in MARKETS:
        m = j[j["market"] == market]
        out["logistic_slope"][market] = {c: logistic_slope(m, c) for c in ("p_over25", "dc_lam_total", "dc_lam_side")}
        for venue, v in ((1, "home"), (0, "away")):
            h = m[(pd.to_numeric(m["is_home"], errors="coerce") == venue) & m["dc_lam_side_grp"].isin([0, 1])]
            b = cluster_bootstrap_diff(h["res"], h["dc_lam_side_grp"], h["fixture_id"], reps, seed)
            out["side_effect_by_venue"].setdefault(market, {})[v] = {k: b[k] for k in ("est", "lo", "hi")} | {
                "rows_top": int((h["dc_lam_side_grp"] == 1).sum()), "rows_bottom": int((h["dc_lam_side_grp"] == 0).sum())}
            r = m[pd.to_numeric(m["is_home"], errors="coerce") == venue]
            out["residual_by_venue"].setdefault(market, {})[v] = float(r["res"].mean())
        for pos in ("F", "M", "D", "G"):
            h = m[(m["position"].astype(str).str[:1] == pos) & m["dc_lam_side_grp"].isin([0, 1])]
            if h["fixture_id"].nunique() < 60:
                continue
            b = cluster_bootstrap_diff(h["res"], h["dc_lam_side_grp"], h["fixture_id"], reps, seed)
            out["side_effect_by_position"].setdefault(market, {})[pos] = {k: b[k] for k in ("est", "lo", "hi")}
        cal = m.groupby("league").agg(mean_p=("p_props", "mean"), mean_y=("y", "mean"), rows=("y", "size"))
        out["calibration_by_league"][market] = cal.round(4).to_dict(orient="index")
    s = j[(j["market"] == "sot")].drop_duplicates(["fixture_id", "is_home"])
    for c in TEAM_CONTEXT_FEATURES:
        if c in s.columns and c != "is_home":
            ok = s[["dc_lam_side", c]].dropna()
            if len(ok) > 100:
                out["spearman_dc_side_vs_props_features"][c] = float(spearmanr(ok["dc_lam_side"], ok[c]).statistic)
    return out


def contradictions(j: pd.DataFrame, fx: pd.DataFrame, z_strong: float = 2.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Q3: fixture-level z-scores (within league), strong disagreements and their diagnostics."""
    f = fx[(fx["date"] < TEAM_HOLDOUT_START) & fx["fixture_id"].notna()].drop_duplicates("fixture_id").copy()
    agg = j[j["market"] == "sot"].groupby("fixture_id").agg(
        V=("V", "first"), n_starters=("n_starters", "first"),
        feat_nan_share=("feat_nan_share", "mean") if "feat_nan_share" in j.columns else ("p_props", "size"))
    f = f.merge(agg.reset_index(), on="fixture_id", how="inner")
    f = f[f["p_over25"].notna() & f["V"].notna() & f["dc_lam_total"].notna()].copy()
    f["team_total"] = [poisson_total_from_over(a, b, c) for a, b, c in zip(f["p_over15"], f["p_over25"], f["p_over35"])]
    for c in ("team_total", "dc_lam_total", "V"):
        f[f"z_{c}"] = f.groupby("league")[c].transform(lambda s: (s - s.mean()) / s.std())
    f["d_team_props"] = f["z_team_total"] - f["z_V"]
    f["d_team_dc"] = f["z_team_total"] - f["z_dc_lam_total"]
    f["goals"] = f["home_goals"] + f["away_goals"]
    f["strong_team_props"] = f["d_team_props"].abs() >= z_strong
    f["strong_team_dc"] = f["d_team_dc"].abs() >= z_strong
    return f, f[f["strong_team_props"] | f["strong_team_dc"]]


def price_inventory(today: str) -> pd.DataFrame:
    """COUNTS ONLY of distinct (match date, match, player, market) props price rows for sot / sot2 /
    goals per league: spent window (from the first price, 2026-10-14, to 2026-11-08) and forward
    (2026-11-09 on), plus the spent window's rate per week. No price value is read into the output."""
    import glob
    fs = [ROOT / "output" / "player_prop_odds_history.csv"] +         [Path(f) for f in glob.glob(str(ROOT / "output" / "player_prop_odds_history_archive" / "*.csv*"))]
    cols = ["match_date", "league", "match", "player", "market"]
    d = pd.concat([pd.read_csv(f, usecols=cols) for f in fs if Path(f).exists()], ignore_index=True)
    d = d[d["market"].isin(MARKETS)].drop_duplicates(cols)
    d["match_date"] = pd.to_datetime(d["match_date"], errors="coerce")
    first = pd.Timestamp("2026-10-14")
    spent_end = pd.Timestamp("2026-11-08")
    sp = d[(d["match_date"] >= first) & (d["match_date"] <= spent_end)]
    fw = d[(d["match_date"] > spent_end) & (d["match_date"] <= pd.Timestamp(today))]
    weeks = ((spent_end - first).days + 1) / 7
    out = sp.groupby(["league", "market"]).size().rename("priced_spent").to_frame()         .join(fw.groupby(["league", "market"]).size().rename("priced_forward_to_date"), how="outer").fillna(0)
    out["priced_per_week"] = out["priced_spent"] / weeks
    return out.reset_index()


def gk_defect_after_refetch(work: Path) -> pd.DataFrame:
    """The goalkeeper goals_conceded check on the counts the props features actually read: raw
    player_history with the re-fetched counts applied (history_refetch.with_refetched). Per month:
    share of mapped fixtures (both GK sides present) whose GK-derived score differs from the real
    v9 score."""
    from player_model.history_refetch import with_refetched
    mp = pd.read_parquet(work / "fixture_map.parquet")
    mp = mp[mp["map_status"].isin(["ok", "ok_gk_score_defect"])]
    h = pd.read_parquet(ROOT / "player_history.parquet")
    h = h[h["fixture_id"].isin(mp["fixture_id"])].copy()
    for c in h.columns:
        if str(h[c].dtype).startswith("int"):
            h[c] = h[c].astype("float64")
    h = with_refetched(h)
    g = h[h["position"] == "G"].groupby(["fixture_id", "team"])["goals_conceded"].sum(min_count=1).reset_index()
    fx = h.drop_duplicates("fixture_id")[["fixture_id", "home_team", "away_team", "date"]]
    fx = fx.merge(g.rename(columns={"team": "away_team", "goals_conceded": "gk_home"}), on=["fixture_id", "away_team"], how="left")
    fx = fx.merge(g.rename(columns={"team": "home_team", "goals_conceded": "gk_away"}), on=["fixture_id", "home_team"], how="left")
    fx = fx.merge(mp[["fixture_id", "home_goals", "away_goals"]], on="fixture_id")
    known = fx["gk_home"].notna() & fx["gk_away"].notna()
    fx = fx[known]
    fx["defect"] = (fx["gk_home"] != fx["home_goals"]) | (fx["gk_away"] != fx["away_goals"])
    fx["gk_zero_vs_real_goals"] = ((fx["gk_home"] == 0) & (fx["home_goals"] > 0)) | ((fx["gk_away"] == 0) & (fx["away_goals"] > 0))
    out = fx.groupby(pd.to_datetime(fx["date"]).dt.to_period("M").astype(str)).agg(
        fixtures=("defect", "size"), defect_share=("defect", "mean"), gk_zero_share=("gk_zero_vs_real_goals", "mean"))
    return out.round(4)


def write_report_data(work: Path, out: Path, today: str) -> dict:
    """Counts and estimates only (no prices, no player-level rows) into reports/combined_models_data."""
    from scipy.stats import spearmanr
    out.mkdir(parents=True, exist_ok=True)
    j = pd.read_parquet(work / "joined.parquet")
    j = j[j["date"] < TEAM_HOLDOUT_START]
    fx = pd.read_parquet(work / "fixtures_joined.parquet")
    tests = pd.read_csv(work / "tests.csv")
    tests.round(6).to_csv(out / "tests.csv", index=False)
    (out / "context.json").write_text((work / "context.json").read_text())
    # inventory per league
    inv = []
    for lg, g in j.groupby("league"):
        s = g[g["market"] == "sot"]
        rec = {"league": lg, "props_rows_sot": len(s), "fixtures": s["fixture_id"].nunique(),
               "q1_rows_team": int(s["p_over25_grp"].isin([0, 1]).sum()),
               "q1_rows_dc_side": int(s["dc_lam_side_grp"].isin([0, 1]).sum()),
               "q2_fixtures_team_high_with_V": int(s.drop_duplicates("fixture_id").pipe(
                   lambda d: ((d["p_over25_grp"] == 1) & d["V_grp"].isin([0, 1])).sum())),
               "q2_props_tips_sot": int(((s["tip"] == 1) & s["p_over25_grp"].isin([0, 1])).sum()),
               "q2_fixtures_dc_ge6_flag": int(s.drop_duplicates("fixture_id")["cand_ge6"].eq(1).sum())}
        inv.append(rec)
    inv = pd.DataFrame(inv)
    inv["has_team_and_dc"] = inv["q1_rows_team"] > 0
    inv.to_csv(out / "inventory_by_league.csv", index=False)
    # fixture mapping
    mp = pd.read_parquet(work / "fixture_map.parquet")
    mp = mp[mp["date"] >= pd.Timestamp("2023-01-01")]
    mp.groupby("league")["map_status"].value_counts().unstack(fill_value=0).to_csv(out / "fixture_mapping_by_league.csv")
    # GK goals_conceded defect: share of mapped fixtures whose GK-derived score disagrees with the real score
    m2 = mp[mp["map_status"].isin(["ok", "ok_gk_score_defect"])].copy()
    m2["defect"] = m2["map_status"] == "ok_gk_score_defect"
    m2.groupby(m2["date"].dt.to_period("M").astype(str))["defect"].agg(["mean", "sum", "size"])         .rename(columns={"mean": "share", "sum": "fixtures_defect", "size": "fixtures"}).round(4)         .to_csv(out / "gk_goals_conceded_defect_by_month.csv")
    gk_defect_after_refetch(work).to_csv(out / "gk_goals_conceded_defect_by_month_after_refetch.csv")
    # what team_goals_pg_roll measures
    s = j[(j["market"] == "sot") & j["dc_lam_side"].notna()].drop_duplicates(["fixture_id", "is_home"]).copy()
    home = pd.to_numeric(s["is_home"], errors="coerce") == 1
    s["own_goals"] = np.where(home, s["home_goals"], s["away_goals"])
    s["opp_goals"] = np.where(home, s["away_goals"], s["home_goals"])
    rows = []
    for c in TEAM_CONTEXT_FEATURES:
        if c == "is_home" or c not in s.columns:
            continue
        ok = s[[c, "dc_lam_side", "dc_lam_opp", "own_goals", "opp_goals"]].dropna()
        if len(ok) < 100:
            continue
        rows.append({"feature": c, "n": len(ok),
                     **{f"spearman_vs_{k}": float(spearmanr(ok[c], ok[k]).statistic)
                        for k in ("dc_lam_side", "dc_lam_opp", "own_goals", "opp_goals")}})
    pd.DataFrame(rows).round(4).to_csv(out / "props_team_features_vs_dc.csv", index=False)
    # Q3 patterns (aggregates)
    q = pd.read_parquet(work / "q3_fixtures.parquet")
    q["pattern"] = np.select(
        [q["strong_team_props"] & (q["d_team_props"] > 0), q["strong_team_props"] & (q["d_team_props"] < 0),
         q["strong_team_dc"] & (q["d_team_dc"] > 0), q["strong_team_dc"] & (q["d_team_dc"] < 0)],
        ["team HIGH / props LOW", "team LOW / props HIGH", "team HIGH / DC LOW", "team LOW / DC HIGH"], "none")
    q["over25"] = (q["goals"] > 2.5).astype(float)
    q["calendar_month"] = pd.to_datetime(q["month"]).dt.month
    q.groupby("pattern").agg(fixtures=("fixture_id", "size"), goals=("goals", "mean"), over25=("over25", "mean"),
                             p_over25=("p_over25", "mean"), dc_lam_total=("dc_lam_total", "mean"), V=("V", "mean"),
                             starters=("n_starters", "mean"), feat_nan_share=("feat_nan_share", "mean"))         .round(4).to_csv(out / "q3_patterns.csv")
    pd.crosstab(q["league"], q["pattern"]).to_csv(out / "q3_patterns_by_league.csv")
    pd.crosstab(q["calendar_month"], q["pattern"]).to_csv(out / "q3_patterns_by_calendar_month.csv")
    corr = q.groupby("league").apply(lambda g: pd.Series({
        "fixtures": len(g), "corr_team_total_vs_props_V": g["team_total"].corr(g["V"]),
        "corr_team_total_vs_dc_total": g["team_total"].corr(g["dc_lam_total"]),
        "corr_dc_total_vs_props_V": g["dc_lam_total"].corr(g["V"])}), include_groups=False).round(3)
    corr.to_csv(out / "q3_model_agreement_by_league.csv")
    # forward volume: counts only
    price_inventory(today).round(1).to_csv(out / "forward_price_volume.csv", index=False)
    summary = {"family_size": int(tests["in_family"].sum()),
               "readable": int((tests["in_family"] & tests["readable"]).sum()),
               "signals": tests.loc[tests["signal"], ["test", "market", "league"]].to_dict(orient="records"),
               "dev_props_rows_joined": int(j["map_status"].isin(["ok", "ok_gk_score_defect"]).sum()),
               "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["props-wf", "join", "analyse", "report-data"])
    ap.add_argument("--today", default=None)
    ap.add_argument("--work", required=True)
    ap.add_argument("--team", nargs="*", default=[])
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args(argv)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    if a.stage == "props-wf":
        months = list(pd.date_range(DEV_START, TEAM_HOLDOUT_START - pd.Timedelta(days=1), freq="MS"))
        props_walk_forward(work, months)
    elif a.stage == "join":
        build_join(work, [Path(p) for p in a.team])
    elif a.stage == "report-data":
        print(json.dumps(write_report_data(work, REPORT_DATA, a.today or str(pd.Timestamp.today().date())), indent=1))
    else:
        j = pd.read_parquet(work / "joined.parquet")
        fx = pd.read_parquet(work / "fixtures_joined.parquet")
        res = run_tests(j, fx, reps=a.reps)
        res.to_csv(work / "tests.csv", index=False)
        (work / "context.json").write_text(json.dumps(context_checks(j), indent=1, default=float))
        allf, strong = contradictions(j, fx)
        allf.to_parquet(work / "q3_fixtures.parquet")
        strong.to_csv(work / "q3_strong.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
