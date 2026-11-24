"""Compact index of every model prediction for the dashboard Predictions page (owner 2026-11-22).

The prediction ledgers are hundreds of small CSVs (output/prediction_ledger/<repo>/<model>/<market>/<league>/
<season>.csv, schema pl/1.0, src/prediction_ledger.py) plus their settled views
(output/prediction_ledger_settled/...). A page that opened them all would be slow on every render, so this
script folds them into TWO committed files the page reads instead:

    reports/dashboard/predictions_index.parquet   one row per ledger row (both snapshots), no price:
                                                   who predicted what for which match, the probability, the
                                                   outcome once settled and whether the model's pick hit
    reports/dashboard/predictions_summary.csv     hit rate / log loss / Brier per repo x model x market x
                                                   league x snapshot, counted from the Clean Record start
                                                   (config/trustworthy_performance.json clean_record_start_utc;
                                                   predicted_at on or after it)

    python scripts/build_predictions_index.py [--pro-root DIR] [--out-dir DIR]

Sources: v9 from this checkout; Pro (pro_bb_dc score grid + pro_bb_combo) from --pro-root, a checkout of
NevixAA/wowzaV9-Pro holding output/prediction_ledger/pro and output/prediction_ledger_settled/pro
(update_results.yml checks it out sparse). Only the repos in PUBLIC_REPOS are read: v11 rows are private
(wowza-exec) and never enter the public dashboard. LEARNING RECORD, NOT BETS: no price, stake or P/L column
is read or written (any column matching src.prediction_ledger.PRICE_LIKE is refused).

Scoring, per QUESTION (one model version x market x line x fixture x player x snapshot x side):
  * complementary selections (OVER/UNDER, YES/NO, H/D/A, the correct-score grid, HOME_/AWAY_ OVER/UNDER) form
    one question: pick = the most likely selection; log loss = -ln p(outcome that happened); Brier = binary
    (p-y)^2 of one selection for a two-way question, the multi-class sum for 1X2 and correct score. Scored
    only when every row settled and exactly one selection happened.
  * a single-selection question (player props YES, Pro goal bands, team 4+, combos) is binary on its row:
    pick = YES when p >= 0.5 else NO; log loss and Brier binary.
Single-selection questions also carry calibration sums (single_p_sum / single_y_sum over single_settled): the
mean probability against how often the selection happened - for rare events the hit rate alone says little.
VOID_DID_NOT_PLAY / NO_RESULT rows are counted as void, never scored. Deterministic: no wall-clock value is
written, so an unchanged ledger gives byte-identical files and no commit.

ONE FIXTURE, ONE IDENTITY (owner bug 2026-11-23: Puebla v Atletico San Luis showed as two matches). Every row also
carries its CANONICAL fixture identity from src.fixture_identity (API-Football ref when any model carries one, the
API-Football kickoff as the authoritative one, club spellings unified by src.team_names; the original fixture_ref,
kickoff_utc and team names stay as they are). The counts of matches the old page key split - per kind (accent,
name variant, kickoff drift) and per league - and every kickoff disagreement go to
reports/dashboard/fixture_identity.json (a data-truth finding, never silently picked).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import fixture_identity as FI  # noqa: E402
from src import prediction_ledger as PL  # noqa: E402

OUT_DIR = ROOT / "reports" / "dashboard"
INDEX_NAME = "predictions_index.parquet"
SUMMARY_NAME = "predictions_summary.csv"
IDENTITY_NAME = "fixture_identity.json"
CLEAN_CFG = ROOT / "config" / "trustworthy_performance.json"

#: Repos the public page may show. v11 is private (wowza-exec) and is never read here.
PUBLIC_REPOS = ("v9", "pro")

INDEX_COLS = ["repo", "model_id", "model_version", "market", "line", "selection", "fixture_ref", "league",
              "league_slug", "kickoff_utc", "home_team", "away_team", "player_id", "player_name", "team",
              "probability", "snapshot", "predicted_at", "ledger_only", "outcome_status", "y", "pick", "hit",
              "q_size", "canonical_fixture_ref", "canonical_kickoff_utc", "kickoff_source", "canonical_home",
              "canonical_away"]
SUMMARY_COLS = ["repo", "model_id", "market", "league_slug", "league", "ledger_only", "snapshot", "predictions",
                "settled", "void", "pending", "hits", "hit_rate", "log_loss", "brier", "mean_pick_prob",
                "log_loss_sum", "brier_sum", "pick_prob_sum", "single_settled", "single_p_sum", "single_y_sum",
                "first_predicted", "data_through", "clean_record_start"]

EPS = PL.EPS
_PAIR = {"OVER", "UNDER", "YES", "NO"}
_SIDE_OU = re.compile(r"^(HOME|AWAY)_(OVER|UNDER)$")
_1X2 = {"H", "D", "A"}


def clean_record_start(path: Path = CLEAN_CFG) -> pd.Timestamp:
    """THE one start (owner 2026-11-18): the go-live merge instant. Never inferred: missing = error."""
    v = json.loads(Path(path).read_text(encoding="utf-8"))["clean_record_start_utc"]
    return pd.Timestamp(v).tz_convert("UTC") if pd.Timestamp(v).tzinfo else pd.Timestamp(v, tz="UTC")


def question_side(market: str, selection: str) -> str:
    """The part of a selection that names a SEPARATE question ('' = all selections of the question compete)."""
    s = str(selection)
    if market == "correct_score" or s in _1X2 or s in _PAIR:
        return ""
    m = _SIDE_OU.match(s)
    if m:
        return m.group(1)
    return s                     # HOME / AWAY team 4+, a combo, anything else: its own binary question


# ── read ───────────────────────────────────────────────────────────────────────────────────────────
def _read_tree(root: Path) -> pd.DataFrame:
    """Every partition CSV under <root>/<repo>/<model>/<market>/<league>/<season>.csv of a public repo."""
    frames = []
    for p in PL.partitions(root):
        part = PL.parse_partition(root, p)
        if part["repo"] not in PUBLIC_REPOS:
            continue
        d = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8")
        if d.empty:
            continue
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _no_price(df: pd.DataFrame) -> None:
    bad = [c for c in df.columns if PL.PRICE_LIKE.search(str(c))]
    if bad:
        raise PL.LedgerError(f"price-like columns refused: {bad}")


def load(ledger_root: Path, settled_root: Path) -> pd.DataFrame:
    """Raw ledger rows + the outcome of each from the settled view (joined 1:1 on repo + the ledger key)."""
    raw = _read_tree(Path(ledger_root))
    if raw.empty:
        return raw
    _no_price(raw)
    raw = raw[raw["repo"].isin(PUBLIC_REPOS)].copy()
    key = ["repo"] + PL.KEY_COLS
    st = _read_tree(Path(settled_root))
    if not st.empty and {"outcome_status", "y"} <= set(st.columns):
        st = st[key + ["outcome_status", "y"]].drop_duplicates(key, keep="last")
        raw = raw.merge(st, on=key, how="left")
    else:
        raw["outcome_status"], raw["y"] = "", ""
    raw["outcome_status"] = raw["outcome_status"].fillna("").replace("", "PENDING")
    return raw


# ── score ──────────────────────────────────────────────────────────────────────────────────────────
def score(rows: pd.DataFrame) -> pd.DataFrame:
    """Typed index rows with pick / hit / q_size and per-question loss columns (_qid, _ll, _br, _pp)."""
    d = rows.copy()
    d["probability"] = pd.to_numeric(d["probability"], errors="coerce")
    d = d[d["probability"].between(0, 1)].copy()
    d["y"] = pd.to_numeric(d["y"].replace("", np.nan), errors="coerce")
    d.loc[d["outcome_status"] != "SETTLED", "y"] = np.nan
    d["ledger_only"] = d["payload_version"].str.contains("ledger_only:", regex=False).fillna(False)
    d["_side"] = [question_side(m, s) for m, s in zip(d["market"], d["selection"])]
    qkey = ["repo", "model_id", "model_version", "market", "line", "fixture_ref", "player_id", "snapshot", "_side"]
    d["_qid"] = d.groupby(qkey, sort=False, dropna=False).ngroup()
    g = d.groupby("_qid", sort=False)
    d["q_size"] = g["probability"].transform("size").astype(int)

    multi = d["q_size"] > 1
    # pick: the most likely selection of a multi-row question (first on a tie), every single row
    top = d[multi].sort_values(["_qid", "probability"], ascending=[True, False], kind="mergesort")
    pick_idx = top.drop_duplicates("_qid").index
    d["pick"] = ~multi
    d.loc[pick_idx, "pick"] = True

    # a multi-row question is scored only when every row settled and exactly one selection happened
    n_set = g["y"].transform("count")
    y_sum = g["y"].transform("sum")
    multi_ok = multi & (n_set == d["q_size"]) & (y_sum == 1)
    single_ok = ~multi & d["y"].notna()

    p, y = d["probability"].astype(float), d["y"]
    d["hit"] = np.nan
    d.loc[multi_ok & d["pick"], "hit"] = y[multi_ok & d["pick"]]
    d.loc[single_ok, "hit"] = ((p[single_ok] >= 0.5).astype(int) == y[single_ok]).astype(int)

    pc = p.clip(EPS, 1 - EPS)
    d["_ll"], d["_br"], d["_pp"] = np.nan, np.nan, np.nan
    # single: binary
    d.loc[single_ok, "_ll"] = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))[single_ok]
    d.loc[single_ok, "_br"] = ((p - y) ** 2)[single_ok]
    d.loc[single_ok, "_pp"] = np.maximum(p, 1 - p)[single_ok]
    # multi: carried on the pick row only (one value per question)
    if multi_ok.any():
        m = d[multi_ok]
        happened = m[m["y"] == 1].set_index("_qid")["probability"]
        ll = -np.log(happened.astype(float).clip(lower=EPS))
        size = m.groupby("_qid")["probability"].size()
        sq = ((m["probability"] - m["y"]) ** 2).groupby(m["_qid"]).sum()
        # two-way: binary Brier of one selection (both rows carry the same error); multi-class: the grid sum
        br = sq.where(size > 2, sq / 2)
        pp = m.groupby("_qid")["probability"].max()
        pr = d.index.isin(pick_idx) & multi_ok
        q = d.loc[pr, "_qid"]
        d.loc[pr, "_ll"] = q.map(ll).values
        d.loc[pr, "_br"] = q.map(br).values
        d.loc[pr, "_pp"] = q.map(pp).values
    return d


def build_index(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored[INDEX_COLS].copy()
    for c in ("kickoff_utc", "predicted_at", "canonical_kickoff_utc"):
        out[c] = pd.to_datetime(out[c], utc=True, errors="coerce")
    out["probability"] = out["probability"].astype("float32")
    out["y"] = out["y"].astype("Int8")
    out["hit"] = out["hit"].astype("Int8")
    out["q_size"] = out["q_size"].astype("int16")
    out["pick"] = out["pick"].astype(bool)
    out["ledger_only"] = out["ledger_only"].astype(bool)
    for c in ("repo", "model_id", "model_version", "market", "line", "selection", "league", "league_slug",
              "snapshot", "outcome_status", "team", "kickoff_source"):
        out[c] = out[c].astype("category")
    out = out.sort_values(["kickoff_utc", "fixture_ref", "repo", "model_id", "market", "line", "player_id",
                           "selection", "snapshot", "model_version"], kind="mergesort").reset_index(drop=True)
    _no_price(out)
    return out


def build_summary(scored: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    d = scored.copy()
    d["_pa"] = pd.to_datetime(d["predicted_at"], utc=True, errors="coerce")
    d = d[d["_pa"] >= start]
    if d.empty:
        return pd.DataFrame(columns=SUMMARY_COLS)
    # league ledger-only = any of its rows came from the ledger-only writer (no tips in that league)
    lo = d.groupby("league_slug")["ledger_only"].any()
    q = d.drop_duplicates("_qid").copy()
    st_ = d.groupby("_qid").agg(_status=("outcome_status",
                                         lambda s: "SETTLED" if (s == "SETTLED").all()
                                         else ("VOID" if s.isin(["VOID_DID_NOT_PLAY", "NO_RESULT"]).any()
                                               else "PENDING")),
                                _ll=("_ll", "max"), _br=("_br", "max"), _pp=("_pp", "max"),
                                _hit=("hit", "max"), _size=("q_size", "first"),
                                _p=("probability", "first"), _y=("y", "first"))
    q = q[["_qid", "repo", "model_id", "market", "league_slug", "league", "snapshot", "_pa"]].merge(
        st_, left_on="_qid", right_index=True)
    # calibration of single-selection questions (props YES, goal bands, combos): mean p vs how often it happened
    single = (q["_size"] == 1) & q["_ll"].notna()
    q["_sp"] = q["_p"].where(single)
    q["_sy"] = q["_y"].where(single)
    k = ["repo", "model_id", "market", "league_slug", "snapshot"]
    agg = q.groupby(k, sort=True).agg(
        league=("league", "first"), predictions=("_qid", "size"),
        settled=("_ll", "count"), void=("_status", lambda s: int((s == "VOID").sum())),
        pending=("_status", lambda s: int((s == "PENDING").sum())),
        hits=("_hit", "sum"), log_loss_sum=("_ll", "sum"), brier_sum=("_br", "sum"),
        pick_prob_sum=("_pp", "sum"), single_settled=("_sp", "count"), single_p_sum=("_sp", "sum"),
        single_y_sum=("_sy", "sum"), first_predicted=("_pa", "min"), data_through=("_pa", "max")).reset_index()
    n = agg["settled"].where(agg["settled"] > 0)
    agg["hits"] = agg["hits"].fillna(0).astype(int)
    agg["hit_rate"] = (agg["hits"] / n).round(4)
    agg["log_loss"] = (agg["log_loss_sum"] / n).round(4)
    agg["brier"] = (agg["brier_sum"] / n).round(4)
    agg["mean_pick_prob"] = (agg["pick_prob_sum"] / n).round(4)
    agg["single_y_sum"] = agg["single_y_sum"].fillna(0).astype(int)
    for c in ("log_loss_sum", "brier_sum", "pick_prob_sum", "single_p_sum"):
        agg[c] = agg[c].round(6)
    agg["ledger_only"] = agg["league_slug"].map(lo).fillna(False).astype(bool)
    for c in ("first_predicted", "data_through"):
        agg[c] = agg[c].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    agg["clean_record_start"] = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = agg[SUMMARY_COLS]
    _no_price(out)
    return out


def with_identity(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ledger rows + their canonical identity columns (src.fixture_identity), and the identity table."""
    ident = FI.resolve_identities(rows)
    out = rows.merge(ident[["fixture_ref", "canonical_fixture_ref", "canonical_kickoff_utc", "kickoff_source",
                            "canonical_home", "canonical_away"]], on="fixture_ref", how="left", validate="m:1")
    return out, ident


def identity_doc(rows: pd.DataFrame, ident: pd.DataFrame) -> dict:
    """Split counts (before / after, per kind, per league) and every kickoff disagreement (data-truth findings)."""
    rep = FI.split_report(rows, ident)
    fx = FI._fixtures(rows).merge(ident, on="fixture_ref")
    drift = []
    for cref, g in fx[fx["kickoff_drift_min"] > 0].groupby("canonical_fixture_ref", sort=True):
        drift.append({"canonical_fixture_ref": cref, "league": str(g["league"].iloc[0]),
                      "match": f"{g['canonical_home'].iloc[0]} v {g['canonical_away'].iloc[0]}",
                      "kickoff_source": str(g["kickoff_source"].iloc[0]),
                      "canonical_kickoff_utc": (g["canonical_kickoff_utc"].iloc[0].strftime("%Y-%m-%dT%H:%M:%SZ")
                                                if pd.notna(g["canonical_kickoff_utc"].iloc[0]) else None),
                      "drift_min": float(g["kickoff_drift_min"].iloc[0]),
                      "sources": {r.fixture_ref: r.kickoff_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
                                  for r in g.sort_values("fixture_ref").itertuples() if pd.notna(r.kickoff_utc)}})
    return {**rep, "kickoff_disagreements": drift}


def run(*, v9_root: Path = ROOT, pro_root: Path | None = None, out_dir: Path = OUT_DIR,
        clean_cfg: Path = CLEAN_CFG) -> dict:
    frames = []
    for r in [Path(v9_root)] + ([Path(pro_root)] if pro_root else []):
        d = load(r / "output" / "prediction_ledger", r / "output" / "prediction_ledger_settled")
        if not d.empty:
            frames.append(d)
    if not frames:
        raise SystemExit("no prediction ledger rows found - nothing written")
    rows, ident = with_identity(pd.concat(frames, ignore_index=True))
    scored = score(rows)
    start = clean_record_start(clean_cfg)
    index, summary = build_index(scored), build_summary(scored, start)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index.to_parquet(out_dir / INDEX_NAME, index=False, compression="zstd")
    summary.to_csv(out_dir / SUMMARY_NAME, index=False, encoding="utf-8", lineterminator="\n")
    idoc = identity_doc(rows, ident)
    (out_dir / IDENTITY_NAME).write_text(json.dumps(idoc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                                         newline="\n")
    return {"rows": int(len(index)), "repos": sorted(index["repo"].astype(str).unique()),
            "summary_rows": int(len(summary)), "settled_questions": int(summary["settled"].sum()),
            "index_bytes": (out_dir / INDEX_NAME).stat().st_size,
            "matches_split_before": idoc["matches_split_before"], "page_keys_before": idoc["page_keys_before"],
            "canonical_fixtures_after": idoc["canonical_fixtures_after"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pro-root", type=Path, default=None,
                    help="checkout of NevixAA/wowzaV9-Pro (output/prediction_ledger[_settled]/pro)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args(argv)
    pro = a.pro_root if a.pro_root and (a.pro_root / "output" / "prediction_ledger").exists() else None
    if a.pro_root and pro is None:
        print(f"::warning::Pro ledger not found under {a.pro_root} - index holds v9 only")
    print(json.dumps(run(pro_root=pro, out_dir=a.out_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
