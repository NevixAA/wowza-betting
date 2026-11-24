"""Success by GAMEWEEK - the stored prediction dataset (owner 2026-11-23).

    python scripts/build_prediction_gw_dataset.py [--pro-root DIR] [--now ISO]     # build / append (update_results)
    python scripts/build_prediction_gw_dataset.py --verify                         # reconcile, exit 1 on a finding

OWNER: "for every GW that passed, i want for each league, market and all the models - what the % success. of course
we need to store it. its not only prediction, its gonna be our data set". LEARNING DATA, NOT BETS: no price, stake or
P/L column is read or written; v11 is private and never read (scripts/build_predictions_index.PUBLIC_REPOS).

ONE ROW per KEY = gw x league_slug x repo x model_id x model_version x market x line x selection_group x snapshot x
clean_record. Each row counts the model's QUESTIONS (src: scripts/build_predictions_index.score - the Predictions
page's own scoring, reused): predictions = settled + void + pending + unscorable; hits of the model's own pick;
hit_rate = hits / settled; Brier and log loss on the settled questions (the sums are stored too, so any roll-up is
exact - never an average of averages); calibration sums of single-selection questions. A rate with no settled
question is MISSING (empty), never 0.

  selection_group  '' when every selection of the market is one question (O/U, BTTS, 1X2, correct score, props
                   YES); HOME / AWAY for team markets with one question per side; the combo legs for a combo.
  model_version    always part of the key: success is NEVER pooled across versions. version_label is the short
                   display name; version_pre_clean_record = the version first predicted before the Clean Record
                   start.
  clean_record     the question was predicted on/after the Clean Record start
                   (config/trustworthy_performance.json clean_record_start_utc, 2026-11-19T06:41:48Z). A GW holding
                   both kinds has two rows, never one mixed row.

GAMEWEEK = the TOP-3 gameweek (src/top3_league.py "League-week = src/gameweek.league_week_id (GW-1, tip day)"):
'<league>|<ISO year>-W<ISO week>' of the fixture's TIP DAY (src/tip_day: the day rolls over at 10:00 Israel time).
gw = the '<ISO year>-W<week>' part; gw_start_date / gw_end_date = the Monday / Sunday tip days; gw_end_utc = the
next Monday 10:00 Israel time. The fixture is its CANONICAL identity (src/fixture_identity: one match = one identity
across models; API-Football kickoff authoritative). When the sources disagree on the kickoff and no API-Football
row exists, the question's own ledger kickoff places it. One model with two ledger refs for one canonical fixture
(e.g. two OddsAPI events) keeps its FIRST-predicted ref; the later ref is counted as duplicate_refs_excluded.

STORAGE (append-only, immutable once closed)
  output/prediction_gw_dataset.parquet + .csv   the dataset (schema_version pgw/1.0, row_id = digest of the key)
  output/prediction_gw_dataset_links/<gw>.parquet   one row per question: row_id -> fixture_ref / canonical
                                                fixture / player and its outcome, hit, log loss, Brier. The dataset
                                                is rebuilt and verified from these (verify()).
  output/prediction_gw_dataset.meta.json        schema, GW rule, Clean Record start, sha256 of every source ledger
                                                tree, counts, fixture-identity split counts.
  output/corrections/prediction_gw_dataset_corrections.csv   a CLOSED row never changes. If the ledgers later give
                                                different numbers for it, the new numbers go here beside the
                                                original with a reason (repo convention output/corrections/).
A row CLOSES when its GW has ended (now >= gw_end_utc) and no question in it is pending; closed_at records when.
Open rows are re-derived on every run. Idempotent: an unchanged ledger rewrites nothing.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gameweek import RULE_VERSION as GW_RULE, league_week_id  # noqa: E402
from src.model_version_label import version_label  # noqa: E402
from src.tip_day import ROLLOVER_HOUR, TZ  # noqa: E402

SCHEMA_VERSION = "pgw/1.0"
OUT = ROOT / "output"
DATASET = OUT / "prediction_gw_dataset.parquet"
DATASET_CSV = OUT / "prediction_gw_dataset.csv"
META = OUT / "prediction_gw_dataset.meta.json"
LINKS_DIR = OUT / "prediction_gw_dataset_links"
CORRECTIONS = OUT / "corrections" / "prediction_gw_dataset_corrections.csv"
#: readable release names per model version (src.model_releases) - metadata BESIDE the immutable rows
VERSIONS = OUT / "prediction_gw_dataset_versions.csv"

KEY = ["gw", "league_slug", "repo", "model_id", "model_version", "market", "line", "selection_group", "snapshot",
       "clean_record"]
COUNTS = ["predictions", "settled", "void", "pending", "unscorable", "hits", "duplicate_refs_excluded"]
SUMS = ["brier_sum", "log_loss_sum", "pick_prob_sum", "single_settled", "single_p_sum", "single_y_sum"]
COLUMNS = (["row_id", "schema_version"] + KEY +
           ["league", "version_label", "version_first_predicted_at", "version_pre_clean_record",
            "gw_start_date", "gw_end_date", "gw_end_utc", "fixtures"] + COUNTS +
           ["hit_rate", "brier", "log_loss", "mean_pick_prob"] + SUMS +
           ["first_kickoff_utc", "last_kickoff_utc", "first_predicted_at", "last_predicted_at", "status", "closed_at",
            "source_digest"])
LINK_COLS = ["row_id", "gw", "canonical_fixture_ref", "fixture_ref", "player_id", "kickoff_utc", "kickoff_basis",
             "predicted_at", "outcome", "hit", "log_loss", "brier", "pick_prob", "single_p", "single_y"]
CORR_COLS = ["row_id", "reason", "detected_at", "field", "original", "recomputed", "original_digest",
             "recomputed_digest"]
OUTCOMES = ("SETTLED", "VOID", "PENDING", "UNSCORABLE", "DUPLICATE_REF")
_ISO = "%Y-%m-%dT%H:%M:%SZ"


def _iso(ts) -> str:
    return pd.Timestamp(ts).strftime(_ISO) if pd.notna(ts) else ""


def _iso_col(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce").dt.strftime(_ISO).fillna("")


# ── gameweek bounds (GW-1 on the tip day) ──────────────────────────────────────────────────────────────────────────
def gw_bounds(gw: str) -> tuple[str, str, str]:
    """('YYYY-MM-DD' Monday tip day, 'YYYY-MM-DD' Sunday tip day, end instant UTC ISO) of '<year>-W<week>'."""
    y, w = gw.split("-W")
    mon = pd.Timestamp.fromisocalendar(int(y), int(w), 1)
    sun = mon + pd.Timedelta(days=6)
    end_local = (mon + pd.Timedelta(days=7)).replace(hour=ROLLOVER_HOUR)
    end_utc = end_local.tz_localize(TZ).tz_convert("UTC")
    return mon.strftime("%Y-%m-%d"), sun.strftime("%Y-%m-%d"), end_utc.strftime(_ISO)


def gw_start_utc(gw: str) -> pd.Timestamp:
    y, w = gw.split("-W")
    mon = pd.Timestamp.fromisocalendar(int(y), int(w), 1).replace(hour=ROLLOVER_HOUR)
    return mon.tz_localize(TZ).tz_convert("UTC")


# ── questions ──────────────────────────────────────────────────────────────────────────────────────────────────────
def questions(scored: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """One row per scored QUESTION (scripts/build_predictions_index.score `_qid`), with its GW and outcome."""
    d = scored
    pa = pd.to_datetime(d["predicted_at"], utc=True, errors="coerce")
    st = d["outcome_status"].astype(str)
    g = d.assign(_pa=pa, _set=(st == "SETTLED"), _void=st.isin(["VOID_DID_NOT_PLAY", "NO_RESULT"])) \
        .groupby("_qid", sort=False)
    q = g.agg(repo=("repo", "first"), model_id=("model_id", "first"), model_version=("model_version", "first"),
              market=("market", "first"), line=("line", "first"), selection_group=("_side", "first"),
              snapshot=("snapshot", "first"), league=("league", "first"), league_slug=("league_slug", "first"),
              fixture_ref=("fixture_ref", "first"), canonical_fixture_ref=("canonical_fixture_ref", "first"),
              canonical_kickoff_utc=("canonical_kickoff_utc", "first"), kickoff_utc=("kickoff_utc", "first"),
              player_id=("player_id", "first"), predicted_at=("_pa", "min"), _all_set=("_set", "all"),
              _any_void=("_void", "any"), hit=("hit", "max"), log_loss=("_ll", "max"), brier=("_br", "max"),
              pick_prob=("_pp", "max"), _size=("q_size", "first"), _p=("probability", "first"),
              _y=("y", "first")).reset_index()
    for c in ("repo", "model_id", "model_version", "market", "line", "selection_group", "snapshot", "league",
              "league_slug", "fixture_ref", "canonical_fixture_ref", "player_id"):
        q[c] = q[c].astype(str).replace("nan", "")
    scored_ok = q["log_loss"].notna()
    q["outcome"] = np.select([scored_ok, q["_any_void"], q["_all_set"]], ["SETTLED", "VOID", "UNSCORABLE"],
                             default="PENDING")
    single = (q["_size"] == 1) & scored_ok
    q["single_p"] = q["_p"].astype(float).where(single)
    q["single_y"] = q["_y"].astype(float).where(single)
    ck = pd.to_datetime(q["canonical_kickoff_utc"], utc=True, errors="coerce")
    ok_ = pd.to_datetime(q["kickoff_utc"], utc=True, errors="coerce")
    q["kickoff_basis"] = np.where(ck.notna(), "canonical", "own")
    q["kickoff_utc"] = ck.where(ck.notna(), ok_)
    q["clean_record"] = q["predicted_at"] >= start
    # one model x version x question x canonical fixture = ONE ref: the first predicted wins
    dk = ["repo", "model_id", "model_version", "market", "line", "selection_group", "canonical_fixture_ref",
          "player_id", "snapshot"]
    q = q.sort_values(dk + ["predicted_at", "fixture_ref"], kind="mergesort")
    dup = q.duplicated(dk, keep="first")
    q.loc[dup, "outcome"] = "DUPLICATE_REF"
    for c in ("hit", "log_loss", "brier", "pick_prob", "single_p", "single_y"):
        q.loc[dup, c] = np.nan
    # gameweek (GW-1) per distinct league x kickoff
    pairs = q[["league", "kickoff_utc"]].drop_duplicates()
    wk = {(lg, ko): league_week_id(lg, ko) for lg, ko in zip(pairs["league"], pairs["kickoff_utc"])}
    q["league_week_id"] = [wk[(lg, ko)] for lg, ko in zip(q["league"], q["kickoff_utc"])]
    q["gw"] = q["league_week_id"].str.split("|", n=1).str[1].fillna("")
    return q.drop(columns=[c for c in q.columns if c.startswith("_")]).reset_index(drop=True)


def row_id(key_vals) -> str:
    return hashlib.sha256("\x1f".join(str(v) for v in key_vals).encode("utf-8")).hexdigest()[:20]


def _fmt(x: pd.Series, spec: str) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce").astype(float)
    return pd.Series(["" if a != a else format(a, spec) for a in v], index=x.index)


def digests(links: pd.DataFrame) -> pd.Series:
    """sha256 per row_id of its question outcomes (the link rows, order-independent)."""
    if links.empty:
        return pd.Series(dtype=str)
    L = links.sort_values(["row_id", "canonical_fixture_ref", "fixture_ref", "player_id"], kind="mergesort")
    line = (L["canonical_fixture_ref"].astype(str) + "|" + L["fixture_ref"].astype(str) + "|"
            + L["player_id"].astype(str) + "|" + L["outcome"].astype(str) + "|" + _fmt(L["hit"], ".0f") + "|"
            + _fmt(L["log_loss"], ".9f") + "|" + _fmt(L["brier"], ".9f"))
    txt = line.groupby(L["row_id"], sort=True).agg("\n".join)
    return txt.map(lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest())


def aggregate(links: pd.DataFrame) -> pd.DataFrame:
    """Dataset metric columns from link rows (THE one aggregation: build and verify both use it)."""
    L = links.copy()
    o = L["outcome"]
    L["_n"] = (o != "DUPLICATE_REF").astype(int)
    for name in ("SETTLED", "VOID", "PENDING", "UNSCORABLE", "DUPLICATE_REF"):
        L["_" + name] = (o == name).astype(int)
    g = L.groupby("row_id", sort=True)
    a = g.agg(predictions=("_n", "sum"), settled=("_SETTLED", "sum"), void=("_VOID", "sum"),
              pending=("_PENDING", "sum"), unscorable=("_UNSCORABLE", "sum"),
              duplicate_refs_excluded=("_DUPLICATE_REF", "sum"), hits=("hit", "sum"),
              brier_sum=("brier", "sum"), log_loss_sum=("log_loss", "sum"), pick_prob_sum=("pick_prob", "sum"),
              single_settled=("single_p", "count"), single_p_sum=("single_p", "sum"),
              single_y_sum=("single_y", "sum"))
    for c in COUNTS + ["single_settled", "single_y_sum"]:
        a[c] = a[c].astype(int)
    for c in ("brier_sum", "log_loss_sum", "pick_prob_sum", "single_p_sum"):
        a[c] = a[c].astype(float).round(9)
    n = a["settled"].where(a["settled"] > 0)
    a["hit_rate"] = (a["hits"] / n).round(6)
    a["brier"] = (a["brier_sum"] / n).round(6)
    a["log_loss"] = (a["log_loss_sum"] / n).round(6)
    a["mean_pick_prob"] = (a["pick_prob_sum"] / n).round(6)
    a["source_digest"] = digests(L)
    return a.reset_index()


def derive(q: pd.DataFrame, start: pd.Timestamp, now: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(rows, links, info) for every GW that has STARTED by `now`. Rows without a GW are counted, never guessed."""
    unassigned = int((q["gw"] == "").sum())
    q = q[q["gw"] != ""].copy()
    started = {gw: gw_start_utc(gw) <= now for gw in q["gw"].unique()}
    q = q[q["gw"].map(started)].copy()
    # version facts (over every question of the version, not only this GW)
    vfirst = q.groupby(["model_id", "model_version"])["predicted_at"].min()
    q["clean_record"] = q["clean_record"].astype(bool)
    keys = q[KEY].drop_duplicates()
    keys["row_id"] = [row_id(k) for k in keys[KEY].itertuples(index=False, name=None)]
    q = q.merge(keys, on=KEY, how="left", validate="m:1")
    links = q[LINK_COLS].copy()
    links["hit"] = links["hit"].astype("Int8")
    agg = aggregate(links)
    g = q.groupby("row_id", sort=True)
    head = g[KEY + ["league"]].first()
    extra = g.agg(fixtures=("canonical_fixture_ref", "nunique"), first_kickoff_utc=("kickoff_utc", "min"),
                  last_kickoff_utc=("kickoff_utc", "max"), first_predicted_at=("predicted_at", "min"),
                  last_predicted_at=("predicted_at", "max"))
    rows = head.join(extra).join(agg.set_index("row_id")).reset_index()
    rows["schema_version"] = SCHEMA_VERSION
    rows["version_label"] = rows["model_version"].map(version_label)
    vf = [vfirst[(m, v)] for m, v in zip(rows["model_id"], rows["model_version"])]
    rows["version_first_predicted_at"] = [_iso(x) for x in vf]
    rows["version_pre_clean_record"] = [bool(x < start) for x in vf]
    b = {gw: gw_bounds(gw) for gw in rows["gw"].unique()}
    rows["gw_start_date"] = rows["gw"].map(lambda x: b[x][0])
    rows["gw_end_date"] = rows["gw"].map(lambda x: b[x][1])
    rows["gw_end_utc"] = rows["gw"].map(lambda x: b[x][2])
    for c in ("first_kickoff_utc", "last_kickoff_utc", "first_predicted_at", "last_predicted_at"):
        rows[c] = _iso_col(rows[c])
    ended = pd.to_datetime(rows["gw_end_utc"], utc=True) <= now
    rows["status"] = np.where(ended & (rows["pending"] == 0), "CLOSED", "OPEN")
    rows["closed_at"] = ""
    links["predicted_at"] = _iso_col(links["predicted_at"])
    links["kickoff_utc"] = _iso_col(links["kickoff_utc"])
    return rows[COLUMNS], links[LINK_COLS], {"unassigned_questions": unassigned}


# ── storage ────────────────────────────────────────────────────────────────────────────────────────────────────────
def read_dataset(path: Path = DATASET) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=COLUMNS)
    return _typed(pd.read_parquet(path))


def _typed(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for c in COUNTS + ["fixtures", "single_settled", "single_y_sum"]:
        d[c] = pd.to_numeric(d[c]).astype(int)
    for c in ("hit_rate", "brier", "log_loss", "mean_pick_prob", "brier_sum", "log_loss_sum", "pick_prob_sum",
              "single_p_sum"):
        d[c] = pd.to_numeric(d[c]).astype(float)
    for c in ("clean_record", "version_pre_clean_record"):
        d[c] = d[c].astype(str).str.lower().isin(["true", "1"])
    for c in COLUMNS:
        if c not in d.columns:
            raise ValueError(f"prediction GW dataset lacks column {c}")
    obj = [c for c in COLUMNS if c not in COUNTS + SUMS + ["fixtures", "hit_rate", "brier", "log_loss",
                                                           "mean_pick_prob", "clean_record",
                                                           "version_pre_clean_record"]]
    for c in obj:
        d[c] = d[c].astype(str).replace({"nan": "", "None": ""})
    return d[COLUMNS]


def read_links(links_dir: Path = LINKS_DIR) -> pd.DataFrame:
    parts = sorted(Path(links_dir).glob("*.parquet")) if Path(links_dir).exists() else []
    if not parts:
        return pd.DataFrame(columns=LINK_COLS)
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def read_corrections(path: Path = CORRECTIONS) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=CORR_COLS)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def merge(old: pd.DataFrame, old_links: pd.DataFrame, new: pd.DataFrame, new_links: pd.DataFrame,
          now: pd.Timestamp, unread_repos=frozenset()) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Closed rows (and their links) are kept verbatim; a recomputed difference becomes a correction. Rows of a repo
    whose ledger was not read this run (unread_repos) are only kept, never compared."""
    closed = old[old["status"] == "CLOSED"].set_index("row_id")
    newi = new.set_index("row_id")
    corr = []
    stamp = now.strftime(_ISO)
    for rid, o in closed.iterrows():
        if rid not in newi.index:
            if o["repo"] in unread_repos:
                continue
            corr.append({"row_id": rid, "reason": "SOURCE_ROWS_MISSING", "detected_at": stamp, "field": "",
                         "original": "", "recomputed": "", "original_digest": o["source_digest"],
                         "recomputed_digest": ""})
            continue
        n = newi.loc[rid]
        if n["source_digest"] == o["source_digest"]:
            continue
        for f in COUNTS + ["hit_rate", "brier", "log_loss"]:
            a, b = o[f], n[f]
            same = (pd.isna(a) and pd.isna(b)) or (pd.notna(a) and pd.notna(b) and np.isclose(float(a), float(b)))
            if not same:
                corr.append({"row_id": rid, "reason": "LEDGER_CHANGED_AFTER_CLOSE", "detected_at": stamp,
                             "field": f, "original": "" if pd.isna(a) else str(a),
                             "recomputed": "" if pd.isna(b) else str(b), "original_digest": o["source_digest"],
                             "recomputed_digest": n["source_digest"]})
        if not any(c["row_id"] == rid and c["recomputed_digest"] == n["source_digest"] for c in corr):
            corr.append({"row_id": rid, "reason": "LEDGER_CHANGED_AFTER_CLOSE", "detected_at": stamp,
                         "field": "source_digest", "original": o["source_digest"], "recomputed": n["source_digest"],
                         "original_digest": o["source_digest"], "recomputed_digest": n["source_digest"]})
    keep_new = new[~new["row_id"].isin(closed.index)].copy()
    newly = keep_new["status"] == "CLOSED"
    keep_new.loc[newly, "closed_at"] = stamp
    rows = pd.concat([old[old["status"] == "CLOSED"], keep_new], ignore_index=True)
    rows = rows.sort_values(["gw"] + KEY[1:], kind="mergesort").reset_index(drop=True)
    links = pd.concat([old_links[old_links["row_id"].isin(closed.index)],
                       new_links[new_links["row_id"].isin(keep_new["row_id"])]], ignore_index=True)
    return _typed(rows), links, corr


def _sha_tree(root: Path) -> str:
    h = hashlib.sha256()
    if Path(root).exists():
        for p in sorted(Path(root).rglob("*.csv")):
            h.update(p.relative_to(root).as_posix().encode("utf-8"))
            h.update(hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).digest())
    return h.hexdigest()


def _cmp(s: pd.Series):
    """A column as comparable text (missing -> '<NA>', whatever the dtype)."""
    return s.astype(object).where(s.notna(), "<NA>").astype(str).to_numpy()


def _write_if_changed_parquet(df: pd.DataFrame, path: Path) -> bool:
    if path.exists():
        try:
            cur = pd.read_parquet(path)
            if list(cur.columns) == list(df.columns) and len(cur) == len(df) and all(
                    (_cmp(cur[c]) == _cmp(df[c])).all() for c in df.columns):
                return False
        except (OSError, ValueError):
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    return True


def _write_text_if_changed(text: str, path: Path) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return True


def store(rows: pd.DataFrame, links: pd.DataFrame, corr: list[dict], meta: dict, *, out: Path = OUT) -> dict:
    out = Path(out)
    ds, csv, mt = out / DATASET.name, out / DATASET_CSV.name, out / META.name
    ldir, cpath = out / LINKS_DIR.name, out / "corrections" / CORRECTIONS.name
    changed = []
    if _write_if_changed_parquet(rows, ds):
        changed.append(ds.name)
    if _write_text_if_changed(rows.to_csv(index=False, lineterminator="\n"), csv):
        changed.append(csv.name)
    for gw, L in links.groupby("gw", sort=True):
        L = L.sort_values(["row_id", "canonical_fixture_ref", "fixture_ref", "player_id"], kind="mergesort").copy()
        L["hit"] = pd.to_numeric(L["hit"], errors="coerce").astype("Int8")
        for c in ("log_loss", "brier", "pick_prob", "single_p", "single_y"):
            L[c] = pd.to_numeric(L[c], errors="coerce").astype(float)
        if _write_if_changed_parquet(L[LINK_COLS].reset_index(drop=True), ldir / f"{gw}.parquet"):
            changed.append(f"links/{gw}")
    if corr:
        old = read_corrections(cpath)
        new = pd.DataFrame(corr, columns=CORR_COLS).astype(str)
        seen = set(zip(old["row_id"], old["field"], old["recomputed_digest"]))
        add = new[[(a, b, c) not in seen for a, b, c in zip(new["row_id"], new["field"], new["recomputed_digest"])]]
        if len(add):
            cpath.parent.mkdir(parents=True, exist_ok=True)
            add.to_csv(cpath, mode="a", header=not cpath.exists(), index=False, lineterminator="\n")
            changed.append(cpath.name)
    if _write_text_if_changed(json.dumps(meta, indent=1, ensure_ascii=False, sort_keys=True) + "\n", mt):
        changed.append(mt.name)
    return {"changed": changed}


# ── verify (reconcile exactly; data truth rule 14) ─────────────────────────────────────────────────────────────────
def verify(rows: pd.DataFrame, links: pd.DataFrame, corrections: pd.DataFrame | None = None) -> list[str]:
    """Findings ([] = clean): schema, keys, no defaults, no price, and every row's numbers re-derived exactly from its
    link rows (the question-level record)."""
    issues = []
    from src import prediction_ledger as PL
    bad = [c for c in list(rows.columns) + list(links.columns) if PL.PRICE_LIKE.search(str(c))]
    if bad:
        issues.append(f"price-like columns {bad}")
    if set(rows["schema_version"]) - {SCHEMA_VERSION}:
        issues.append(f"schema_version {sorted(set(rows['schema_version']))} != {SCHEMA_VERSION}")
    if rows["row_id"].duplicated().any():
        issues.append(f"{int(rows['row_id'].duplicated().sum())} duplicate row_id")
    if rows.duplicated(KEY).any():
        issues.append(f"{int(rows.duplicated(KEY).sum())} duplicate key rows")
    rid = [row_id(k) for k in rows[KEY].itertuples(index=False, name=None)]
    if list(rows["row_id"]) != rid:
        issues.append(f"{sum(a != b for a, b in zip(rows['row_id'], rid))} row_id not the digest of its key")
    for c in ("gw", "league_slug", "league", "repo", "model_id", "model_version", "market", "snapshot"):
        if (rows[c].astype(str) == "").any():
            issues.append(f"{int((rows[c].astype(str) == '').sum())} rows with empty {c}")
    if (~rows["gw"].str.fullmatch(r"\d{4}-W\d{2}")).any():
        issues.append("gw not '<ISO year>-W<week>'")
    if (rows["repo"].isin(["v11"])).any():
        issues.append("v11 rows in the public dataset")
    tot = rows["settled"] + rows["void"] + rows["pending"] + rows["unscorable"]
    if (tot != rows["predictions"]).any():
        issues.append(f"{int((tot != rows['predictions']).sum())} rows: predictions != settled+void+pending+unscorable")
    none = rows["settled"] == 0
    for c in ("hit_rate", "brier", "log_loss"):
        if rows.loc[none, c].notna().any():
            issues.append(f"{c} filled on a row with no settled question (a default)")
        if rows.loc[~none, c].isna().any():
            issues.append(f"{c} missing on a row with settled questions")
    hr = (rows["hits"] / rows["settled"].where(~none)).round(6)
    both = hr.notna() & rows["hit_rate"].notna()
    if (hr.isna() != rows["hit_rate"].isna()).any() or \
            (~np.isclose(hr[both].astype(float), rows.loc[both, "hit_rate"].astype(float))).any():
        issues.append("hit_rate != hits / settled")
    if ((rows["hit_rate"] < 0) | (rows["hit_rate"] > 1)).any():
        issues.append("hit_rate outside [0, 1]")
    cl = rows["status"] == "CLOSED"
    if (rows.loc[cl, "pending"] > 0).any():
        issues.append("a CLOSED row still has pending questions")
    if (rows.loc[cl, "closed_at"] == "").any():
        issues.append("a CLOSED row without closed_at")
    if set(rows["status"]) - {"OPEN", "CLOSED"}:
        issues.append(f"unknown status {sorted(set(rows['status']) - {'OPEN', 'CLOSED'})}")
    if set(links["outcome"]) - set(OUTCOMES):
        issues.append(f"unknown link outcome {sorted(set(links['outcome']) - set(OUTCOMES))}")
    orphan = set(links["row_id"]) - set(rows["row_id"])
    missing = set(rows["row_id"]) - set(links["row_id"])
    if orphan:
        issues.append(f"{len(orphan)} link row_id(s) without a dataset row")
    if missing:
        issues.append(f"{len(missing)} dataset row(s) without link rows")
    lk = links[links["outcome"] != "DUPLICATE_REF"]
    if lk.duplicated(["row_id", "canonical_fixture_ref", "player_id"]).any():
        issues.append(f"{int(lk.duplicated(['row_id', 'canonical_fixture_ref', 'player_id']).sum())} questions "
                      "counted twice in one row (one fixture, one identity)")
    if len(links):
        a = aggregate(links).set_index("row_id")
        r = rows.set_index("row_id").loc[a.index]
        for c in COUNTS + ["single_settled", "single_y_sum"]:
            n = int((a[c] != r[c]).sum())
            if n:
                issues.append(f"{n} rows: {c} does not reconcile with the link rows")
        for c in ("brier_sum", "log_loss_sum", "pick_prob_sum", "single_p_sum"):
            n = int((~np.isclose(a[c], r[c], atol=1e-6)).sum())
            if n:
                issues.append(f"{n} rows: {c} does not reconcile with the link rows")
        n = int((a["source_digest"] != r["source_digest"]).sum())
        if n:
            issues.append(f"{n} rows: source_digest does not match the link rows")
    return issues


def reconcile_with_ledger(rows: pd.DataFrame, recomputed: pd.DataFrame, corrections: pd.DataFrame,
                          repos=("v9",)) -> list[str]:
    """CLOSED rows of `repos` against a fresh derivation from the ledgers: equal, or a correction beside it.
    Totals of OPEN rows lag the ledgers between runs and are not compared here."""
    issues = []
    c = rows[(rows["status"] == "CLOSED") & rows["repo"].isin(repos)].set_index("row_id")
    n = recomputed.set_index("row_id")
    corrected = set(zip(corrections["row_id"], corrections["recomputed_digest"]))
    gone = {r for r, _ in corrected} | set()
    for rid, o in c.iterrows():
        if rid not in n.index:
            if rid not in gone:
                issues.append(f"closed row {rid} no longer derivable from the ledger and no correction")
            continue
        if n.loc[rid, "source_digest"] != o["source_digest"] and (rid, n.loc[rid, "source_digest"]) not in corrected:
            issues.append(f"closed row {rid} ({o['gw']} {o['league_slug']} {o['model_id']} {o['market']}) differs "
                          "from the ledger and no correction is recorded")
    return issues


def version_names(rows: pd.DataFrame) -> pd.DataFrame:
    """model_id x market x model_version -> release name, install date, serving windows (src.model_releases;
    config/model_release_registry.json). 'unknown release' where no install record exists - never a guess."""
    from src.model_releases import UNKNOWN, Releases
    R = Releases()
    out = []
    for m, k, v in rows[["model_id", "market", "model_version"]].drop_duplicates().sort_values(
            ["model_id", "market", "model_version"]).itertuples(index=False, name=None):
        rec = R.records(m, k, v)
        name, day = R.name(m, k, v)
        out.append({"model_id": m, "market": k, "model_version": v, "release_name": name,
                    "install_date": min(r["installed_at"] for r in rec)[:10] if rec and name != UNKNOWN else "",
                    "role": rec[0].get("role", "champion") if rec else "",
                    "served_windows": "; ".join(f"{r['served_from']}..{r.get('served_to') or ''}" for r in rec)})
    return pd.DataFrame(out, columns=["model_id", "market", "model_version", "release_name", "install_date", "role",
                                      "served_windows"])


# ── orchestration ─────────────────────────────────────────────────────────────────────────────────────────────────
def build_from_roots(v9_root: Path, pro_root: Path | None, clean_cfg: Path | None = None):
    """(questions, start, identity split report, source hashes) from the ledgers (the Predictions page's sources)."""
    from scripts import build_predictions_index as B
    frames, sources = [], {}
    for name, r in (("v9", Path(v9_root)), ("pro", Path(pro_root) if pro_root else None)):
        if r is None:
            continue
        led, st = r / "output" / "prediction_ledger", r / "output" / "prediction_ledger_settled"
        d = B.load(led, st)
        if not d.empty:
            frames.append(d)
        for repo in B.PUBLIC_REPOS:
            if (led / repo).exists():
                sources[f"{repo}:prediction_ledger"] = _sha_tree(led / repo)
                sources[f"{repo}:prediction_ledger_settled"] = _sha_tree(st / repo)
    if not frames:
        raise SystemExit("no prediction ledger rows found - nothing written")
    rows, ident = B.with_identity(pd.concat(frames, ignore_index=True))
    scored = B.score(rows)
    start = B.clean_record_start(clean_cfg or B.CLEAN_CFG)
    from src import fixture_identity as FI
    rep = FI.split_report(rows, ident)
    rep.pop("per_league", None)
    return questions(scored, start), start, rep, dict(sorted(sources.items()))


def run(*, v9_root: Path = ROOT, pro_root: Path | None = None, out: Path = OUT, now=None,
        clean_cfg: Path | None = None) -> dict:
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    q, start, ident, sources = build_from_roots(v9_root, pro_root, clean_cfg)
    new, new_links, info = derive(q, start, now)
    out = Path(out)
    old = read_dataset(out / DATASET.name)
    old_links = read_links(out / LINKS_DIR.name)
    unread = set() if (pro_root is not None or (new["repo"] == "pro").any()) else {"pro"}
    if unread and len(old):
        # Pro ledger unreachable this run: its open rows stay as they were (never dropped, never re-derived from v9)
        keep = old[old["repo"].isin(unread) & (old["status"] == "OPEN")]
        new = pd.concat([new, keep], ignore_index=True)
        new_links = pd.concat([new_links, old_links[old_links["row_id"].isin(keep["row_id"])]], ignore_index=True)
    rows, links, corr = merge(old, old_links, new, new_links, now, unread_repos=unread)
    issues = verify(rows, links)
    if issues:
        raise SystemExit("prediction GW dataset does not reconcile - nothing written: " + "; ".join(issues[:10]))
    meta = {"artifact": "prediction_gw_dataset", "schema_version": SCHEMA_VERSION,
            "gameweek_rule": f"{GW_RULE}: src/gameweek.league_week_id on the tip day (src/tip_day, rollover "
                             f"{ROLLOVER_HOUR}:00 {TZ.key}) - the TOP-3 gameweek (src/top3_league.py)",
            "clean_record_start_utc": start.strftime(_ISO), "sources_sha256": sources,
            "repos": sorted(rows["repo"].unique().tolist()), "rows": int(len(rows)),
            "closed_rows": int((rows["status"] == "CLOSED").sum()), "link_rows": int(len(links)),
            "gws": sorted(rows["gw"].unique().tolist()), "unassigned_questions": info["unassigned_questions"],
            "fixture_identity": ident, "private": "v11 never read; no price column",
            # owner 2026-11-24: Pro combos have no bookmaker builder price anywhere (API-Football, OddsAPI)
            "price_basis": {"pro:pro_bb_combo": "no market price: hit rate only"}}
    res = store(rows, links, corr, meta, out=out)
    if _write_text_if_changed(version_names(rows).to_csv(index=False, lineterminator="\n"), out / VERSIONS.name):
        res["changed"].append(VERSIONS.name)
    return {"rows": int(len(rows)), "closed": meta["closed_rows"], "links": int(len(links)),
            "corrections_new": len(corr), **res}
