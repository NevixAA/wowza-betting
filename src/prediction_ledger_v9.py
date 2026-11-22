"""v9 writers of the prediction ledger (src/prediction_ledger.py) - LEARNING ONLY, never bets.
=============================================================================================

    python -m src.prediction_ledger_v9 board      # team models: every board fixture x market (predict.yml)
    python -m src.prediction_ledger_v9 settle     # rebuild the settled views (update_results.yml)
    python -m src.prediction_ledger_v9 registry   # rewrite output/prediction_ledger/registry.json
    python -m src.prediction_ledger_v9 validate   # Data truth 12 on the files on disk (exit 1 on a violation)

Props are recorded from inside player_model/predict.py through `PropsRecorder` (the per-player
probabilities only exist in memory there), next to the existing first-sight recorders.

OWNER RULE (2026-11-17): every model predicts every match of the tracked leagues, for every market it
serves, and the prediction is stored - not only the tipped ones - so the bars can be judged against the
fixtures that were NOT tipped.

WHAT THIS NEVER DOES. It reads output/predictions.csv (already written by predict) and the props model's
in-memory probabilities; it writes ONLY under output/prediction_ledger/ and output/prediction_ledger_settled/.
No tip, tier, gate, stake, threshold, served board or Telegram file is read for a decision or written.
No bookmaker price is stored (prices are moving to the private store). No API call is made.
Every writer is a no-op unless the repo variable PREDICTION_LEDGER_ENABLED is 'true' (default OFF), and
none of them can raise into the pipeline that calls it.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import prediction_ledger as PL  # noqa: E402

REPO = "v9"
#: the parked props challengers (player_model.props_gate challenger slot), owner 2026-11-18
CHALLENGER_MODEL_ID = "v9_props_challenger"
BOARD = ROOT / "output" / "predictions.csv"
FD_HISTORY = ROOT / "output" / "fd_history.parquet"
AF_HISTORY = ROOT / "output" / "af_history.parquet"
#: API-Football 90-minute + HT scores of every finished fixture in a config league, written daily from the
#: /fixtures?date answers by scripts/collect_standard_team_cards.py (no extra call).
AF_RESULTS = ROOT / "output" / "af_results_recent.parquet"
#: API-Football FT scores of the Standard leagues' stat-answered fixtures (same collector, since 2026-11-16).
AF_STD_CARDS = ROOT / "output" / "af_team_cards_std.parquet"
PLAYER_HISTORY = ROOT / "player_history.parquet"
#: board model_type -> ledger model_id
TRACK_MODEL = {"standard": "v9_standard", "new_format": "v9_new_format"}
#: The board's feature block (first .. last feature column). Hashed, never stored.
FEATURE_FIRST, FEATURE_LAST = "home_scored_last5", "table_pos_gap"
#: How long after kickoff a fixture with no result stays PENDING before it is reported NO_RESULT.
NO_RESULT_AFTER_H = 72


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _run_id() -> str:
    return os.getenv("GITHUB_RUN_ID", "")


def _git_sha() -> str:
    return (os.getenv("GITHUB_SHA") or "")[:12]


_SHA_CACHE: dict = {}


def file_sha(rel: str | None) -> str:
    """sha256[:12] of a model file's CONTENT ('' when there is no such file). v9's own board stamp
    `model_sha` hashes size+mtime and changes on every CI checkout, so it is not used as the sha."""
    if not rel:
        return ""
    if rel not in _SHA_CACHE:
        p = ROOT / rel
        _SHA_CACHE[rel] = hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.is_file() else ""
    return _SHA_CACHE[rel]


def _model_file(entry: dict, recipe: str) -> str | None:
    mf = entry.get("model_file") or {}
    return mf.get(recipe) or mf.get("*")


def _hash_values(vals) -> str:
    h = hashlib.sha1()
    for v in vals:
        try:
            f = float(v)
            h.update(b"nan|" if f != f else f"{f:.10g}|".encode())
        except (TypeError, ValueError):
            h.update(f"{v}|".encode("utf-8", "replace"))
    return h.hexdigest()[:12]


# ── TEAM: the board ──────────────────────────────────────────────────────────────────────────────

def team_rows(board: pd.DataFrame, *, markets: dict | None = None, slugs: PL.LeagueSlugs | None = None,
              now: datetime | None = None, coverage: dict | None = None) -> list:
    """Ledger rows for every board fixture x active v9 team entry x selection x due snapshot.

    `coverage` (a dict, filled in place when given): every board fixture with the reason it got NO row
    (owner rule 2026-11-17/18: every model predicts every match of every tracked league for every market it
    can - a fixture that is dropped is reported with its reason, never silently)."""
    drops = coverage.setdefault("dropped", []) if coverage is not None else None

    def _drop(r, why):
        if drops is not None:
            drops.append({"league": PL._s(r.get("league")), "date": PL._s(r.get("date"))[:10],
                          "home_team": PL._s(r.get("home_team")), "away_team": PL._s(r.get("away_team")),
                          "reason": why})
    markets = markets or PL.load_markets()
    slugs = slugs or PL.LeagueSlugs()
    now = now or _now()
    ents = [e for e in PL.entries(markets, repo=REPO, status="active") if e.get("family") == "team"]
    cols = list(board.columns)
    feat_cols = (cols[cols.index(FEATURE_FIRST): cols.index(FEATURE_LAST) + 1]
                 if FEATURE_FIRST in cols and FEATURE_LAST in cols else [])
    feat = board[feat_cols].to_numpy() if feat_cols else None
    rows = []
    for i, r in enumerate(board.to_dict("records")):
        model_id = TRACK_MODEL.get(str(r.get("model_type", "")).strip())
        ko = PL.utc_ts(r.get("kickoff_utc"))
        at = PL.utc_ts(r.get("generated_at")) or pd.Timestamp(now)
        if ko is None:
            _drop(r, "no kickoff_utc on the board row (cannot be a pre-kickoff snapshot)")
            continue                                   # no kickoff: cannot be a pre-kickoff snapshot
        mtk = (ko - at).total_seconds() / 60.0
        due = PL.snapshots_due(mtk)
        if model_id is None:
            _drop(r, f"model_type {PL._s(r.get('model_type')) or 'missing'!r} is not a v9 track")
            continue
        if not due:
            _drop(r, "predicted at or after kickoff (no pre-kickoff snapshot due)")
            continue
        lentry = slugs.entry(r.get("league"))
        league = slugs.canonical(r.get("league")) or PL._s(r.get("league"))
        recipe = PL._s(r.get("serving_recipe")) or "*"
        fhash = _hash_values(feat[i]) if feat is not None else ""
        oa = PL._s(r.get("_oa_event_id"))
        date = PL._s(r.get("date"))[:10] or ko.strftime("%Y-%m-%d")
        fref = PL.fixture_ref(None, oa, (lentry["id"] if lentry else "", date, r.get("home_team"),
                                         r.get("away_team")))
        for e in ents:
            if e["model_id"] != model_id:
                continue
            p = pd.to_numeric(pd.Series([r.get(e["prob_col"])]), errors="coerce").iloc[0]
            ver = PL._s(r.get(e["version_col"])) if e.get("version_col") else ""
            mfile = _model_file(e, recipe)
            reason = "" if pd.notna(p) else (
                f"board row has no {e['prob_col']} (model_type={r.get('model_type')}, "
                f"serving_recipe={PL._s(r.get('serving_recipe')) or 'n/a'})")
            for sel in e["selections"]:
                prob = None if pd.isna(p) else (float(p) if sel == e["positive_selection"] else 1.0 - float(p))
                for snap in due:
                    rows.append(PL.make_row(
                        repo=REPO, model_id=model_id, model_version=ver, model_sha=file_sha(mfile),
                        payload_version=PL._s(r.get("serving_recipe")), feature_hash=fhash,
                        track=e.get("track"), market=e["market"], line=e.get("line"), selection=sel,
                        fixture_ref=fref, fixture_id="", oddsapi_event_id=oa, league=league,
                        season=PL.season_of(lentry, ko), kickoff_utc=PL.iso(ko),
                        home_team=r.get("home_team"), away_team=r.get("away_team"),
                        probability=prob, snapshot=snap, predicted_at=PL.iso(at),
                        minutes_to_kickoff=round(mtk, 1), not_scored_reason=reason,
                        git_sha=PL._s(r.get("git_sha"))[:12] or _git_sha(), run_id=_run_id()))
    return rows


#: Identity / lineage columns read as TEXT: an OddsAPI id or a sha like '8e12...' must never be parsed
#: as a number (it would become 8e12 or inf and silently change the fixture key).
TEXT_COLS = ("_oa_event_id", "git_sha", "model_sha", "date", "kickoff_utc", "generated_at", "serving_recipe",
             "model_type", "league", "home_team", "away_team", "model_version", "model_version_btts",
             "model_version_over15", "model_version_over35", "model_version_ht_over05", "model_version_ht_over15")


def coverage_report(board: pd.DataFrame, rows: list, dropped: list, *, markets: dict | None = None,
                    slugs: PL.LeagueSlugs | None = None) -> dict:
    """Per run: did every fixture of every tracked league on the board get a ledger row per ACTIVE v9 team
    model x market? Statistics only (no prices). Lists
      * dropped: board fixtures with no row at all, with the reason (team_rows);
      * unscored: rows written without a probability, by reason (the model returned none - e.g. a required
        input missing; the row is kept so the gap is visible);
      * missing_cells: a written fixture lacking an active entry of its own track (must be 0);
      * tracked_leagues_without_board_fixtures: tracked registry leagues with no fixture on this board
        (no OddsAPI event / odds in the board window, or a provider gap - league_scope gives the reason)."""
    markets = markets or PL.load_markets()
    slugs = slugs or PL.LeagueSlugs()
    ents = [e for e in PL.entries(markets, repo=REPO, status="active") if e.get("family") == "team"]
    want = {}
    for e in ents:
        want.setdefault(e["model_id"], set()).add(e["market"])
    have: dict = {}
    unscored: Counter = Counter()
    for r in rows:
        have.setdefault((r["fixture_ref"], r["model_id"]), set()).add(r["market"])
        why = PL._s(r.get("not_scored_reason"))
        if why and r.get("selection") and r.get("snapshot") == "first_sight":
            k = f"{r['model_id']}|{r['market']}: {why.split(' (')[0]}"
            unscored[k] += 1
    missing = [{"fixture_ref": f, "model_id": m, "markets": sorted(want.get(m, set()) - got)}
               for (f, m), got in have.items() if want.get(m, set()) - got]
    on_board = {slugs.canonical(x) or PL._s(x) for x in board.get("league", pd.Series(dtype=str)).tolist()}
    scope = PL.league_scope(slugs)
    no_board = [{"league": s["league"], "reason": s["not_scored_reason"] or
                 "no fixture of this tracked league on this board (no OddsAPI event with odds in the window)"}
                for s in scope if s["scope"].startswith("tracked") and s["league"] not in on_board]
    rep = {"board_fixtures": int(len(board)), "fixtures_with_rows": len({f for f, _ in have}),
           "active_team_entries": {m: sorted(v) for m, v in want.items()},
           "dropped": dropped, "dropped_n": len(dropped), "unscored_first_sight_rows": dict(unscored),
           "missing_cells": missing, "missing_cells_n": len(missing),
           "tracked_leagues_without_board_fixtures": no_board}
    return rep


def _write_coverage(root: Path, rep: dict, now: datetime | None = None) -> Path:
    now = now or _now()
    p = Path(root) / REPO / "_coverage" / "team_board" / f"{now:%Y-%m}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), **rep}, sort_keys=True, default=str)
    with p.open("ab") as fh:
        fh.write(line.encode("utf-8") + b"\n")
    return p


def read_board(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or BOARD, low_memory=False, dtype={c: str for c in TEXT_COLS},
                       keep_default_na=True)


def record_board(path: Path | None = None, *, root: Path | None = None, now: datetime | None = None,
                 force: bool = False) -> dict:
    """Append the board's due snapshots. No-op (and no file touched) unless the flag is on."""
    if not (force or PL.enabled()):
        return {"skipped": f"{PL.FLAG_ENV} is not 'true'"}
    board = read_board(path)
    cov: dict = {}
    rows = team_rows(board, now=now, coverage=cov)
    stats = PL.append(rows, repo=REPO, writer="team_board", run_id=_run_id(), root=root, now=now)
    # coverage check (owner rule 2026-11-18): reported every run, never blocks the predict run
    try:
        rep = coverage_report(board, rows, cov.get("dropped", []))
        _write_coverage(Path(root or PL.LEDGER_DIR), rep, now)
        stats["coverage"] = {k: rep[k] for k in ("board_fixtures", "fixtures_with_rows", "dropped_n",
                                                  "missing_cells_n")}
        if rep["dropped_n"] or rep["missing_cells_n"]:
            why = dict(Counter(d["reason"] for d in rep["dropped"]))
            print(f"::warning title=Prediction ledger coverage::{rep['dropped_n']} board fixture(s) without a "
                  f"row {why}; {rep['missing_cells_n']} fixture x model with a missing market")
    except Exception as e:                                              # noqa: BLE001
        print(f"::warning title=Prediction ledger coverage::check failed: {type(e).__name__}: {e}")
    return stats


# ── PROPS: recorder called from player_model/predict.py ─────────────────────────────────────────

class PropsRecorder:
    """Collects every scored player x market of a props predict run; `flush()` appends the due
    snapshots. Holds the model's OWN probability (predict_proba of the serving payload), before the
    team-strength multiplier, caps and the tip filters. Never raises into predict."""

    def __init__(self, *, now: datetime | None = None):
        self.now = now
        self.rows: list = []
        self.markets = PL.load_markets()
        self.slugs = PL.LeagueSlugs()
        self.ents = {e["market"]: e for e in PL.entries(self.markets, repo=REPO, status="active")
                     if e.get("model_id") == "v9_props"}
        #: the parked clean ML challengers beside the served player base rate (owner 2026-11-18)
        self.ch_ents = {e["market"]: e for e in PL.entries(self.markets, repo=REPO, status="active")
                        if e.get("model_id") == CHALLENGER_MODEL_ID}
        self._parked: dict = {}

    def _challenger(self, market: str):
        """(payload, meta) of the market's parked challenger (player_model.props_gate.load_parked: bytes
        verified against their manifest), or None. Loaded once per run."""
        if market not in self._parked:
            try:
                from player_model.props_gate import load_parked
                self._parked[market] = load_parked(market)
            except Exception as ex:                                    # noqa: BLE001
                print(f"[prediction_ledger] props challenger {market}: not loaded ({type(ex).__name__}: {ex})")
                self._parked[market] = None
        return self._parked[market]

    def add(self, market: str, feat_df: pd.DataFrame, probs, *, league: str, kickoff_utc: str = "",
            fixture_id=None, home: str = "", away: str = "", payload: dict | None = None) -> None:
        try:
            e = self.ents.get(market)
            if e is None or feat_df is None or feat_df.empty:
                return
            now = self.now or _now()
            ko = PL.utc_ts(kickoff_utc)
            if ko is None:
                return
            mtk = (ko - pd.Timestamp(now)).total_seconds() / 60.0
            due = PL.snapshots_due(mtk)
            if not due:
                return
            mfile = _model_file(e, "*")
            sha = file_sha(mfile)
            recipe = str((payload or {}).get("recipe") or "legacy")
            ver = f"props_{market}_{recipe}_{sha[:8]}" if sha else ""
            fcols = list((payload or {}).get("feature_cols") or [])
            fvals = feat_df.reindex(columns=fcols).to_numpy() if fcols else None
            lentry = self.slugs.entry(league)
            canon = self.slugs.canonical(league) or league
            fref = PL.fixture_ref(fixture_id)
            p_arr = np.asarray(probs, dtype=float)
            pid = feat_df["player_id"].tolist() if "player_id" in feat_df else [None] * len(feat_df)
            pname = feat_df["player_name"].tolist() if "player_name" in feat_df else [""] * len(feat_df)
            team = feat_df["team"].tolist() if "team" in feat_df else [""] * len(feat_df)
            models = [("v9_props", ver, sha, recipe, e, p_arr, fvals)]
            ce, ch = self.ch_ents.get(market), None
            if ce is not None:
                ch = self._challenger(market)
            if ch is not None:
                from player_model.model import predict_proba
                cpl, cmeta = ch
                ccols = list(cpl.get("feature_cols") or [])
                models.append((CHALLENGER_MODEL_ID, cmeta.get("model_version", ""), cmeta.get("artifact_sha256", ""),
                               str(cpl.get("recipe") or ""), ce,
                               np.asarray(predict_proba(feat_df, cpl), dtype=float),
                               feat_df.reindex(columns=ccols).to_numpy() if ccols else None))
            for mid, mver, msha, mrec, me, mp, mf in models:
                for i in range(len(feat_df)):
                    p = mp[i] if i < len(mp) else float("nan")
                    ok = bool(np.isfinite(p))
                    for snap in due:
                        self.rows.append(PL.make_row(
                            repo=REPO, model_id=mid, model_version=mver, model_sha=msha,
                            payload_version=mrec, feature_hash=_hash_values(mf[i]) if mf is not None else "",
                            track="props", market=market, line=me.get("line"), selection=me["selections"][0],
                            fixture_ref=fref, fixture_id=fixture_id, oddsapi_event_id="", league=canon,
                            season=PL.season_of(lentry, ko), kickoff_utc=PL.iso(ko), home_team=home,
                            away_team=away, player_id=pid[i], player_name=pname[i], team=team[i],
                            probability=float(min(max(p, 0.0), 1.0)) if ok else None, snapshot=snap,
                            predicted_at=PL.iso(pd.Timestamp(now)), minutes_to_kickoff=round(mtk, 1),
                            not_scored_reason="" if ok else "props model returned no probability",
                            git_sha=_git_sha(), run_id=_run_id()))
        except Exception as ex:                                        # noqa: BLE001
            print(f"[prediction_ledger] props {market}: not recorded ({type(ex).__name__}: {ex})")

    def flush(self, root: Path | None = None) -> dict:
        try:
            if not self.rows:
                return {"offered": 0}
            st = PL.append(self.rows, repo=REPO, writer="props", run_id=_run_id(), root=root, now=self.now)
            self.rows = []
            print(f"[prediction_ledger] props: {st}")
            return st
        except Exception as ex:                                        # noqa: BLE001
            print(f"[prediction_ledger] props flush failed ({type(ex).__name__}: {ex})")
            return {"error": str(ex)}


def props_recorder():
    """The recorder predict.py uses, or None when the flag is off / the ledger cannot load."""
    if not PL.enabled():
        return None
    try:
        return PropsRecorder()
    except Exception as ex:                                            # noqa: BLE001
        print(f"[prediction_ledger] props recorder unavailable ({type(ex).__name__}: {ex})")
        return None


# ── SETTLE ───────────────────────────────────────────────────────────────────────────────────────

class TeamResults:
    """Played scores from the committed history banks (football-data first, API-Football for the
    leagues football-data does not carry). Matched on registry league + date (+-1 day) + team names
    (exact, else src/team_names.resolve_result).

    STANDARD-LEAGUE GAP (fixed 2026-11-22). The Standard leagues are in neither API-Football bank above
    (af_history carries only the new-format / AF-only leagues), and football-data uploads a round days
    after it is played, so every v9_standard row waited for that upload (2,964 of 2,964 past-kickoff
    rows PENDING on 2026-11-22). Two more VERIFIED provider banks are now read AFTER those two, each on
    its own: af_results_recent (API-Football 90-minute + HT score of every finished fixture) and
    af_team_cards_std (API-Football FT score, status FT only, no HT). When football-data uploads the
    match it is searched first and wins, as before. A row that carries an API-Football fixture id
    (fixture_ref af:<id>) is matched on that id in the API-Football banks before names."""

    def __init__(self, slugs: PL.LeagueSlugs | None = None):
        self.slugs = slugs or PL.LeagueSlugs()
        frames = []
        # Go-live merge 2026-11-17 (data-truth-100 fix16/readers): a settlement reader reads the history AS THE
        # LOADER SERVES IT - the owner's corrections (90-minute scores, proven cup / awarded rows and duplicate
        # copies out, provider dates) through serve_history / apply_af_history_corrections, never the raw banks.
        if FD_HISTORY.exists():
            from src.history_corrections import serve_history
            fd = serve_history(pd.read_parquet(FD_HISTORY), tag="prediction_ledger.fd_history")
            fd = fd.rename(columns={"ht_home_goals": "hth", "ht_away_goals": "hta"})
            fd["src"] = "football-data"
            frames.append(fd[[c for c in ("league", "date", "home_team", "away_team", "home_goals",
                                          "away_goals", "hth", "hta", "src") if c in fd.columns]])
        if AF_HISTORY.exists():
            from src.af_history_corrections import apply_af_history_corrections
            af = apply_af_history_corrections(pd.read_parquet(AF_HISTORY), tag="prediction_ledger.af_history")
            af = af.rename(columns={"FTHG": "home_goals", "FTAG": "away_goals"})
            af["src"] = "api-football"
            frames.append(af[["league", "date", "home_team", "away_team", "home_goals", "away_goals", "src"]])
        if AF_RESULTS.exists():
            r = pd.read_parquet(AF_RESULTS)
            # score.fulltime of a finished fixture = the regular-time (90-minute) score, also after AET / PEN
            r = r[r["status"].isin(["FT", "AET", "PEN"])].rename(columns={
                "match_date": "date", "ft_home": "home_goals", "ft_away": "away_goals", "ht_home": "hth",
                "ht_away": "hta"})
            r["src"] = "api-football-results"
            frames.append(r[["league", "date", "home_team", "away_team", "home_goals", "away_goals", "hth",
                             "hta", "fixture_id", "src"]])
        if AF_STD_CARDS.exists():
            c = pd.read_parquet(AF_STD_CARDS)
            # `goals` there include extra time: only status FT (no extra time played) is a 90-minute score
            c = c[c["status"] == "FT"].rename(columns={"match_date": "date", "FTHG": "home_goals",
                                                       "FTAG": "away_goals"})
            c["src"] = "api-football-std-cards"
            frames.append(c[["league", "date", "home_team", "away_team", "home_goals", "away_goals",
                             "fixture_id", "src"]])
        frames = [f.astype({k: "float64" for k in ("home_goals", "away_goals", "hth", "hta") if k in f.columns})
                  for f in frames]
        d = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["league", "date", "home_team", "away_team", "home_goals", "away_goals", "src"])
        if "fixture_id" not in d.columns:
            d["fixture_id"] = np.nan
        d["slug"] = d["league"].map(self.slugs.slug)
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        d = d.dropna(subset=["slug", "date", "home_goals", "away_goals"])
        self.srcs = [x for x in self.SOURCES if x in set(d["src"])]
        self.by = {k: g for k, g in d.groupby(["slug", "date", "src"])}
        fid = pd.to_numeric(d["fixture_id"], errors="coerce")
        self.by_fid = {(s, int(f)): r for s, f, r in zip(d["src"], fid, d.to_dict("records"))
                       if pd.notna(f) and s in self.ID_SOURCES}

    #: search order; football-data wins whenever it carries the match
    SOURCES = ("football-data", "api-football", "api-football-results", "api-football-std-cards")
    #: banks keyed by the API-Football fixture id
    ID_SOURCES = ("api-football-results", "api-football-std-cards")

    def find(self, slug: str, kickoff, home: str, away: str, fixture_id=None, need_ht: bool = False):
        """The first bank's row for the match (SOURCES order), or None. need_ht: a bank whose row has no
        half-time score is passed over for the next bank (an HT market waits for a bank that HAS it; the
        HT score is never inferred)."""
        ko = PL.utc_ts(kickoff)
        if ko is None:
            return None
        fid = pd.to_numeric(pd.Series([fixture_id]), errors="coerce").iloc[0]
        for src in self.srcs:
            m = self._find_in(src, slug, ko, home, away, fid)
            if m is None:
                continue
            if need_ht and (pd.isna(m.get("hth")) or pd.isna(m.get("hta"))):
                continue
            return m                                 # football-data is searched first and wins
        return None

    def _find_in(self, src: str, slug: str, ko, home: str, away: str, fid):
        from src import team_names as TN
        if src in self.ID_SOURCES and pd.notna(fid):
            m = self.by_fid.get((src, int(fid)))
            if m is not None and m.get("slug") == slug:
                return m
        # each bank on its own (the banks spell clubs differently; a pair is never split across them)
        for off in (0, -1, 1):
            g = self.by.get((slug, (ko + pd.Timedelta(days=off)).strftime("%Y-%m-%d"), src))
            if g is None:
                continue
            m = g[(g["home_team"] == home) & (g["away_team"] == away)]
            if m.empty:
                rh = TN.resolve_result(home, g["home_team"].tolist())
                ra = TN.resolve_result(away, g["away_team"].tolist())
                m = g[(g["home_team"] == rh) & (g["away_team"] == ra)] if rh and ra else m
            if len(m):
                return m.iloc[0].to_dict()
        return None


class PropsResults:
    """Per-player targets from player_history.parquet (the props models' own training targets)."""

    def __init__(self, fixture_ids: set):
        cols = ["fixture_id", "player_id", "minutes", "target_goals", "target_goals2", "target_assists",
                "target_sot", "target_sot2", "target_sot3", "target_sot4", "target_cards", "goals"]
        parts = sorted(PLAYER_HISTORY.glob("*.parquet")) if PLAYER_HISTORY.is_dir() else (
            [PLAYER_HISTORY] if PLAYER_HISTORY.exists() else [])
        frames = []
        fids = {int(f) for f in fixture_ids}
        for p in parts[-4:]:                          # the ledger starts 2026-11; recent months suffice
            try:
                d = pd.read_parquet(p, columns=[c for c in cols])
            except Exception:                                          # noqa: BLE001
                continue
            d = d[pd.to_numeric(d["fixture_id"], errors="coerce").isin(fids)]
            frames.append(d)
        d = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
        d["fixture_id"] = pd.to_numeric(d["fixture_id"], errors="coerce").astype("Int64")
        d["player_id"] = pd.to_numeric(d["player_id"], errors="coerce").astype("Int64")
        # goals3 (hat-trick) has no stored target: the same _label rule from the provider's goals count
        # (missing count -> missing label, never 0)
        g = pd.to_numeric(d["goals"], errors="coerce") if "goals" in d.columns else pd.Series(np.nan, index=d.index)
        d["target_goals3"] = (g >= 3).astype(float).where(g.notna())
        self.fixtures = set(d["fixture_id"].dropna().astype(int))
        self.rows = {(int(a), int(b)): r for a, b, r in zip(d["fixture_id"], d["player_id"],
                                                            d.to_dict("records")) if pd.notna(a) and pd.notna(b)}


#: Owner settlement rule: a fixture moved by more than 48 h from the kickoff a snapshot was made for is
#: VOID for that snapshot (the opinion was about a different match day).
RESCHEDULE_VOID_H = 48

#: Model ids settled by src/side_models/ledger.py (match statistics), never by settle() below.
SIDE_MODEL_IDS = ("v9_corners", "v9_cards")


def latest_kickoffs(root: Path) -> dict:
    """{fixture_ref: kickoff of its most recent snapshot} over every v9 partition (reschedules)."""
    best: dict = {}
    for p in PL.partitions(root):
        if PL.parse_partition(root, p)["repo"] != REPO:
            continue
        d = pd.read_csv(p, usecols=["fixture_ref", "kickoff_utc", "predicted_at"], dtype=str,
                        keep_default_na=False)
        for ref, ko, at in zip(d["fixture_ref"], d["kickoff_utc"], d["predicted_at"]):
            if ref and (ref not in best or at > best[ref][0]):
                best[ref] = (at, ko)
    return {k: v[1] for k, v in best.items()}


def outcome_fn(markets: dict | None = None, *, now: datetime | None = None,
               team: TeamResults | None = None, props: PropsResults | None = None,
               latest_ko: dict | None = None):
    idx = PL.entry_index(markets)
    latest_ko = latest_ko or {}
    now_ts = pd.Timestamp(now or _now())
    stamp = PL.iso(now_ts)

    def pending_or_missing(ko, why):
        if ko is not None and (now_ts - ko).total_seconds() / 3600 >= NO_RESULT_AFTER_H:
            return {"outcome_status": "NO_RESULT", "outcome_value": why}
        return {"outcome_status": "PENDING"}

    def outcome_of(r: dict) -> dict:
        e = idx.get((r["repo"], r["model_id"], r["market"]))
        ko_snap = PL.utc_ts(r.get("kickoff_utc"))
        ko = PL.utc_ts(latest_ko.get(r.get("fixture_ref"))) or ko_snap
        if e is None:
            return {"outcome_status": "UNREGISTERED"}
        if ko is not None and ko_snap is not None and                 abs((ko - ko_snap).total_seconds()) / 3600 > RESCHEDULE_VOID_H:
            return {"outcome_status": "VOID_RESCHEDULED",
                    "outcome_value": f"kickoff moved {PL.iso(ko_snap)} -> {PL.iso(ko)}"}
        if ko is None or ko > now_ts:
            return {"outcome_status": "PENDING"}
        st = e.get("settle", {})
        if st.get("rule") == "player_target":
            if props is None:
                return pending_or_missing(ko, "no props results loaded")
            fid, pid = PL._s(r.get("fixture_id")), PL._s(r.get("player_id"))
            if not fid or not pid:
                return pending_or_missing(ko, "no fixture/player id")
            if int(fid) not in props.fixtures:
                return pending_or_missing(ko, "fixture not in player_history yet")
            pr = props.rows.get((int(fid), int(pid)))
            mins = None if pr is None else pd.to_numeric(pd.Series([pr.get("minutes")]), errors="coerce").iloc[0]
            if pr is None or pd.isna(mins) or float(mins) <= 0:
                return {"outcome_status": "VOID_DID_NOT_PLAY", "result_source": "player_history",
                        "settled_at": stamp}
            t = pd.to_numeric(pd.Series([pr.get(st["target_col"])]), errors="coerce").iloc[0]
            if pd.isna(t):
                return {"outcome_status": "NULL_STAT", "outcome_value": f"{st['target_col']} missing",
                        "result_source": "player_history", "settled_at": stamp}
            return {"outcome_status": "SETTLED", "y": int(t >= 1), "outcome_value": int(t),
                    "result_source": "player_history", "settled_at": stamp}
        if team is None:
            return pending_or_missing(ko, "no team results loaded")
        ht = st.get("period", "ft") == "ht"
        m = team.find(r["league_slug"], PL.iso(ko), r["home_team"], r["away_team"],
                      fixture_id=r.get("fixture_id") if str(r.get("fixture_ref", "")).startswith("af:") else None,
                      need_ht=ht)
        if m is None:
            return pending_or_missing(ko, "no HT score in the history banks" if ht else
                                      "no result in the history banks")
        y, val = PL.match_outcome(st, r["selection"], r.get("line"), m.get("home_goals"), m.get("away_goals"),
                                  m.get("hth"), m.get("hta"))
        if y is None:
            return {"outcome_status": "NO_RESULT", "outcome_value": val, "result_source": m.get("src")}
        return {"outcome_status": "SETTLED", "y": y, "home_goals": int(m["home_goals"]),
                "away_goals": int(m["away_goals"]), "outcome_value": val, "result_source": m.get("src"),
                "settled_at": stamp}
    return outcome_of


def settle(*, root: Path | None = None, settled_root: Path | None = None, now: datetime | None = None,
           force: bool = False) -> dict:
    if not (force or PL.enabled()):
        return {"skipped": f"{PL.FLAG_ENV} is not 'true'"}
    root = Path(root or PL.LEDGER_DIR)
    fids: set = set()
    for p in PL.partitions(root):
        part = PL.parse_partition(root, p)
        if part["repo"] == REPO and part["model_id"] in ("v9_props", CHALLENGER_MODEL_ID):
            f = pd.read_csv(p, usecols=["fixture_id"], dtype=str, keep_default_na=False)["fixture_id"]
            fids |= {int(x) for x in f if x}
    # The corners / cards partitions (v9_corners / v9_cards) settle from match STATISTICS, which only the
    # side-models settle step reads (src/side_models/ledger.py settle, run after this one in
    # update_results.yml). Settling them here would overwrite that view with "no statistics" rows.
    models = tuple(sorted({PL.parse_partition(root, p)["model_id"] for p in PL.partitions(root)
                           if PL.parse_partition(root, p)["repo"] == REPO} - set(SIDE_MODEL_IDS)))
    if not models:
        return {"partitions": 0, "rows": 0, "written": 0, "settled": 0}
    fn = outcome_fn(now=now, team=TeamResults(), props=PropsResults(fids) if fids else None,
                    latest_ko=latest_kickoffs(root))
    return PL.write_settled(fn, repo=REPO, root=root, settled_root=settled_root, models=models)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="v9 prediction ledger writers (learning only)")
    ap.add_argument("cmd", choices=["board", "settle", "registry", "validate"])
    ap.add_argument("--force", action="store_true", help="run even with the flag off (local dry runs)")
    ap.add_argument("--root", default=None, help="ledger root (default output/prediction_ledger)")
    ap.add_argument("--settled-root", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root) if a.root else None
    sroot = Path(a.settled_root) if a.settled_root else None
    if a.cmd == "validate":
        issues = PL.validate(root)
        for i in issues:
            print(f"PREDICTION LEDGER VIOLATION: {i}")
        print(f"prediction ledger: {len(PL.partitions(root))} partition(s), {len(issues)} violation(s)")
        return 1 if issues else 0
    if not (a.force or PL.enabled()):
        print(f"[prediction_ledger] {PL.FLAG_ENV} is not 'true' - nothing written")
        return 0
    if a.cmd == "board":
        print(f"[prediction_ledger] board: {record_board(root=root, force=True)}")
    elif a.cmd == "settle":
        print(f"[prediction_ledger] settle: {settle(root=root, settled_root=sroot, force=True)}")
    elif a.cmd == "registry":
        reg = PL.write_registry(path=(root / 'registry.json') if root else None, root=root, settled_root=sroot)
        print(f"[prediction_ledger] registry: {len(reg['models'])} model x market entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
