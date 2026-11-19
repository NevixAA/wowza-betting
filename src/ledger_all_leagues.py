"""Ledger-only predictions for the TRAINING_ONLY Standard leagues - LEARNING ONLY, never a tip.
=============================================================================================

    python -m src.ledger_all_leagues board     # predict.yml: due snapshots -> prediction ledger
    python -m src.ledger_all_leagues preview   # same scoring, prints a summary, writes nothing

OWNER 2026-11-19 ("So do it"; earlier rule "predict every match ... not as a bet"): the Standard models
predict every match of the ten TRAINING_ONLY Standard leagues (Scottish Premiership / Championship /
League One / League Two, Belgian First Division A, Dutch Eredivisie, Greek Super League, Portuguese
Primeira Liga, Turkish Super Lig, National League) into the prediction ledger, in every team market they
serve (O/U 1.5 / 2.5 / 3.5, BTTS, HT 0.5 / 1.5). The Standard models were trained on these leagues
(config.STANDARD_FORMAT_LEAGUES), so serving them is in-distribution.

SOURCES (no new paid call, by construction)
  * fixtures: output/collect/upcoming_fixtures.parquet - API-Football fixtures the daily collector
    (daily_injuries_lineups.yml) already fetches for every registry league; NS / TBD, next 7 days
    (the board's window, pipeline.mode_predict days_ahead=7).
  * team identity: the API-Football team id -> the club's name in the served history, PROVEN by
    results: finished fixtures of output/strength/competition_fixtures.parquet that meet a served
    history match on (league, date, full-time score) where that key is unique in both sources. The
    latest MIN_AGREE pieces of evidence must all name the same club and no other team id of the league
    may claim it. Anything else is NOT guessed: the fixture is stored unscored with the reason.
  * features + probabilities: src.predict.serve_fixtures - the very code predict_upcoming scores the
    board with (build_upcoming_features + the HT block + every model's routing), on the history
    data_loader.load_all_matches serves. The process blocks the network (socket) while it runs, so
    none of the display-only API enrichments in build_upcoming_features can spend a call; no model
    feature comes from them (tests/test_ledger_all_leagues.py checks the payloads' feature_cols).

WHAT IT WRITES - and nothing else
  * output/prediction_ledger/v9/v9_standard/<market>/<league>/<season>.csv (append-only, first write
    wins), fixture_ref 'af:<API-Football fixture id>', payload_version '<recipe>|ledger_only:<state>'
    (the league's registry production_state is recorded on every row);
  * output/prediction_ledger/v9/_coverage/ledger_only/<YYYY-MM>.jsonl (every fixture, with the reason
    when it got no probability);
  * output/ledger_only/board.csv - the model probabilities of the fixtures in the window (no price
    column: no price is read at all), the input of Pro's ledger-only score grid.
It never reads or writes predictions.csv, bets.csv, side_bets.csv, any tip / tier / stake / TOP-3 /
Telegram / dashboard-tip file: these leagues stay out of every bet path by construction (their
production_state keeps them out of config.ENABLED_LEAGUES, so the OddsAPI board never carries them).

SWITCH: writes only when the repo variable LEDGER_ALL_LEAGUES_ENABLED is 'true' AND the ledger is on
(PREDICTION_LEDGER_ENABLED). Cheap when nothing is due: the history is loaded only when a fixture has a
snapshot (first_sight / final) the ledger does not hold yet. Never raises into the workflow.
"""
from __future__ import annotations

import json
import os
import socket
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import prediction_ledger as PL  # noqa: E402
from src import prediction_ledger_v9 as V  # noqa: E402

FLAG_ENV = "LEDGER_ALL_LEAGUES_ENABLED"
#: registry production states predicted into the ledger only (never tipped)
LEDGER_ONLY_STATES = ("TRAINING_ONLY",)
#: the tracks whose served models cover them (the Standard models were trained on these leagues)
LEDGER_ONLY_MODEL_TYPES = ("standard",)
UPCOMING = ROOT / "output" / "collect" / "upcoming_fixtures.parquet"
AF_FIXTURES = ROOT / "output" / "strength" / "competition_fixtures.parquet"
BOARD_OUT = ROOT / "output" / "ledger_only" / "board.csv"
UPCOMING_STATUSES = ("NS", "TBD")
HORIZON_DAYS = 7
WRITER = "ledger_only"
#: identity proof: the latest MIN_AGREE unique-key result matches must all name the same club
MIN_AGREE = 3
#: evidence window for the identity proof (clubs are renamed; old spellings stay in the history)
EVIDENCE_SINCE = "2022-07-01"
#: board columns Pro's grid and the audit read (probabilities and identity only - never a price)
BOARD_COLS = ["fixture_id", "league", "league_state", "date", "kickoff_utc", "home_team", "away_team",
              "af_home_id", "af_away_id", "af_home_name", "af_away_name", "model_type", "serving_recipe",
              "p_over25", "p_over15", "p_over35", "p_btts", "p_ht_over05", "p_ht_over15",
              "model_version", "model_version_over15", "model_version_over35", "model_version_btts",
              "model_version_ht_over05", "model_version_ht_over15", "generated_at", "git_sha",
              "not_scored_reason"]


def enabled() -> bool:
    return os.getenv(FLAG_ENV, "").strip().lower() == "true" and PL.enabled()


def _now() -> datetime:
    return datetime.now(timezone.utc)


@contextmanager
def network_blocked():
    """No socket may connect inside this block (no API call, paid or free, can happen)."""
    orig_connect, orig_create = socket.socket.connect, socket.create_connection

    def _no_net(*a, **k):
        raise OSError("network blocked (ledger-only writer: no API call)")
    socket.socket.connect = _no_net
    socket.create_connection = _no_net
    try:
        yield
    finally:
        socket.socket.connect, socket.create_connection = orig_connect, orig_create


# ── leagues + fixtures ───────────────────────────────────────────────────────────────────────────

def ledger_only_leagues(slugs: PL.LeagueSlugs | None = None) -> list:
    """Registry entries predicted into the ledger only: TRAINING_ONLY leagues of a served track."""
    slugs = slugs or PL.LeagueSlugs()
    return [e for e in slugs.leagues
            if e.get("production_state") in LEDGER_ONLY_STATES
            and e.get("model_type") in LEDGER_ONLY_MODEL_TYPES
            and e.get("api_football_id") is not None]


def upcoming(now: datetime, leagues: list, path: Path | None = None,
             horizon_days: float = HORIZON_DAYS) -> pd.DataFrame:
    """NS / TBD API-Football fixtures of `leagues` kicking off in (now, now + horizon]."""
    cols = ["fixture_id", "league", "league_id", "league_state", "kickoff_utc", "date", "home_id", "away_id"]
    u = pd.read_parquet(path or UPCOMING)
    u = u[u["status"].isin(UPCOMING_STATUSES)].drop_duplicates("fixture_id")
    by_af = {int(e["api_football_id"]): e for e in leagues}
    u = u[pd.to_numeric(u["league_id"], errors="coerce").isin(list(by_af))].copy()
    ko = pd.to_datetime(u["kickoff"], utc=True, errors="coerce")
    t0 = pd.Timestamp(now)
    u = u[(ko > t0) & (ko <= t0 + pd.Timedelta(days=horizon_days))].copy()
    if u.empty:
        return pd.DataFrame(columns=cols)
    u["kickoff_utc"] = pd.to_datetime(u["kickoff"], utc=True)
    u["league"] = [by_af[int(x)]["name"] for x in u["league_id"]]
    u["league_state"] = [by_af[int(x)]["production_state"] for x in u["league_id"]]
    u["date"] = u["kickoff_utc"].dt.strftime("%Y-%m-%d")
    for c in ("fixture_id", "league_id", "home_id", "away_id"):
        u[c] = u[c].astype(int)
    return u[cols].sort_values(["kickoff_utc", "fixture_id"]).reset_index(drop=True)


def af_names(fixtures: pd.DataFrame) -> dict:
    """{API-Football team id: its latest API-Football name} (display / unresolved rows only)."""
    long = pd.concat([fixtures[["date", "home_id", "home_name"]].set_axis(["date", "id", "name"], axis=1),
                      fixtures[["date", "away_id", "away_name"]].set_axis(["date", "id", "name"], axis=1)])
    long = long.dropna(subset=["id"]).sort_values("date").drop_duplicates("id", keep="last")
    return dict(zip(long["id"].astype(int), long["name"].astype(str)))


def identity_map(history: pd.DataFrame, fixtures: pd.DataFrame, leagues: list,
                 since: str = EVIDENCE_SINCE, min_agree: int = MIN_AGREE) -> tuple[dict, dict]:
    """({(league, af_team_id): history club name}, {(league, af_team_id): reason it is not proven}).

    Evidence = a finished API-Football fixture and a served history match with the same league, date
    and full-time score, where that (league, date, score) is unique in BOTH sources (two 1-1 draws on
    one day prove nothing). The club a team id is mapped to is the one its latest `min_agree` pieces
    of evidence all name; a club claimed by two team ids of one league is refused for both."""
    names = {int(e["api_football_id"]): e["name"] for e in leagues}
    fx = fixtures[pd.to_numeric(fixtures["league_id"], errors="coerce").isin(list(names))].copy()
    fx = fx[pd.to_numeric(fx["ft_home"], errors="coerce").notna()
            & pd.to_numeric(fx["ft_away"], errors="coerce").notna()]
    fx["lg"] = fx["league_id"].astype(int).map(names)
    fx["d"] = pd.to_datetime(fx["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    fx["gh"], fx["ga"] = fx["ft_home"].astype(int), fx["ft_away"].astype(int)
    fx = fx[fx["d"] >= since]
    h = history[history["league"].isin(list(names.values()))].copy()
    h = h[pd.to_numeric(h["home_goals"], errors="coerce").notna()
          & pd.to_numeric(h["away_goals"], errors="coerce").notna()]
    h["lg"] = h["league"]
    h["d"] = pd.to_datetime(h["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    h["gh"], h["ga"] = h["home_goals"].astype(int), h["away_goals"].astype(int)
    h = h[h["d"] >= since]
    k = ["lg", "d", "gh", "ga"]
    fx1 = fx[~fx.duplicated(k, keep=False)]
    h1 = h[~h.duplicated(k, keep=False)]
    m = fx1.merge(h1[k + ["home_team", "away_team"]], on=k)
    ev: dict = defaultdict(list)                     # (league, id) -> [(date, club)]
    for lg, d, a, b in zip(m["lg"], m["d"], m["home_id"], m["home_team"]):
        ev[(lg, int(a))].append((d, str(b)))
    for lg, d, a, b in zip(m["lg"], m["d"], m["away_id"], m["away_team"]):
        ev[(lg, int(a))].append((d, str(b)))
    proven, why = {}, {}
    for key, lst in ev.items():
        last = [c for _, c in sorted(lst)[-min_agree:]]
        if len(last) < min_agree:
            why[key] = f"identity not proven: {len(last)} unique-key result match(es) since {since} (< {min_agree})"
        elif len(set(last)) != 1:
            why[key] = f"identity not proven: the latest {min_agree} result matches name {sorted(set(last))}"
        else:
            proven[key] = last[0]
    claimed = Counter((lg, c) for (lg, _), c in proven.items())
    for key in [k2 for k2, c in proven.items() if claimed[(k2[0], c)] > 1]:
        why[key] = f"identity not proven: {proven.pop(key)!r} is claimed by more than one API-Football team id"
    return proven, why


# ── due check (skip the history load when the ledger already holds every due snapshot) ─────────────

def _std_entries(markets: dict | None = None) -> list:
    return [e for e in PL.entries(markets or PL.load_markets(), repo=V.REPO, status="active")
            if e.get("family") == "team" and e.get("model_id") == V.TRACK_MODEL["standard"]]


def _versions() -> dict:
    """The version stamp of every Standard team market, as predict's provenance stamps the board."""
    from src import provenance
    stub = provenance.stamp_versions(pd.DataFrame({"model_type": ["standard"]}))
    return {c: PL._s(stub[c].iloc[0]) for c in stub.columns if c.startswith("model_version")}


def due_keys(fx: pd.DataFrame, now: datetime, *, slugs: PL.LeagueSlugs | None = None,
             markets: dict | None = None, versions: dict | None = None) -> dict:
    """{partition path: {ledger key}} of every row this run would offer (all selections, due snapshots)."""
    slugs = slugs or PL.LeagueSlugs()
    ents = _std_entries(markets)
    versions = versions if versions is not None else _versions()
    out: dict = defaultdict(set)
    for r in fx.to_dict("records"):
        ko = PL.utc_ts(r["kickoff_utc"])
        due = PL.snapshots_due((ko - pd.Timestamp(now)).total_seconds() / 60.0)
        lentry = slugs.entry(r["league"])
        season = PL.season_of(lentry, ko)
        if not due or lentry is None or not season:
            continue
        fref = PL.fixture_ref(r["fixture_id"])
        for e in ents:
            ver = versions.get(e.get("version_col") or "", "")
            p = PL.partition_path(PL.LEDGER_DIR, V.REPO, e["model_id"], e["market"], lentry["id"], season)
            for sel in e["selections"]:
                for snap in due:
                    out[p].add(PL.key_of({"model_id": e["model_id"], "model_version": ver, "market": e["market"],
                                          "line": PL._line(e.get("line")), "selection": sel,
                                          "fixture_ref": fref, "player_id": "", "snapshot": snap}))
    return out


def missing_due(fx: pd.DataFrame, now: datetime, root: Path | None = None, **kw) -> int:
    """How many due rows the ledger at `root` does not hold yet."""
    root = Path(root or PL.LEDGER_DIR)
    n = 0
    for p, keys in due_keys(fx, now, **kw).items():
        rp = root / Path(p).relative_to(PL.LEDGER_DIR)
        n += len(keys - PL._read_keys(rp))
    return n


# ── scoring ────────────────────────────────────────────────────────────────────────────────────────

def board_frame(fx: pd.DataFrame, ident: dict, why: dict, names: dict) -> pd.DataFrame:
    """The board-shaped frame src.predict.serve_fixtures scores (no price: odds columns are NaN)."""
    rows = []
    for r in fx.to_dict("records"):
        lg, hid, aid = r["league"], int(r["home_id"]), int(r["away_id"])
        h, a = ident.get((lg, hid)), ident.get((lg, aid))
        reasons = [f"{side} team (API-Football id {tid}, {names.get(tid, '?')!r}): "
                   + why.get((lg, tid), "identity not proven: no unique-key result match in the served history")
                   for side, tid, club in (("home", hid, h), ("away", aid, a)) if club is None]
        ko = pd.Timestamp(r["kickoff_utc"]).tz_convert("UTC").tz_localize(None)
        rows.append({"league": lg, "league_state": r["league_state"], "date": ko.normalize(),
                     "kickoff_utc": ko, "home_team": h or names.get(hid, str(hid)),
                     "away_team": a or names.get(aid, str(aid)), "odds_over25": np.nan, "odds_under25": np.nan,
                     "fixture_id": int(r["fixture_id"]), "af_home_id": hid, "af_away_id": aid,
                     "af_home_name": names.get(hid, ""), "af_away_name": names.get(aid, ""),
                     "not_scored_reason": "; ".join(reasons)})
    return pd.DataFrame(rows)


def load_payloads() -> tuple[dict, dict | None, dict]:
    """The payloads pipeline.mode_predict loads (same files, same loader)."""
    import config
    from src.model import load_models
    std = load_models(model_file=config.MODEL_FILE_STANDARD)
    nf = load_models(model_file=config.MODEL_FILE_NEWFORMAT) if config.MODEL_FILE_NEWFORMAT.exists() else None
    side = {t: load_models(model_file=f) for t, f in config.SIDE_MARKETS.items() if f.exists()}
    return std, nf, side


def score(board: pd.DataFrame, historical: pd.DataFrame, payloads: tuple | None = None) -> pd.DataFrame:
    """Every model's probability for the resolved fixtures (serve_fixtures, network blocked); the
    unresolved ones keep NaN probabilities and their reason. provenance.stamp names the models."""
    from src import provenance
    from src.predict import serve_fixtures
    std, nf, side = payloads or load_payloads()
    ok = board["not_scored_reason"] == ""
    parts = []
    if ok.any():
        with network_blocked():
            feat = serve_fixtures(board[ok].drop(columns=["not_scored_reason"]).reset_index(drop=True),
                                  historical, std, nf, side, record_health=False)
        feat["not_scored_reason"] = ""
        parts.append(feat)
    if (~ok).any():
        bad = board[~ok].copy()
        bad["model_type"] = "standard"
        parts.append(bad)
    out = pd.concat(parts, ignore_index=True, sort=False)
    for c in ("p_over25", "p_over15", "p_over35", "p_btts", "p_ht_over05", "p_ht_over15"):
        if c not in out.columns:
            out[c] = np.nan
    return provenance.stamp(out)


def ledger_rows(scored: pd.DataFrame, now: datetime, *, slugs: PL.LeagueSlugs | None = None,
                coverage: dict | None = None) -> list:
    """src.prediction_ledger_v9.team_rows on the scored frame (the board writer's own row builder), then
    the API-Football identity (fixture_ref af:<id>) and the league state on every row."""
    slugs = slugs or PL.LeagueSlugs()
    frame = scored.copy()
    frame["kickoff_utc"] = [PL.iso(PL.utc_ts(x)) for x in frame["kickoff_utc"]]
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    rows = V.team_rows(frame, now=now, slugs=slugs, coverage=coverage)
    by_teams = {(PL._s(r["league"]), PL.iso(PL.utc_ts(r["kickoff_utc"])), PL._s(r["home_team"]),
                 PL._s(r["away_team"])): r for r in frame.to_dict("records")}
    out = []
    for row in rows:
        src = by_teams.get((PL._s(row["league"]), PL._s(row["kickoff_utc"]), PL._s(row["home_team"]),
                            PL._s(row["away_team"])))
        if src is None:
            continue
        row = dict(row)
        row["fixture_id"] = str(int(src["fixture_id"]))
        row["fixture_ref"] = PL.fixture_ref(src["fixture_id"])
        pv = PL._s(row.get("payload_version"))
        row["payload_version"] = (pv + "|" if pv else "") + f"ledger_only:{src['league_state']}"
        why = PL._s(src.get("not_scored_reason"))
        if why and row["scored"] == 0:
            row["not_scored_reason"] = why
        out.append(row)
    return out


def write_board(scored: pd.DataFrame, path: Path | None = None) -> Path:
    path = Path(path or BOARD_OUT)
    d = scored.copy()
    d["kickoff_utc"] = [PL.iso(PL.utc_ts(x)) for x in d["kickoff_utc"]]
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    for c in BOARD_COLS:
        if c not in d.columns:
            d[c] = ""
    d = d[BOARD_COLS]
    assert not [c for c in d.columns if PL.PRICE_LIKE.search(c)], "ledger-only board must carry no price"
    path.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(path, index=False, lineterminator="\n")
    return path


def run(now: datetime | None = None, *, write: bool = True, historical: pd.DataFrame | None = None,
        root: Path | None = None, board_path: Path | None = None, payloads: tuple | None = None,
        upcoming_path: Path | None = None, fixtures_path: Path | None = None) -> dict:
    now = now or _now()
    slugs = PL.LeagueSlugs()
    leagues = ledger_only_leagues(slugs)
    fx = upcoming(now, leagues, upcoming_path)
    summ = {"leagues": [e["name"] for e in leagues], "fixtures_in_window": int(len(fx)),
            "by_league": {k: int(v) for k, v in fx["league"].value_counts().sort_index().items()}}
    if fx.empty:
        summ["skipped"] = "no fixture of a ledger-only league in the window"
        return summ
    if write:
        n_missing = missing_due(fx, now, root=root, slugs=slugs)
        summ["due_rows_missing"] = int(n_missing)
        if not n_missing:
            summ["skipped"] = "every due snapshot is already in the ledger"
            return summ
    if historical is None:
        with network_blocked():
            from src.data_loader import load_all_matches
            historical = load_all_matches()
    afx = pd.read_parquet(fixtures_path or AF_FIXTURES)
    ident, why = identity_map(historical, afx, leagues)
    board = board_frame(fx, ident, why, af_names(afx))
    scored = score(board, historical, payloads)
    scored["generated_at"] = PL.iso(pd.Timestamp(now))       # the run's instant (the snapshot clock)
    cov: dict = {}
    rows = ledger_rows(scored, now, slugs=slugs, coverage=cov)
    p = pd.to_numeric(scored["p_over25"], errors="coerce")
    summ.update({"identity_unproven_fixtures": int((board["not_scored_reason"] != "").sum()),
                 "scored_fixtures": int(p.notna().sum()), "rows": len(rows),
                 "unscored_reasons": dict(Counter(PL._s(x).split(":")[0] for x in
                                                  scored.loc[p.isna(), "not_scored_reason"])),
                 "dropped": cov.get("dropped", [])})
    if not write:
        return summ
    st = PL.append(rows, repo=V.REPO, writer=WRITER, run_id=os.getenv("GITHUB_RUN_ID", ""), root=root, now=now)
    summ["append"] = {k: st[k] for k in ("offered", "written", "duplicate_skipped", "late_skipped", "refused")}
    write_board(scored, board_path)
    cp = Path(root or PL.LEDGER_DIR) / V.REPO / "_coverage" / "ledger_only" / f"{now:%Y-%m}.jsonl"
    cp.parent.mkdir(parents=True, exist_ok=True)
    with cp.open("ab") as fh:
        fh.write(json.dumps({"at": PL.iso(pd.Timestamp(now)), **summ}, sort_keys=True,
                            default=str).encode("utf-8") + b"\n")
    return summ


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "preview"
    try:
        if cmd == "board":
            if not enabled():
                print(f"[ledger_only] {FLAG_ENV} / {PL.FLAG_ENV} not both 'true': nothing written")
                return 0
            print(f"[ledger_only] {json.dumps(run(), default=str)}")
        elif cmd == "preview":
            print(json.dumps(run(write=False), indent=1, default=str))
        else:
            print(f"unknown command {cmd!r} (board | preview)")
            return 2
    except Exception as e:                                             # noqa: BLE001
        print(f"::warning::ledger-only predictions failed (learning only - tips unaffected): "
              f"{type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
