"""RETROSPECTIVE replay: legacy raw-edge selection vs market_edge_v2 at FIRST SIGHT (2026-10-25).

    python scripts/edge_engine_replay.py            # -> output/edge_engine_backtest.csv
                                                    #    output/edge_engine_comparison.json
                                                    #    output/edge_engine_proposed_params.json

Every number this module produces is RETROSPECTIVE. It can describe what the two engines would have
done on fixtures that are already played; it can never confirm either of them. The forward test that
can is preregistered in Pro (registry/edge2_hypotheses_v1.json) and reads output/edge_engine_shadow.csv.

INPUTS (all committed v9 data, nothing synthetic, nothing post-kickoff)
  board      every committed output/predictions.csv since 2026-08-14, read from git history. One row per
             fixture per predict run, carrying the model probabilities and the board prices production
             decided at. Observation time = the run's `generated_at` (capped at the commit time).
  first      FIRST SIGHT per fixture x market family: the earliest committed run, strictly before
             kickoff, that carries the family's model probability AND a real board price for it (OU25:
             both sides). Both engines are judged at that same instant - never at a later, better-
             informed run.
  quotes     output/book_odds_snapshots.csv + book_odds_snapshots_archive/ (OddsAPI, per book, a
             CHANGE-LOG). Pairs are formed ONLY within one book (src.edge_engine_shadow.book_pairs):
             last observation carried forward per book and side up to the entry instant, so a quote
             written after the entry instant can never reach the decision.
  results    output/af_history.parquet + fd_history.parquet, joined league-scoped through
             src.team_names.resolve_result (unique match within +-1 day or the fixture is unsettled).

ENGINES
  LEGACY  OU25: the tier production RECORDED at first sight (signal_tier / best_side in that run's
          predictions.csv) - by definition the legacy logic with the thresholds in force then. It is
          independently reproduced (`legacy_ou25_base_tier`) from the threshold regime of the time
          (predict.yml env, git b82cec9f 2026-08-21: VALUABLE 0.04->0.03, MARKSMAN 0.14->0.08, league
          SNIPER cap 1.0->0.12; models/best_params_standard.json read from git as of the entry time)
          and the reproduction rate is reported. BTTS-YES / OVER1.5: the legacy side-market rule
          (`p - (1/odds)/1.08`, bars 0.10/0.08/0.04 unless the league is `approved` in
          models/best_params_side_markets.json as of the entry time) - reproduced, because the board
          does not record side tiers.
  NEW     src.market_edge.evaluate with config/market_edge_v2.json EXACTLY as committed
          (UNVALIDATED_DEFAULTS). Never tuned here. Point-in-time inputs only: the track CLV record
          from ledger rows whose match date precedes the entry date, league history counted before
          the entry date, no prospective evidence file (none existed) -> STRONG unreachable.

CLV is CERTIFIED only: a close exists when the fixture/market's book archive has a write inside the
30 minutes before kickoff; the close is the per-book LOCF pair at kickoff, power de-vigged, median of
>= 3 two-sided books (src.market_anchor.MIN_BOOKS_FOR_MEDIAN). CLV% = entry_odds * p_close_fair - 1.
No close -> NaN, never an earlier price.
"""
from __future__ import annotations

import io
import json
import logging
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import market_edge as ME
from src.edge_engine_shadow import book_pairs

log = logging.getLogger(__name__)

LABEL = "RETROSPECTIVE"
GIT = os.environ.get("WOWZA_GIT", "git")
REPO = Path(__file__).resolve().parents[2]
TIERS = ("SNIPER", "MARKSMAN", "VALUABLE")
CLOSE_WINDOW_MIN = 30
CLOSE_MIN_BOOKS = 3
SIDE_OVERROUND = 1.08

#: family -> board columns. p is always P(side A).
FAMILY_SPEC = {
    "OU25": {"p": "p_over25", "a": "odds_over25", "b": "odds_under25", "ver": "model_version",
             "legacy": "ou25", "both_prices": True},
    "BTTS": {"p": "p_btts", "a": "odds_btts", "b": "odds_btts_no", "ver": "model_version_btts",
             "legacy": "btts", "both_prices": False},
    "OU15": {"p": "p_over15", "a": "odds_over15", "b": "odds_under15", "ver": "model_version_over15",
             "legacy": "over15", "both_prices": False},
}
#: The six track x market cells the brief asks for.
CELLS = [(t, f) for f in ("OU25", "BTTS", "OU15") for t in ("new_format", "standard")]

BOARD_COLS = ["league", "date", "kickoff_utc", "home_team", "away_team", "odds_over25", "odds_under25",
              "odds_over15", "odds_under15", "odds_over35", "odds_btts", "odds_btts_no", "p_over25",
              "p_btts", "p_over15", "p_over35", "model_type", "edge_over", "edge_under", "tier_over",
              "tier_under", "best_side", "best_edge", "signal_tier", "bet", "no_form_data",
              "drift_signal", "generated_at", "model_version", "model_version_btts",
              "model_version_over15", "_oa_event_id", "price_synthetic"]

#: Threshold regime of predict.yml env (git log -G of .github/workflows/predict.yml since 2026-08-10
#: shows ONE change: b82cec9f, committed 2026-08-21T19:06:04+03:00). Before it the config.py defaults
#: applied (no env override existed); after it the env values.
REGIMES = [
    (pd.Timestamp("2000-01-01T00:00:00Z"), {"valuable": 0.04, "marksman": 0.14, "cap": 1.0,
                                             "id": "config_defaults_pre_b82cec9f"}),
    (pd.Timestamp("2026-08-21T16:06:04Z"), {"valuable": 0.03, "marksman": 0.08, "cap": 0.12,
                                             "id": "predict_yml_env_b82cec9f"}),
]


# ═══════════════════════════════ git-backed inputs ═══════════════════════════════
def _git(*args: str, repo: Path = REPO) -> bytes:
    return subprocess.run([GIT, "-C", str(repo), *args], check=True, capture_output=True).stdout


def commits_touching(path: str, since: str = "2026-08-14", repo: Path = REPO) -> list[tuple[str, pd.Timestamp]]:
    out = _git("log", f"--since={since}", "--format=%H %cI", "--", path, repo=repo).decode()
    rows = []
    for line in out.splitlines():
        sha, ts = line.split(" ", 1)
        rows.append((sha, pd.Timestamp(ts).tz_convert("UTC")))
    return sorted(rows, key=lambda r: r[1])


def board_history(repo: Path = REPO, since: str = "2026-08-14", cache: Path | None = None) -> pd.DataFrame:
    """Every committed predictions.csv since `since`, one row per fixture per run. Cached by commit sha
    (a commit's content never changes, so a cache hit is exact)."""
    commits = commits_touching("output/predictions.csv", since, repo)
    have = pd.DataFrame()
    if cache is not None and Path(cache).exists():
        have = pd.read_parquet(cache)
    done = set(have["commit_sha"]) if len(have) else set()
    parts = [have] if len(have) else []
    for sha, cts in commits:
        if sha in done:
            continue
        try:
            raw = _git("show", f"{sha}:output/predictions.csv", repo=repo)
            d = pd.read_csv(io.BytesIO(raw), usecols=lambda c: c in BOARD_COLS, low_memory=False)
        except Exception as e:                                        # noqa: BLE001
            log.warning(f"[edge_replay] skip {sha[:8]}: {type(e).__name__}")
            continue
        d = d.reindex(columns=BOARD_COLS)
        d["commit_sha"], d["committed_at"] = sha, cts
        parts.append(d)
    b = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=BOARD_COLS)
    for c in ("generated_at", "model_version", "model_version_btts", "model_version_over15",
              "_oa_event_id", "tier_over", "tier_under", "best_side", "signal_tier", "bet",
              "drift_signal", "no_form_data", "price_synthetic", "model_type", "kickoff_utc", "date"):
        b[c] = b[c].astype("string")
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        b.to_parquet(cache, index=False)
    return b


def observation_time(b: pd.DataFrame) -> pd.Series:
    """When the run's opinion existed: generated_at, never later than the commit that published it."""
    g = pd.to_datetime(b["generated_at"], errors="coerce", utc=True, format="mixed")
    c = pd.to_datetime(b["committed_at"], utc=True)
    return g.where(g.notna() & (g <= c), c)


def fixture_key(b: pd.DataFrame) -> pd.Series:
    return (b["league"].astype(str) + "|" + b["date"].astype(str).str[:10] + "|"
            + b["home_team"].astype(str) + "|" + b["away_team"].astype(str))


def _flag(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def first_sight(board: pd.DataFrame, family: str) -> pd.DataFrame:
    """Earliest pre-kickoff run per fixture that carries the family's model probability and a real
    board price (OU25: both sides). Rows flagged `price_synthetic` are dropped before selection."""
    s = FAMILY_SPEC[family]
    b = board.copy()
    for c in b.columns:                       # pandas StringDtype NA -> None (NA has no truth value)
        if isinstance(b[c].dtype, pd.StringDtype):
            b[c] = b[c].astype(object).where(b[c].notna(), None)
    if "price_synthetic" in b.columns:
        b = b[~b["price_synthetic"].map(_flag)]
    b["obs_ts"] = observation_time(b)
    b["kickoff"] = pd.to_datetime(b["kickoff_utc"], errors="coerce", utc=True, format="mixed")
    p = pd.to_numeric(b[s["p"]], errors="coerce")
    oa = pd.to_numeric(b[s["a"]], errors="coerce")
    ob = pd.to_numeric(b.get(s["b"]), errors="coerce") if s["b"] in b else pd.Series(np.nan, index=b.index)
    ok = b["kickoff"].notna() & (b["obs_ts"] < b["kickoff"]) & p.between(0, 1, inclusive="neither") & (oa > 1)
    if s["both_prices"]:
        ok &= ob > 1
    b = b[ok].copy()
    b["fx"] = fixture_key(b)
    b = b.sort_values(["obs_ts", "commit_sha"], kind="mergesort").drop_duplicates("fx", keep="first")
    b["family"] = family
    b["track"] = b["model_type"].where(b["model_type"].isin(ME.TRACKS),
                                       b["league"].map(lambda lg: config.model_type_for_league(str(lg))))
    return b.reset_index(drop=True)


@lru_cache(maxsize=64)
def _artifact_versions(path: str, repo: str = str(REPO)) -> tuple:
    try:
        return tuple(commits_touching(path, "2026-01-01", Path(repo)))
    except Exception:                                                 # noqa: BLE001
        return ()


@lru_cache(maxsize=256)
def _artifact_json(path: str, sha: str, repo: str = str(REPO)) -> dict:
    try:
        return json.loads(_git("show", f"{sha}:{path}", repo=Path(repo)).decode("utf-8"))
    except Exception:                                                 # noqa: BLE001
        return {}


def artifact_at(path: str, ts: pd.Timestamp, repo: Path = REPO) -> dict:
    """The committed JSON `path` as it stood at `ts` (latest commit at or before it); {} if none."""
    vs = [sha for sha, cts in _artifact_versions(path, str(repo)) if cts <= ts]
    return _artifact_json(path, vs[-1], str(repo)) if vs else {}


# ═══════════════════════════════ quotes / close ═══════════════════════════════
def load_book_quotes(output_dir: Path) -> pd.DataFrame:
    """OddsAPI per-book change-log, archive + current, exact duplicates removed."""
    fs = sorted((Path(output_dir) / "book_odds_snapshots_archive").glob("*.csv"))
    cur = Path(output_dir) / "book_odds_snapshots.csv"
    if cur.exists():
        fs.append(cur)
    cols = ["snapshot_ts", "match_date", "kickoff_utc", "match", "bookmaker", "market", "side", "odds"]
    if not fs:
        return pd.DataFrame(columns=cols + ["_ts"])
    q = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols) for f in fs], ignore_index=True)
    q = q.drop_duplicates()
    q["market"] = q["market"].astype(str).str.upper()
    q["side"] = q["side"].astype(str).str.upper()
    q["odds"] = pd.to_numeric(q["odds"], errors="coerce")
    q["_ts"] = pd.to_datetime(q["snapshot_ts"], errors="coerce", utc=True)
    q["match_date"] = q["match_date"].astype(str).str[:10]
    return q.dropna(subset=["_ts"])


def index_quotes(q: pd.DataFrame) -> dict:
    """{(match_date, 'home vs away', family): frame} for O(1) lookup per entry."""
    if q is None or q.empty:
        return {}
    return {k: g for k, g in q.groupby(["match_date", "match", "market"], sort=False)}


def quotes_for(qidx: dict, row, family: str) -> pd.DataFrame | None:
    return qidx.get((str(row["date"])[:10], f"{row['home_team']} vs {row['away_team']}", family))


def pairs_at(sub: pd.DataFrame | None, t: pd.Timestamp, family: str) -> list:
    """Same-book two-sided (or one-sided) BookQuotes as they stood at `t` (LOCF, nothing after t)."""
    if sub is None or sub.empty:
        return []
    d = book_pairs(sub, t)
    return next(iter(d.values()), []) if d else []


def certified_close(sub: pd.DataFrame | None, kickoff: pd.Timestamp, family: str,
                    params: dict) -> dict:
    """Consensus close for side A, or a status. See the module docstring."""
    out = {"close_status": "NO_QUOTES", "p_close_fair": np.nan, "close_books": 0,
           "close_last_write_min": np.nan}
    if sub is None or sub.empty or kickoff is None:
        return out
    pre = sub[sub["_ts"] < kickoff]
    if pre.empty:
        out["close_status"] = "POST_KICKOFF_ONLY"
        return out
    last = pre["_ts"].max()
    gap = (kickoff - last).total_seconds() / 60.0
    out["close_last_write_min"] = round(gap, 1)
    if gap > CLOSE_WINDOW_MIN:
        out["close_status"] = "NO_CLOSE_IN_WINDOW"
        return out
    t = kickoff - pd.Timedelta(seconds=1)
    ps = []
    for qq in pairs_at(pre, t, family):
        ovr = ME.overround(qq.odds_a, qq.odds_b)
        if ovr is None or not (float(params["overround_min"]) <= ovr <= float(params["overround_max"])):
            continue
        p = ME.power_devig(qq.odds_a, qq.odds_b)
        if p is not None:
            ps.append(p)
    out["close_books"] = len(ps)
    if len(ps) < CLOSE_MIN_BOOKS:
        out["close_status"] = "INSUFFICIENT_BOOKS"
        return out
    out["close_status"] = "OK"
    out["p_close_fair"] = float(np.median(ps))
    return out


# ═══════════════════════════════ legacy ═══════════════════════════════
def regime_at(ts: pd.Timestamp) -> dict:
    cur = REGIMES[0][1]
    for start, r in REGIMES:
        if ts >= start:
            cur = r
    return cur


def legacy_ou25_base_tier(edge: float, side: str, league: str, ts: pd.Timestamp,
                          std_params: dict | None = None) -> str:
    """src.betting._base_tier re-expressed against the threshold regime in force at `ts`.
    Base tier only (before the min-odds / blind-fixture / drift steps), i.e. production's
    tier_over / tier_under columns."""
    if edge is None or not np.isfinite(edge):
        return "AVOID"
    r = regime_at(ts)
    cap = r["cap"]
    opt = (std_params or {}).get(str(league))
    if opt and opt.get("approved") and opt.get("sniper_th") is not None:
        has_league, sniper = True, float(opt["sniper_th"])
    elif league in config._LEAGUE_SNIPER_CALIBRATED:
        has_league, sniper = True, min(config._LEAGUE_SNIPER_CALIBRATED[league], cap)
    else:
        has_league = False
        sniper = config.SNIPER_THRESHOLD_OVER if side == "OVER" else config.SNIPER_THRESHOLD_UNDER
    if edge >= sniper:
        return "SNIPER"
    if not has_league:
        ceiling = config.LEAGUE_EDGE_CEILING.get(league, config.EDGE_CEILING)
        if edge > ceiling:
            return "MARKSMAN"
    if opt and opt.get("approved") and opt.get("marksman_th") is not None:
        mm = float(opt["marksman_th"])
    else:
        lm = config._LEAGUE_MARKSMAN_CALIBRATED.get(league)
        mm = min(lm, cap) if lm is not None else r["marksman"]
    if edge >= mm:
        return "MARKSMAN"
    if edge >= r["valuable"]:
        return "VALUABLE"
    return "AVOID"


def legacy_side_tier(p_model: float, odds: float, league: str, market: str,
                     side_params: dict | None = None) -> tuple[str, float]:
    """pipeline._generate_side_bets for one row: (tier, edge). One-sided (BTTS-YES / OVER)."""
    try:
        p, o = float(p_model), float(odds)
    except (TypeError, ValueError):
        return "NOT_EVALUATED", float("nan")
    if not (np.isfinite(p) and np.isfinite(o) and o > 1):
        return "NOT_EVALUATED", float("nan")
    edge = p - (1.0 / o) / SIDE_OVERROUND
    lp = ((side_params or {}).get(market) or {}).get(league, {})
    appr = lp.get("approved")
    use_own = (lp.get("approved", not lp.get("drop", False)) if appr is not None
               else not lp.get("drop", False))
    if use_own and lp:
        sn, mm = lp.get("sniper_th", 0.10), lp.get("marksman_th", 0.08)
    else:
        sn, mm = 0.10, 0.08
    if edge >= sn:
        return "SNIPER", edge
    if edge >= mm:
        return "MARKSMAN", edge
    if edge >= 0.04:
        return "VALUABLE", edge
    return "AVOID", edge


def _tier_str(v) -> str:
    s = str(v).strip().upper()
    return {"VALUE": "VALUABLE", "NAN": "AVOID", "<NA>": "AVOID", "": "AVOID", "NONE": "AVOID"}.get(s, s)


def legacy_decision(row: pd.Series, family: str, ts: pd.Timestamp, repo: Path = REPO) -> dict:
    """Legacy decision at first sight: {legacy_tier, legacy_side, legacy_edge, legacy_odds,
    legacy_repro_tier (OU25 only: reproduction of the recorded base tier of the chosen side)}."""
    if family == "OU25":
        side = str(row.get("best_side") or "")
        tier = _tier_str(row.get("signal_tier"))
        eo, eu = _num(row.get("edge_over")), _num(row.get("edge_under"))
        edge = eo if side == "OVER" else eu
        odds = _num(row.get("odds_over25" if side == "OVER" else "odds_under25"))
        stdp = artifact_at("models/best_params_standard.json", ts, repo)
        rep_o = legacy_ou25_base_tier(eo, "OVER", str(row["league"]), ts, stdp)
        rep_u = legacy_ou25_base_tier(eu, "UNDER", str(row["league"]), ts, stdp)
        return {"legacy_tier": tier if tier in TIERS else "AVOID", "legacy_side": side,
                "legacy_edge": edge, "legacy_odds": odds,
                "legacy_base_over": _tier_str(row.get("tier_over")),
                "legacy_base_under": _tier_str(row.get("tier_under")),
                "legacy_repro_over": rep_o, "legacy_repro_under": rep_u,
                "legacy_regime": regime_at(ts)["id"]}
    s = FAMILY_SPEC[family]
    sp = artifact_at("models/best_params_side_markets.json", ts, repo)
    tier, edge = legacy_side_tier(row.get(s["p"]), row.get(s["a"]), str(row["league"]), s["legacy"], sp)
    return {"legacy_tier": tier if tier in TIERS else "AVOID",
            "legacy_side": "YES" if family == "BTTS" else "OVER",
            "legacy_edge": edge, "legacy_odds": _num(row.get(s["a"])),
            "legacy_regime": "side_fixed_bars_0.10_0.08_0.04"}


def _num(v) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


# ═══════════════════════════════ point-in-time engine context ═══════════════════════════════
class ClvContext:
    """Track CLV record (src.market_edge.clv_track_stats) using only ledger rows whose match date is
    strictly before the entry date. Context for the engine's evidence gate - point in time."""

    def __init__(self, bets_ledger: pd.DataFrame | None, side_ledger: pd.DataFrame | None, clip: float):
        self.clip = clip
        self.b = self._prep(bets_ledger, ["model_type", "clv_pct", "match_date"])
        self.s = self._prep(side_ledger, ["model_type", "market", "clv_pct", "match_date"])
        self._cache: dict = {}

    @staticmethod
    def _prep(d, cols):
        if d is None or not len(d) or not set(cols) <= set(d.columns):
            return None
        d = d[cols].copy()
        d["_md"] = pd.to_datetime(d["match_date"], errors="coerce").dt.normalize()
        return d.dropna(subset=["_md"])

    def at(self, day: pd.Timestamp) -> dict:
        day = pd.Timestamp(day).tz_localize(None).normalize() if pd.Timestamp(day).tzinfo else pd.Timestamp(day).normalize()
        if day not in self._cache:
            b = self.b[self.b["_md"] < day] if self.b is not None else None
            s = self.s[self.s["_md"] < day] if self.s is not None else None
            self._cache[day] = ME.clv_track_stats(b, s, clip_pct=self.clip)
        return self._cache[day]


class LeagueHistory:
    """Rows of league history strictly before a date (af_history + fd_history)."""

    def __init__(self, hist: pd.DataFrame):
        self.h = {lg: np.sort(g["_d"].values) for lg, g in hist.groupby("league")} if len(hist) else {}

    def n_before(self, league: str, day: pd.Timestamp) -> int:
        a = self.h.get(league)
        if a is None:
            return 0
        return int(np.searchsorted(a, np.datetime64(pd.Timestamp(day).tz_localize(None) if pd.Timestamp(day).tzinfo else pd.Timestamp(day)), side="left"))


def load_results(output_dir: Path) -> pd.DataFrame:
    parts = []
    af = Path(output_dir) / "af_history.parquet"
    if af.exists():
        a = pd.read_parquet(af, columns=["league", "date", "home_team", "away_team", "FTHG", "FTAG"])
        parts.append(a.rename(columns={"FTHG": "hg", "FTAG": "ag"}))
    fd = Path(output_dir) / "fd_history.parquet"
    if fd.exists():
        f = pd.read_parquet(fd, columns=["league", "date", "home_team", "away_team", "home_goals", "away_goals"])
        parts.append(f.rename(columns={"home_goals": "hg", "away_goals": "ag"}))
    if not parts:
        return pd.DataFrame(columns=["league", "date", "home_team", "away_team", "hg", "ag", "_d"])
    r = pd.concat(parts, ignore_index=True)
    r["_d"] = pd.to_datetime(r["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    return r.dropna(subset=["_d"])


def settle(entries: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Attach hg/ag: same league, kickoff date within +-1 day, both names resolved league-scoped,
    exactly one result row with goals. Anything else stays unsettled (NaN)."""
    from src.team_names import resolve_result
    e = entries.copy()
    e["hg"], e["ag"] = np.nan, np.nan
    r = results.dropna(subset=["hg", "ag"])
    by_lg = {lg: g for lg, g in r.groupby("league")}
    cache: dict = {}
    for i, row in e.iterrows():
        lg = str(row["league"])
        g = by_lg.get(lg)
        if g is None:
            continue
        if lg not in cache:
            cache[lg] = sorted(set(g["home_team"].astype(str)) | set(g["away_team"].astype(str)))
        h = resolve_result(str(row["home_team"]), cache[lg])
        a = resolve_result(str(row["away_team"]), cache[lg])
        if not h or not a:
            continue
        d = pd.Timestamp(str(row["date"])[:10])
        m = g[(g["home_team"] == h) & (g["away_team"] == a) & ((g["_d"] - d).abs() <= pd.Timedelta(days=1))]
        m = m.drop_duplicates(["hg", "ag"])
        if len(m) == 1:
            e.at[i, "hg"], e.at[i, "ag"] = float(m["hg"].iloc[0]), float(m["ag"].iloc[0])
    return e


def outcome_a(family: str, hg: float, ag: float) -> float:
    """1 if side A (OVER / YES) happened, else 0; NaN if unsettled."""
    if not (np.isfinite(hg) and np.isfinite(ag)):
        return float("nan")
    if family == "BTTS":
        return float(hg > 0 and ag > 0)
    line = {"OU25": 2.5, "OU15": 1.5, "OU35": 3.5}[family]
    return float(hg + ag > line)


# ═══════════════════════════════ replay ═══════════════════════════════
def replay(board: pd.DataFrame, quotes: pd.DataFrame, results: pd.DataFrame,
           bets_ledger: pd.DataFrame | None, side_ledger: pd.DataFrame | None,
           params: dict | None = None, families=("OU25", "BTTS", "OU15"),
           repo: Path = REPO) -> pd.DataFrame:
    """One row per fixture x family at first sight: legacy decision, NEW engine output, outcome,
    certified close. Pure function of committed inputs."""
    P = params or ME.load_params()
    qidx = index_quotes(quotes)
    clv = ClvContext(bets_ledger, side_ledger, float(P["clv_clip_pct"]))
    lh = LeagueHistory(results)
    rows = []
    for fam in families:
        fs = first_sight(board, fam)
        spec = FAMILY_SPEC[fam]
        for _, r in fs.iterrows():
            t, ko = r["obs_ts"], r["kickoff"]
            sub = quotes_for(qidx, r, fam)
            sub_pre = sub[sub["_ts"] <= t] if sub is not None else None
            day = t.tz_convert(None).normalize()
            c = ME.Candidate(
                family=fam, track=str(r["track"]), p_model=_num(r[spec["p"]]), kickoff=ko, now=t,
                quotes=pairs_at(sub_pre, t, fam),
                row_odds_a=_num(r.get(spec["a"])), row_odds_b=_num(r.get(spec["b"])),
                league=str(r["league"]), league_history_n=lh.n_before(str(r["league"]), day),
                model_trained_at=ME.trained_at_from_version(r.get(spec["ver"])),
                clv=clv.at(day).get(f"{r['track']}|{fam}"), evidence=None,
                price_synthetic=_flag(r.get("price_synthetic")))
            e = ME.evaluate(c, P)
            sides = e.pop("_sides", [])
            sa = sides[0] if sides else {}
            leg = legacy_decision(r, fam, t, repo)
            cl = certified_close(sub, ko, fam, P)
            rows.append({
                "label": LABEL, "fx": r["fx"], "family": fam, "track": r["track"], "league": r["league"],
                "date": str(r["date"])[:10], "home_team": r["home_team"], "away_team": r["away_team"],
                "entry_ts": t, "kickoff": ko, "commit_sha": r["commit_sha"],
                "minutes_to_kickoff": (ko - t).total_seconds() / 60.0,
                "p_model_a": _num(r[spec["p"]]), "board_odds_a": _num(r.get(spec["a"])),
                "board_odds_b": _num(r.get(spec["b"])),
                "no_form_data": r.get("no_form_data"), "drift_signal": r.get("drift_signal"),
                **leg,
                "new_state": e["decision_state"], "new_reason": e["decision_reason"], "new_side": e["side"],
                "new_odds": e["odds"], "new_ev_point": e["ev_point"], "new_ev_lb": e["ev_lower_bound"],
                "new_p_decision": e["p_decision"], "new_p_lb": e["p_lower_bound"],
                "new_penalty": e["uncertainty_penalty"], "market_source": e["market_source"],
                "n_books": e["n_books"], "n_one_sided_books": e["n_one_sided_books"],
                "n_invalid_books": e["n_invalid_books"], "price_age_minutes": e["price_age_minutes"],
                "market_overround": e["market_overround"], "clv_track_n": e["clv_track_n"],
                "p_mkt_a_power": sa.get("p_market_fair_power", np.nan),
                "p_mkt_a_prop": sa.get("p_market_fair_proportional", np.nan),
                "p_mkt_a": sa.get("p_market_fair", np.nan), "p_dec_a": sa.get("p_decision", np.nan),
                "new_side_a_odds": sa.get("odds", np.nan),
                "new_side_b_odds": sides[1]["odds"] if len(sides) > 1 else np.nan,
                **cl,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = settle(out, results)
    out["y_a"] = [outcome_a(f, h, a) for f, h, a in zip(out["family"], out["hg"], out["ag"])]
    out["two_sided"] = out["market_source"].isin(["MEDIAN_TWO_SIDED_BOOKS", "SINGLE_TWO_SIDED_BOOK"])
    # selections
    out["legacy_bet"] = out["legacy_tier"].isin(TIERS) & out["legacy_side"].isin(["OVER", "UNDER", "YES"])
    out["new_qualified"] = out["new_state"].isin(["QUALIFIED", "STRONG"])
    caps_ok = (pd.to_numeric(out["new_odds"], errors="coerce") <= float(P["max_odds"])) & \
              ~out["new_reason"].astype(str).str.contains("EXTREME_DISAGREEMENT|MARKET_BLOCKED")
    out["new_evlb"] = (pd.to_numeric(out["new_ev_lb"], errors="coerce") > 0) & \
                      (pd.to_numeric(out["n_books"], errors="coerce") >= int(P["min_books_qualified"])) & caps_ok
    out["new_observe_pool"] = out["new_state"].isin(["OBSERVE", "QUALIFIED", "STRONG"]) & out["two_sided"]
    return out


def settle_bets(df: pd.DataFrame, engine: str) -> pd.DataFrame:
    """Rows selected by `engine` with side, odds, won, pnl, certified CLV. Unsettled rows dropped."""
    if engine == "LEGACY":
        m = df["legacy_bet"]
        side, odds = df["legacy_side"], df["legacy_odds"]
    else:
        m = df[{"NEW_QUALIFIED": "new_qualified", "NEW_EVLB": "new_evlb",
                "NEW_OBSERVE_POOL": "new_observe_pool"}[engine]]
        side, odds = df["new_side"], df["new_odds"]
    x = df[m & df["y_a"].notna()].copy()
    x["bet_side"] = side[x.index]
    x["bet_odds"] = pd.to_numeric(odds[x.index], errors="coerce")
    x = x[x["bet_odds"] > 1]
    is_a = x["bet_side"].isin(["OVER", "YES"])
    x["won"] = np.where(is_a, x["y_a"], 1 - x["y_a"]).astype(float)
    x["pnl"] = np.where(x["won"] == 1, x["bet_odds"] - 1, -1.0)
    pc = np.where(is_a, x["p_close_fair"], 1 - x["p_close_fair"])
    x["clv_pct"] = np.where(x["close_status"] == "OK", (x["bet_odds"] * pc - 1) * 100, np.nan)
    x["md"] = pd.to_datetime(x["kickoff"], utc=True).dt.tz_convert(None).dt.normalize()
    return x


# ═══════════════════════════════ metrics ═══════════════════════════════
def block_bootstrap(v: np.ndarray, days: np.ndarray, stat=np.mean, n: int = 2000, seed: int = 20261025,
                    q=(2.5, 97.5)) -> tuple[float, float]:
    """Matchday (kickoff date) block bootstrap."""
    v = np.asarray(v, float)
    if len(v) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    days = np.asarray(days)
    u = np.unique(days)
    idx = [np.where(days == k)[0] for k in u]
    bs = []
    for _ in range(n):
        pick = rng.integers(0, len(u), len(u))
        bs.append(stat(v[np.concatenate([idx[k] for k in pick])]))
    lo, hi = np.percentile(bs, q)
    return float(lo), float(hi)


def bet_metrics(x: pd.DataFrame, window_days: float) -> dict:
    n = len(x)
    if not n:
        return {"n": 0}
    x = x.sort_values("kickoff")
    pnl, won = x["pnl"].to_numpy(float), x["won"].to_numpy(float)
    cum = np.cumsum(pnl)
    dd = float(np.max(np.maximum.accumulate(np.r_[0, cum]) - np.r_[0, cum]))
    streak = best = 0
    for w in won:
        streak = streak + 1 if w == 0 else 0
        best = max(best, streak)
    lo, hi = block_bootstrap(pnl, x["md"].to_numpy())
    clv = pd.to_numeric(x["clv_pct"], errors="coerce")
    return {
        "n": int(n), "matchdays": int(x["md"].nunique()),
        "bets_per_day": round(n / max(window_days, 1.0), 2),
        "hit_pct": round(float(won.mean() * 100), 1),
        "mean_odds": round(float(x["bet_odds"].mean()), 3),
        "roi_pct": round(float(pnl.mean() * 100), 1),
        "roi_ci95": [round(lo * 100, 1), round(hi * 100, 1)],
        "pnl_u": round(float(pnl.sum()), 2), "max_drawdown_u": round(dd, 2),
        "longest_losing_streak": int(best),
        "clv_coverage_pct": round(float(clv.notna().mean() * 100), 1),
        "clv_n": int(clv.notna().sum()),
        "clv_mean_pct": round(float(clv.mean()), 2) if clv.notna().any() else None,
        "clv_median_pct": round(float(clv.median()), 2) if clv.notna().any() else None,
        "clv_positive_pct": round(float((clv[clv.notna()] > 0).mean() * 100), 1) if clv.notna().any() else None,
    }


def _clipp(p):
    return np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)


def brier(y, p) -> float:
    return float(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2))


def logloss(y, p) -> float:
    y, p = np.asarray(y, float), _clipp(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_slope(y, p) -> float:
    """Slope of a logistic regression of y on logit(p) (1 = calibrated, <1 = overconfident)."""
    y = np.asarray(y, float)
    x = np.log(_clipp(p) / (1 - _clipp(p)))
    if len(y) < 20 or y.min() == y.max():
        return float("nan")
    a, b = 0.0, 1.0
    for _ in range(50):                                               # Newton-Raphson, 2 params
        z = a + b * x
        mu = 1 / (1 + np.exp(-z))
        w = mu * (1 - mu)
        g = np.array([np.sum(y - mu), np.sum((y - mu) * x)])
        h = np.array([[np.sum(w), np.sum(w * x)], [np.sum(w * x), np.sum(w * x * x)]])
        try:
            step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            return float("nan")
        a, b = a + step[0], b + step[1]
        if np.max(np.abs(step)) < 1e-9:
            break
    return float(b)


def ece(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if not len(y):
        return float("nan")
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for k in range(bins):
        m = idx == k
        if m.any():
            tot += m.sum() * abs(y[m].mean() - p[m].mean())
    return float(tot / len(y))


def prob_metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 5:
        return {"n": int(len(y))}
    return {"n": int(len(y)), "brier": round(brier(y, p), 5), "logloss": round(logloss(y, p), 5),
            "calibration_slope": round(calibration_slope(y, p), 3), "ece": round(ece(y, p), 4)}


def agreement(df: pd.DataFrame, new_col: str) -> dict:
    lb, nb = df["legacy_bet"], df[new_col]
    same = lb & nb & (df["legacy_side"] == df["new_side"])
    return {"legacy_only": int((lb & ~nb).sum()), "new_only": int((~lb & nb).sum()),
            "both_same_side": int(same.sum()), "disagree_side": int((lb & nb & ~same).sum()),
            "neither": int((~lb & ~nb).sum())}


def bh(pvals: list) -> list:
    v = np.array([np.nan if p is None else p for p in pvals], float)
    ok = np.isfinite(v)
    q = np.full(len(v), np.nan)
    if ok.sum():
        w = v[ok]
        o = np.argsort(w)
        qq = w[o] * len(w) / (np.arange(len(w)) + 1)
        qq = np.minimum.accumulate(qq[::-1])[::-1]
        r = np.empty(len(w))
        r[o] = np.minimum(qq, 1)
        q[ok] = r
    return [None if not np.isfinite(x) else round(float(x), 4) for x in q]


MIN_CELL_N = 30          # below this a cell is reported INSUFFICIENT, never as a rate


def compare_cell(df: pd.DataFrame) -> dict:
    """Legacy vs NEW for one track x family (settled rows only)."""
    s = df[df["y_a"].notna()]
    two = s[s["two_sided"]]
    if s.empty:
        return {"status": "INSUFFICIENT", "reason": "no settled first-sight rows"}
    days = max((s["entry_ts"].max() - s["entry_ts"].min()).total_seconds() / 86400.0, 1.0)
    res = {
        "status": "OK" if len(two) >= MIN_CELL_N else "INSUFFICIENT",
        "window": [str(s["entry_ts"].min()), str(s["entry_ts"].max())],
        "first_sight_settled": int(len(s)), "two_sided_at_entry": int(len(two)),
        "two_sided_share": round(len(two) / len(s), 3),
        "new_states_all": s["new_state"].value_counts().to_dict(),
        "new_reasons_top": s["new_reason"].astype(str).str.split(";").str[0]
                           .str.replace(r"\(.*", "", regex=True).value_counts().head(8).to_dict(),
        "certified_close_coverage_two_sided_pct": round(float((two["close_status"] == "OK").mean() * 100), 1)
        if len(two) else None,
    }
    if res["status"] == "INSUFFICIENT":
        res["reason"] = f"{len(two)} rows with a same-book two-sided price at first sight (< {MIN_CELL_N})"
    res["engines_full_universe"] = {"LEGACY": bet_metrics(settle_bets(s, "LEGACY"), days)}
    eng = {}
    for e in ("LEGACY", "NEW_QUALIFIED", "NEW_EVLB", "NEW_OBSERVE_POOL"):
        eng[e] = bet_metrics(settle_bets(two, e), days) if len(two) else {"n": 0}
    res["engines_two_sided_universe"] = eng
    res["agreement_two_sided"] = {"NEW_EVLB": agreement(two, "new_evlb"),
                                  "NEW_QUALIFIED": agreement(two, "new_qualified")}
    if len(two):
        y = two["y_a"].to_numpy(float)
        res["probability_quality_side_a"] = {
            "model": prob_metrics(y, two["p_model_a"]),
            "market_power": prob_metrics(y, two["p_mkt_a_power"]),
            "market_proportional": prob_metrics(y, two["p_mkt_a_prop"]),
            "decision": prob_metrics(y, two["p_dec_a"]),
        }
        d = (y - _clipp(two["p_mkt_a"])) ** 2 - (y - _clipp(two["p_model_a"])) ** 2
        lo, hi = block_bootstrap(d, two["kickoff"].dt.tz_convert(None).dt.normalize().to_numpy())
        res["model_minus_market_brier_gain"] = {"mean": round(float(d.mean()), 5),
                                                "ci95": [round(lo, 5), round(hi, 5)],
                                                "note": "positive = model beats the de-vigged market"}
    if s["family"].iloc[0] == "OU25":
        rec = s.apply(lambda r: r["legacy_base_over"] == r["legacy_repro_over"]
                      and r["legacy_base_under"] == r["legacy_repro_under"], axis=1)
        res["legacy_reproduction_match_pct"] = round(float(rec.mean() * 100), 1)
    return res


# ═══════════════════════════════ disagreement test ═══════════════════════════════
def disagreement_frame(df: pd.DataFrame) -> pd.DataFrame:
    s = df[df["y_a"].notna() & df["two_sided"]].copy()
    if s.empty:
        return s
    pm, pk = _clipp(s["p_model_a"]), _clipp(s["p_mkt_a"])
    y = s["y_a"].to_numpy(float)
    sign = np.sign(pm - pk)
    s["d"] = np.abs(pm - pk)
    s["adv"] = sign * (y - pk)
    s["dbrier"] = (y - pk) ** 2 - (y - pm) ** 2
    s["dll"] = -(y * np.log(pk) + (1 - y) * np.log(1 - pk)) + (y * np.log(pm) + (1 - y) * np.log(1 - pm))
    s["clv_dir"] = np.where(s["close_status"] == "OK", sign * (s["p_close_fair"].to_numpy(float) - pk), np.nan)
    fav_a = sign > 0
    odds = np.where(fav_a, s["new_side_a_odds"], s["new_side_b_odds"]).astype(float)
    won = np.where(fav_a, y, 1 - y)
    s["roi"] = np.where(np.isfinite(odds) & (odds > 1), np.where(won == 1, odds - 1, -1.0), np.nan)
    s["md"] = s["kickoff"].dt.tz_convert(None).dt.normalize()
    return s.sort_values("entry_ts")


def disagreement_cell(df: pd.DataFrame, n_boot: int = 2000) -> dict:
    s = disagreement_frame(df)
    if len(s) < 2 * MIN_CELL_N:
        return {"status": "INSUFFICIENT", "n": int(len(s)),
                "reason": f"{len(s)} settled two-sided rows (< {2 * MIN_CELL_N} for a half/half split)"}
    half = len(s) // 2
    disc, hold = s.iloc[:half], s.iloc[half:]
    edges = np.quantile(disc["d"], [1 / 3, 2 / 3])
    lab = np.where(hold["d"] <= edges[0], "small", np.where(hold["d"] <= edges[1], "moderate", "extreme"))
    hold = hold.assign(bin=lab)
    out = {"status": "OK", "n_discovery": int(len(disc)), "n_holdout": int(len(hold)),
           "discovery_window": [str(disc["entry_ts"].min()), str(disc["entry_ts"].max())],
           "holdout_window": [str(hold["entry_ts"].min()), str(hold["entry_ts"].max())],
           "bin_edges_from_discovery": [round(float(e), 4) for e in edges], "bins": {}, "tests": {}}
    for b in ("small", "moderate", "extreme"):
        g = hold[hold["bin"] == b]
        row = {"n": int(len(g))}
        for m in ("adv", "dbrier", "dll", "clv_dir", "roi"):
            v = g[m].dropna()
            if len(v) >= 5:
                lo, hi = block_bootstrap(v.to_numpy(), g.loc[v.index, "md"].to_numpy(), n=n_boot)
                row[m] = {"mean": round(float(v.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)], "n": int(len(v))}
            else:
                row[m] = {"mean": None, "n": int(len(v))}
        out["bins"][b] = row
    # moderate vs extreme: one-sided bootstrap p for "moderate > extreme" on each metric
    rng = np.random.default_rng(7)
    mo, ex = hold[hold["bin"] == "moderate"], hold[hold["bin"] == "extreme"]
    for m in ("adv", "dll", "dbrier", "roi", "clv_dir"):
        a, b = mo[m].dropna(), ex[m].dropna()
        if len(a) < 10 or len(b) < 10:
            out["tests"][m] = {"diff": None, "p_one_sided": None}
            continue
        ua, ub = a.index.map(hold["md"]), b.index.map(hold["md"])
        da, db = a.groupby(np.asarray(ua)), b.groupby(np.asarray(ub))
        ga, gb = [g.to_numpy() for _, g in da], [g.to_numpy() for _, g in db]
        diffs = []
        for _ in range(n_boot):
            sa = np.concatenate([ga[i] for i in rng.integers(0, len(ga), len(ga))])
            sb = np.concatenate([gb[i] for i in rng.integers(0, len(gb), len(gb))])
            diffs.append(sa.mean() - sb.mean())
        diffs = np.asarray(diffs)
        out["tests"][m] = {"diff_moderate_minus_extreme": round(float(a.mean() - b.mean()), 4),
                           "ci95": [round(float(np.percentile(diffs, 2.5)), 4),
                                    round(float(np.percentile(diffs, 97.5)), 4)],
                           "p_one_sided": round(float((diffs <= 0).mean()), 4)}
    # isotonic: fit adv ~ d on discovery (direction chosen by discovery), score holdout
    try:
        from sklearn.isotonic import IsotonicRegression
        from scipy.stats import spearmanr
        iso = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(disc["d"], disc["adv"])
        pred = iso.predict(hold["d"])
        rho, p = spearmanr(pred, hold["adv"]) if np.std(pred) > 0 else (float("nan"), float("nan"))
        out["isotonic"] = {"direction_fitted_on_discovery": "increasing" if iso.increasing_ else "decreasing",
                           "holdout_spearman_pred_vs_realised_adv": None if not np.isfinite(rho) else round(float(rho), 4),
                           "p": None if not np.isfinite(p) else round(float(p), 4)}
    except Exception as e:                                            # noqa: BLE001
        out["isotonic"] = {"error": type(e).__name__}
    return out


# ═══════════════════════════════ discovery-only parameter fit ═══════════════════════════════
W_GRID = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.05)]


def _decision(pk, pm, w, clip):
    lk = np.log(_clipp(pk) / (1 - _clipp(pk)))
    raw = np.log(_clipp(pm) / (1 - _clipp(pm))) - lk
    return 1 / (1 + np.exp(-(lk + w * np.clip(raw, -clip, clip))))


def fit_params_cell(df: pd.DataFrame, clip: float, default_w: float, disc_frac: float = 0.6) -> dict:
    s = df[df["y_a"].notna() & df["two_sided"]].sort_values("entry_ts")
    if len(s) < 2 * MIN_CELL_N:
        return {"status": "INSUFFICIENT", "n": int(len(s))}
    k = int(len(s) * disc_frac)
    disc, hold = s.iloc[:k], s.iloc[k:]
    yd, yh = disc["y_a"].to_numpy(float), hold["y_a"].to_numpy(float)
    ll_pow, ll_prop = logloss(yd, disc["p_mkt_a_power"]), logloss(yd, disc["p_mkt_a_prop"])
    method = "power" if ll_pow <= ll_prop else "proportional"
    col = "p_mkt_a_power" if method == "power" else "p_mkt_a_prop"
    lls = {w: logloss(yd, _decision(disc[col], disc["p_model_a"], w, clip)) for w in W_GRID}
    w_fit = min(lls, key=lambda w: (lls[w], w))
    hm = logloss(yh, hold[col])
    hf = logloss(yh, _decision(hold[col], hold["p_model_a"], w_fit, clip))
    hd = logloss(yh, _decision(hold[col], hold["p_model_a"], default_w, clip))
    dd = -(yh * np.log(_clipp(_decision(hold[col], hold["p_model_a"], w_fit, clip))) +
           (1 - yh) * np.log(1 - _clipp(_decision(hold[col], hold["p_model_a"], w_fit, clip)))) + \
        (yh * np.log(_clipp(hold[col])) + (1 - yh) * np.log(1 - _clipp(hold[col])))
    days = hold["kickoff"].dt.tz_convert(None).dt.normalize().to_numpy()
    lo, hi = block_bootstrap(dd, days)
    dist = block_bootstrap_dist(dd, days)
    # w = 0 IS the market: there is nothing to test (float noise only)
    p_one = float((dist >= 0).mean()) if len(dist) and w_fit > 0 else None
    return {
        "status": "PROPOSED_DISCOVERY_ONLY",
        "discovery_window": [str(disc["entry_ts"].min()), str(disc["entry_ts"].max())], "n_discovery": int(len(disc)),
        "holdout_window": [str(hold["entry_ts"].min()), str(hold["entry_ts"].max())], "n_holdout": int(len(hold)),
        "devig_method_discovery": method,
        "discovery_logloss": {"market_power": round(ll_pow, 5), "market_proportional": round(ll_prop, 5),
                              "fitted_w": round(lls[w_fit], 5), "w0": round(lls[0.0], 5)},
        "residual_weight_fitted": w_fit, "residual_n_eff_proposed": int(len(disc)),
        "holdout_logloss": {"market": round(hm, 5), "fitted_w": round(hf, 5),
                            f"current_default_w_{default_w}": round(hd, 5)},
        "holdout_fitted_minus_market_ll": {"mean": round(float(dd.mean()), 5), "ci95": [round(lo, 5), round(hi, 5)],
                                           "p_one_sided_fitted_better": None if p_one is None else round(p_one, 4),
                                           "note": "negative = fitted residual beats the market on holdout"},
    }


def block_bootstrap_dist(v, days, n: int = 2000, seed: int = 20261025) -> np.ndarray:
    """Bootstrap distribution of the mean (matchday blocks)."""
    v = np.asarray(v, float)
    if len(v) < 5:
        return np.array([])
    rng = np.random.default_rng(seed)
    days = np.asarray(days)
    u = np.unique(days)
    idx = [np.where(days == k)[0] for k in u]
    return np.array([v[np.concatenate([idx[k] for k in rng.integers(0, len(u), len(u))])].mean()
                     for _ in range(n)])


def legacy_vs_ledger(df: pd.DataFrame, bets_ledger: pd.DataFrame, side_ledger: pd.DataFrame) -> dict:
    """Validate the legacy decision against the committed ledgers on the rows where both describe the
    SAME moment: the ledger's entry odds equal the first-sight board odds (same fixture, side/market).
    Ledger tiers can be upgraded in place later, so a mismatch is reported, never silently 'fixed'."""
    out = {}
    lb = df[df["legacy_bet"]].copy()
    lb["md"] = lb["date"].astype(str).str[:10]
    o = lb[lb["family"] == "OU25"]
    if len(o) and bets_ledger is not None and len(bets_ledger):
        b = bets_ledger.copy()
        b["md"] = b["match_date"].astype(str).str[:10]
        m = o.merge(b, left_on=["md", "home_team", "away_team", "legacy_side"],
                    right_on=["md", "home_team", "away_team", "side"], how="inner", suffixes=("", "_l"))
        m = m[np.isclose(pd.to_numeric(m["odds"], errors="coerce"), m["legacy_odds"])]
        t = m["signal_tier"].map(_tier_str)
        out["OU25"] = {"matched_same_price": int(len(m)), "tier_equal": int((t == m["legacy_tier"]).sum()),
                       "tier_equal_pct": round(float((t == m["legacy_tier"]).mean() * 100), 1) if len(m) else None,
                       **_rank_diff(t, m["legacy_tier"])}
    if side_ledger is not None and len(side_ledger):
        s = side_ledger.copy()
        s["md"] = s["match_date"].astype(str).str[:10]
        s["market"] = s["market"].astype(str).str.lower()
        for fam in ("BTTS", "OU15"):
            x = lb[lb["family"] == fam].assign(market=FAMILY_SPEC[fam]["legacy"])
            if x.empty:
                continue
            m = x.merge(s, on=["md", "home_team", "away_team", "market"], how="inner", suffixes=("", "_l"))
            m = m[np.isclose(pd.to_numeric(m["odds"], errors="coerce"), m["legacy_odds"])]
            t = m["signal_tier"].map(_tier_str)
            out[fam] = {"matched_same_price": int(len(m)), "tier_equal": int((t == m["legacy_tier"]).sum()),
                        "tier_equal_pct": round(float((t == m["legacy_tier"]).mean() * 100), 1) if len(m) else None,
                        **_rank_diff(t, m["legacy_tier"])}
    return out


_RANK = {"AVOID": 0, "VALUABLE": 1, "MARKSMAN": 2, "SNIPER": 3}


def _rank_diff(ledger: pd.Series, first: pd.Series) -> dict:
    a, b = ledger.map(_RANK), first.map(_RANK)
    return {"ledger_tier_higher": int((a > b).sum()), "ledger_tier_lower": int((a < b).sum())}
