"""market_edge_v2 SHADOW LOG — legacy and market-first decisions side by side. OUTPUT ONLY.

    output/edge_engine_shadow.csv    append-only, FIRST SIGHT per (selection_engine_version,
                                     fixture_id, market): the earliest pre-kickoff opinion is kept
                                     and existing bytes are never rewritten
    output/edge_engine_health.json   this run: coverage of two-sided prices per market/track,
                                     state counts, one-sided counts, runtime (recorded generated_at)

WHAT IT DOES NOT DO. It never changes a tip, a tier, a threshold, a stake, a ledger row or a
Telegram message. The legacy engine (src/betting.py + pipeline._generate_side_bets) stays the
production/reference selection path. `run` cannot raise; a failure logs a warning and returns None.

PRICES. Per-book two-sided quotes only (src/market_edge.py never pairs book A's over with book B's
under). Sources, newest wins per (fixture, market, book, side):
  * this run's OddsAPI response (src/predict._LAST_BOOK_ROWS, O/U 1.5/2.5/3.5, snapshot = now)
  * this run's per-event BTTS fetch (src/btts_odds._LAST_BOOK_ROWS)
  * output/book_odds_snapshots.csv, carried forward per book and side (it is a CHANGE-LOG, so an
    unchanged price is still the price; the pair's timestamp is the later side's last change,
    which makes price_age an upper bound — conservative).
  * output/book_quotes/<date>.csv — API-Football per-book quotes committed by the batch capture jobs
    (added 2026-10-25, owner-approved; src/edge_af_quotes.py). Read from disk only, no API call.
    Paired ONLY within the same book AND the same snapshot_ts, strictly pre-kickoff, one line per
    family, clubs resolved league-scoped with src/team_names (ambiguity refused). Books are named
    "af:<book>" in books_used; a physical book quoted by both sources votes once (fresher wins).
    First-sight immutability means this coverage applies only to rows first logged after the change.
The board price (`odds_over25`, `odds_btts`, ...) is used only as the execution price, never as
a market: the BTTS board pair is a median-yes / median-no from different books.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import edge_af_quotes as EAF
from src import market_edge as ME

log = logging.getLogger(__name__)

SHADOW_FILE = config.OUTPUT_DIR / "edge_engine_shadow.csv"
HEALTH_FILE = config.OUTPUT_DIR / "edge_engine_health.json"
SNAPSHOTS = config.OUTPUT_DIR / "book_odds_snapshots.csv"
KEY = ["selection_engine_version", "fixture_id", "market"]

#: family -> (p_model column, board price side A, board price side B, model_version column,
#:            legacy side-market name in pipeline._generate_side_bets / stake plan market name)
FAMILIES = {
    "OU25": ("p_over25", "odds_over25", "odds_under25", "model_version", None, "OU25"),
    "BTTS": ("p_btts", "odds_btts", "odds_btts_no", "model_version_btts", "btts", "BTTS"),
    "OU15": ("p_over15", "odds_over15", "odds_under15", "model_version_over15", "over15", "OVER15"),
    "OU35": ("p_over35", "odds_over35", "odds_under35", "model_version_over35", "over35", "OVER35"),
}
LEGACY_SIDE_OVERROUND = 1.08          # pipeline._generate_side_bets' OVERROUND, mirrored for logging

SHADOW_COLS = [
    "selection_engine_version", "prediction_ts", "fixture_id", "match_date", "kickoff_utc",
    "league", "home_team", "away_team", "model_type", "market",
    "legacy_engine_version", "legacy_raw_edge", "legacy_tier", "legacy_side", "legacy_threshold",
    "legacy_odds", "new_state", "new_side", "new_reason",
    *[c for c in ME.OUTPUT_COLS if c not in ("selection_engine_version", "market")],
]


# ── quotes ────────────────────────────────────────────────────────────────────────────────────
def _quote_frame(now: pd.Timestamp, live_rows: list | None, path: Path | None) -> pd.DataFrame:
    cols = ["snapshot_ts", "match_date", "match", "bookmaker", "market", "side", "odds"]
    parts = []
    p = Path(path) if path else SNAPSHOTS
    if p.exists():
        f = pd.read_csv(p, usecols=lambda c: c in cols)
        lo = (now - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        parts.append(f[f["match_date"].astype(str) >= lo])
    if live_rows:
        parts.append(pd.DataFrame(live_rows).reindex(columns=cols))
    if not parts:
        return pd.DataFrame(columns=cols)
    q = pd.concat(parts, ignore_index=True)
    q["market"] = q["market"].astype(str).str.upper()
    q["side"] = q["side"].astype(str).str.upper()
    q["_ts"] = pd.to_datetime(q["snapshot_ts"], errors="coerce", utc=True)
    q["odds"] = pd.to_numeric(q["odds"], errors="coerce")
    return q.dropna(subset=["_ts"])


def book_pairs(q: pd.DataFrame, now: pd.Timestamp) -> dict:
    """{(match_date, match, family): [BookQuote]} — last observation carried forward per book and
    side up to `now`, paired ONLY within the same book. A book with one side only yields a
    one-sided BookQuote (the engine counts it and never uses it as a market)."""
    out: dict = {}
    if q is None or q.empty:
        return out
    q = q[(q["_ts"] <= now) & q["market"].isin(list(ME.SIDES))]
    if q.empty:
        return out
    k = ["match_date", "match", "market", "bookmaker"]
    last = q.sort_values("_ts", kind="mergesort").drop_duplicates(k + ["side"], keep="last")
    odds = last.pivot_table(index=k, columns="side", values="odds", aggfunc="last")
    ts = last.pivot_table(index=k, columns="side", values="_ts", aggfunc="max")
    for idx, orow in odds.iterrows():
        md, m, fam, bk = idx
        a, b = ME.SIDES[fam]
        oa, ob = orow.get(a, np.nan), orow.get(b, np.nan)
        trow = ts.loc[idx]
        stamps = [trow.get(s) for s in (a, b) if s in trow.index and pd.notna(trow.get(s))]
        out.setdefault((str(md)[:10], str(m), fam), []).append(ME.BookQuote(
            str(bk), None if pd.isna(oa) else float(oa), None if pd.isna(ob) else float(ob),
            max(stamps) if stamps else None))
    return out


# ── legacy side ───────────────────────────────────────────────────────────────────────────────
class _Gates:
    def __init__(self):
        self._c: dict = {}

    def threshold(self, league: str, market: str, side: str, tier: str) -> float:
        k = (league, market, side)
        if k not in self._c:
            try:
                from src.gate_resolver import resolve_effective_tier_gates
                self._c[k] = resolve_effective_tier_gates(league, market, side=side)
            except Exception:                                         # noqa: BLE001
                self._c[k] = None
        g = self._c[k]
        if g is None:
            return float("nan")
        return float({"SNIPER": g.sniper, "MARKSMAN": g.marksman}.get(tier, g.valuable))


def _num(v) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _fixture_id(r) -> str:
    eid = r.get("_oa_event_id")
    if eid is not None and str(eid).strip() not in ("", "nan", "None"):
        return str(eid)
    return f"{str(r.get('date'))[:10]}|{r.get('home_team')}|{r.get('away_team')}"


# ── main ──────────────────────────────────────────────────────────────────────────────────────
def evaluate_board(preds: pd.DataFrame, side_bets: pd.DataFrame | None, now: pd.Timestamp,
                   pairs: dict, params: dict, league_n: dict | None = None,
                   clv: dict | None = None, evidence: dict | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every pre-kickoff fixture x family -> (shadow rows, per-side rows for the stake plan)."""
    league_n = league_n or {}
    clv = clv or {}
    gates = _Gates()
    tipped = {}
    if side_bets is not None and len(side_bets):
        for r in side_bets.itertuples(index=False):
            tipped[(str(r.date)[:10], str(r.home_team), str(r.away_team), str(r.market).lower())] = r
    rows, per_side = [], []
    pts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    for r in preds.to_dict("records"):
        ko = ME._utc(r.get("kickoff_utc"))
        if ko is None or ko <= now:                                   # pre-kickoff only
            continue
        md = str(r.get("date"))[:10]
        home, away, league = str(r.get("home_team")), str(r.get("away_team")), str(r.get("league"))
        match = f"{home} vs {away}"
        track = str(r.get("model_type") or config.model_type_for_league(league))
        for fam, (pcol, oa_col, ob_col, vcol, legacy_mkt, plan_mkt) in FAMILIES.items():
            if pcol not in r:
                continue
            c = ME.Candidate(
                family=fam, track=track, p_model=r.get(pcol), kickoff=ko, now=now,
                quotes=pairs.get((md, match, fam), []),
                row_odds_a=_num(r.get(oa_col)), row_odds_b=_num(r.get(ob_col)), league=league,
                league_history_n=league_n.get(league),
                model_trained_at=ME.trained_at_from_version(r.get(vcol)),
                clv=clv.get(f"{track}|{fam}"), evidence=evidence)
            e = ME.evaluate(c, params)
            sides = e.pop("_sides", [])
            # legacy, exactly as production computed it
            if fam == "OU25":
                l_edge, l_tier = _num(r.get("best_edge")), str(r.get("signal_tier", ""))
                l_side = str(r.get("best_side", ""))
                l_odds = _num(r.get(oa_col if l_side == "OVER" else ob_col))
                l_th = gates.threshold(league, "ou25", l_side, l_tier)
            else:
                po, oo = _num(r.get(pcol)), _num(r.get(oa_col))
                l_side = "YES" if fam == "BTTS" else "OVER"
                l_odds = oo
                if np.isnan(po) or np.isnan(oo):
                    l_edge, l_tier = float("nan"), "NOT_EVALUATED"
                else:
                    l_edge = po - (1.0 / oo) / LEGACY_SIDE_OVERROUND
                    t = tipped.get((md, home, away, legacy_mkt))
                    l_tier = str(t.signal_tier) if t is not None else "AVOID"
                l_th = gates.threshold(league, legacy_mkt, l_side, l_tier)
            rows.append({
                "prediction_ts": pts, "fixture_id": _fixture_id(r), "match_date": md,
                "kickoff_utc": ko.strftime("%Y-%m-%dT%H:%M:%SZ"), "league": league,
                "home_team": home, "away_team": away, "model_type": track,
                "legacy_engine_version": ME.LEGACY_ENGINE_VERSION, "legacy_raw_edge": l_edge,
                "legacy_tier": l_tier, "legacy_side": l_side, "legacy_threshold": l_th,
                "legacy_odds": l_odds, "new_state": e["decision_state"], "new_side": e["side"],
                "new_reason": e["decision_reason"], **e})
            fixture = f"{md}|{home}|{away}"
            if not sides:                    # early exit (no price / stale / one-sided / no model)
                nan = float("nan")
                sides = [{"side": s, "p_market_fair": nan, "p_decision": nan, "p_lower_bound": nan,
                          "ev_lower_bound": nan} for s in ME.SIDES.get(fam, ())]
            for s in sides:
                per_side.append({
                    "fixture": fixture, "market": plan_mkt, "side": s["side"],
                    # the engine decides ONE side per market; the other side is not a selection
                    "decision_state": (e["decision_state"] if (not e["side"] or s["side"] == e["side"])
                                       else "NO_BET"),
                    "engine_side": e["side"], "engine_version": ME.ENGINE_VERSION,
                    "p_market_fair": round(s["p_market_fair"], 6),
                    "p_decision": round(s["p_decision"], 6),
                    "p_lower_bound": round(s["p_lower_bound"], 6),
                    "ev_lower_bound": round(s["ev_lower_bound"], 6),
                    "market_source": e["market_source"],
                    "clv_evidence_state": e["clv_evidence_state"]})
    return (pd.DataFrame(rows).reindex(columns=SHADOW_COLS) if rows else pd.DataFrame(columns=SHADOW_COLS),
            pd.DataFrame(per_side))


def first_sight_store(path: Path | None = None):
    """The shared first-sight store (src/research/first_sight.py) for this log."""
    from src.research.first_sight import FirstSightStore
    return FirstSightStore(Path(path) if path else SHADOW_FILE, key=KEY, columns=SHADOW_COLS,
                           observed_col="prediction_ts", kickoff_col="kickoff_utc")


def append_first_sight(rows: pd.DataFrame, path: Path | None = None) -> int:
    """Append rows whose (engine version, fixture, market) was never logged. Existing bytes are
    never rewritten; within one batch the first row wins. Migrated 2026-10-25 onto the shared
    FirstSightStore; byte-identical to the previous inline implementation on committed inputs
    (evaluate_board already drops post-kickoff fixtures, so the store's kickoff refusal is a no-op)."""
    if rows is None or rows.empty:
        return 0
    return first_sight_store(path).append(rows)


def _source_mix(books_used) -> str:
    """'oddsapi' | 'apifootball' | 'both' | '' for one row's books_used."""
    srcs = {EAF.source_of(b) for b in str(books_used or "").split(";") if b and b != "nan"}
    return "both" if len(srcs) > 1 else (srcs.pop() if srcs else "")


def health(rows: pd.DataFrame, now: pd.Timestamp, runtime_s: float, appended: int,
           params: dict, n_live_quotes: int, af_stats: dict | None = None) -> dict:
    cov: dict = {}
    if rows is not None and len(rows):
        for (fam, tr), g in rows.groupby(["market", "model_type"]):
            two = g["market_source"].isin(["MEDIAN_TWO_SIDED_BOOKS", "SINGLE_TWO_SIDED_BOOK"])
            mix = g.loc[two, "books_used"].map(_source_mix)
            nb = g.loc[two, "books_used"].astype(str).str.split(";")
            n_af = nb.map(lambda xs: sum(1 for b in xs if EAF.source_of(b) == "apifootball"))
            n_oa = nb.map(len) - n_af
            age = pd.to_numeric(g.loc[two, "price_age_minutes"], errors="coerce")
            cov[f"{fam}|{tr}"] = {
                "fixtures": int(len(g)),
                "two_sided": int(two.sum()),
                "two_sided_share": round(float(two.mean()), 4),
                "multi_book": int((g["market_source"] == "MEDIAN_TWO_SIDED_BOOKS").sum()),
                "one_sided": int((g["market_source"] == "ONE_SIDED").sum()),
                "states": {k: int(v) for k, v in g["new_state"].value_counts().items()},
                "legacy_tipped": int(g["legacy_tier"].isin(["SNIPER", "MARKSMAN", "VALUABLE"]).sum()),
                "median_books": float(pd.to_numeric(g["n_books"], errors="coerce").median()),
                "two_sided_by_source": {k: int((mix == k).sum()) for k in ("oddsapi", "apifootball", "both")},
                "median_books_two_sided": {
                    "oddsapi": float(n_oa.median()) if len(n_oa) else None,
                    "apifootball": float(n_af.median()) if len(n_af) else None},
                "median_price_age_minutes": float(age.median()) if age.notna().any() else None,
                "reasons": {k: int(v) for k, v in
                            g["new_reason"].astype(str).str.split("(").str[0].value_counts().items()},
            }
    return {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection_engine_version": ME.ENGINE_VERSION,
        "config_status": params.get("status", "?"),
        "devig_method": params.get("devig_method"),
        "shadow_only": True,
        "runtime_seconds": round(runtime_s, 3),
        "rows_scored": int(len(rows)) if rows is not None else 0,
        "rows_appended_first_sight": int(appended),
        "current_run_book_quotes": int(n_live_quotes),
        "apifootball_book_quotes": af_stats or {},
        "prospective_evidence_loaded": bool(params.get("_evidence_loaded")),
        "coverage": cov,
    }


def run(preds: pd.DataFrame, bets: pd.DataFrame | None, side_bets: pd.DataFrame | None,
        historical: pd.DataFrame | None = None, now: pd.Timestamp | None = None,
        shadow_path: Path | None = None, health_path: Path | None = None,
        snapshots_path: Path | None = None, live_rows: list | None = None,
        af_quotes_dir: Path | None = None, use_af_quotes: bool = True) -> pd.DataFrame | None:
    """Score the board, append first-sight rows, write health. Returns the per-side frame the stake
    plan attaches as shadow columns, or None. NEVER RAISES."""
    t0 = time.perf_counter()
    try:
        now = now or pd.Timestamp.now(tz="UTC")
        params = ME.load_params()
        if live_rows is None:
            live_rows = []
            try:
                from src import predict as _p
                live_rows += list(getattr(_p, "_LAST_BOOK_ROWS", []) or [])
            except Exception:                                         # noqa: BLE001
                pass
            try:
                from src import btts_odds as _b
                live_rows += list(getattr(_b, "_LAST_BOOK_ROWS", []) or [])
            except Exception:                                         # noqa: BLE001
                pass
        q = _quote_frame(now, live_rows, snapshots_path)
        pairs = book_pairs(q, now)
        af_stats = {"enabled": bool(use_af_quotes)}
        if use_af_quotes:                       # API-Football batch-captured quotes (disk only)
            af_pairs, st = EAF.af_book_pairs(preds, now, af_quotes_dir)
            af_stats.update(st)
            pairs = EAF.merge_pairs(pairs, af_pairs)
        league_n = (historical["league"].astype(str).value_counts().to_dict()
                    if historical is not None and "league" in getattr(historical, "columns", []) else {})
        clv = {}
        try:
            bl = config.OUTPUT_DIR / "bets_ledger.csv"
            sl = config.OUTPUT_DIR / "side_bets_ledger.csv"
            clv = ME.clv_track_stats(
                pd.read_csv(bl, usecols=["model_type", "clv_pct"]) if bl.exists() else None,
                pd.read_csv(sl, usecols=["model_type", "market", "clv_pct"]) if sl.exists() else None,
                clip_pct=float(params["clv_clip_pct"]))
        except Exception as e:                                        # noqa: BLE001
            log.warning(f"[edge_shadow] CLV context unavailable ({type(e).__name__})")
        ev_path = Path(params["evidence_path"])
        if not ev_path.is_absolute():
            ev_path = Path(config.BASE_DIR) / ev_path if hasattr(config, "BASE_DIR") else ev_path
        evidence = ME.load_evidence(ev_path)
        params["_evidence_loaded"] = evidence is not None
        rows, per_side = evaluate_board(preds, side_bets, now, pairs, params, league_n, clv, evidence)
        n = append_first_sight(rows, shadow_path)
        try:                                     # SHADOW contract check (src/contracts); observe only
            from src.contracts.shadow import observe
            observe("edge_engine_shadow.csv (scored rows)", rows, "SELECTION_DECISION")
        except Exception:                                             # noqa: BLE001
            pass
        dt = time.perf_counter() - t0
        h = health(rows, now, dt, n, params, len(live_rows), af_stats)
        hp = Path(health_path) if health_path else HEALTH_FILE
        hp.write_text(json.dumps(h, indent=2, default=str), encoding="utf-8")
        st = rows["new_state"].value_counts().to_dict() if len(rows) else {}
        log.info(f"[edge_shadow] {ME.ENGINE_VERSION}: {len(rows)} fixture x market row(s), "
                 f"+{n} first-sight, states {st}, {dt:.2f}s (shadow only)")
        return per_side
    except Exception as e:                                            # noqa: BLE001 - never break predict
        log.warning(f"[edge_shadow] skipped ({type(e).__name__}: {e}) - tips are unaffected "
                    f"({time.perf_counter() - t0:.2f}s)")
        return None
