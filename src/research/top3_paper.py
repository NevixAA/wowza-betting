"""TOP-3 PAPER TRACKS R1 and R2 - PAPER / RESEARCH ONLY (protocol config/top3_ranking_paper_protocol_v2.json).

    python -m src.research.top3_paper run       # predict.yml, after src.top3_league run: freeze due league-weeks
    python -m src.research.top3_paper settle    # update_results.yml: grade finished paper picks
    python -m src.research.top3_paper summary   # rewrite the dashboard summary

Owner 2026-11-11: "Add the 3 tracks", then "Clean rebuild": the live TOP-3 is ranked by R3 (src/top3_league.py,
the third arm of the protocol); R1 and R2 are recorded HERE, beside it, as paper tracks. Flat 1 unit, the
production settlement, every row labelled PAPER/RESEARCH.

NOTHING HERE REACHES MONEY OR THE PUBLIC. This module writes ONLY under output/research/top3_paper/. It never
writes a tip, a stake, an order, a production ledger or a Telegram message, and nothing in production reads its
files (tests/test_top3_paper.py proves both).

THE POOL (owner 2026-11-11: "the top 3 should be based on our retrain fixed models with the correct optimized
bars per league"). A league-week's pool is the production TOP-3 pool (src.top3_league: candidates first seen
strictly before FREEZE = first kickoff - TOP3_FREEZE_LEAD_MIN, priced two-sided, at or above the league's OWN
owner-applied bar) restricted, per candidate, to:
  * MODEL_NOT_CLEAN   - its prediction_engine_version must be a model whose artifact sha256 is listed in a stage
                        of the owner's clean-data-switch override (config/promotion_overrides.json: first,
                        second or final refit). Any other model (pre-repair, v9_repaired, later) is refused.
  * CELL_NOT_APPROVED - its served bar must be a bars_opt cell (no legacy / global bar) that is IDENTICAL to the
                        cell of an approved proposal listed in config/top3_paper_tracks.json (hash-verified), and
                        the league x market must have been applied from that proposal (config/threshold_applied.json).
OFF cells give no candidates (production already returns NO_LEAGUE_BAR for them).

RANKING. src/research/top3_ranking.rank_order(rule) on the frozen per-cell statistics (scripts/top3_paper_freeze.py).
The statistics, the protocol and the ranking code are verified by hash on every run; any mismatch -> nothing is
frozen (status says why). Cap 3, never padded, direction rule against the track's own picks of the week AND the
fixture's live production tip, STARTED_BEFORE_ISSUE skipped - exactly as src.top3_league.select_week.

FLAG. env TOP3_PAPER_ENABLED (true/1/yes). Off (default): every entrypoint returns at once and writes nothing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = Path("config") / "top3_paper_tracks.json"
OUT_SUBDIR = Path("research") / "top3_paper"
FLAG_ENV = "TOP3_PAPER_ENABLED"
LABEL = "PAPER/RESEARCH"
STAKE_EFFECT = "NONE - PAPER/RESEARCH track: never staked, never executed, never published"
LEDGER_NAME = "paper_ledger.csv"
WEEKS_NAME = "paper_weeks.csv"
SUMMARY_NAME = "summary.json"
TOP_N = 3

R_NOT_CLEAN = "MODEL_NOT_CLEAN"
R_CELL = "CELL_NOT_APPROVED"
R_NOT_APPLIED_FROM = "NOT_APPLIED_FROM_APPROVED_PROPOSAL"

PICK_KEY = ["track", "league_week_id", "rank"]
PICK_FROZEN = [
    "pick_id", "label", "track", "protocol_sha256", "stats_sha256", "league_week_id", "rank", "week_lock_ts",
    "freeze_ts", "issued_at", "league", "model_type", "fixture_id", "match_date", "home_team", "away_team",
    "kickoff_utc", "market", "side", "odds", "p_model_raw", "p_market_fair", "p_used", "ev_est", "rank_key",
    "no_stat", "cell_track", "cell_mside", "cell_config", "cell_proposal_id", "prediction_engine_version",
    "model_artifact_sha256", "first_seen_at", "stake_effect",
]
PICK_GRADING = ["home_goals", "away_goals", "result", "won", "flat_pnl", "settle_status", "settled_at"]
PICK_COLS = PICK_FROZEN + PICK_GRADING
WEEK_KEY = ["track", "league_week_id"]
WEEK_COLS = ["label", "track", "league_week_id", "league", "week_lock_ts", "freeze_ts", "issued_at", "n_candidates",
             "n_pool", "n_picks", "excluded", "frozen_at"]
CELL_FIELDS = ("mside", "league", "track", "config", "source", "measure", "mode", "valuable_th", "marksman_th",
               "sniper_th", "odds_min", "odds_max", "source_params")

log = logging.getLogger(__name__)


# ── files, flag, hashes ─────────────────────────────────────────────────────────────────────────
def sha256_lf(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def enabled(env: dict | None = None) -> bool:
    return str((env if env is not None else os.environ).get(FLAG_ENV, "")).strip().lower() in ("1", "true", "yes", "on")


def prod_output(base: Path | None = None) -> Path:
    """The production output folder (read-only here: the candidate store, the live TOP-3 ledger, results)."""
    if base is not None:
        return Path(base)
    import config
    return Path(config.OUTPUT_DIR)


def out_dir(base: Path | None = None) -> Path:
    """This module's ONLY write location: <production output>/research/top3_paper."""
    return prod_output(base) / OUT_SUBDIR


def _json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:                                                  # noqa: BLE001
        return {}


def load_config(root: Path | None = None) -> dict:
    return _json((Path(root) if root else ROOT) / CONFIG_FILE)


def hash_problems(cfg: dict, root: Path | None = None) -> list[str]:
    """The protocol file and the ranking code must be exactly the pinned ones."""
    r = Path(root) if root else ROOT
    why = []
    pf = r / str(cfg.get("protocol_file") or "")
    if not cfg.get("protocol_file") or not pf.is_file():
        return [f"protocol file {cfg.get('protocol_file')!r} missing"]
    if sha256_lf(pf) != cfg.get("protocol_sha256"):
        why.append(f"protocol {pf.name} sha256 {sha256_lf(pf)[:12]} is not the pinned {str(cfg.get('protocol_sha256'))[:12]}")
    code = (_json(pf).get("rules_common") or {}).get("code") or {}
    cf = r / str(code.get("file") or "")
    if not code.get("file") or not cf.is_file() or sha256_lf(cf) != code.get("sha256_lf"):
        why.append(f"ranking code {code.get('file')} does not match the protocol's sha256")
    return why


def load_freeze(cfg: dict, root: Path | None = None):
    """(stats frame, record, sha256, problems). The ledger names the ONE frozen record; the record's own hash,
    protocol hash and code hash must all match."""
    from src.research import top3_ranking as R
    r = Path(root) if root else ROOT
    why = hash_problems(cfg, r)
    led = _json(r / str(cfg.get("freeze_ledger") or ""))
    rec_rel = led.get("record")
    if not rec_rel:
        return None, {}, "", why + ["statistics not frozen (no record in the freeze ledger)"]
    rp = r / rec_rel
    if not rp.is_file():
        return None, {}, "", why + [f"frozen record {rec_rel} missing"]
    sha = sha256_lf(rp)
    if sha != led.get("sha256"):
        why.append(f"frozen record sha256 {sha[:12]} is not the ledger's {str(led.get('sha256'))[:12]}")
    rec = _json(rp)
    if rec.get("protocol_sha256") != cfg.get("protocol_sha256"):
        why.append("frozen record was made under another protocol")
    return (R.stats_from_freeze(rec) if not why else None), rec, sha, why


# ── the owner's pool guards ─────────────────────────────────────────────────────────────────────
def clean_artifacts(overrides: dict, override_id: str) -> dict:
    """{champion slot: {sha256}} of every stage of the owner's clean-data-switch override."""
    out: dict = {}
    for o in overrides.get("overrides") or []:
        if o.get("id") != override_id:
            continue
        for st in o.get("stages") or []:
            for slot, art in (st.get("artifacts") or {}).items():
                if isinstance(art, dict) and art.get("sha256"):
                    out.setdefault(slot, set()).add(str(art["sha256"]))
    return out


def model_guard(cand: dict, clean: dict, champions: dict) -> tuple[str, str]:
    """("", artifact sha256) when the candidate's model is a clean-switch artifact, else (reason, "")."""
    from src.research import league_weekly as LW
    fam = str(cand.get("engine_market") or "")
    slot = LW.CHAMP_SLOT.get(fam) or ("standard" if str(cand.get("model_type")) == "standard" else "newformat")
    ver = str(cand.get("prediction_engine_version") or "").strip()
    if not ver or "_" not in ver:
        return R_NOT_CLEAN, ""
    short = ver.rsplit("_", 1)[-1].lower()
    hits = [s for s in clean.get(slot, set()) if len(short) >= 8 and s.startswith(short)]
    ch = champions.get(slot) or {}
    if ch.get("model_version") == ver:                      # the full hash is known: it must be listed itself
        hits = [s for s in hits if s == ch.get("artifact_sha256")]
    return ("", hits[0]) if len(hits) == 1 else (R_NOT_CLEAN, "")


def approved_cells(cfg: dict, root: Path | None = None) -> tuple[dict, list[str]]:
    """{(params market, league, cell side): (cell, proposal id)} of every approved proposal, each file
    verified against its pinned id and hash."""
    from src import threshold_proposals as TP
    r = Path(root) if root else ROOT
    out, why = {}, []
    for a in cfg.get("approved_proposals") or []:
        p = _json(r / str(a.get("file") or ""))
        if not p or p.get("proposal_id") != a.get("proposal_id") or p.get("proposal_hash") != a.get("proposal_hash") \
                or TP.proposal_hash(p) != a.get("proposal_hash"):
            why.append(f"approved proposal {a.get('proposal_id')} file/hash does not verify")
            continue
        for e in p.get("entries") or []:
            for side, cell in (((e.get("proposed") or {}).get("bars_opt")) or {}).items():
                out[(str(e.get("market")), str(e.get("league")), str(side))] = (cell, p["proposal_id"])
    return out, why


def applied_from(root: Path | None = None) -> dict:
    """{"<market>|<league>": proposal id of its LAST application} from config/threshold_applied.json."""
    led = _json((Path(root) if root else ROOT) / "config" / "threshold_applied.json")
    out = {}
    for h in led.get("history") or []:
        for e in h.get("entries") or []:
            out[str(e)] = str(h.get("proposal_id"))
    return out


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12)
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def cell_guard(cell: dict | None, key: tuple, approved: dict, applied: dict) -> tuple[str, str]:
    """("", proposal id) when the served cell is IDENTICAL to an approved proposal's cell and the entry was
    applied from an approved proposal; else (reason, "")."""
    if not isinstance(cell, dict):
        return R_CELL, ""
    hit = approved.get(key)
    if hit is None:
        return R_CELL, ""
    ref, pid = hit
    if any(not _same(cell.get(f), ref.get(f)) for f in CELL_FIELDS):
        return R_CELL, ""
    ids = {a for a in {v[1] for v in approved.values()}}
    if applied.get(f"{key[0]}|{key[1]}") not in ids:
        return R_NOT_APPLIED_FROM, ""
    return "", pid


# ── the pool and the selection ──────────────────────────────────────────────────────────────────
def pool(g: pd.DataFrame, ctx: dict, guards: dict) -> tuple[pd.DataFrame, Counter]:
    """The league-week's paper pool: production-qualified candidates that pass the owner's guards, with the
    cell (track, mside), p_used and ev_est of the cell's served source. Pure apart from `guards` inputs."""
    from src import top3_league as TL
    from src.bars_opt import serve as BS
    excl: Counter = Counter()
    q = TL.qualify(g, ctx)
    rows = []
    for r in q.to_dict("records"):
        if r.get("reason"):
            excl[r["reason"]] += 1
            continue
        why, sha = model_guard(r, guards["clean"], guards["champions"])
        if why:
            excl[why] += 1
            continue
        key = TL.MARKETS[str(r["market"])][0]
        cs = TL._CELL_SIDE.get((key, str(r["side"]).upper()))
        b, _ = TL.league_bars(ctx, str(r["league"]), str(r["market"]), str(r["side"]))
        why, pid = cell_guard((b or {}).get("cell"), (key, str(r["league"]), str(cs)), guards["approved"],
                              guards["applied"])
        if why:
            excl[why] += 1
            continue
        cell = b["cell"]
        res = BS.score(pd.DataFrame([{"league": str(r["league"]), "p_side": TL._num(r.get("p_model_raw")),
                                      "fair": TL._num(r.get("p_market_fair")), "odds": TL._num(r.get("odds"))}]), cell)
        if str(res["tier"].iloc[0] or "") == "":
            excl[str(res["reason"].iloc[0] or "BELOW_LEAGUE_BAR")] += 1
            continue
        rows.append(r | {"track": cell["track"], "mside": cell["mside"], "fair": TL._num(r.get("p_market_fair")),
                         "p_used": float(res["p_used"].iloc[0]), "ev_est": float(res["ev_est"].iloc[0]),
                         "cell_config": cell["config"], "cell_proposal_id": pid, "model_artifact_sha256": sha})
    return pd.DataFrame(rows), excl


def take(order: pd.DataFrame, freeze: pd.Timestamp, issue: pd.Timestamp, committed: dict) -> tuple[list[int], Counter]:
    """src.top3_league.select_week's taking rules on an already ranked pool (row positions of `order`)."""
    from src import top3_league as TL
    from src.research import league_weekly as LW
    excl: Counter = Counter()
    fs, ko = LW._ts(order["first_seen_at"]), LW._ts(order["kickoff_utc"])
    taken, dirs = [], {}
    for i, r in enumerate(order.to_dict("records")):
        if not (pd.notna(fs.iloc[i]) and fs.iloc[i] < freeze):
            excl[TL.R_LATE] += 1
            continue
        if len(taken) >= TOP_N:
            excl[TL.R_RANK] += 1
            continue
        if not (pd.notna(ko.iloc[i]) and ko.iloc[i] > issue):
            excl[TL.R_STARTED] += 1
            continue
        fx = (str(r["match_date"])[:10], str(r["home_team"]), str(r["away_team"]))
        dr = TL._direction(r["market"], r["side"])
        stood = dirs.get(fx) or (committed.get(fx) or {}).get("direction")
        if dr and stood and dr != stood:
            excl[TL.R_DIRECTION] += 1
            continue
        if dr:
            dirs[fx] = dr
        taken.append(i)
    return taken, excl


def freeze_due(cands: pd.DataFrame, done: set, ctx: dict, now: pd.Timestamp, lead_min: float, committed: dict,
               stats: pd.DataFrame, guards: dict, cfg: dict, stats_sha: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pick rows, week rows) for every (track, league-week) whose FREEZE has passed, is on/after the protocol
    start and is not decided yet. A week with 0 picks is still recorded. Pure apart from injected inputs."""
    from src.research import league_weekly as LW
    from src.research import top3_ranking as R
    if cands is None or cands.empty:
        return pd.DataFrame(columns=PICK_COLS), pd.DataFrame(columns=WEEK_COLS)
    cands = LW._plain(cands)
    start = pd.Timestamp(cfg["start_utc"])
    locks = LW.week_locks(cands)
    stamp = LW._stamp(now)
    picks, weeks = [], []
    for week, g in cands.groupby("league_week_id", sort=True):
        lock = locks.get(week)
        if not week or lock is None:
            continue
        freeze = lock - pd.Timedelta(minutes=float(lead_min))
        if now < freeze or freeze < start or all((t, week) in done for t in cfg["tracks"]):
            continue
        pl, excl0 = pool(g, ctx, guards)
        for track in cfg["tracks"]:
            if (track, week) in done:
                continue
            excl = Counter(excl0)
            chosen = pd.DataFrame()
            if len(pl):
                order = R.rank_order(pl, stats, track, tiebreak=("kickoff_utc", "fixture_id")).reset_index(drop=True)
                idx, ex = take(order, freeze, now, committed)
                excl.update(ex)
                chosen = order.iloc[idx]
            weeks.append({"label": LABEL, "track": track, "league_week_id": week, "league": str(g["league"].iloc[0]),
                          "week_lock_ts": LW._stamp(lock), "freeze_ts": LW._stamp(freeze), "issued_at": stamp,
                          "n_candidates": int(len(g)), "n_pool": int(len(pl)), "n_picks": int(len(chosen)),
                          "excluded": json.dumps(dict(excl), sort_keys=True), "frozen_at": stamp})
            for k, r in enumerate(chosen.to_dict("records"), start=1):
                picks.append({c: r.get(c, "") for c in PICK_FROZEN if c in r} | {
                    "pick_id": LW.research_id(f"PAPER-{track}", week, str(r["market"]), str(r["fixture_id"]), str(r["side"])),
                    "label": LABEL, "track": track, "protocol_sha256": cfg["protocol_sha256"], "stats_sha256": stats_sha,
                    "league_week_id": week, "rank": k, "week_lock_ts": LW._stamp(lock), "freeze_ts": LW._stamp(freeze),
                    "issued_at": stamp, "p_market_fair": LW._r(r["fair"]), "p_used": LW._r(r["p_used"]),
                    "ev_est": LW._r(r["ev_est"]), "rank_key": LW._r(r["key"]) if pd.notna(r["key"]) else "",
                    "no_stat": int(r["no_stat"]), "cell_track": r["track"], "cell_mside": r["mside"],
                    "stake_effect": STAKE_EFFECT})
    return (pd.DataFrame(picks).reindex(columns=PICK_COLS) if picks else pd.DataFrame(columns=PICK_COLS),
            pd.DataFrame(weeks).reindex(columns=WEEK_COLS) if weeks else pd.DataFrame(columns=WEEK_COLS))


# ── stores ──────────────────────────────────────────────────────────────────────────────────────
def picks_store(od: Path):
    from src.research.first_sight import FirstSightStore
    return FirstSightStore(Path(od) / LEDGER_NAME, key=PICK_KEY, columns=PICK_COLS, observed_col="issued_at",
                           kickoff_col="kickoff_utc", frozen_cols=PICK_FROZEN, digest=True)


def weeks_store(od: Path):
    from src.research.first_sight import FirstSightStore
    return FirstSightStore(Path(od) / WEEKS_NAME, key=WEEK_KEY, columns=WEEK_COLS)


def guards_from(cfg: dict, root: Path | None = None) -> tuple[dict, list[str]]:
    r = Path(root) if root else ROOT
    approved, why = approved_cells(cfg, r)
    return {"clean": clean_artifacts(_json(r / "config" / "promotion_overrides.json"), cfg.get("clean_override_id", "")),
            "champions": _json(r / "models" / "registry" / "current_champions.json"),
            "approved": approved, "applied": applied_from(r)}, why


# ── entrypoints ─────────────────────────────────────────────────────────────────────────────────
def run(now=None, out: Path | None = None, root: Path | None = None, env: dict | None = None,
        committed: dict | None = None, cands: pd.DataFrame | None = None) -> dict:
    """`out` = the production output folder (default config.OUTPUT_DIR): read for the candidate store, written
    only under its research/top3_paper subfolder."""
    """Flag OFF -> {'enabled': False}, nothing written. Never raises."""
    if not enabled(env):
        return {"enabled": False}
    from src import top3_league as TL
    from src.research import league_weekly as LW
    now = LW._now(now)
    od = out_dir(out)
    res = {"enabled": True, "label": LABEL, "picks_appended": 0, "weeks_decided": 0}
    try:
        cfg = load_config(root)
        lead = TL.freeze_lead_min(env)
        stats, _rec, sha, why = load_freeze(cfg, root)
        guards, gwhy = guards_from(cfg, root)
        why += gwhy
        if lead is None:
            why.append("TOP3_FREEZE_LEAD_MIN not set")
        if why:
            res["status"] = "NOT_RUN: " + "; ".join(why)
        else:
            if cands is None:
                cands = TL.candidate_store(prod_output(out)).read()
            if committed is None:
                from src.direction_guard import committed as _committed
                committed = _committed()
            od.mkdir(parents=True, exist_ok=True)
            ws = weeks_store(od)
            w0 = ws.read()
            done = set(zip(w0["track"], w0["league_week_id"])) if len(w0) else set()
            picks, weeks = freeze_due(cands, done, TL.bars_context(root), now, lead, committed, stats, guards, cfg, sha)
            res["picks_appended"] = picks_store(od).append(picks)
            res["weeks_decided"] = ws.append(weeks)
            res["status"] = "OK"
        write_summary(now, out, cfg, res.get("status"))
    except Exception as e:                                             # noqa: BLE001
        log.warning(f"[top3_paper] run skipped ({type(e).__name__}: {e}) - production unaffected")
        res["error"] = f"{type(e).__name__}: {e}"[:300]
    return res


def settle_picks(picks: pd.DataFrame, v91: pd.DataFrame | None, now: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    """Production TOP-3 settlement (src.top3_league.settle_picks rules) on the paper columns; frozen columns
    hashed before and after."""
    from src.research import league_weekly as LW
    from src.research.first_sight import frozen_digest
    from src.shadow_settlement import grade
    out = picks.copy().reindex(columns=PICK_COLS).fillna("")
    before = frozen_digest(out, PICK_FROZEN)
    by_id, by_key = LW._scores(v91)
    n, stamp = 0, LW._stamp(now)
    for i in range(len(out)):
        r = out.iloc[i]
        if LW._s(r["settle_status"]):
            continue
        ko = pd.to_datetime(r["kickoff_utc"], utc=True, errors="coerce")
        if pd.isna(ko) or now < ko + pd.Timedelta(minutes=LW.SETTLE_DELAY_MIN):
            continue
        g = by_id.get(str(r["fixture_id"]).strip()) or by_key.get((str(r["match_date"])[:10], str(r["home_team"]),
                                                                   str(r["away_team"])))
        if g is None:
            continue
        hg, ag, _src = g
        _outcome, result = grade(LW.SETTLE_MARKET[str(r["market"])], str(r["side"]), hg, ag)
        try:
            odds = float(r["odds"])
        except ValueError:
            odds = float("nan")
        pnl = ((odds - 1.0) if result == "WIN" else -1.0) if result in ("WIN", "LOSS") and odds > 1 else float("nan")
        vals = {"home_goals": hg, "away_goals": ag, "result": result or "",
                "won": "1" if result == "WIN" else "0" if result == "LOSS" else "", "flat_pnl": LW._r(pnl),
                "settle_status": "SETTLED" if result else "SETTLED_NO_SIDE", "settled_at": stamp}
        for k, v in vals.items():
            out.iat[i, out.columns.get_loc(k)] = "" if v == "" else str(v)
        n += 1
    if frozen_digest(out, PICK_FROZEN) != before:
        raise RuntimeError("top3_paper settlement would alter a frozen column - refused")
    return out, n


def settle(now=None, out: Path | None = None, env: dict | None = None, v91: pd.DataFrame | None = None) -> dict:
    if not enabled(env):
        return {"enabled": False}
    from src.research import league_weekly as LW
    now = LW._now(now)
    od = out_dir(out)
    ps = picks_store(od)
    picks = ps.read()
    if not len(picks):
        return {"enabled": True, "settled": 0}
    v = ps.verify()
    if v["ok"] is False:
        raise RuntimeError(f"paper ledger TAMPERED ({v['reason']}) - not settling")
    if v91 is None:
        v91 = LW._read(prod_output(out) / "v91_shadow.csv")
    res, n = settle_picks(picks, v91, now)
    if n:
        p = od / LEDGER_NAME
        tmp = p.with_suffix(".csv.tmp")
        res.to_csv(tmp, index=False, lineterminator="\n")
        tmp.replace(p)
    write_summary(now, out, load_config(), "OK")
    return {"enabled": True, "settled": n}


def track_summary(picks: pd.DataFrame) -> dict:
    """Picks, settled, won, flat P/L, ROI and the study's 95% CI - no prices."""
    from src.bars_opt import study as S
    st = picks[picks["settle_status"].astype(str) == "SETTLED"] if len(picks) else picks
    pnl = pd.to_numeric(st["flat_pnl"], errors="coerce") if len(st) else pd.Series(dtype=float)
    ok = st[pnl.notna()] if len(st) else st
    sc = S.score(pd.DataFrame({"pnl": pnl[pnl.notna()].to_numpy(float),
                               "date": pd.to_datetime(ok["match_date"].astype(str).str[:10]) if len(ok) else pd.Series(dtype="datetime64[ns]"),
                               "odds": np.nan, "odds_c": np.nan})) if len(ok) else {"ci95": [None, None]}
    return {"picks": int(len(picks)), "settled": int(len(ok)), "won": int((ok["won"].astype(str) == "1").sum()) if len(ok) else 0,
            "flat_pnl_units": round(float(pnl.sum()), 3) if len(ok) else 0.0,
            "roi": round(float(pnl.mean()), 4) if len(ok) else None, "ci95": sc.get("ci95"),
            "no_stat_picks": int((pd.to_numeric(picks.get("no_stat"), errors="coerce") == 1).sum()) if len(picks) and "no_stat" in picks else 0,
            "reached_min_picks": bool(len(ok) >= 150)}


def summary(picks: pd.DataFrame, now: pd.Timestamp, cfg: dict, status: str | None, live: pd.DataFrame | None = None) -> dict:
    from src.research import league_weekly as LW
    picks = picks if picks is not None and len(picks) else pd.DataFrame(columns=PICK_COLS)
    tracks = {t: track_summary(picks[picks["track"] == t]) for t in cfg.get("tracks", [])}
    out = {"generated_at": LW._stamp(now), "label": LABEL, "status": status,
           "protocol_sha256": cfg.get("protocol_sha256"), "start_utc": cfg.get("start_utc"),
           "note": "PAPER/RESEARCH: flat 1 unit, never staked, never published. R3 is the live TOP-3 (its own "
                   "ledger); evaluation dates and test in the protocol file.",
           "tracks": tracks}
    if live is not None and len(live):
        lv = live[pd.to_datetime(live["kickoff_utc"], utc=True, errors="coerce") >= pd.Timestamp(cfg.get("start_utc"))]
        out["live_R3_from_production_ledger"] = track_summary(lv)
    return out


def write_summary(now, out: Path | None, cfg: dict, status: str | None) -> dict:
    """Dashboard summary per track (picks, ROI, CI; no prices) + the live R3 arm read from the production
    TOP-3 ledger (read-only)."""
    from src.research import league_weekly as LW
    od = out_dir(out)
    live = LW._read(prod_output(out) / "top3_league_ledger.csv")
    rep = summary(picks_store(od).read() if (od / LEDGER_NAME).exists() else None, LW._now(now), cfg, status,
                  live if len(live) and "settle_status" in live else None)
    od.mkdir(parents=True, exist_ok=True)
    LW._write_json(od / SUMMARY_NAME, rep)
    return rep


def _cli(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    cmd = argv[0] if argv else "run"
    if cmd == "run":
        print(json.dumps(run(), default=str))
    elif cmd == "settle":
        print(json.dumps(settle(), default=str))
    elif cmd == "summary":
        if not enabled():
            print(json.dumps({"enabled": False}))
        else:
            print(json.dumps(write_summary(None, None, load_config(), "OK")["tracks"], default=str))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
