"""PROPS TOP-3 PER LEAGUE PER GAMEWEEK (owner 2026-11-22, chat: "Like TOP-3: 3 per league, stake by proof").

Replaces the old daily props feed (top 20 priced props by EV, labelled with the retired SNIPER / MARKSMAN /
VALUABLE tiers) and the board's top-N "PAPER" promotion. Per prop league per gameweek, three player props:

  * T1 BEST VALUE    the highest VALUE among the week's eligible props.
  * T2 SAFEST        the highest model probability among the rest.
  * T3 RISKY VALUE   the longest odds among the rest.

ELIGIBLE = a priced prop (a real captured bookmaker price, market_odds > 1; no tip without a price) with POSITIVE
value, in a registry prop league, not a rare market (goals2 / goals3 / sot3 / sot4 stay in the rare-props research,
player_model/rare_props.py, max 3 per gameweek, owner 2026-11-09), not a goalkeeper on an attacking market.

VALUE (stated on every pick, column value_basis):
  * "vs de-vigged fair"  when the prop has a REAL two-sided price (fair_basis two_sided_devig): the model's EV at
                         the de-vigged fair price, p_model / p_fair - 1 (= the board's edge_rel).
  * "vs quoted price"    otherwise: the model's EV at the quoted price, p_model x odds - 1 (= the board's ev). An
                         assumed margin is never used to make a fair price here.

CONVENTIONS - the same as src/top3_league.py (team TOP-3):
  * League-week = src/gameweek.league_week_id (GW-1, tip day). LOCK = the earliest kickoff among the week's prop
    candidates; FREEZE = LOCK - TOP3_FREEZE_LEAD_MIN (env, minutes; no default: unset -> nothing is frozen).
  * FIRST SIGHT: a candidate is stored once, the first time it is seen priced (output/props_top3_candidates.csv);
    its values are never rewritten. Only candidates first seen strictly before FREEZE are members of the week.
  * The week is decided ONCE, at the first run at/after FREEZE; a candidate whose kickoff is not after the moment
    of issue is skipped (never sent after kickoff) and the next one takes the slot. Never padded: fewer than 3
    eligible -> fewer picks, with the reasons recorded on the week row.
  * Flag: env TOP3_LEAGUE_ENABLED (true/1/yes); off -> every entrypoint returns at once and writes nothing.
  * Start: config/props_top3.json start_utc - a week whose freeze is before it is never decided.
  * COHERENCE: the shared fixture gate (src/tip_coherence). A prop that mismatches a tip already SENT on the same
    match (e.g. anytime scorer vs Under 0.5 / Under 2.5 / BTTS No) is never a pick; the next one takes the slot.
    The prop's match is resolved to the team board's names (output/predictions.csv: same league, same day,
    fuzzy team names) so the gate sees the team tips; an unresolved match keeps its own names (logged).

STAKE BY PROOF. No prop cell is proven, so every pick carries stake 0 and the label "No stake: props not proven
yet" (config/props_top3.json). Nothing here sizes, writes or places a stake.

LEDGER. output/props_top3_ledger.csv - append-only first-sight picks (frozen columns hashed, sidecar) + grading
columns filled once by settle() from output/player_ledger.csv (the same player result the prediction ledger uses),
flat 1u at the pick's odds for the record only. Every prop prediction still goes to the prediction ledger
(player_model/predict.py, unchanged).

    python -m src.props_top3 run       # player_props.yml: store first-sight candidates, freeze due weeks
    python -m src.props_top3 settle    # player_props.yml: grade finished picks from player_ledger.csv
    python -m src.props_top3 notify    # player_props.yml: send each frozen league-week once
    python -m src.props_top3 report    # rewrite output/props_top3_report.json

Prices are never printed in logs.
"""
from __future__ import annotations

import difflib
import json
from collections import Counter
import logging
import os
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

import config
from src.gameweek import league_week_id
from src.research import league_weekly as LW
from src.research.first_sight import FirstSightStore

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT = config.OUTPUT_DIR
FLAG_ENV = "TOP3_LEAGUE_ENABLED"            # one switch for both TOP-3s (owner: "Like TOP-3")
LEAD_ENV = "TOP3_FREEZE_LEAD_MIN"
SELECTION_MODE = "PROPS_TOP3_LEAGUE_GW"
VERSION = "PROPS-TOP3-1"
SETTINGS_FILE = Path("config") / "props_top3.json"

ROLES = ("T1_BEST_VALUE", "T2_SAFEST", "T3_RISKY_VALUE")
ROLE_TEXT = {"T1_BEST_VALUE": "T1 best value", "T2_SAFEST": "T2 safest", "T3_RISKY_VALUE": "T3 risky value"}
STAKE_LABEL_DEFAULT = "No stake: props not proven yet"     # also pinned in config/props_top3.json
BASIS_FAIR = "vs de-vigged fair"
BASIS_QUOTE = "vs quoted price"
SETTLE_DELAY_MIN = 150                       # a finished match: kickoff + 150 min, as the team ledgers

CANDIDATES_NAME = "props_top3_candidates.csv"
PICKS_NAME = "props_top3_ledger.csv"
WEEKS_NAME = "props_top3_weeks.csv"
REPORT_NAME = "props_top3_report.json"
NOTIFIED_NAME = "props_top3_notified.json"
BOARD_NAME = "player_tips.csv"
PLAYER_LEDGER_NAME = "player_ledger.csv"

MARKET_TEXT = {"goals": "Anytime scorer", "assists": "Assist", "sot": "1+ shot on target",
               "sot2": "2+ shots on target", "cards": "Booked"}

# exclusion reasons (one per candidate, the first that applies)
R_NO_VALUE = "NO_POSITIVE_VALUE"
R_LATE = "FIRST_SEEN_AFTER_FREEZE"
R_STARTED = "STARTED_BEFORE_ISSUE"
R_MISMATCH = "MISMATCH_WITH_SENT_TIP"
R_RANK = "RANKED_BELOW_TOP3"

CAND_KEY = ["league_week_id", "fixture_id", "player_id", "market"]
CAND_COLS = CAND_KEY + [
    "league", "match_date", "kickoff_utc", "match", "home_team", "away_team", "player_name", "team", "position",
    "model_prob", "odds", "fair_prob", "fair_basis", "value", "value_basis", "first_seen_at",
    # owner 2026-11-23 "Yes, do all" (D6), appended: the book + capture time of `odds`, carried from the board row
    # (player_tips.csv entry_bookmaker / entry_captured_at, player_model/odds_fetcher.match_meta_to_tips). Blank when
    # the board carries none; older rows stay blank (FirstSightStore.widen), never back-filled.
    "entry_bookmaker", "entry_captured_at"]

PICK_KEY = ["selection_mode", "league_week_id", "props_rank"]
PICK_FROZEN = [
    "pick_id", "selection_mode", "version", "league_week_id", "props_rank", "role", "week_lock_ts", "freeze_ts",
    "issued_at", "league", "fixture_id", "match_date", "kickoff_utc", "match", "home_team", "away_team",
    "board_home_team", "board_away_team", "player_id", "player_name", "team", "position", "market", "odds",
    "model_prob", "fair_prob", "fair_basis", "value", "value_basis", "first_seen_at", "cell_status", "stake",
    "stake_label",
    # owner 2026-11-23 "Yes, do all" (D6): who quoted `odds` and when (from the candidate). Frozen like the rest, but
    # stored AFTER the grading columns so the written ledger's header stays a prefix (FirstSightStore.widen).
    "entry_bookmaker", "entry_captured_at"]
PICK_GRADING = ["result", "won", "flat_pnl", "settle_status", "settled_at"]
ENTRY_COLS = ("entry_bookmaker", "entry_captured_at")
PICK_COLS = [c for c in PICK_FROZEN if c not in ENTRY_COLS] + PICK_GRADING + list(ENTRY_COLS)

WEEK_KEY = ["selection_mode", "league_week_id"]
WEEK_COLS = ["selection_mode", "version", "league_week_id", "league", "week_lock_ts", "freeze_ts", "issued_at",
             "n_candidates", "n_eligible", "n_picks", "excluded", "frozen_at"]


# ── switches ────────────────────────────────────────────────────────────────────────────────────
def enabled(env: dict | None = None) -> bool:
    v = (env if env is not None else os.environ).get(FLAG_ENV, "")
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def freeze_lead_min(env: dict | None = None) -> float | None:
    """The owner's lead in minutes (the team TOP-3's), or None when unset/unusable (no default)."""
    v = str((env if env is not None else os.environ).get(LEAD_ENV, "")).strip()
    try:
        x = float(v) if v else None
    except ValueError:
        return None
    return x if x is not None and x > 0 else None


def settings(root: Path | None = None) -> dict:
    try:
        return json.loads(((Path(root) if root else ROOT) / SETTINGS_FILE).read_text(encoding="utf-8"))
    except Exception:                                                  # noqa: BLE001
        return {}


def start(root: Path | None = None) -> pd.Timestamp | None:
    v = settings(root).get("start_utc")
    try:
        ts = pd.Timestamp(v) if v else None
    except (TypeError, ValueError):
        return None
    if ts is None or pd.isna(ts):
        return None
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def stake_label(root: Path | None = None) -> str:
    return str(settings(root).get("stake_label") or STAKE_LABEL_DEFAULT)


# ── candidates ──────────────────────────────────────────────────────────────────────────────────
def _num(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _id(v) -> str:
    s = LW._s(v)
    return s[:-2] if s.endswith(".0") else s


def _split_match(m: str) -> tuple[str, str]:
    p = re.split(r"\s+vs\.?\s+|\s+v\s+", str(m or "").strip(), maxsplit=1)
    return (p[0].strip(), p[1].strip()) if len(p) == 2 else ("", "")


def value_of(model_prob: float, odds: float, fair_prob: float, fair_basis: str) -> tuple[float, str]:
    """(value, basis). Two-sided price -> EV at the de-vigged fair price; else EV at the quoted price."""
    from player_model.fair_basis import TWO_SIDED_DEVIG
    if str(fair_basis or "").strip() == TWO_SIDED_DEVIG and 0.0 < fair_prob < 1.0:
        return model_prob / fair_prob - 1.0, BASIS_FAIR
    return model_prob * odds - 1.0, BASIS_QUOTE


def build_candidates(board: pd.DataFrame, now: pd.Timestamp, prop_league=None) -> pd.DataFrame:
    """Eligible-shape rows of the props board (player_tips.csv) at `now`: priced, upcoming, prop league, not a rare
    market, no goalkeeper on an attacking market, one row per (fixture, player, market). Value is computed here;
    positivity is decided at freeze (a non-positive row is stored and counted, never picked)."""
    if board is None or board.empty:
        return pd.DataFrame(columns=CAND_COLS)
    from player_model import config as PC
    d = board.copy()
    rare = set(getattr(PC, "HIDDEN_PENDING_CLEAN_RETRAIN", ()))
    gk_blocked = set(getattr(PC, "GK_BLOCKED_MARKETS", ()))
    d["_odds"] = pd.to_numeric(d.get("market_odds"), errors="coerce")
    d["_p"] = pd.to_numeric(d.get("model_prob"), errors="coerce")
    d["_fair"] = pd.to_numeric(d.get("fair_implied"), errors="coerce")
    d["_ko"] = pd.to_datetime(d.get("kickoff_utc"), errors="coerce", utc=True)
    keep = (d["_odds"] > 1.0) & d["_p"].between(0, 1, inclusive="neither") & d["_ko"].notna() & (d["_ko"] > now)
    keep &= ~d["market"].astype(str).isin(rare)
    pos = d.get("position", pd.Series("", index=d.index)).astype(str).str.strip().str.upper()
    keep &= ~(pos.str.startswith("G") & d["market"].astype(str).isin(gk_blocked))
    if prop_league is not None:
        keep &= prop_league(d["league"]).astype(bool)
    d = d[keep]
    if d.empty:
        return pd.DataFrame(columns=CAND_COLS)
    rows = []
    stamp = LW._stamp(now)
    for r in d.to_dict("records"):
        home, away = _split_match(r.get("match"))
        val, basis = value_of(r["_p"], r["_odds"], r["_fair"], r.get("fair_basis"))
        ko = r["_ko"]
        rows.append({
            "league_week_id": league_week_id(r.get("league"), ko, str(r.get("date"))[:10]),
            "fixture_id": _id(r.get("fixture_id")), "player_id": _id(r.get("player_id")),
            "market": str(r.get("market")), "league": str(r.get("league")),
            "match_date": ko.strftime("%Y-%m-%d"), "kickoff_utc": LW._stamp(ko), "match": str(r.get("match")),
            "home_team": home, "away_team": away, "player_name": str(r.get("player_name")),
            "team": LW._s(r.get("team")), "position": LW._s(r.get("position")),
            "model_prob": LW._r(r["_p"]), "odds": LW._r(r["_odds"]),
            "fair_prob": LW._r(r["_fair"]), "fair_basis": LW._s(r.get("fair_basis")),
            "value": LW._r(val), "value_basis": basis, "first_seen_at": stamp,
            # D6: the board row's own book + capture time of market_odds; blank when it carries none
            "entry_bookmaker": LW._s(r.get("entry_bookmaker")),
            "entry_captured_at": LW._s(r.get("entry_captured_at"))})
    out = pd.DataFrame(rows).reindex(columns=CAND_COLS)
    out = out[(out["league_week_id"] != "") & (out["fixture_id"] != "") & (out["player_id"] != "")]
    return out.drop_duplicates(CAND_KEY, keep="first")


# ── the team board's names for a prop's match (for the coherence gate) ──────────────────────────
_STOP = {"fc", "cf", "sc", "ac", "afc", "cd", "sd", "ud", "club", "de", "del", "la", "the"}


def _tokens(name) -> tuple[str, ...]:
    s = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode().lower()
    return tuple(t for t in re.split(r"[^a-z0-9]+", s) if t and t not in _STOP)


def name_similarity(a, b) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    if ta == tb:
        return 1.0
    if set(ta) <= set(tb) or set(tb) <= set(ta):
        return 0.9
    return difflib.SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()


#: both sides must reach this to call two spellings the same club; ambiguity (a second fixture as close) -> none
NAME_MATCH_MIN = 0.6


def board_names(board: pd.DataFrame | None, league: str, match_date: str, home: str, away: str
                ) -> tuple[str, str] | None:
    """(home, away) as the team board (output/predictions.csv) spells them, for the same league and day; None when
    no single fixture matches both names."""
    if board is None or board.empty:
        return None
    b = board[(board["league"].astype(str) == str(league)) & (board["date"].astype(str).str[:10] == match_date)]
    best, second = None, 0.0
    for r in b.to_dict("records"):
        s = min(name_similarity(home, r["home_team"]), name_similarity(away, r["away_team"]))
        if best is None or s > best[0]:
            second = best[0] if best else 0.0
            best = (s, str(r["home_team"]), str(r["away_team"]))
        elif s > second:
            second = s
    if best is None or best[0] < NAME_MATCH_MIN or second >= best[0]:
        return None
    return best[1], best[2]


def _team_board(od: Path) -> pd.DataFrame | None:
    p = Path(od) / "predictions.csv"
    if not p.exists():
        return None
    try:
        hdr = pd.read_csv(p, nrows=0).columns
        cols = [c for c in ("league", "date", "home_team", "away_team") if c in hdr]
        return pd.read_csv(p, usecols=cols, dtype=str) if len(cols) == 4 else None
    except Exception:                                                  # noqa: BLE001
        return None


# ── selection ───────────────────────────────────────────────────────────────────────────────────
def _gate_cand(r: dict):
    from src.tip_coherence import Candidate, fixture_id, prop_event
    ev = prop_event(r.get("market"))
    if ev is None:
        return None
    fx = fixture_id(r.get("match_date"), r.get("board_home_team") or r.get("home_team"),
                    r.get("board_away_team") or r.get("away_team"))
    return Candidate(fx, ev, "props TOP-3", label=f"{r.get('player_name')} {MARKET_TEXT.get(r.get('market'), '')}",
                     uid=f"props|{r.get('league_week_id')}|{r.get('fixture_id')}|{r.get('player_id')}|{r.get('market')}")


def gate_blocks(gate, r: dict) -> str:
    if not gate:
        return ""
    c = _gate_cand(r)
    return gate.check(c) if c is not None else ""


def gate_take(gate, r: dict) -> None:
    if gate:
        c = _gate_cand(r)
        if c is not None:
            gate.commit(c)


#: rule 2 of src/tip_truth for props: one pick per (canonical match, player, market)
TRUTH_KEY = ("player_id", "market")


def truth_bars_ok(league, market, side=None) -> bool:
    """Rule 4 of src/tip_truth for props: the market must carry an owner-applied bar. No prop market has one
    (config/threshold_applied.json holds team markets only; no prop cell is proven), so every prop fails closed until
    the owner applies a props bar."""
    return False


def select_week(g: pd.DataFrame, freeze: pd.Timestamp, issue: pd.Timestamp, gate=None
                ) -> tuple[list[dict], dict]:
    """(picks in role order, excluded counts) for one league-week's candidates. Pure apart from the gate."""
    counts: Counter = Counter()

    def bump(k):
        counts[k] += 1
    seen = pd.to_datetime(g["first_seen_at"], utc=True, errors="coerce")
    ko = pd.to_datetime(g["kickoff_utc"], utc=True, errors="coerce")
    val = pd.to_numeric(g["value"], errors="coerce")
    pool = []
    for i, r in enumerate(g.to_dict("records")):
        if not (seen.iloc[i] < freeze):
            bump(R_LATE)
            continue
        if not (val.iloc[i] > 0):
            bump(R_NO_VALUE)
            continue
        if not (ko.iloc[i] > issue):
            bump(R_STARTED)
            continue
        pool.append({**r, "_v": float(val.iloc[i]), "_p": _num(r["model_prob"]), "_o": _num(r["odds"]),
                     "_ko": ko.iloc[i]})
    orders = {
        "T1_BEST_VALUE": lambda x: (-x["_v"], x["_ko"], x["player_id"], x["market"]),
        "T2_SAFEST": lambda x: (-x["_p"], -x["_v"], x["_ko"], x["player_id"], x["market"]),
        "T3_RISKY_VALUE": lambda x: (-x["_o"], -x["_v"], x["_ko"], x["player_id"], x["market"]),
    }
    taken, picks, blocked = set(), [], set()
    for role in ROLES:
        for x in sorted(pool, key=orders[role]):
            k = (x["fixture_id"], x["player_id"], x["market"])
            if k in taken or k in blocked:
                continue
            why = gate_blocks(gate, x)
            if why:
                blocked.add(k)
                bump(R_MISMATCH)
                log.warning(f"[props_top3] SKIP {x['player_name']} {x['market']} ({x['match']}): {why}")
                continue
            taken.add(k)
            picks.append({**x, "role": role})
            gate_take(gate, x)
            break
    n_rest = len(pool) - len(picks) - len(blocked)
    if n_rest > 0:
        counts[R_RANK] = n_rest
    return picks, dict(counts)


def freeze_due(cands: pd.DataFrame, weeks_done: set, now: pd.Timestamp, lead_min: float,
               start_ts: pd.Timestamp | None, gate=None, team_board: pd.DataFrame | None = None,
               label: str = STAKE_LABEL_DEFAULT, truth=None, prior: set | None = None
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(new pick rows, new week rows) for every league-week whose FREEZE has passed, not decided yet, and whose
    FREEZE is at/after the props TOP-3 start. A week with 0 picks is still recorded once, with its reasons.
    `truth` (src/tip_truth, owner 2026-11-24; run() always passes it): a candidate failing the real fixture / real
    price / real market checks, or repeating a (canonical match, player, market) already picked (`prior`), is never a
    member (counted under its truth reason)."""
    empty = (pd.DataFrame(columns=PICK_COLS), pd.DataFrame(columns=WEEK_COLS))
    if cands is None or cands.empty or start_ts is None:
        return empty
    cands = cands.copy()
    if team_board is not None:
        names = [board_names(team_board, lg, md, h, a) for lg, md, h, a in
                 zip(cands["league"], cands["match_date"], cands["home_team"], cands["away_team"])]
        cands["board_home_team"] = [n[0] if n else "" for n in names]
        cands["board_away_team"] = [n[1] if n else "" for n in names]
        unresolved = cands.loc[[not n for n in names], "match"].nunique()
        if unresolved:
            log.info(f"[props_top3] {unresolved} prop match(es) not found on the team board - gate uses their own names")
    else:
        cands["board_home_team"] = ""
        cands["board_away_team"] = ""
    stamp = LW._stamp(now)
    ko_all = pd.to_datetime(cands["kickoff_utc"], utc=True, errors="coerce")
    picks_out, weeks_out = [], []
    for week, g in cands.groupby("league_week_id", sort=True):
        if not week or week in weeks_done:
            continue
        lock = ko_all.loc[g.index].min()
        if pd.isna(lock):
            continue
        freeze = lock - pd.Timedelta(minutes=float(lead_min))
        if now < freeze or freeze < start_ts:
            continue
        truth_excl: Counter = Counter()
        g0 = g
        if truth is not None:
            from src import tip_truth as TT
            gq = truth.gate(g, now, "props", prior=prior, key_cols=TRUTH_KEY)
            bad = gq[gq["truth_reason"] != ""]
            TT.log_blocks(bad, f"props TOP-3 {week}")
            truth_excl.update(bad["truth_reason"])
            g = gq[gq["truth_reason"] == ""]
        picks, excl = (select_week(g, freeze, now, gate) if len(g) else ([], {}))
        excl = dict(Counter(excl) + truth_excl)
        if prior is not None:
            prior.update((str(r.get("canonical_match")), str(r["player_id"]), str(r["market"])) for r in picks
                         if str(r.get("canonical_match") or ""))
        league = str(g0["league"].iloc[0])
        n_elig = int((pd.to_numeric(g0["value"], errors="coerce") > 0).sum())
        weeks_out.append({"selection_mode": SELECTION_MODE, "version": VERSION, "league_week_id": week,
                          "league": league, "week_lock_ts": LW._stamp(lock), "freeze_ts": LW._stamp(freeze),
                          "issued_at": stamp, "n_candidates": int(len(g0)), "n_eligible": n_elig,
                          "n_picks": len(picks), "excluded": json.dumps(excl, sort_keys=True), "frozen_at": stamp})
        for rank, r in enumerate(picks, start=1):
            picks_out.append({
                **{k: r.get(k, "") for k in PICK_FROZEN if k in r},
                "pick_id": LW.research_id(VERSION, week, str(r["market"]), str(r["fixture_id"]), str(r["player_id"])),
                "selection_mode": SELECTION_MODE, "version": VERSION, "league_week_id": week, "props_rank": rank,
                "role": r["role"], "week_lock_ts": LW._stamp(lock), "freeze_ts": LW._stamp(freeze),
                "issued_at": stamp, "cell_status": "UNPROVEN", "stake": 0.0, "stake_label": label})
    return (pd.DataFrame(picks_out).reindex(columns=PICK_COLS) if picks_out else empty[0],
            pd.DataFrame(weeks_out).reindex(columns=WEEK_COLS) if weeks_out else empty[1])


# ── stores ──────────────────────────────────────────────────────────────────────────────────────
def candidate_store(od: Path) -> FirstSightStore:
    return FirstSightStore(Path(od) / CANDIDATES_NAME, key=CAND_KEY, columns=CAND_COLS, observed_col="first_seen_at",
                           kickoff_col="kickoff_utc")


def picks_store(od: Path) -> FirstSightStore:
    return FirstSightStore(Path(od) / PICKS_NAME, key=PICK_KEY, columns=PICK_COLS, observed_col="issued_at",
                           kickoff_col="kickoff_utc", frozen_cols=PICK_FROZEN, digest=True)


def weeks_store(od: Path) -> FirstSightStore:
    return FirstSightStore(Path(od) / WEEKS_NAME, key=WEEK_KEY, columns=WEEK_COLS)


def _prop_league_mask():
    try:
        from telegram_bot.notifier import _is_prop_league
        return _is_prop_league
    except Exception as e:                                             # noqa: BLE001
        log.warning(f"[props_top3] prop-league registry unavailable ({type(e).__name__}) - no candidates")
        return lambda s: pd.Series(False, index=s.index)


# ── entrypoints ─────────────────────────────────────────────────────────────────────────────────
def run(now=None, out_dir: Path | None = None, root: Path | None = None, env: dict | None = None,
        gate=None, prop_league=None) -> dict:
    """Flag OFF -> {'enabled': False}, nothing written. Never raises."""
    if not enabled(env):
        return {"enabled": False}
    now = LW._now(now)
    od = Path(out_dir) if out_dir else OUT
    res = {"enabled": True, "candidates_appended": 0, "picks_appended": 0, "weeks_decided": 0}
    lead = freeze_lead_min(env)
    try:
        board = LW._read(od / BOARD_NAME)
        cands = build_candidates(board, now, prop_league or _prop_league_mask())
        cs = candidate_store(od)
        res["candidates_appended"] = cs.append(cands)
        st = start(root)
        if lead is None:
            res["status"] = "FREEZE_LEAD_NOT_SET"
        elif st is None:
            res["status"] = "PROPS_TOP3_START_NOT_SET"
        else:
            if gate is None:
                try:
                    from src.tip_coherence import CoherenceGate
                    gate = CoherenceGate.load(out_dir=od)
                except Exception as e:                                 # noqa: BLE001
                    log.warning(f"[props_top3] coherence gate unavailable ({type(e).__name__}: {e})")
                    gate = None
            ws = weeks_store(od)
            w = ws.read()
            done = set(w["league_week_id"]) if len(w) else set()
            # the truth gate (src/tip_truth, owner 2026-11-24): FAIL CLOSED - unavailable -> nothing frozen
            from src import tip_truth as TT
            truth = TT.default(out_dir=od, root=root, bars_ok=truth_bars_ok)
            prior = TT.prior_keys(truth, picks_store(od).read(), "props", key_cols=TRUTH_KEY)
            picks, weeks = freeze_due(cs.read(), done, now, lead, st, gate, _team_board(od), stake_label(root),
                                      truth, prior)
            res["picks_appended"] = picks_store(od).append(picks)
            res["weeks_decided"] = ws.append(weeks)
            res["status"] = "OK"
        write_report(now, od, lead)
    except Exception as e:                                             # noqa: BLE001
        log.warning(f"[props_top3] run skipped ({type(e).__name__}: {e}) - other tips unaffected")
        res["error"] = f"{type(e).__name__}: {e}"[:300]
    return res


def _results(player_ledger: pd.DataFrame) -> dict:
    """(fixture_id, player_id, market) -> WIN / LOSS / VOID from the player ledger (a resolved row wins)."""
    if player_ledger is None or player_ledger.empty:
        return {}
    d = player_ledger
    need = {"fixture_id", "player_id", "market", "result"}
    if not need <= set(d.columns):
        return {}
    out = {}
    for f, p, m, r in zip(d["fixture_id"], d["player_id"], d["market"], d["result"]):
        r = LW._s(r).upper()
        if r in ("WIN", "LOSS", "VOID"):
            out.setdefault((_id(f), _id(p), str(m)), r)
    return out


def settle_picks(picks: pd.DataFrame, player_ledger: pd.DataFrame | None, now: pd.Timestamp
                 ) -> tuple[pd.DataFrame, int]:
    """Fill the grading columns of finished picks once; frozen columns are hashed before/after."""
    from src.research.first_sight import frozen_digest
    out = picks.copy().reindex(columns=PICK_COLS).fillna("")
    before = frozen_digest(out, PICK_FROZEN)
    res = _results(player_ledger)
    n, stamp = 0, LW._stamp(now)
    for i in range(len(out)):
        r = out.iloc[i]
        if LW._s(r["settle_status"]):
            continue
        ko = pd.to_datetime(r["kickoff_utc"], utc=True, errors="coerce")
        if pd.isna(ko) or now < ko + pd.Timedelta(minutes=SETTLE_DELAY_MIN):
            continue
        result = res.get((_id(r["fixture_id"]), _id(r["player_id"]), str(r["market"])))
        if not result:
            continue
        odds = _num(r["odds"])
        pnl = (odds - 1.0) if result == "WIN" else -1.0 if result == "LOSS" else 0.0
        vals = {"result": result, "won": "1" if result == "WIN" else "0" if result == "LOSS" else "",
                "flat_pnl": LW._r(pnl) if odds > 1 else "", "settle_status": "SETTLED", "settled_at": stamp}
        for k, v in vals.items():
            out.iat[i, out.columns.get_loc(k)] = "" if v == "" else str(v)
        n += 1
    if frozen_digest(out, PICK_FROZEN) != before:
        raise RuntimeError("props_top3 settlement would alter a frozen column - refused")
    return out, n


def settle(now=None, out_dir: Path | None = None, env: dict | None = None) -> dict:
    if not enabled(env):
        return {"enabled": False}
    now = LW._now(now)
    od = Path(out_dir) if out_dir else OUT
    ps = picks_store(od)
    picks = ps.read()
    if not len(picks):
        write_report(now, od, freeze_lead_min(env))
        return {"enabled": True, "settled": 0}
    v = ps.verify()
    if v["ok"] is False:
        raise RuntimeError(f"props TOP-3 ledger TAMPERED ({v['reason']}) - not settling")
    out, n = settle_picks(picks, LW._read(od / PLAYER_LEDGER_NAME), now)
    if n:
        p = od / PICKS_NAME
        tmp = p.with_suffix(".csv.tmp")
        out.to_csv(tmp, index=False, lineterminator="\n")
        tmp.replace(p)
    write_report(now, od, freeze_lead_min(env))
    return {"enabled": True, "settled": n}


def report(picks: pd.DataFrame, weeks: pd.DataFrame, now: pd.Timestamp, lead: float | None) -> dict:
    """Per league: the latest decided week's picks (or why there are none) + settled totals (flat 1u record)."""
    picks = picks if picks is not None and len(picks) else pd.DataFrame(columns=PICK_COLS)
    weeks = weeks if weeks is not None and len(weeks) else pd.DataFrame(columns=WEEK_COLS)
    leagues = {}
    for lg in sorted(set(weeks["league"]) | set(picks["league"])):
        w = weeks[weeks["league"] == lg].sort_values("league_week_id")
        p = picks[picks["league"] == lg]
        latest = w["league_week_id"].iloc[-1] if len(w) else None
        cur = p[p["league_week_id"] == latest].sort_values("props_rank", key=lambda s: s.astype(int))
        st = p[p["settle_status"].astype(str) == "SETTLED"]
        pnl = pd.to_numeric(st["flat_pnl"], errors="coerce")
        leagues[lg] = {
            "latest_league_week_id": latest,
            "latest_excluded": json.loads(w["excluded"].iloc[-1]) if len(w) and w["excluded"].iloc[-1] else {},
            "latest_picks": [{k: r.get(k) for k in ("role", "player_name", "team", "match", "kickoff_utc", "market",
                                                    "value_basis", "stake_label", "result")}
                             for r in cur.to_dict("records")],
            "weeks_decided": int(len(w)), "picks": int(len(p)), "settled": int(len(st)),
            "won": int((st["won"].astype(str) == "1").sum()),
            "flat_pnl_units": round(float(pnl.sum()), 3) if pnl.notna().any() else 0.0}
    return {"generated_at": LW._stamp(now), "selection_mode": SELECTION_MODE, "version": VERSION,
            "freeze_lead_min": lead, "status": "OK" if lead is not None else "FREEZE_LEAD_NOT_SET",
            "note": ("Each prop league's 3 player props of the gameweek: T1 best value, T2 safest (highest model "
                     "probability), T3 risky value (longest odds), all priced with positive value. "
                     + STAKE_LABEL_DEFAULT + "; the record is flat 1u for tracking."),
            "leagues": leagues}


def write_report(now=None, out_dir: Path | None = None, lead: float | None = None) -> dict:
    od = Path(out_dir) if out_dir else OUT
    rep = report(picks_store(od).read(), weeks_store(od).read(), LW._now(now), lead)
    LW._write_json(od / REPORT_NAME, rep)
    return rep


def _pct(x) -> str:
    v = _num(x)
    return "" if v != v else f"{v * 100:+.0f}%"


def message(picks: pd.DataFrame, label: str = STAKE_LABEL_DEFAULT) -> str:
    """Telegram text (HTML) for one league-week's picks."""
    from html import escape
    r0 = picks.iloc[0]
    lines = [f"👤 <b>PROPS TOP-3 {escape(str(r0['league']))}</b> · "
             f"{escape(str(r0['league_week_id']).split('|')[-1])}"]
    for r in picks.sort_values("props_rank", key=lambda s: s.astype(int)).to_dict("records"):
        ko = pd.to_datetime(r.get("kickoff_utc"), utc=True, errors="coerce")
        when = ko.strftime("%a %H:%M UTC") if pd.notna(ko) else str(r.get("match_date"))[:10]
        team = f" ({escape(str(r['team']))})" if LW._s(r.get("team")) else ""
        lines.append(f"<b>{ROLE_TEXT.get(str(r['role']), str(r['role']))}</b>: {escape(str(r['player_name']))}{team} · "
                     f"{escape(MARKET_TEXT.get(str(r['market']), str(r['market'])))} @ {_num(r['odds']):.2f}\n"
                     f"   {escape(str(r['match']))} · {when} · model {_num(r['model_prob']) * 100:.0f}% · "
                     f"value {_pct(r['value'])} {escape(str(r['value_basis']))}")
    if len(picks) < len(ROLES):
        lines.append(f"Only {len(picks)} priced prop(s) with positive value this week - never padded.")
    lines.append(f"<i>{escape(label)}. T1 = best value, T2 = highest model probability, T3 = longest odds, "
                 "each with positive value at a real bookmaker price.</i>")
    return "\n".join(lines)


def table_rows(picks: pd.DataFrame, label: str = STAKE_LABEL_DEFAULT) -> list[dict]:
    """Rows of the all-leagues table (src.top3_league.combined_messages), kickoff order."""
    out = []
    for r in picks.to_dict("records"):
        ko = pd.to_datetime(r.get("kickoff_utc"), utc=True, errors="coerce")
        role = str(r.get("role"))
        out.append({"ko": ko, "league": str(r["league"]), "match": str(r["match"]).replace(" vs ", " v "),
                    "tier": {"T1_BEST_VALUE": "T1", "T2_SAFEST": "T2", "T3_RISKY_VALUE": "T3"}.get(role, role),
                    "pick": f"{r['player_name']} {MARKET_TEXT.get(str(r['market']), str(r['market']))}",
                    "odds": f"{_num(r['odds']):.2f}", "stake": "No stake" if label else "-",
                    "pick_id": str(r.get("pick_id")), "week": str(r.get("league_week_id")),
                    "_order": (ko.value if pd.notna(ko) else float("inf"), str(r["league"]), role)})
    return sorted(out, key=lambda x: x["_order"])


def _mark_player_notified(g: pd.DataFrame) -> None:
    """Record the sent picks in telegram_bot/player_notified.json (the props cash-out alert reads it)."""
    try:
        from telegram_bot.notifier import PLAYER_KICKOFF_FILE, PLAYER_NOTIFIED_FILE, _load_notified, _save_notified
        keys = _load_notified(PLAYER_NOTIFIED_FILE)
        kc = json.loads(PLAYER_KICKOFF_FILE.read_text()) if PLAYER_KICKOFF_FILE.exists() else {}
        for r in g.to_dict("records"):
            k = f"PLAYER|{str(r['match_date'])[:10]}|{r['player_name']}|{r['market']}"
            keys.add(k)
            kc[k] = str(r["kickoff_utc"])
        _save_notified(keys, PLAYER_NOTIFIED_FILE)
        PLAYER_KICKOFF_FILE.write_text(json.dumps(kc))
    except Exception as e:                                             # noqa: BLE001
        log.warning(f"[props_top3] player_notified not updated ({type(e).__name__})")


def notify(out_dir: Path | None = None, env: dict | None = None, sender=None, now=None, gate=None,
           truth=None) -> dict:
    """Send each frozen league-week's picks once, never after a pick's kickoff. Needs TELEGRAM_TOKEN /
    TELEGRAM_CHAT_ID (or `sender`). Every pick passes the truth gate (src/tip_truth) at send time; the gate
    unavailable -> nothing is sent."""
    if not enabled(env):
        return {"enabled": False}
    od = Path(out_dir) if out_dir else OUT
    e = env if env is not None else os.environ
    now = LW._now(now)
    mark = sender is None
    if sender is None:
        token, chat = e.get("TELEGRAM_TOKEN", ""), e.get("TELEGRAM_CHAT_ID", "")
        if not token or not chat:
            return {"enabled": True, "sent": 0, "status": "NO_TELEGRAM_CREDENTIALS"}
        sys.path.insert(0, str(ROOT))
        from telegram_bot.notifier import _send

        def sender(text):
            return _send(token, chat, text)
    # PAPER leagues (config.PAPER_LEAGUES) are never sent - ledgered and graded only, as on every other send path
    picks = config.drop_paper_leagues(picks_store(od).read())
    nf = od / NOTIFIED_NAME
    doc = LW._json(nf)
    sent = set(doc.get("weeks", []))
    sent_ids = set(doc.get("picks", []))                  # once per PICK (a week may span several messages)
    before, before_ids = set(sent), set(sent_ids)
    n = 0
    label = stake_label()
    keep_all, week_ids = [], {}
    pending = picks[~picks["league_week_id"].isin(sent)] if len(picks) else picks
    if len(pending):
        from src import tip_truth as TT
        try:
            truth = truth or TT.default(out_dir=od, bars_ok=truth_bars_ok)
        except Exception as ex:                                        # noqa: BLE001
            log.warning(f"[props_top3] truth gate unavailable ({type(ex).__name__}: {ex}) - nothing sent")
            return {"enabled": True, "sent": 0, "status": "TRUTH_GATE_UNAVAILABLE"}
        prior = TT.prior_keys(truth, picks[picks["league_week_id"].isin(sent)], "props", key_cols=TRUTH_KEY)
    for week, g in (picks.groupby("league_week_id", sort=True) if len(picks) else []):
        if week in sent:
            continue
        g = g[~g["pick_id"].astype(str).isin(sent_ids)]
        ko = pd.to_datetime(g["kickoff_utc"], utc=True, errors="coerce")
        g = g[ko > now]                                   # never sent after kickoff
        if len(g):                                        # the truth gate at send time (owner 2026-11-24)
            tg = truth.gate(g, now, "props", prior=prior, key_cols=TRUTH_KEY)
            TT.log_blocks(tg, f"props TOP-3 send {week}")
            g = g.loc[tg.index[tg["truth_reason"] == ""]]
            prior |= TT.pick_keys(tg[tg["truth_reason"] == ""], TRUTH_KEY)
        if gate is None:
            try:
                from src.tip_coherence import CoherenceGate
                gate = CoherenceGate.load(out_dir=od)
            except Exception as ex:                                    # noqa: BLE001
                log.warning(f"[props_top3] coherence gate unavailable ({type(ex).__name__}: {ex})")
                gate = False
        if gate and len(g):
            keep = []
            for i, r in g.sort_values("props_rank", key=lambda x: pd.to_numeric(x, errors="coerce")).iterrows():
                why = gate_blocks(gate, r.to_dict())
                if why:
                    log.warning(f"[props_top3] SKIP {r['player_name']} {r['market']} ({week}): {why}")
                else:
                    keep.append(i)
            g = g.loc[keep]
        week_ids[week] = set(g["pick_id"].astype(str))
        if len(g):
            keep_all.append(g)
    # owner 2026-11-24 "one table for all": every league's new props picks in ONE table per run (split > 4096 chars)
    if keep_all:
        from src import top3_league as TL
        allp = pd.concat(keep_all)
        for text, ids in TL.combined_messages(table_rows(allp, label), title="Props TOP-3 · all leagues",
                                              foot=f"{label}. T1 best value, T2 safest, T3 risky value - each with "
                                                   "positive value at a real bookmaker price."):
            if sender(text):
                n += 1
                done_ids = set(ids)
                g = allp[allp["pick_id"].astype(str).isin(done_ids)]
                if gate:
                    for r in g.to_dict("records"):
                        gate_take(gate, r)
                if mark:
                    _mark_player_notified(g)
                sent_ids.update(done_ids)
    for week, ids in week_ids.items():
        if ids <= sent_ids:                               # all its picks went out, or none was left to send
            sent.add(week)
    if sent != before or sent_ids != before_ids:
        LW._write_json(nf, {"weeks": sorted(sent), "picks": sorted(sent_ids)})
    return {"enabled": True, "sent": n}


def _cli(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cmd = argv[0] if argv else "report"
    if cmd == "run":
        print(json.dumps(run(), default=str))
    elif cmd == "settle":
        print(json.dumps(settle(), default=str))
    elif cmd == "notify":
        print(json.dumps(notify(), default=str))
    elif cmd == "report":
        print(json.dumps({"leagues": len(write_report(lead=freeze_lead_min())["leagues"])}))
    else:
        print(f"unknown command {cmd!r}: run | settle | notify | report")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
