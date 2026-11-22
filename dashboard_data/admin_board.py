"""Admin Board - bars, TOP-3 and the research tracker in one place (owner 2026-11-22). VIEWER ONLY.

Three pure builders the page (pages/29_🧭_Admin_Board.py) renders; tests/test_admin_board.py pins them.

  bars_frame()    every league x market x side, read from the APPLIED config (the files the tips are served
                  from, never a copy): models/best_params_standard.json, models/best_params_side_markets.json
                  and the owner-apply ledger config/threshold_applied.json, through src/bars_opt/serve.served_cell
                  (the one serving rule). A cell without its own owner-applied bar is shown as NO BAR with the
                  reason. Statistics and rules only: bar, measure, odds range (a rule, not a price), counts, ROI, CI.
  top3_frame()    every TOP-3 pick of output/top3_league_ledger.csv (old shape: stake_effect; new shape
                  "3 per league, stake by proof": stake / stake_reason / value_basis), odds supplied separately
                  by dashboard_data/private_prices.py (never read from here).
  top3_weeks()    each league-week of output/top3_league_weeks.csv with its exclusion reasons, so an empty slot
                  is explained.
  tracker rows    config/research_tracker.json (curated) + live GitHub Actions status and each tool's record
                  file, fetched by the page with a cache. Fetching is injected (fetch_json / fetch_text), so tests
                  never touch the network.

Nothing has a default: a missing file is a named problem, a missing value stays empty.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STD_PARAMS = ROOT / "models" / "best_params_standard.json"
SIDE_PARAMS = ROOT / "models" / "best_params_side_markets.json"
APPLIED = ROOT / "config" / "threshold_applied.json"
LEAGUES = ROOT / "config" / "leagues.json"
LEDGER = ROOT / "output" / "top3_league_ledger.csv"
WEEKS = ROOT / "output" / "top3_league_weeks.csv"
TRACKER = ROOT / "config" / "research_tracker.json"
TRACKER_SCHEMA = "wowza.research_tracker/1"

TRACK_LABEL = {"standard": "Standard", "new_format": "New-Format"}
#: (market label, side label, params key, bars_opt cell side or None = the serving code has no cell for this side)
MARKET_SIDES = [
    ("O/U 2.5", "OVER", "ou25", "over"), ("O/U 2.5", "UNDER", "ou25", "under"),
    ("BTTS", "YES", "btts", "yes"), ("BTTS", "NO", "btts", None),
    ("O/U 1.5", "OVER", "over15", "yes"), ("O/U 1.5", "UNDER", "over15", None),
    ("O/U 3.5", "OVER", "over35", "yes"), ("O/U 3.5", "UNDER", "over35", None),
    ("HT O/U 0.5", "OVER", "ht_ou05", "over"), ("HT O/U 0.5", "UNDER", "ht_ou05", "under"),
    ("HT O/U 1.5", "OVER", "ht_ou15", "over"), ("HT O/U 1.5", "UNDER", "ht_ou15", "under"),
]
NO_BAR, NOT_APPLIED, OFF = "NO BAR", "NOT APPLIED", "OFF"
STATUS_ORDER = ["PROVEN", "PROMISING", "UNPROVEN", OFF, NOT_APPLIED, NO_BAR]
BARS_COLUMNS = ["Track", "League", "Market", "Side", "Status", "Bar %", "Measure", "Odds range", "Mode",
                "WF bets", "WF ROI %", "Study bets", "Study ROI %", "ROI 95% CI", "CLV %", "Proposal", "Why no bar"]


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_json(p: Path) -> dict:
    """{} when missing or unreadable (callers show the problem)."""
    return _read_json(p) or {}


def read_csv(p: Path) -> pd.DataFrame:
    """An empty frame when missing or unreadable, never an exception."""
    try:
        return pd.read_csv(p)
    except Exception:                                                  # noqa: BLE001
        return pd.DataFrame()


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _pct(v, dp: int = 1):
    x = _num(v)
    return None if x is None else round(100 * x, dp)


def _odds_range(lo, hi) -> str:
    lo, hi = _num(lo), _num(hi)
    if lo is None and hi is None:
        return "any"
    if lo is None:
        return f"≤ {hi:.2f}"
    if hi is None:
        return f"≥ {lo:.2f}"
    return f"{lo:.2f} – {hi:.2f}"


def _ci(ci) -> str:
    if not isinstance(ci, (list, tuple)) or len(ci) != 2 or _num(ci[0]) is None or _num(ci[1]) is None:
        return "—"
    return f"{100 * float(ci[0]):+.1f}% … {100 * float(ci[1]):+.1f}%"


# ── 1. BARS ─────────────────────────────────────────────────────────────────────────────────────
def universe(leagues_doc: dict | None) -> list[tuple[str, str]]:
    """(track, league) of every league with a production model (config/leagues.json model_type)."""
    out = []
    for lg in (leagues_doc or {}).get("leagues") or []:
        t = lg.get("model_type")
        if t in TRACK_LABEL and lg.get("name"):
            out.append((t, str(lg["name"])))
    return sorted(out, key=lambda x: (x[0] != "standard", x[1]))


def applied_index(applied_doc: dict | None) -> tuple[frozenset, dict]:
    """(owner-applied entry ids, {entry id: proposal id of its latest apply})."""
    ids, prop = set(), {}
    for h in (applied_doc or {}).get("history") or []:
        for e in h.get("entries") or []:
            ids.add(str(e))
            prop[str(e)] = str(h.get("proposal_id") or "")
    return frozenset(ids), prop


def load_bars_inputs(root: Path | None = None) -> dict:
    r = Path(root) if root else ROOT
    names = {"standard": r / STD_PARAMS.relative_to(ROOT), "side": r / SIDE_PARAMS.relative_to(ROOT),
             "applied": r / APPLIED.relative_to(ROOT), "leagues": r / LEAGUES.relative_to(ROOT)}
    out, problems = {}, []
    for k, p in names.items():
        d = _read_json(p)
        if d is None:
            problems.append(f"{p.relative_to(r).as_posix()} cannot be read")
        out[k] = d
    out["problems"] = problems
    return out


def bars_frame(inp: dict) -> pd.DataFrame:
    """One row per league x market x side. The status is what serving would do today (served_cell)."""
    from src.bars_opt import serve as BS
    ids, prop = applied_index(inp.get("applied"))
    std, side = inp.get("standard") or {}, inp.get("side") or {}
    sides = list(MARKET_SIDES)
    known = {k for _, _, k, _ in MARKET_SIDES}
    # any other market of the side-markets file that carries optimized cells gets its own rows
    for key, by_league in sorted(side.items()):
        if key in known or not isinstance(by_league, dict):
            continue
        found = sorted({s for lp in by_league.values() if isinstance(lp, dict)
                        for s in ((lp.get(BS.CELL_KEY) or {}) if isinstance(lp.get(BS.CELL_KEY), dict) else {})})
        sides += [(key, s.upper(), key, s) for s in found]
    rows = []
    for track, league in universe(inp.get("leagues")):
        for mlabel, slabel, key, cside in sides:
            lp = (std if key == "ou25" else (side.get(key) or {})).get(league)
            row = {"Track": TRACK_LABEL[track], "League": league, "Market": mlabel, "Side": slabel,
                   "Status": NO_BAR, "Bar %": None, "Measure": None, "Odds range": None, "Mode": None,
                   "WF bets": None, "WF ROI %": None, "Study bets": None, "Study ROI %": None, "ROI 95% CI": None,
                   "CLV %": None, "Proposal": None, "Why no bar": None}
            entry = f"{key}|{league}"
            if cside is None:
                row["Why no bar"] = "the serving code has no cell for this side"
                rows.append(row)
                continue
            try:
                cell, why = BS.served_cell(lp, key, league, cside, ids)
            except Exception as e:                                     # noqa: BLE001 - a broken entry is shown, not hidden
                cell, why = None, f"unreadable entry ({type(e).__name__})"
            if cell is not None:
                sf = cell.get("study_final") or {}
                row.update({"Status": str(cell.get("verdict") or "UNKNOWN").upper(),
                            "Bar %": _pct(cell.get("valuable_th")), "Measure": cell.get("measure"),
                            "Odds range": _odds_range(cell.get("odds_min"), cell.get("odds_max")),
                            "Mode": cell.get("mode"), "WF bets": cell.get("wf_bets"), "WF ROI %": _pct(cell.get("wf_roi")),
                            "Study bets": sf.get("bets"), "Study ROI %": _pct(sf.get("roi")), "ROI 95% CI": _ci(sf.get("ci95")),
                            "CLV %": _pct(sf.get("clv"), 2), "Proposal": prop.get(entry)})
            else:
                why = re.sub(r"^NO_LEAGUE_BAR:\s*", "", str(why or ""))
                if isinstance(lp, dict) and lp.get("drop") is True:
                    row["Status"] = OFF
                elif "not applied" in why:
                    row["Status"] = NOT_APPLIED
                row["Why no bar"] = why or "no owner-applied bar"
            rows.append(row)
    d = pd.DataFrame(rows, columns=BARS_COLUMNS)
    for c in ("WF bets", "Study bets"):
        d[c] = pd.to_numeric(d[c], errors="coerce").astype("Int64")
    return d


def bars_counts(d: pd.DataFrame) -> dict:
    n = Counter(d["Status"]) if len(d) else Counter()          # a Counter counts an absent status as none
    return {s: n[s] for s in STATUS_ORDER} | {k: n[k] for k in n if k not in STATUS_ORDER}


# ── 2. TOP-3 ────────────────────────────────────────────────────────────────────────────────────
MARKET_LABEL = {"OU25": "O/U 2.5", "BTTS": "BTTS", "OVER15": "O/U 1.5", "OVER35": "O/U 3.5"}
TIER_LABEL = {"1": "T1 best value", "2": "T2 safest", "3": "T3 risky value",
              "T1_BEST_VALUE": "T1 best value", "T2_SAFEST": "T2 safest", "T3_RISKY_VALUE": "T3 risky value"}
#: the stake-by-proof labels (src/top3_league.NO_STAKE_LABEL, owner 2026-11-22); an unknown reason is shown as is
NO_STAKE_LABEL = {"NO_STAKE_BELOW_BAR": "No stake: below bar", "NO_STAKE_NO_APPROVED_BAR": "No stake: no approved bar",
                  "NO_STAKE_NO_VALUE": "No stake: no value at the bar",
                  "NO_STAKE_PAPER_LEAGUE": "No stake: league not staked",
                  "NO_STAKE_NOT_CONFIGURED": "No stake: no TOP-3 stake set", "LEAGUE_RULE": "League rule line"}
PRICE_COLUMNS = ("odds", "opposite_side_odds", "p_market_fair")   # never taken from the ledger by top3_frame


def week_of(league_week_id) -> str:
    s = str(league_week_id or "")
    return s.split("|", 1)[1] if "|" in s else ""


def _stake_text(fraction: float) -> str:
    return f"{fraction * 100:.2f}".rstrip("0").rstrip(".") + "% stake"


def stake_label(r: dict) -> tuple[bool | None, str]:
    """(staked?, label) of one ledger row, either ledger shape. None = cannot tell (shown as such)."""
    if "stake" in r and _num(r.get("stake")) is not None:
        f = float(r["stake"])
        if f > 0:
            return True, _stake_text(f)
        reason = str(r.get("stake_reason") or "").strip()
        return False, NO_STAKE_LABEL.get(reason, f"No stake: {reason.lower().replace('_', ' ')}" if reason else "No stake")
    eff = str(r.get("stake_effect") or "").strip()
    m = re.match(r"REAL\s+([0-9.]+)%", eff)
    if m:
        return True, f"{m.group(1)}% stake"
    if eff.upper().startswith("NONE"):
        return False, "No stake"
    return None, "stake unknown"


def top3_frame(ledger: pd.DataFrame, odds: dict | None = None) -> pd.DataFrame:
    """Display rows of every TOP-3 pick. `odds` = {pick_id: odds at tip} from the private reader, or None."""
    cols = ["Week", "League", "Tier", "Match", "Kickoff (UTC)", "Pick", "Odds at tip", "Calibrated value pp",
            "Staked", "Stake", "Result", "P/L (units)", "pick_id"]
    if ledger is None or ledger.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for r in ledger.to_dict("records"):
        tier_raw = str(r.get("tier") or r.get("tier_name") or "").strip()
        rank = r.get("top3_rank")
        rank_s = str(int(rank)) if _num(rank) is not None else ""
        side = str(r.get("side") or "")
        staked, label = stake_label(r)
        pnl = _num(r.get("flat_pnl"))
        res = str(r.get("result") or "").strip()
        if res.lower() == "nan":
            res = ""
        rows.append({
            "Week": week_of(r.get("league_week_id")), "League": r.get("league"),
            "Tier": TIER_LABEL.get(tier_raw) or TIER_LABEL.get(rank_s) or (f"#{rank_s}" if rank_s else ""),
            "Match": f"{r.get('home_team')} v {r.get('away_team')}",
            "Kickoff (UTC)": str(r.get("kickoff_utc") or "")[:16].replace("T", " "),
            "Pick": f"{MARKET_LABEL.get(str(r.get('market')), str(r.get('market')))} "
                    f"{'Yes' if side == 'YES' else ('No' if side == 'NO' else side.title())}",
            "Odds at tip": (odds or {}).get(str(r.get("pick_id"))) if odds else None,
            "Calibrated value pp": _pct(r.get("edge_calibrated")),
            "Staked": {True: "Yes", False: "No", None: "?"}[staked], "Stake": label,
            "Result": res or ("pending" if str(r.get("settle_status") or "") in ("", "nan") else str(r.get("settle_status"))),
            "P/L (units)": None if pnl is None else round(pnl, 2), "pick_id": str(r.get("pick_id"))})
    d = pd.DataFrame(rows, columns=cols)
    return d.sort_values(["Week", "League", "Tier"], ascending=[False, True, True], kind="mergesort").reset_index(drop=True)


def odds_from_csv(text: str) -> dict:
    """{pick_id: odds} from a TOP-3 ledger CSV text (the private store copy or the public file). Pure."""
    out = {}
    for r in csv.DictReader(io.StringIO(text)):
        x = _num(r.get("odds"))
        if r.get("pick_id") and x is not None and x > 1:
            out[str(r["pick_id"])] = x
    return out


def exclusions_text(raw) -> str:
    try:
        d = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
    except ValueError:
        return str(raw)
    if not isinstance(d, dict) or not d:
        return ""
    order = sorted(d.items(), key=lambda kv: (-(_num(kv[1]) if _num(kv[1]) is not None else -1.0), str(kv[0])))
    return " · ".join(f"{k.replace('_', ' ').lower()} {v}" for k, v in order)


def top3_weeks(weeks: pd.DataFrame, ledger: pd.DataFrame | None = None) -> pd.DataFrame:
    cols = ["Week", "League", "Candidates", "Qualifying", "Picks", "Staked", "Empty slots", "Why (excluded candidates)"]
    if weeks is None or weeks.empty:
        return pd.DataFrame(columns=cols)
    staked = Counter()
    if ledger is not None and len(ledger):
        staked.update(str(r.get("league_week_id")) for r in ledger.to_dict("records") if stake_label(r)[0])
    rows = []
    for r in weeks.to_dict("records"):
        n = _num(r.get("n_picks"))
        rows.append({"Week": week_of(r.get("league_week_id")), "League": r.get("league"),
                     "Candidates": _num(r.get("n_candidates")), "Qualifying": _num(r.get("n_qualifying")),
                     "Picks": n, "Staked": staked[str(r.get("league_week_id"))],
                     "Empty slots": None if n is None else max(0, 3 - int(n)),
                     "Why (excluded candidates)": exclusions_text(r.get("excluded"))})
    d = pd.DataFrame(rows, columns=cols)
    for c in ("Candidates", "Qualifying", "Picks", "Empty slots"):
        d[c] = pd.to_numeric(d[c], errors="coerce").astype("Int64")
    return d.sort_values(["Week", "League"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


# ── 3. RESEARCH & PRO TOOLS TRACKER ─────────────────────────────────────────────────────────────
def load_tracker(path: Path | None = None) -> tuple[dict | None, list[str]]:
    p = Path(path) if path else TRACKER
    d = _read_json(p)
    if d is None:
        return None, [f"{p.name} cannot be read"]
    if d.get("schema") != TRACKER_SCHEMA:
        return None, [f"{p.name}: unexpected schema {d.get('schema')!r}"]
    probs = []
    repos = d.get("repos") or {}
    for t in d.get("tools") or []:
        for k in ("id", "name", "repo", "purpose"):
            if not t.get(k):
                probs.append(f"tool {t.get('id')!r} lacks {k}")
        if t.get("repo") not in repos:
            probs.append(f"tool {t.get('id')!r}: unknown repo {t.get('repo')!r}")
        rec = t.get("record")
        if rec and rec.get("repo", t.get("repo")) not in repos:
            probs.append(f"tool {t.get('id')!r}: record repo unknown")
    return d, probs


_SEL = re.compile(r"^([^\[#]+)(?:\[([^\]]*)\])?(#len)?$")


def pick(doc, key: str):
    """Value at a dotted key. 'a.b' ; 'models[model_id=x,market=y].rows' (matches summed); 'items#len'."""
    cur = [doc]
    parts = key.split(".")
    for i, part in enumerate(parts):
        m = _SEL.match(part)
        if not m:
            return None
        name, filt, ln = m.groups()
        nxt = []
        for c in cur:
            v = c.get(name) if isinstance(c, dict) else None
            if v is None:
                continue
            if filt is not None:
                conds = [f.split("=", 1) for f in filt.split(",") if "=" in f]
                v = [x for x in (v if isinstance(v, list) else []) if isinstance(x, dict)
                     and all(str(x.get(a)) == b for a, b in conds)]
                nxt.extend(v)
            else:
                nxt.append(v)
        if ln:
            nxt = [len(x) if isinstance(x, (list, dict)) else None for x in nxt]
        cur = nxt
    if not cur:
        return None
    if len(cur) == 1:
        return cur[0]
    nums = [_num(x) for x in cur]
    return sum(nums) if all(x is not None for x in nums) else None


def csv_count(text: str, where: dict | None = None, nonempty: str | None = None) -> int:
    n = 0
    for r in csv.DictReader(io.StringIO(text)):
        if where and any(str(r.get(k, "")) != str(v) for k, v in where.items()):
            continue
        if nonempty and not str(r.get(nonempty) or "").strip():
            continue
        n += 1
    return n


def record_text(rec: dict, body: str | None, err: str | None) -> str:
    """'label value / gate · ...' for one tool's record file, or the reason it cannot be read."""
    if body is None:
        return f"not readable ({err})" if err else "not readable"
    parts = []
    try:
        if rec.get("kind") == "csv":
            for f in rec.get("fields") or []:
                v = csv_count(body, f.get("where"), f.get("nonempty"))
                parts.append(_fmt_field(f, v))
        else:
            doc = json.loads(body)
            for f in rec.get("fields") or []:
                parts.append(_fmt_field(f, pick(doc, f["key"])))
    except Exception as e:                                             # noqa: BLE001
        return f"not readable ({type(e).__name__})"
    return " · ".join(parts)


def _fmt_field(f: dict, v) -> str:
    if v is None:
        s = "missing"
    elif isinstance(v, dict):
        s = ", ".join(f"{k} {vv}" for k, vv in v.items())
    elif isinstance(v, float) and v.is_integer():
        s = str(int(v))
    else:
        s = str(v)
    gate = f.get("gate")
    return f"{f['label']} {s}" + (f" / {gate}" if gate is not None else "")


def latest_runs(runs_doc: dict | None) -> dict:
    """{workflow path: newest run} from /actions/runs (newest first)."""
    out = {}
    for r in (runs_doc or {}).get("workflow_runs") or []:
        p = r.get("path")
        if p and p not in out:
            out[p] = r
    return out


def workflow_states(wf_doc: dict | None) -> dict:
    return {w.get("path"): w for w in (wf_doc or {}).get("workflows") or [] if w.get("path")}


def tracker_frame(reg: dict, states: dict, runs: dict, records: dict, problems: dict) -> pd.DataFrame:
    """states/runs: {repo key: {workflow path: obj}}; records: {tool id: text}; problems: {repo key: reason}."""
    repos = reg.get("repos") or {}
    rows = []
    for t in reg.get("tools") or []:
        rk = t["repo"]
        slug = repos.get(rk, {}).get("slug", "")
        wfs = t.get("workflows") or []
        state, last, concl, link = "", "", "", f"https://github.com/{slug}"
        if wfs:
            path = ".github/workflows/" + wfs[0]
            # the page relabels every shown string that holds the retired word, which would break such a URL
            link = (f"https://github.com/{slug}/actions" if "paper" in wfs[0].lower()
                    else f"https://github.com/{slug}/actions/workflows/{wfs[0]}")
            w = (states.get(rk) or {}).get(path)
            if w is None:
                state = f"unknown ({problems[rk]})" if rk in problems else "not found"
            else:
                state = "enabled" if w.get("state") == "active" else ("disabled" if "disabled" in str(w.get("state"))
                                                                     else str(w.get("state")))
            run = (runs.get(rk) or {}).get(path)
            if run:
                last = str(run.get("created_at") or "")[:16].replace("T", " ")
                concl = str(run.get("conclusion") or run.get("status") or "")
            elif rk in problems:
                concl = "unknown"
            else:
                concl = "no recent run"
        rows.append({"Tool": t["name"], "Repo": repos.get(rk, {}).get("label", rk), "What it does": t["purpose"],
                     "Schedule": t.get("schedule") or "", "Enabled": state or "n/a (no workflow)",
                     "Last run (UTC)": last, "Conclusion": concl,
                     "Record": records.get(t["id"], "no record file" if not t.get("record") else ""),
                     "Record note": t.get("record_note") or "", "Link": link})
    return pd.DataFrame(rows)
