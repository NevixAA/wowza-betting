"""Every league x market cell with its owner-approved bar, status and stake (VIEWER ONLY, owner 2026-11-11).

"the best bar for every league and market": one row per league x market-side cell of the APPROVED bars
proposal, never a global bar.

SOURCES (read, never written):
  config/threshold_approvals.json                 the owner's approvals; the newest one names the proposal
                                                  (id + sha256) that is approved
  output/threshold_proposals/bars_oddsapi_hist/latest.json -> path of that proposal; its id and hash must
                                                  equal the approval's, or the page says so and shows nothing
  <proposal>.json  entries[].proposed.bars_opt    each served cell: qualifying bar (valuable_th, the ONE bar a
                                                  tip must clear since 2026-11-11), TIER2 boundary
                                                  (marksman_th, the TOP-3 TIER2 slot rule only), measure, odds
                                                  range, mode (MODEL / LEAGUE_WIDE), verdict
  evidence_<month>.json league_market_status      every cell of the study, OFF ones included (reason)
  config/bankroll_policy.json cell_staking        the stake per cell; a cell without an entry is PAPER

A served cell whose proposal verdict and evidence status disagree is listed in `problems` (never silently
picked). Nothing here has a default: a missing file is a named problem and an empty table.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APPROVALS = ROOT / "config" / "threshold_approvals.json"
POLICY = ROOT / "config" / "bankroll_policy.json"
PROPOSAL_DIR = ROOT / "output" / "threshold_proposals" / "bars_oddsapi_hist"
LATEST = PROPOSAL_DIR / "latest.json"

MSIDE_LABEL = {"over25_over": "O/U 2.5 Over", "over25_under": "O/U 2.5 Under", "btts_yes": "BTTS Yes",
               "over15_over": "Over 1.5", "over35_over": "Over 3.5", "ht_over05_over": "HT Over 0.5",
               "ht_over15_over": "HT Over 1.5"}
#: study market-side -> the policy track's market part (src/bankroll_policy.track_key); HT cells have no track
MSIDE_TRACK = {"over25_over": "OU25_OVER", "over25_under": "OU25_UNDER", "btts_yes": "BTTS",
               "over15_over": "OVER15", "over35_over": "OVER35"}
STATUS_ORDER = {"PROVEN": 0, "PROMISING": 1, "UNPROVEN": 2, "OFF": 3}
COLUMNS = ["League", "Market", "Track", "Status", "Qualifying bar", "Measure", "Odds range", "Mode",
           "TIER2 boundary", "Stake", "Stake %", "Basis / reason"]


def _json(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:                                                  # noqa: BLE001 - absent = named problem
        return None


def approved() -> dict | None:
    """The newest owner approval (by approved_at), or None."""
    a = _json(APPROVALS) or {}
    rows = [x for x in (a.get("approvals") or []) if x.get("proposal_id") and x.get("approved_at")]
    return max(rows, key=lambda x: str(x["approved_at"])) if rows else None


def load() -> dict:
    """{'approval', 'proposal', 'evidence', 'policy', 'problems'}; proposal/evidence None when not provable."""
    out = {"approval": approved(), "proposal": None, "evidence": None, "policy": _json(POLICY), "problems": []}
    ap = out["approval"]
    if not ap:
        out["problems"].append("no owner approval in config/threshold_approvals.json")
        return out
    lt = _json(LATEST) or {}
    if lt.get("proposal_id") != ap["proposal_id"] or not lt.get("path"):
        out["problems"].append(f"output/threshold_proposals/bars_oddsapi_hist/latest.json does not point at the "
                               f"approved proposal {ap['proposal_id']}")
        return out
    pr = _json(ROOT / lt["path"])
    if not pr or pr.get("proposal_id") != ap["proposal_id"] or pr.get("proposal_hash") != ap.get("proposal_hash"):
        out["problems"].append(f"{lt['path']}: id / sha256 differ from the approval of {ap['proposal_id']}")
        return out
    out["proposal"] = pr
    ev = _json(PROPOSAL_DIR / f"evidence_{pr.get('month')}.json")
    if not ev or not isinstance(ev.get("league_market_status"), list):
        out["problems"].append(f"evidence_{pr.get('month')}.json (every league x market status) is missing")
    else:
        out["evidence"] = ev
    return out


def _odds_range(lo, hi) -> str:
    if lo is None and hi is None:
        return "any"
    if lo is None:
        return f"≤ {float(hi):.2f}"
    if hi is None:
        return f"≥ {float(lo):.2f}"
    return f"{float(lo):.2f} – {float(hi):.2f}"


def _stake(policy: dict | None, track: str, mside: str, league: str) -> tuple[str, float | None]:
    """(state, base fraction %) from config/bankroll_policy.json cell_staking; PAPER when no entry."""
    if mside not in MSIDE_TRACK:
        return "PAPER (no policy track)", 0.0
    cs = (policy or {}).get("cell_staking") or {}
    if cs.get("enabled") is not True:
        return "per-cell staking off", None
    e = (cs.get("cells") or {}).get(f"{track}|{MSIDE_TRACK[mside]}|{league}")
    if not e:
        return "PAPER", 0.0
    bf = e.get("base_fraction")
    return str(e.get("state")), (None if bf is None else round(100 * float(bf), 3))


def served_cells(proposal: dict) -> dict:
    """{(track, mside, league): bars_opt cell} of every proposal entry."""
    out = {}
    for e in proposal.get("entries") or []:
        for _side, c in (((e.get("proposed") or {}).get("bars_opt")) or {}).items():
            out[(c.get("track"), c.get("mside"), c.get("league"))] = c
    return out


def cells_frame(data: dict) -> pd.DataFrame:
    """One row per league x market-side cell (every status, OFF included). Appends disagreements to
    data['problems']."""
    pr, ev = data.get("proposal"), data.get("evidence")
    if not pr or not ev:
        return pd.DataFrame(columns=COLUMNS)
    served = served_cells(pr)
    rows, seen = [], set()
    for x in ev["league_market_status"]:
        k = (x.get("track"), x.get("mside"), x.get("league"))
        seen.add(k)
        c = served.get(k)
        status = str(x.get("status"))
        if c is not None and str(c.get("verdict")) != status:
            data["problems"].append(f"{k}: proposal verdict {c.get('verdict')} vs evidence status {status}")
        state, frac = _stake(data.get("policy"), k[0], k[1], k[2])
        if c is None:
            reason = " / ".join(str(v) for v in (x.get("reason"), x.get("detail")) if v) or "—"
            rows.append({"League": k[2], "Market": MSIDE_LABEL.get(k[1], k[1]), "Track": k[0], "Status": status,
                         "Qualifying bar": None, "Measure": "—", "Odds range": "—", "Mode": "—",
                         "TIER2 boundary": None, "Stake": "OFF (no tip)" if status == "OFF" else state,
                         "Stake %": 0.0 if status == "OFF" else frac, "Basis / reason": reason})
            continue
        rows.append({"League": k[2], "Market": MSIDE_LABEL.get(k[1], k[1]), "Track": k[0], "Status": status,
                     "Qualifying bar": float(c["valuable_th"]), "Measure": str(c.get("measure")),
                     "Odds range": _odds_range(c.get("odds_min"), c.get("odds_max")), "Mode": str(c.get("mode")),
                     "TIER2 boundary": float(c["marksman_th"]), "Stake": state, "Stake %": frac,
                     "Basis / reason": f"{c.get('config_basis')} ({c.get('bar_basis')})"})
    for k in set(served) - seen:
        data["problems"].append(f"{k}: in the proposal but not in the evidence status list")
    d = pd.DataFrame(rows, columns=COLUMNS)
    d["_o"] = d["Status"].map(STATUS_ORDER).fillna(9)
    return d.sort_values(["_o", "League", "Market"]).drop(columns="_o").reset_index(drop=True)


def counts(d: pd.DataFrame) -> dict:
    """{status: n} over the cells, PROVEN first (0 is shown, never hidden)."""
    return {s: int((d["Status"] == s).sum()) for s in STATUS_ORDER}


def staked(d: pd.DataFrame) -> pd.DataFrame:
    """The cells with a real stake (state not PAPER / OFF and a fraction > 0)."""
    f = pd.to_numeric(d["Stake %"], errors="coerce")
    return d[(f > 0)]


def key_of(track: str, market: str, side, league: str) -> tuple:
    """(track, mside, league) of a ledger tip (market OU25/BTTS/OVER15/OVER35 or the ledgers' side-market
    names), for joining a tip to its cell."""
    mk = str(market or "").strip().upper().replace(" ", "")
    sd = str(side or "").strip().upper()
    if mk in ("OU25", "O/U2.5", "OVER25"):
        ms = "over25_over" if sd.startswith("OVER") else "over25_under" if sd.startswith("UNDER") else None
    else:
        ms = {"BTTS": "btts_yes", "OVER15": "over15_over", "OVER1.5": "over15_over", "OVER35": "over35_over",
              "OVER3.5": "over35_over"}.get(mk)
    return (str(track), ms, str(league))
