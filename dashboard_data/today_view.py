"""Read-only views for the Home page (visual redesign 2026-10-25).

The Home page used to compute its numbers inline. They now live here so the page and its
non-regression test (tests/test_home_tips_values.py) read the SAME functions. The logic of the
figures Home already showed is moved verbatim — not changed:

  * staked basis: SNIPER + MARKSMAN, live rows, settled, match date >= PERFORMANCE_CUTOFF_DATE, every
    staked market (O/U 2.5 ledger, side-market ledger, HT ledger), paper leagues removed;
  * upcoming staked tips: bets.csv SNIPER + MARKSMAN with an OVER/UNDER bet.

New Home figures are derived from those same frames (ROI = P/L / staked bets at flat 1u; drawdown
from the running total; last-7-days P/L) or read from artifacts CI already writes
(output/system_health.json, output/artifact_manifest.json, output/stake_plan.csv,
output/api_usage_log.csv). Nothing here sizes, selects, settles or re-derives a status.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import units as U

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

# ── THE definition of "Staked" (owner 2026-11-04; supersedes the tier basis of 2026-11-02) ─────────
# A staked bet is a ledger row the bankroll policy RECORDED a stake > 0 for on a real-money track
# (src/staked.py: stake_audit.csv / stake_plan.csv, REAL_STATES, base_fraction > 0), source == live,
# settled (pnl recorded), match date on or after PERFORMANCE_CUTOFF_DATE, outside config.PAPER_LEAGUES,
# on a team-market ledger (SOURCES). Until 2026-11-04 it was "SNIPER or MARKSMAN on any track", which
# counted 495 of 805 bets on PAPER tracks as money. Player props and HT O/U have no policy track and
# are never staked. Every page that says "Staked" reads staked_settled() / is_staked(); P/L stays
# flat 1u per bet (labelled so), with the P/L at the recorded stake beside it.
#: TIER words only (the old basis) — kept for tier displays, never to decide "Staked"
STAKED = ("QUALIFIED", "SNIPER", "MARKSMAN")   # QUALIFIED = tips from 2026-11-21 (owner 2026-11-11)
SOURCES = [("bets_ledger.csv", None, "O/U 2.5"), ("side_bets_ledger.csv", "market", None),
           ("ht_ledger.csv", None, "HT O/U")]
#: ledgers that can never hold a staked bet (props are paper-only)
PAPER_ONLY_LEDGERS = ("player_ledger.csv",)
#: CLAUDE.md guardrail: alert at 45,000 of the 75,000/day API-Football plan.
API_BUDGET_WARN_SHARE = 0.60


def _read_csv(*args, **kwargs) -> pd.DataFrame:
    """pd.read_csv with paper leagues removed — collected, never counted (config.PAPER_LEAGUES)."""
    import config as _cfg
    return _cfg.drop_paper_leagues(pd.read_csv(*args, **kwargs))


def _out(out_dir) -> Path:
    if out_dir is not None:
        return Path(out_dir)
    import config
    return Path(config.OUTPUT_DIR)


def _json(p: Path) -> dict:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:                                                 # noqa: BLE001
        return {}


# ── performance (Home "Current performance") ─────────────────────────────────────────────────
def load_bets(out_dir=None) -> pd.DataFrame:
    f = _out(out_dir) / "bets.csv"
    if not f.exists():
        return pd.DataFrame()
    df = _read_csv(f)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


SETTLED_COLS = ["signal_tier", "_mkt", "_pnl", "_dt", "league", "_odds", "_ledger",
                 "_stake_status", "_stake_units", "_staked"]


def is_staked(labels, source=None, dt=None, league=None, ledger=None, pnl=None) -> pd.Series:
    """Row mask for THE staked definition above. `labels` is src.staked.label()'s frame (or its
    `stake_status` Series); the optional Series are the extra clauses (live, cutoff, paper league,
    paper-only ledger, settled) on already-aligned rows (None = the caller has already applied it).
    A tier is NOT an argument any more: SNIPER / MARKSMAN never prove a stake (owner 2026-11-04)."""
    import config
    from src import staked as S
    m = S.is_staked(labels).reset_index(drop=True)
    idx = m.index

    def _al(x):
        return pd.Series(pd.Series(x).to_numpy(), index=idx)
    if source is not None:
        m &= _al(source).fillna("live").astype(str).eq("live")
    if dt is not None:
        cut = pd.Timestamp(getattr(config, "PERFORMANCE_CUTOFF_DATE", "1970-01-01"))
        m &= pd.to_datetime(_al(dt), errors="coerce") >= cut
    if league is not None:
        m &= ~_al(league).map(config.is_paper_league).fillna(False).astype(bool)
    if ledger is not None:
        m &= ~_al(ledger).astype(str).isin(PAPER_ONLY_LEDGERS)
    if pnl is not None:
        m &= pd.to_numeric(_al(pnl), errors="coerce").notna()
    lab = labels if isinstance(labels, (pd.Series, pd.DataFrame)) else pd.Series(labels)
    return pd.Series(m.to_numpy(), index=lab.index)


def staked_settled(out_dir=None) -> pd.DataFrame:
    """THE staked rows (see the definition at the top of this module)."""
    s = load_settled(out_dir)
    return s[s["_staked"]].reset_index(drop=True)


def load_settled(out_dir=None) -> pd.DataFrame:
    """Every settled live row on every staked market since the cutoff (all tiers; callers split), each
    labelled with its recorded policy stake (src/staked.py): `_stake_status`, `_stake_units`, `_staked`."""
    import config
    from src import staked as S
    parts = []
    cut = pd.Timestamp(getattr(config, "PERFORMANCE_CUTOFF_DATE", "1970-01-01"))
    records = S.load_records(_out(out_dir))
    for f, mkc, const in SOURCES:
        p = _out(out_dir) / f
        if not p.exists():
            continue
        try:
            d = _read_csv(p, low_memory=False)
            if "source" in d.columns:
                d = d[d["source"].fillna("live").astype(str) == "live"]
            if "signal_tier" not in d.columns:
                continue                                  # HT O/U: no tier and no policy track - never staked
            oc = "odds" if "odds" in d.columns else "entry_odds"
            lab = S.label_ledger(d, f, records)
            d = d.assign(_dt=pd.to_datetime(d.get("match_date"), errors="coerce"),
                         _pnl=pd.to_numeric(d.get("pnl"), errors="coerce"),
                         _odds=pd.to_numeric(d[oc], errors="coerce") if oc in d.columns else float("nan"),
                         _ledger=f,
                         league=d["league"] if "league" in d.columns else pd.NA,
                         _mkt=d[mkc].astype(str) if mkc and mkc in d.columns else const,
                         _stake_status=lab["stake_status"], _stake_units=lab["stake_units"],
                         _staked=S.is_staked(lab))
            parts.append(d[d["_pnl"].notna() & (d["_dt"] >= cut)][SETTLED_COLS])
        except Exception:                                             # noqa: BLE001 — a viewer never dies on one file
            continue
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SETTLED_COLS)


def upcoming_staked_ou(plan: pd.DataFrame) -> int:
    """O/U 2.5 tips in the current stake plan that the policy stakes (src.staked.plan_staked). Until
    2026-11-04 this counted SNIPER + MARKSMAN in bets.csv, i.e. tiers, PAPER tracks included."""
    from src import staked as S
    if plan is None or plan.empty or "market" not in plan.columns:
        return 0
    return int((S.plan_staked(plan) & plan["market"].astype(str).str.upper().eq("OU25")).sum())


def performance(settled: pd.DataFrame, today=None) -> dict:
    """Staked headline figures (THE definition: `_staked`). `pnl` is flat 1u per staked bet (the
    ledgers' convention, labelled "flat 1u"); `stake_pnl` / `stake_units` are the P/L and units at the
    stake the policy recorded (notional 100-unit bankroll). `unstaked_*` = every other settled tip."""
    staked = settled[settled["_staked"].astype(bool)] if "_staked" in settled.columns else settled.iloc[0:0]
    paper = settled.drop(index=staked.index)
    n = int(len(staked))
    su = pd.to_numeric(staked.get("_stake_units"), errors="coerce") if n else pd.Series(dtype=float)
    out = {"n": n, "pnl": U.total(staked["_pnl"]) if n else None,
           "win_rate": float((staked["_pnl"] > 0).mean()) if n else None,
           "roi": U.total(staked["_pnl"]) / n if n else None,
           "stake_units": U.total(su) if n else None,
           "stake_pnl": U.total(staked["_pnl"] * su) if n else None,
           "paper_n": int(len(paper)), "paper_pnl": U.total(paper["_pnl"]) if len(paper) else None,
           # of the not-staked tips, those from BEFORE the first recorded policy decision (no stake could
           # exist yet): the line mixes them with policy-era NOT_STAKED tips, so the split is shown
           "paper_pre_policy_n": (int(paper["_stake_status"].eq("PRE_POLICY_NO_RECORD").sum())
                                  if "_stake_status" in paper.columns else None),
           "drawdown": None, "peak": None, "last7_pnl": None, "last7_n": 0, "curve": pd.DataFrame(),
           "by_market": pd.DataFrame()}
    if not n:
        return out
    s = staked.sort_values("_dt", kind="mergesort")
    cum = pd.Series(U.running(s["_pnl"]), index=s.index)
    peak = max(0.0, float(cum.max()))
    out["peak"] = peak
    out["drawdown"] = U.total([cum.iloc[-1], -peak])                        # <= 0: units below the peak
    daily = s.groupby(s["_dt"].dt.normalize())["_pnl"].agg(U.total)
    out["curve"] = pd.Series(U.running(daily), index=daily.index).rename("cum").reset_index().rename(columns={"_dt": "date"})
    t = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()   # UTC day
    w = s[s["_dt"] >= t - pd.Timedelta(days=6)]
    out["last7_pnl"] = U.total(w["_pnl"]) if len(w) else 0.0
    out["last7_n"] = int(len(w))
    bm = (s.groupby("_mkt")["_pnl"].agg(Bets="size", PL=U.total).reset_index()
          .rename(columns={"_mkt": "Market", "PL": "P/L"}))
    run = {k: [round(v, 3) for v in U.running(g["_pnl"])] for k, g in s.groupby("_mkt")}
    bm["Run"] = bm["Market"].map(lambda k: run.get(k) if len(run.get(k, [])) >= 3 else None)
    out["by_market"] = bm.sort_values("Bets", ascending=False).reset_index(drop=True)
    return out


# ── today's action ───────────────────────────────────────────────────────────────────────────
def action(plan: pd.DataFrame) -> dict:
    """Counts from the stake plan (production's own sizing; same reader as the Production page)."""
    if plan is None or plan.empty:
        return {"tips": 0, "active": 0, "paper": 0, "by_state": {}, "shadow_bet": 0, "shadow_n": 0}
    from src import staked as S
    staked = S.plan_staked(plan)
    ds = plan.get("decision_state", pd.Series(dtype=str)).astype(str)
    return {"tips": int(len(plan)), "active": int(staked.sum()), "paper": int((~staked).sum()),
            "by_state": plan.loc[~staked, "deployment_state"].fillna("?").value_counts().to_dict(),
            "shadow_bet": int((ds == "BET").sum()), "shadow_n": int(ds.isin(["BET", "NO_BET", "PAPER"]).sum())}


def kickoffs(out_dir=None) -> pd.DataFrame:
    """fixture key ('date|home|away', the stake plan's own key) -> kickoff_utc, from predictions.csv."""
    p = _out(out_dir) / "predictions.csv"
    cols = {"date", "home_team", "away_team", "kickoff_utc"}
    if not p.exists():
        return pd.DataFrame(columns=["fixture", "kickoff_utc"])
    try:
        k = pd.read_csv(p, usecols=lambda c: c in cols)
    except Exception:                                                 # noqa: BLE001
        return pd.DataFrame(columns=["fixture", "kickoff_utc"])
    if not cols <= set(k.columns):
        return pd.DataFrame(columns=["fixture", "kickoff_utc"])
    k["fixture"] = k["date"].astype(str).str[:10] + "|" + k["home_team"].astype(str) + "|" + k["away_team"].astype(str)
    return k[["fixture", "kickoff_utc"]].drop_duplicates("fixture")


def next_kickoff(plan: pd.DataFrame, ko: pd.DataFrame, now=None, active_only: bool = True):
    """Earliest kickoff still ahead among the stake plan's tips (active ones by default)."""
    if plan is None or plan.empty or ko is None or ko.empty:
        return pd.NaT
    from src import staked as S
    d = plan[S.plan_staked(plan)] if active_only else plan
    t = pd.to_datetime(d["fixture"].map(ko.set_index("fixture")["kickoff_utc"]), errors="coerce", utc=True)
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    return t[t > now].min()


# ── system state + attention ─────────────────────────────────────────────────────────────────
def health_files(out_dir=None) -> tuple[dict, dict]:
    o = _out(out_dir)
    return _json(o / "system_health.json"), _json(o / "artifact_manifest.json")


def check(sh: dict, name: str) -> dict:
    for c in sh.get("checks") or []:
        if c.get("check") == name:
            return c
    return {}


def age_hours(ts) -> float | None:
    from dashboard_data.freshness import age_hours as _age
    return _age(ts)


def api_budget(out_dir=None) -> dict:
    """Latest API-Football usage snapshot (requests used today vs the plan's daily limit)."""
    p = _out(out_dir) / "api_usage_log.csv"
    if not p.exists():
        return {}
    try:
        d = pd.read_csv(p, usecols=["snapshot_ts", "date", "requests_used", "limit_day"]).tail(1)
    except Exception:                                                 # noqa: BLE001
        return {}
    if d.empty:
        return {}
    r = d.iloc[0]
    used, lim = pd.to_numeric(r["requests_used"], errors="coerce"), pd.to_numeric(r["limit_day"], errors="coerce")
    return {"snapshot_ts": str(r["snapshot_ts"]), "date": str(r["date"]), "used": used, "limit": lim,
            "share": float(used / lim) if lim and lim == lim and used == used else None}


def attention(sh: dict, man: dict, budget: dict | None = None, health_max_age_h: float = 36) -> list[dict]:
    """Only meaningful warnings. Healthy (GREEN) checks are never returned.

    * any system-health check that is not GREEN (stale artifacts, CLV coverage, workflow failures,
      challenger coverage, research panel incl. repair leagues, contracts, ...) — status and value
      exactly as CI recorded them, with the rule;
    * system_health.json itself missing or older than its limit;
    * decision-facing artifacts the manifest lists as stale;
    * API-Football usage at or above the 60% (45,000 / 75,000) guardrail.
    """
    items: list[dict] = []
    if not sh:
        items.append({"status": "ERROR", "title": "System health unavailable",
                      "detail": "output/system_health.json is missing or unreadable — health is UNKNOWN",
                      "source": "evidence_refresh.yml"})
    else:
        a = age_hours(sh.get("generated_at"))
        if a is None or a > health_max_age_h:
            items.append({"status": "STALE", "title": "System health record is stale",
                          "detail": "no recorded timestamp" if a is None else f"{a:.1f} h old (limit {health_max_age_h:g} h)",
                          "source": "output/system_health.json"})
        for c in sh.get("checks") or []:
            st_ = str(c.get("status", "")).upper()
            if st_ == "GREEN":
                continue
            items.append({"status": st_ or "NO_DATA", "title": str(c.get("check", "")),
                          "detail": str(c.get("value", "")), "source": str(c.get("rule", ""))})
    for s in (man or {}).get("stale_decision_facing") or []:
        name = s.get("artifact") if isinstance(s, dict) else str(s)
        items.append({"status": "STALE", "title": f"Stale artifact: {name}",
                      "detail": (s.get("reason") if isinstance(s, dict) else "") or "not valid for current decisions",
                      "source": "output/artifact_manifest.json"})
    b = budget or {}
    if b.get("share") is not None and b["share"] >= API_BUDGET_WARN_SHARE:
        items.append({"status": "AMBER", "title": "API-Football budget",
                      "detail": f"{int(b['used']):,} of {int(b['limit']):,} requests used ({b['share']:.0%}) on {b['date']}",
                      "source": f"output/api_usage_log.csv · guardrail {API_BUDGET_WARN_SHARE:.0%}"})
    return items
