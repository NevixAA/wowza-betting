"""Dashboard truth reproducer (Phase 2, KEY=dashtruth, 2026-11-05).

    python scripts/dashboard_truth.py                          # this checkout: render + recompute + compare
    python scripts/dashboard_truth.py --root <dir> --json f    # another tree (e.g. origin/main via git archive)

WHAT IT DOES. For every headline figure the v9-local dashboard pages show, two values are put side by side:

  shown        the value the page DISPLAYS, read from a headless render of the page itself
               (streamlit.testing AppTest, run in a child process against the target tree's own code and
               committed output/). Nothing is taken from the dashboard functions: the text on the page is
               the value.
  independent  the same figure recomputed here from the target tree's committed files with plain pandas.
               It imports NO dashboard module (dashboard_data, pages, telegram_bot). The only shared code is
               src/staked.py, which is THE definition of "Staked" (owner 2026-11-04) and must be shared,
               and src/bankroll_policy.REAL_STATES through it. The context (performance cutoff, paper
               leagues, thresholds) is read from the target tree's config.py in the render child.

  MATCH          shown == independent within the display's own rounding (half a unit of the last digit)
  MISMATCH       they differ - the cause is named in reports/DASHBOARD_TRUTH_VALIDATION.md
  UNIDENTIFIABLE the page does not show the figure under that label (NOT_SHOWN) or the input is missing

Clock-bound figures (stake-plan rows dated today or later, last 7 days, the Success Rates window) use the
same clock in both halves: the child records `now` before it renders and the recompute uses that value.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

#: the pages the reproducer renders (v9-local: their figures come from v9's committed files). Pro, V11,
#: Fantasy, Model Lab, Bet Builder, League Research, Portfolio, Agent Analysis and Live read other repos,
#: a live API or user input, and are covered in the report, not here.
PAGES = {"home": "00_🎯_Home.py", "overview": "0_🏠_Overview.py", "tips": "1_📊_Dashboard.py",
         "success": "5_🎯_Success_Rates.py", "modelperf": "9_🎓_Model_Performance.py",
         "props": "11_👤_Player_Props.py", "leagues": "17_🏟️_Leagues.py", "health": "19_🩺_System_Health.py",
         "production": "22_🏭_Production.py", "intel": "21_📡_Market_Intelligence.py",
         "modelinfo": "4_ℹ️_Model_Info.py"}
STAT = re.compile(r"<span class='wz-stat-l'>(.*?)</span><span class='wz-stat-v'>(.*?)</span>")
NUM = re.compile(r"[-+−]?\d[\d,]*(?:\.\d+)?")
MATCH, MISMATCH, UNIDENTIFIABLE = "MATCH", "MISMATCH", "UNIDENTIFIABLE"
TIERS = ("SNIPER", "MARKSMAN", "VALUABLE")


# ── child: render pages of a tree, record context ──────────────────────────────────────────────────
def _child(root: Path, pages: list[str]) -> dict:
    """Runs inside the child process with `root` first on sys.path (its own code, its own output/)."""
    sys.path.insert(0, str(root))
    os.chdir(root)
    os.environ["WOWZA_TIER_OVERRIDE"] = "admin"
    import warnings
    warnings.filterwarnings("ignore")
    import config
    now = pd.Timestamp.now(tz="UTC")
    ctx = {"now_utc": now.isoformat(), "today_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
           "cutoff": str(getattr(config, "PERFORMANCE_CUTOFF_DATE", "")),
           "paper_leagues": sorted(getattr(config, "PAPER_LEAGUES", []) or []),
           "thresholds": {k: getattr(config, k, None) for k in
                          ("SNIPER_THRESHOLD", "SNIPER_THRESHOLD_OVER", "SNIPER_THRESHOLD_UNDER")}}
    from streamlit.testing.v1 import AppTest
    out = {"ctx": ctx, "pages": {}}
    for key in pages:
        p = root / "pages" / PAGES[key]
        if not p.exists():
            out["pages"][key] = {"error": "page missing"}
            continue
        try:
            at = AppTest.from_file(str(p), default_timeout=900).run()
        except Exception as e:                                         # noqa: BLE001 - recorded, not hidden
            out["pages"][key] = {"error": f"{type(e).__name__}: {e}"[:300]}
            continue
        r = {"exception": [str(e.value)[:300] for e in at.exception], "metrics": [], "stats": [],
             "captions": [c.value for c in at.caption], "markdown": [m.value for m in at.markdown]}
        for m in at.metric:
            r["metrics"].append([m.label, m.value])
        for m in at.markdown:
            for k, v in STAT.findall(m.value):
                r["stats"].append([html.unescape(k), html.unescape(v)])
        out["pages"][key] = r
    return out


def render(root: Path, pages: list[str] | None = None) -> dict:
    """Render `pages` of the tree at `root` in a child process; returns {'ctx', 'pages'}."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", str(root), "--pages",
           ",".join(pages or list(PAGES))]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env, cwd=str(root))
    if r.returncode != 0:
        raise RuntimeError(f"render child failed: {r.stderr[-2000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


# ── reading what a page shows ─────────────────────────────────────────────────────────────────────
def _find(page: dict, kind: str, label: str, nth: int = 0):
    vals = [v for k, v in page.get(kind, []) if k == label]
    return vals[nth] if len(vals) > nth else None


def num(s, i: int = 0):
    """(value, decimals) of the i-th number in a displayed string; (None, None) when there is none."""
    if s is None:
        return None, None
    m = NUM.findall(str(s).replace("−", "-"))
    if len(m) <= i:
        return None, None
    t = m[i].replace(",", "")
    return float(t), (len(t.split(".")[1]) if "." in t else 0)


# ── independent recomputation (pandas only; src/staked.py is THE shared definition) ─────────────────
def _csv(out: Path, name: str) -> pd.DataFrame | None:
    p = out / name
    return pd.read_csv(p, low_memory=False) if p.exists() else None


def _total(xs) -> float:
    return float(sum(Decimal(str(x)) for x in xs))


def settled_rows(out: Path, ctx: dict) -> pd.DataFrame:
    """Every live settled row on the two team ledgers since the cutoff, paper leagues out, in ledger order
    (O/U 2.5 first, then side markets), with THE staked label."""
    from src import staked as S
    records = S.load_records(out)
    cut = pd.Timestamp(ctx["cutoff"][:10]) if ctx.get("cutoff") else None
    paper = set(ctx.get("paper_leagues") or [])
    parts = []
    for name, const in (("bets_ledger.csv", "O/U 2.5"), ("side_bets_ledger.csv", None)):
        d = _csv(out, name)
        if d is None or d.empty:
            continue
        d = d[d["source"].astype(str).eq("live") & d["source"].notna()] if "source" in d.columns else d
        lab = S.label_ledger(d, name, records)
        f = pd.DataFrame({"ledger": name, "league": d["league"],
                          "mkt": const if const else d["market"].astype(str),
                          "model_type": d.get("model_type"), "tier": d.get("signal_tier"),
                          "dt": pd.to_datetime(d["match_date"], errors="coerce"),
                          "pnl": pd.to_numeric(d["pnl"], errors="coerce"),
                          "odds": pd.to_numeric(d["odds"], errors="coerce"),
                          "result": d.get("result"),
                          "status": lab["stake_status"], "units": lab["stake_units"]}, index=d.index)
        keep = f["pnl"].notna() & ~f["league"].isin(paper)
        if cut is not None:
            keep &= f["dt"] >= cut
        parts.append(f[keep])
    if not parts:
        return pd.DataFrame(columns=["ledger", "league", "mkt", "model_type", "tier", "dt", "pnl", "odds",
                                     "result", "status", "units"])
    r = pd.concat(parts, ignore_index=True)
    r["staked"] = r["status"].eq(S.STAKED)
    return r


def _plan(out: Path, today: str) -> pd.DataFrame:
    p = _csv(out, "stake_plan.csv")
    if p is None:
        return pd.DataFrame()
    p = p[p["date"].astype(str).str[:10] >= today].reset_index(drop=True)
    return p


def book(s: pd.DataFrame, mask: pd.Series, now, ctx: dict) -> dict:
    """The staked-book figures over the rows `mask` selects (THE definition, or the old tier basis)."""
    v: dict = {}
    st_ = s[mask.to_numpy()]
    n = len(st_)
    v["staked_n"] = n
    if n:
        v["staked_pnl"] = _total(st_["pnl"])
        v["staked_roi_pct"] = 100 * v["staked_pnl"] / n
        v["staked_win_pct"] = 100 * float((st_["pnl"] > 0).mean())
        v["stake_units"] = _total(st_["units"])
        v["stake_pnl"] = _total([a * b for a, b in zip(st_["pnl"], st_["units"])])
        o = st_.sort_values("dt", kind="mergesort")
        cum = o["pnl"].cumsum()
        peak = max(0.0, float(cum.max()))
        v["drawdown"] = float(cum.iloc[-1]) - peak
        w = o[o["dt"] >= pd.Timestamp(ctx["today_utc"]) - pd.Timedelta(days=6)]
        v["last7_pnl"], v["last7_n"] = _total(w["pnl"]), len(w)
        v["staked_pnl_ex_ou25"] = _total(st_.loc[st_["mkt"] != "O/U 2.5", "pnl"])
        for mt in ("new_format", "standard"):                   # a track with no staked bet sums to 0
            g = st_[st_["model_type"] == mt]
            v[f"track_{mt}_pnl"], v[f"track_{mt}_n"] = _total(g["pnl"]), len(g)
            if not len(g):
                continue
            v[f"track_{mt}_hit_pct"] = 100 * float((g["pnl"] > 0).mean())
            v[f"track_{mt}_roi_pct"] = 100 * _total(g["pnl"]) / len(g)
            ok = g["odds"] > 1
            v[f"track_{mt}_be_pct"] = 100 * float((1 / g.loc[ok, "odds"]).mean())
    ns = s[~mask.to_numpy() & s["tier"].notna().to_numpy()]
    v["not_staked_n"], v["not_staked_pnl"] = len(ns), _total(ns["pnl"])
    v["not_staked_pre_policy_n"] = int(ns["status"].eq("PRE_POLICY_NO_RECORD").sum())
    # Leagues: staked tips of a sent tier, paper leagues out (already out of `s`)
    lg = st_[st_["tier"].isin(TIERS)]
    v["leagues_n"] = len(lg)
    if len(lg):
        v["leagues_pnl"], v["leagues_hit_pct"] = _total(lg["pnl"]), 100 * float((lg["pnl"] > 0).mean())
    # Success Rates, default window (first entry of the admin tier's windows: 7 days), staked scope
    since = pd.Timestamp(now).tz_convert(None).normalize() - pd.Timedelta(days=7)
    v["success_obs"] = int((st_["dt"] >= since).sum())
    return v


def independent(out: Path, ctx: dict) -> dict:
    """figure id -> recomputed value (float / int / str); a missing input is absent from the dict."""
    from src.staked import plan_staked
    v: dict = {}
    now = pd.Timestamp(ctx["now_utc"])
    s = settled_rows(out, ctx)
    v.update(book(s, s["staked"], now, ctx))
    # the OLD basis (SNIPER + MARKSMAN tiers, any track): what origin/main's pages still call "Staked".
    # Recomputed so a main figure can be told apart as "reproduces, but on the tier basis".
    v.update({"tier:" + k: x for k, x in book(s, s["tier"].isin(("SNIPER", "MARKSMAN")), now, ctx).items()})
    # stake plan (Home today's action, Production)
    today = now.strftime("%Y-%m-%d")
    plan = _plan(out, today)
    if len(plan):
        stk = plan_staked(plan)
        v["plan_tips"], v["plan_active"], v["plan_paper"] = len(plan), int(stk.sum()), int((~stk).sum())
        v["plan_days"] = int(plan["date"].astype(str).str[:10].nunique())
        v["plan_up_ou"] = int((stk & plan["market"].astype(str).str.upper().eq("OU25")).sum())
        ds = plan["decision_state"].astype(str) if "decision_state" in plan.columns else pd.Series(dtype=str)
        v["plan_shadow_bet"] = int((ds == "BET").sum())
        v["plan_shadow_n"] = int(ds.isin(["BET", "NO_BET", "PAPER"]).sum())
    else:
        # no upcoming plan rows (e.g. an international break): every plan count is a true 0 of an
        # empty plan, the same as the Home page shows, not a missing input
        v.update(plan_tips=0, plan_active=0, plan_paper=0, plan_days=0, plan_up_ou=0,
                 plan_shadow_bet=0, plan_shadow_n=0)
    # system health: overall and the status counts recounted from the recorded checks
    p = out / "system_health.json"
    if p.exists():
        sh = json.loads(p.read_text(encoding="utf-8"))
        v["health_overall"] = str(sh.get("overall"))
        for k in ("GREEN", "AMBER", "RED", "NO_DATA"):
            v[f"health_{k}"] = sum(1 for c in sh.get("checks") or [] if c.get("status") == k)
        listed = sum(len(f.get("violations") or []) for f in ((sh.get("contracts") or {}).get("files") or {}).values())
        known = (sh.get("contracts") or {}).get("n_known_historical")
        v["health_contract_listed"] = listed                      # every violation row the page's expander lists
        if known is not None:
            v["health_contract_known"] = int(known)
            v["health_contract_new"] = listed - int(known)        # listed rows minus the recorded known ones
    # calibration (Model Performance): v9.ledger() rows (any source), cutoff with unknown dates kept
    cal = []
    for name in ("bets_ledger.csv", "side_bets_ledger.csv"):
        d = _csv(out, name)
        if d is None:
            continue
        dd = pd.to_datetime(d["match_date"], errors="coerce")
        d = d[(dd.isna() | (dd >= pd.Timestamp(ctx["cutoff"][:10]))) & ~d["league"].isin(set(ctx["paper_leagues"]))]
        d = d[d["result"].isin(["WIN", "LOSS"])]
        odds, edge = pd.to_numeric(d["odds"], errors="coerce"), pd.to_numeric(d["edge_pct"], errors="coerce")
        claim = (edge / 100 + 1 / odds) if name == "bets_ledger.csv" else pd.to_numeric(d.get("model_prob"),
                                                                                         errors="coerce")
        cal.append(pd.DataFrame({"claimed": claim, "won": (d["result"] == "WIN").astype(float)}))
    if cal:
        c = pd.concat(cal, ignore_index=True)
        c = c[c["claimed"].between(0, 1)].dropna()
        if len(c):
            v["cal_n"], v["cal_claimed"], v["cal_realised"] = len(c), float(c["claimed"].mean()), float(c["won"].mean())
            v["cal_gap_pp"] = 100 * (v["cal_claimed"] - v["cal_realised"])
    # props (paper): SNIPER + MARKSMAN, WIN/LOSS
    pl = _csv(out, "player_ledger.csv")
    if pl is not None and "tier" in pl.columns:
        q = pl[pl["tier"].astype(str).str.upper().isin(["SNIPER", "MARKSMAN"])]
        q = q[q["result"].astype(str).str.upper().isin(["WIN", "LOSS"])]
        if len(q):
            w_ = int((q["result"].astype(str).str.upper() == "WIN").sum())
            v["props_n"], v["props_win_pct"] = len(q), 100 * w_ / len(q)
            v["props_roi_pct"] = 100 * float(pd.to_numeric(q["pnl"], errors="coerce").mean())
            v["props_wins"] = w_
        # rows of that record a correction sidecar flags (exclude=True, same ledger row and player)
        sc = _csv(out, "corrections/props_sentinel_prob.csv")
        if sc is not None and len(q):
            sc = sc[sc["exclude"].astype(str).str.lower().eq("true")].drop_duplicates("ledger_row")
            m = q.join(sc.set_index("ledger_row")[["player_name"]].rename(columns={"player_name": "_p"}), how="inner")
            v["props_flagged_sentinel"] = int((m["player_name"].astype(str) == m["_p"].astype(str)).sum())
    # data truth panel inputs, counted from the files themselves
    cup = _csv(out, "corrections/history_cup_unproven_rows.csv")
    if cup is not None:
        v["truth_cup_unproven"] = len(cup)
    ref = _csv(out, "corrections/history_score_refused.csv")
    if ref is not None:
        v["truth_scores_refused"] = len(ref)
    a = out / "collect" / "player_history_refetch_audit.json"
    if a.exists():
        j = json.loads(a.read_text(encoding="utf-8"))
        v["truth_zeros_unverified"] = int(j["plan"]["rows_to_verify"]) - int(j["stored_rows"])
    # Market Intelligence: suppressed tips = recorded direction conflicts
    dc = _csv(out, "direction_conflicts.csv")
    if dc is not None:
        v["intel_suppressed"] = len(dc)
    # Model Info thresholds (config)
    for k, x in (ctx.get("thresholds") or {}).items():
        if x is not None:
            v[f"cfg_{k}"] = 100 * float(x)
    return v


# ── the figure table: (id, page, label, how the page shows it) ────────────────────────────────────
#: (figure id, page key, label on the page, kind, nth occurrence, number index in the string)
FIGURES = [
    ("staked_pnl", "home", "Staked P/L (flat 1u)|Staked P/L", "metrics", 0, 0),
    ("staked_n", "home", "Settled staked bets", "stats", 0, 0),
    ("staked_win_pct", "home", "Win rate — staked", "stats", 0, 0),
    ("staked_roi_pct", "home", "ROI", "stats", 0, 0),
    ("stake_pnl", "home", "P/L at recorded stakes", "stats", 0, 0),
    ("stake_units", "home", "P/L at recorded stakes", "stats", 0, 1),
    ("drawdown", "home", "Drawdown from peak", "stats", 0, 0),
    ("last7_pnl", "home", "Last 7 days", "stats", 0, 0),
    ("last7_n", "home", "Last 7 days", "stats", 0, 1),
    ("plan_active", "home", "Active tips", "metrics", 0, 0),
    ("plan_up_ou", "home", "Upcoming staked tips (O/U 2.5)", "stats", 0, 0),
    ("plan_paper", "home", "Paper / no stake", "stats", 0, 0),
    ("plan_shadow_bet", "home", "Shadow engine BET", "stats", 0, 0),
    ("plan_shadow_n", "home", "Shadow engine BET", "stats", 0, 1),
    ("health_overall", "home", "System health", "metrics", 0, None),
    ("not_staked_pnl", "home", "Not staked P/L (caption)", "caption:Not staked|Paper Small gap P/L", 0, 0),
    ("not_staked_n", "home", "Not staked tips (caption)", "caption:Not staked|Paper Small gap P/L", 0, 1),
    ("not_staked_pre_policy_n", "home", "Not staked tips before the first policy record (caption)",
     "caption:Not staked", 0, 2),
    ("truth_zeros_unverified", "home", "Player rows zeros_unverified", "stats", 0, 0),
    ("truth_cup_unproven", "home", "Unproven cup rows", "stats", 0, 0),
    ("truth_scores_refused", "home", "Refused scores", "stats", 0, 0),
    ("staked_pnl", "overview", "Staked P/L", "metrics", 0, 0),
    ("staked_n", "overview", "Staked bets", "stats", 0, 0),
    ("stake_pnl", "overview", "P/L at recorded stakes", "stats", 0, 0),
    ("track_new_format_pnl", "overview", "new-format", "metrics", 0, 0),
    ("track_standard_pnl", "overview", "standard", "metrics", 0, 0),
    ("track_new_format_n", "overview", "Staked (new-format card)", "stats:Staked", 0, 0),
    ("track_new_format_hit_pct", "overview", "Hit (new-format card)", "stats:Hit", 0, 0),
    ("track_new_format_be_pct", "overview", "Break-even (new-format card)", "stats:Break-even", 0, 0),
    ("track_new_format_roi_pct", "overview", "ROI (new-format card)", "stats:ROI", 0, 0),
    ("staked_pnl", "tips", "P/L — all staked markets (flat 1u)|P/L — all staked markets", "metrics", 0, 0),
    ("staked_pnl_ex_ou25", "tips", "Excluding O/U 2.5", "metrics", 0, 0),
    ("staked_n", "tips", "Settled bets", "metrics", 0, 0),
    ("success_obs", "success", "Settled observations", "metrics", 0, 0),
    ("cal_gap_pp", "modelperf", "Overconfident by", "metrics", 0, 0),
    ("cal_claimed", "modelperf", "Model claimed", "metrics", 0, 0),
    ("cal_realised", "modelperf", "Actually happened", "metrics", 0, 0),
    ("cal_n", "modelperf", "Settled sample", "stats", 0, 0),
    ("props_win_pct", "props", "Win rate", "metrics", 0, 0),
    ("props_n", "props", "Settled", "stats", 0, 0),
    ("props_roi_pct", "props", "Flat ROI (1u, paper)", "stats", 0, 0),
    ("props_flagged_sentinel", "props", "Includes rows flagged (caption)", "caption:Includes rows flagged", 0, 0),
    ("leagues_pnl", "leagues", "P/L", "metrics", 0, 0),
    ("leagues_n", "leagues", "Staked tips settled", "stats", 0, 0),
    ("leagues_hit_pct", "leagues", "Hit", "stats", 0, 0),
    ("health_GREEN", "health", "GREEN", "metrics", 0, 0),
    ("health_AMBER", "health", "AMBER", "metrics", 0, 0),
    ("health_RED", "health", "RED", "metrics", 0, 0),
    ("health_NO_DATA", "health", "NO_DATA", "metrics", 0, 0),
    ("health_contract_new", "health", "Contract violations (new)|Contract violations", "metrics", 0, 0),
    ("health_contract_known", "health", "Known (baseline)", "metrics", 0, 0),
    ("plan_tips", "production", "Tips sized", "metrics", 0, 0),
    ("plan_active", "production", "With a stake", "metrics", 0, 0),
    ("plan_days", "production", "Days covered", "metrics", 0, 0),
    ("intel_suppressed", "intel", "Suppressed tips (all time)", "metrics", 0, 0),
    ("cfg_SNIPER_THRESHOLD", "modelinfo", "Global Big gap", "metrics", 0, 0),
    ("cfg_SNIPER_THRESHOLD_OVER", "modelinfo", "OVER threshold", "metrics", 0, 0),
    ("cfg_SNIPER_THRESHOLD_UNDER", "modelinfo", "UNDER threshold", "metrics", 0, 0),
]
#: where each recomputation comes from (the report's "source" column)
SOURCES = {
    "staked": "bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, paper leagues out) x "
              "src/staked.py labels from stake_audit.csv + stake_plan.csv",
    "plan": "stake_plan.csv rows dated today (UTC) or later x src/staked.plan_staked",
    "health": "system_health.json checks[] recounted",
    "cal": "bets_ledger (edge_pct/100 + 1/odds) + side_bets_ledger (model_prob), WIN/LOSS, >= cutoff",
    "props": "player_ledger.csv SNIPER + MARKSMAN, WIN/LOSS",
    "intel": "direction_conflicts.csv rows",
    "cfg": "config.py thresholds",
    "truth": "output/corrections/history_cup_unproven_rows.csv, history_score_refused.csv (rows); "
             "output/collect/player_history_refetch_audit.json (plan.rows_to_verify - stored_rows)",
}


def _source(fid: str) -> str:
    for k, s in (("plan_", "plan"), ("health_", "health"), ("cal_", "cal"), ("props_", "props"),
                 ("intel_", "intel"), ("cfg_", "cfg"), ("truth_", "truth")):
        if fid.startswith(k):
            return SOURCES[s]
    return SOURCES["staked"]


def _shown(page: dict, label: str, kind: str, nth: int):
    """The displayed string. `label` / the caption key may list alternatives with '|' (origin/main's older
    wording of the same figure); the first one the page shows is used and returned as the label."""
    if kind.startswith("caption:"):
        for key in kind.split(":", 1)[1].split("|"):
            hits = [c for c in page.get("captions", []) if c.startswith(key)]
            if len(hits) > nth:
                return hits[nth], key
        return None, None
    for lab in (kind.split(":", 1)[1] if kind.startswith("stats:") else label).split("|"):
        got = _find(page, "stats" if kind.startswith("stats:") else kind, lab, nth)
        if got is not None:
            return got, lab
    return None, None


def compare(rendered: dict, indep: dict) -> list[dict]:
    rows = []
    for fid, pg, label, kind, nth, idx in FIGURES:
        if pg not in rendered["pages"]:
            continue                                            # page not rendered in this run
        page = rendered["pages"][pg] or {}
        raw, lab = _shown(page, label, kind, nth)
        want = indep.get(fid)
        tier = indep.get("tier:" + fid)
        row = {"page": PAGES[pg], "label": lab or label.split("|")[0], "figure": fid, "shown": raw,
               "independent": want, "source": _source(fid)}
        if page.get("error") or page.get("exception"):
            row.update(status=UNIDENTIFIABLE, why=f"page did not render: {page.get('error') or page.get('exception')}")
        elif raw is None:
            row.update(status=UNIDENTIFIABLE, why="NOT_SHOWN: the page shows no figure under this label")
        elif want is None:
            row.update(status=UNIDENTIFIABLE, why="input missing: nothing to recompute from")
        elif idx is None:
            ok = str(raw).strip() == str(want).strip()
            row.update(status=MATCH if ok else MISMATCH, why="")
        else:
            got, dec = num(raw, idx)
            if got is None:
                row.update(status=UNIDENTIFIABLE, why="NOT_SHOWN: no number in the displayed value")
            else:
                tol = 0.5 * 10 ** (-dec) + 1e-9
                ok = abs(float(want) - got) <= tol
                row.update(status=MATCH if ok else MISMATCH, why="" if ok else f"shown {got} vs {float(want):.6g}",
                           independent=round(float(want), 6))
                if not ok and tier is not None:
                    tok = abs(float(tier) - got) <= tol
                    row["tier_basis"] = round(float(tier), 6)
                    row["why"] += ("; REPRODUCES on the old tier basis (SNIPER + MARKSMAN, any track)" if tok
                                   else "; does not reproduce on the tier basis either")
        rows.append(row)
    return rows


def run(root: Path, out: Path | None = None, pages: list[str] | None = None) -> dict:
    """Render `pages` (default all) of `root`, recompute from `out` (default root/output), compare."""
    rendered = render(root, pages)
    indep = independent(Path(out or root / "output"), rendered["ctx"])
    rows = compare(rendered, indep)
    counts = {k: sum(r["status"] == k for r in rows) for k in (MATCH, MISMATCH, UNIDENTIFIABLE)}
    return {"root": str(root), "ctx": rendered["ctx"], "counts": counts, "figures": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--json")
    ap.add_argument("--child")
    ap.add_argument("--pages")
    a = ap.parse_args(argv)
    if a.child:
        res = _child(Path(a.child), a.pages.split(","))
        sys.stdout.write("\n" + json.dumps(res, ensure_ascii=False, default=str) + "\n")
        return 0
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    res = run(Path(a.root))
    for r in res["figures"]:
        print(f"{r['status']:14} {r['page'][:26]:26} {r['label'][:40]:40} shown={str(r['shown'])[:40]!s:40} "
              f"indep={r['independent']}")
    print("COUNTS", res["counts"])
    if a.json:
        Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return 1 if res["counts"][MISMATCH] else 0


if __name__ == "__main__":
    raise SystemExit(main())
