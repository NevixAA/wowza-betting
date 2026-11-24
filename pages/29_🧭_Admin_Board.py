"""Admin Board - bars, TOP-3 and the research tracker on one page (owner 2026-11-22). ADMIN only.

  1. Bars       every league x market x side from the APPLIED config (models/best_params_*.json + the owner-apply
                ledger config/threshold_applied.json, read through src/bars_opt/serve.served_cell), cells without
                their own bar shown as NO BAR with the reason.
  2. TOP-3      every league's T1/T2/T3 for the current and recent gameweeks, stake label, result and P/L, plus
                each league-week's exclusion reasons (output/top3_league_weeks.csv).
  3. Research   config/research_tracker.json (curated) + live GitHub Actions state and each tool's record file.
  4. Evaluation the committed post-repair evaluation (src/clean_record_eval.py -> output/clean_record_evaluation.json
                + output/clean_record_evaluation_cells.csv), every record apart, every cell.

PRICES. The tip odds are read server-side from the PRIVATE price store (wowza-exec af-prices-live, the layout of
price_store_sync) with the read-only Streamlit secret PRICE_STORE_RO_TOKEN (dashboard_data/private_prices.py),
falling back to the public file only while it still carries them. Nothing read here is written anywhere. The
token is used only when login is configured (or WOWZA_TIER_OVERRIDE=admin): with dormant tiers everyone counts
as admin, so the private store stays closed then (fail closed, like the Daily Progress page).
"""
import os
import sys
import urllib.error
import urllib.request
import json
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard_access import require as _require_tier, login_configured  # noqa: E402
_require_tier("admin")  # tier guard (dashboard_access.PAGES) - owner only, never a public tier

st.set_page_config(page_title="Wowza | Admin Board", page_icon="🧭", layout="wide")

import dashboard_ui as ui  # noqa: E402
from dashboard_data import admin_board as AB  # noqa: E402
from dashboard_data import private_prices as PP  # noqa: E402

ui.page_header("Admin Board",
               "Bars of every league × market × side as they serve today, every league's TOP-3 with its stake "
               "label and why slots are empty, and every research tool's live state and record.",
               eyebrow_text="System · admin")

PRIVATE_OK = login_configured() or os.getenv("WOWZA_TIER_OVERRIDE", "").strip().lower() == "admin"
TTL = 1800      # GitHub API: 60 anonymous calls an hour; 2 calls per repo per refresh (+ capped extras)
MAX_EXTRA = 6   # per repo: workflows with no run among the last 100


# ── fetchers (cached; failures are cached too so a rate limit is not hammered) ────────────────────
@st.cache_data(ttl=TTL, show_spinner=False)
def _api_json(url: str, use_token: bool) -> tuple[dict | None, str]:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
               "User-Agent": "wowza-dashboard-admin"}
    tok = PP.token() if use_token else ""
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=15) as r:
            return json.loads(r.read().decode("utf-8")), ""
    except urllib.error.HTTPError as e:
        if tok and e.code in (401, 403) and "/NevixAA/wowza-exec/" not in url:
            return _api_json(url, False)         # the store token may not cover this repo: anonymous instead
        return None, {403: "rate-limited or no access", 404: "no access"}.get(e.code, f"HTTP {e.code}")
    except Exception as e:                                             # noqa: BLE001
        return None, type(e).__name__


@st.cache_data(ttl=TTL, show_spinner=False)
def _record_body(slug: str, private: bool, ref: str, path: str) -> tuple[str | None, str]:
    if private:
        if not PRIVATE_OK:
            return None, "private: login not configured"
        try:
            return PP.get(path, repo=slug, ref=ref).decode("utf-8"), ""
        except PP.StoreUnavailable as e:
            return None, f"private: {e}"
    url = f"https://raw.githubusercontent.com/{slug}/{ref}/{path}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "wowza-dashboard-admin"}),
                                    timeout=15) as r:
            return r.read().decode("utf-8"), ""
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"
    except Exception as e:                                             # noqa: BLE001
        return None, type(e).__name__


@st.cache_data(ttl=900, show_spinner=False)
def _store_top3_odds() -> tuple[dict | None, str]:
    if not PRIVATE_OK:
        return None, "login not configured"
    try:
        return AB.odds_from_csv(PP.public_file_text("wowza-betting", "output/top3_league_ledger.csv")), ""
    except PP.StoreUnavailable as e:
        return None, str(e)


tab_bars, tab_top3, tab_res, tab_eval, tab_side = st.tabs(["Bars", "TOP-3", "Research & Pro tools",
                                                          "Clean Record evaluation", "Side-model candidate"])

# ══ 1. BARS ═════════════════════════════════════════════════════════════════════════════════════
with tab_bars:
    inp = AB.load_bars_inputs()
    if inp["problems"]:
        ui.empty_state("ERROR", "; ".join(inp["problems"]) + ". Nothing is shown in its place.")
    else:
        bars = AB.bars_frame(inp)
        c = AB.bars_counts(bars)
        ui.stat_line([(k, str(v)) for k, v in c.items()] + [("Cells", str(len(bars)))])
        last = (inp["applied"].get("history") or [{}])[-1]
        st.caption(f"Applied config: {last.get('proposal_id', 'unknown')} applied {str(last.get('applied_at', ''))[:16]} "
                   f"by {last.get('approved_by', 'unknown')} ({len(last.get('entries') or [])} entries). Status = "
                   "what serving does today: PROVEN / PROMISING / UNPROVEN = the cell's verdict; OFF = switched off; "
                   "NOT APPLIED = a proposed cell the owner has not applied; NO BAR = no cell of its own, so no tip.")
        f1, f2, f3, f4, f5 = st.columns(5)
        tr = f1.multiselect("Track", sorted(bars["Track"].unique()), key="b_tr")
        lg = f2.multiselect("League", sorted(bars["League"].unique()), key="b_lg")
        mk = f3.multiselect("Market", list(dict.fromkeys(bars["Market"])), key="b_mk")
        sd = f4.multiselect("Side", list(dict.fromkeys(bars["Side"])), key="b_sd")
        stt = f5.multiselect("Status", [s for s in AB.STATUS_ORDER if c.get(s)], key="b_st")
        v = bars
        for col, sel in (("Track", tr), ("League", lg), ("Market", mk), ("Side", sd), ("Status", stt)):
            if sel:
                v = v[v[col].isin(sel)]
        ui.compact_table(v.reset_index(drop=True), {
            "Bar %": ui.num_col("Bar %", fmt="%.1f", help="The cell's one qualifying bar in its measure (EDGE = model "
                                "minus fair probability, EV = expected value, ALL = every tip of a league-wide cell)."),
            "Odds range": ui.text_col("Odds range", help="The cell's odds filter: a rule, not a price."),
            "WF bets": ui.num_col("WF bets", fmt="%d", help="Walk-forward out-of-sample bets."),
            "WF ROI %": ui.num_col("WF ROI %", fmt="%+.1f"),
            "Study bets": ui.num_col("Study bets", fmt="%d", help="Validation bets of the selection study."),
            "Study ROI %": ui.num_col("Study ROI %", fmt="%+.1f"),
            "ROI 95% CI": ui.text_col("ROI 95% CI", help="PROVEN needs the lower end above 0."),
            "CLV %": ui.num_col("CLV %", fmt="%+.2f")}, height=560)
        st.caption("Source: models/best_params_standard.json, models/best_params_side_markets.json and "
                   "config/threshold_applied.json, read through the serving rule (src/bars_opt/serve.py). "
                   f"{len(v)} of {len(bars)} rows shown.")

# ══ 2. TOP-3 ════════════════════════════════════════════════════════════════════════════════════
with tab_top3:
    led = AB.read_csv(AB.LEDGER)
    wk = AB.read_csv(AB.WEEKS)
    if led.empty and wk.empty:
        ui.empty_state("NO_DATA", "output/top3_league_ledger.csv and output/top3_league_weeks.csv are empty or absent.")
    else:
        odds, why = _store_top3_odds()
        src = "private price store"
        if odds is None and "odds" in led.columns:
            odds = {str(k): float(x) for k, x in zip(led["pick_id"], pd.to_numeric(led["odds"], errors="coerce"))
                    if pd.notna(x) and x > 1}
            src = "public file (until the price cut-over)"
        picks = AB.top3_frame(led, odds)
        weeks = AB.top3_weeks(wk, led)
        all_weeks = sorted(set(picks["Week"]) | set(weeks["Week"]) - {""}, reverse=True)
        w1, w2 = st.columns([1, 2])
        sel_w = w1.multiselect("Gameweeks", all_weeks, default=all_weeks[:2], key="t_wk")
        sel_l = w2.multiselect("League", sorted(set(picks["League"].dropna()) | set(weeks["League"].dropna())),
                               key="t_lg")
        pv = picks[picks["Week"].isin(sel_w)] if sel_w else picks
        wv = weeks[weeks["Week"].isin(sel_w)] if sel_w else weeks
        if sel_l:
            pv, wv = pv[pv["League"].isin(sel_l)], wv[wv["League"].isin(sel_l)]
        stk = pv[pv["Staked"] == "Yes"]
        pl = pd.to_numeric(stk["P/L (units)"], errors="coerce")
        ui.stat_line([("Picks", str(len(pv))), ("Staked", str(len(stk))),
                      ("Staked settled", str(int(pl.notna().sum()))),
                      ("Staked flat P/L", ui.fmt_units(pl.sum()) if pl.notna().any() else "—"),
                      ("League-weeks", str(len(wv))), ("Empty slots", str(int(wv["Empty slots"].fillna(0).sum())))])
        if odds is None:
            pv = pv.assign(**{"Odds at tip": "odds: private (token missing)" if why == "token missing"
                              else f"odds: private ({why})"})
            st.caption(f"Odds at tip: private ({why}).")
        else:
            st.caption(f"Odds at tip: from the {src}. Shown here only; never written to a public file.")
        ui.section_header("Picks", "T1 best value · T2 safest · T3 risky value. 'Stake' is the frozen stake decision "
                                   "of the pick; P/L is flat units at the tip's odds.")
        ui.compact_table(pv.drop(columns=["pick_id"]).reset_index(drop=True), {
            "Calibrated value pp": ui.num_col("Calibrated value pp", fmt="%+.1f",
                                              help="Calibrated value in percentage points (empty = not calibrated)."),
            "P/L (units)": ui.num_col("P/L (units)", fmt="%+.2f")})
        ui.section_header("League-weeks and why slots are empty",
                          "One row per frozen league-week: candidates seen, how many cleared the league's bar, picks, "
                          "and the exclusion count of every candidate that did not become a pick.")
        ui.compact_table(wv.reset_index(drop=True))
        led_leagues = set(weeks["League"].dropna())
        uni = {lg for _, lg in AB.universe(AB.read_json(AB.LEAGUES))}
        missing = sorted(uni - led_leagues)
        if missing:
            st.caption(f"{len(missing)} leagues have no frozen league-week yet (no fixture in a frozen window): "
                       + ", ".join(missing) + ".")
        st.caption("Source: output/top3_league_ledger.csv and output/top3_league_weeks.csv.")

# ══ 3. RESEARCH & PRO TOOLS ═════════════════════════════════════════════════════════════════════
with tab_res:
    reg, probs = AB.load_tracker()
    if reg is None:
        ui.empty_state("ERROR", "; ".join(probs) + ". Nothing is shown in its place.")
    else:
        if probs:
            st.warning("Tracker registry problems: " + "; ".join(probs))
        repos = reg["repos"]
        used = sorted({t["repo"] for t in reg["tools"] if t.get("workflows")})
        states, runs, rproblems = {}, {}, {}
        for rk in used:
            slug, private = repos[rk]["slug"], bool(repos[rk].get("private"))
            use_tok = PRIVATE_OK and bool(PP.token())
            wf, e1 = _api_json(f"https://api.github.com/repos/{slug}/actions/workflows?per_page=100", use_tok)
            rn, e2 = _api_json(f"https://api.github.com/repos/{slug}/actions/runs?per_page=100", use_tok)
            if wf is None or rn is None:
                rproblems[rk] = ("private repo, " if private else "") + (e1 or e2)
            states[rk], runs[rk] = AB.workflow_states(wf), AB.latest_runs(rn)
            if rk in rproblems:
                continue
            # a tracked workflow with no run among the last 100: ask for its own newest run (capped calls)
            missing = [w for t in reg["tools"] if t["repo"] == rk for w in (t.get("workflows") or [])[:1]
                       if ".github/workflows/" + w not in runs[rk]]
            for w in missing[:MAX_EXTRA]:
                one, _ = _api_json(f"https://api.github.com/repos/{slug}/actions/workflows/{w}/runs?per_page=1", use_tok)
                for r in (one or {}).get("workflow_runs") or []:
                    runs[rk].setdefault(".github/workflows/" + w, r)
        records = {}
        for t in reg["tools"]:
            rec = t.get("record")
            if not rec:
                continue
            rk = rec.get("repo", t["repo"])
            body, err = _record_body(repos[rk]["slug"], bool(repos[rk].get("private")), rec.get("ref", "main"),
                                     rec["path"])
            records[t["id"]] = AB.record_text(rec, body, err)
        tf = AB.tracker_frame(reg, states, runs, records, rproblems)
        sel = st.multiselect("Repo", sorted(tf["Repo"].unique()), key="r_repo")
        if sel:
            tf = tf[tf["Repo"].isin(sel)]
        n_fail = int((tf["Conclusion"] == "failure").sum())
        ui.stat_line([("Tools", str(len(tf))), ("Enabled", str(int((tf["Enabled"] == "enabled").sum()))),
                      ("Disabled", str(int((tf["Enabled"] == "disabled").sum()))), ("Last run failed", str(n_fail))])
        ui.compact_table(tf.reset_index(drop=True), {
            "Link": st.column_config.LinkColumn("Link", display_text="open"),
            "Record": ui.text_col("Record", help="Read live from the tool's record file; 'x / 150' = count / gate.")})
        if rproblems:
            st.caption("Live state unknown for: " + "; ".join(f"{repos[k]['label']} ({v})" for k, v in rproblems.items()))
        st.caption(f"Registry: config/research_tracker.json (updated {reg.get('updated', 'unknown')}). Live state: "
                   f"GitHub Actions API, cached {TTL // 60} min; last run = newest of the repo's last 100 runs.")

# ══ 4. CLEAN RECORD EVALUATION (committed; src/clean_record_eval.py) ══════════════════════════════════
with tab_eval:
    from dashboard_data import clean_record_eval as CRE  # noqa: E402
    CRE.render(st, ui, admin=True)

# ══ 5. SIDE-MODEL CANDIDATE (monthly corners / cards refit; owner 2026-11-24) ════════════════════════════
with tab_side:
    from dashboard_data import side_candidate_view as SCV  # noqa: E402
    ui.section_header("Side-model candidate",
                      "Corners / cards: the serving model and the latest monthly candidate, compared on recent held-out "
                      "games neither model trained on. A candidate is installed only when the owner merges its "
                      "'awaiting owner approval' PR.")
    try:
        _reg = json.loads(Path("config/track_registry.json").read_text(encoding="utf-8"))
    except Exception:                                                  # noqa: BLE001
        _reg = None
    ui.stat_line(SCV.serving_lines(_reg))
    _sb, _se = _record_body("NevixAA/wowza-betting", False, SCV.CANDIDATE_REF, SCV.SUMMARY_PATH)
    _summ = SCV.parse_summary(_sb)
    if _summ is None:
        ui.empty_state("NO_DATA", f"No monthly candidate yet ({_se or 'no summary on ' + SCV.CANDIDATE_REF}).")
    else:
        ui.stat_line(SCV.candidate_lines(_summ))
        _hb, _ = _record_body("NevixAA/wowza-betting", False, SCV.CANDIDATE_REF, SCV.HELDOUT_PATH)
        _hf = SCV.heldout_frame(_hb)
        if len(_hf):
            ui.compact_table(_hf.reset_index(drop=True))
        st.caption("Delta = candidate minus serving (negative = candidate better). Source: branch "
                   f"{SCV.CANDIDATE_REF}, written by .github/workflows/train_side_models.yml.")

from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()
