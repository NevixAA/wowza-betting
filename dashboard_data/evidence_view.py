"""Presentation for the four evidence panels (owner brief 2026-10-26, part D). DISPLAY ONLY.

Every value is handed in by the page from dashboard_data/evidence_panels.py; nothing here computes a
number, writes a file or imports a module that sizes, settles, selects or promotes. Design system:
dashboard_ui only (cards, chips, compact tables; no hex, no inline style).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import dashboard_ui as ui
from dashboard_data.evidence_panels import NO_POLICY_CHANGE, SMALL_N
from src.tier_labels import tier_label


def _none(x) -> bool:
    return x is None or (isinstance(x, float) and x != x) or (not isinstance(x, (str, dict, list)) and pd.isna(x))


def _roi(x) -> str:
    return "—" if _none(x) else ui.fmt_roi(float(x))


def _roi_n(x, n) -> str:
    return "—" if _none(n) or int(n) == 0 else f"{_roi(x)} · n={int(n):,}"


def _clv_frac(x) -> str:
    """CLV stored as a fraction (Pro review convention) shown as a percent."""
    return "—" if _none(x) else ui.fmt_clv(100 * float(x), dp=2)


def _share(x) -> str:
    return "—" if _none(x) else f"{100 * float(x):.0f}%"


def _track(t: str) -> str:
    return str(t).replace("|", " · ").replace("_", " ")


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# 1. BANKROLL — since the last formal review
# ═════════════════════════════════════════════════════════════════════════════════════════════════
_DIRECTION_CHIP = {"IMPROVING": ("PASS", "IMPROVING", "POSITIVE"), "STABLE": ("OBSERVE", "STABLE", "INFO"),
                   "DETERIORATING": ("FAIL", "DETERIORATING", "NEGATIVE"),
                   "INSUFFICIENT": ("SMALL_SAMPLE", "INSUFFICIENT SAMPLE", "NO_DATA")}


def direction_chip(state: str) -> str:
    key, word, tone = _DIRECTION_CHIP.get(str(state), ("UNKNOWN", str(state), "NO_DATA"))
    return ui.status_chip(key, label=word, tone=tone)


SINCE_CONFIG = {
    "review_roi": ui.pct_col("Review ROI"), "since_roi": ui.pct_col("Since ROI"),
    "roll14_roi": ui.pct_col("14d ROI"), "roll30_roi": ui.pct_col("30d ROI"),
    "review_clv_median": ui.pct_col("Review CLV (median)"), "since_clv_median": ui.pct_col("Since CLV (median)"),
    "review_clv_coverage": ui.pct_col("Review CLV coverage"), "since_clv_coverage": ui.pct_col("Since CLV coverage"),
    "review_n": ui.num_col("Review n", fmt="%d"), "since_n": ui.num_col("Since n", fmt="%d"),
    "since_matchdays": ui.num_col("Since matchdays", fmt="%d"), "roll14_n": ui.num_col("14d n", fmt="%d"),
    "roll30_n": ui.num_col("30d n", fmt="%d"), "direction_reason": ui.text_col("Why", width="large")}
SINCE_COLS = ["track", "state", "direction", "review_n", "review_roi", "review_clv_median", "review_clv_coverage",
              "since_n", "since_matchdays", "since_roi", "since_clv_median", "since_clv_coverage", "roll14_n",
              "roll14_roi", "roll30_n", "roll30_roi", "direction_reason"]
REAL_STATES = ("ACTIVE", "ACTIVE_LOW", "ACTIVE_STRONG", "PROBATION", "DEGRADED")


def no_policy_change_notice() -> None:
    with st.container(border=True):
        st.markdown(ui.status_chip("REVIEW", label="READ ONLY · NO POLICY CHANGE", tone="INFO"))
        st.markdown(f"**{NO_POLICY_CHANGE}**")


def bankroll_since_panel(meta: dict, rows: pd.DataFrame) -> None:
    ui.section_header("Since last formal bankroll review",
                      "Pro's latest formal review against what the live ledgers recorded after its evidence "
                      "cutoff, per production track. Read only.")
    no_policy_change_notice()
    if not (meta or {}).get("evidence_cutoff"):
        ui.empty_state("NOT_CONNECTED", "Pro's formal bankroll review (output/bankroll/track_evidence.json + "
                       "current_policy_recommendation.json) could not be read, so there is no review date to "
                       "compare against.")
        return
    if rows is None or rows.empty:
        ui.empty_state("NO_DATA", "No production tracks to compare.")
        return
    st.caption(f"Formal review as of **{meta.get('as_of') or '—'}** (evidence cutoff "
               f"{meta.get('evidence_cutoff') or '—'}, generated {meta.get('generated_at') or '—'}; its own verdict: "
               f"policy change recommended **{meta.get('policy_change_recommended') or '—'}**). Since-review = settled "
               f"bets dated after the cutoff, last settled {meta.get('last_settled') or '—'}; rolling windows end "
               f"{meta.get('today')}. Same bets as the review counted: live tips, paper leagues out, the tiers the "
               "review's evidence used.")
    real = rows[rows["state"].isin(REAL_STATES)]
    recs = real.to_dict("records")
    for i in range(0, len(recs), 3):
        for col, r in zip(st.columns(3), recs[i:i + 3]):
            ui.metric_card("Since-review ROI", _roi(r["since_roi"]), title=_track(r["track"]),
                           supporting=[("Since n · days", f"{int(r['since_n'])} · {int(r['since_matchdays'])}"),
                                       ("Review ROI", _roi_n(r["review_roi"], r["review_n"])),
                                       ("CLV med. review → since",
                                        f"{_clv_frac(r['review_clv_median'])} → {_clv_frac(r['since_clv_median'])}"),
                                       ("Rolling 14d", _roi_n(r["roll14_roi"], r["roll14_n"])),
                                       ("Rolling 30d", _roi_n(r["roll30_roi"], r["roll30_n"]))],
                           status=direction_chip(r["direction"]) + " " + ui.status_chip(r["state"]),
                           n=int(r["since_n"]), container=col)
    with st.expander("All production tracks — table (paper and blocked tracks included)"):
        ui.compact_table(rows[[c for c in SINCE_COLS if c in rows.columns]], SINCE_CONFIG)
    st.caption("Direction rule: " + str(meta.get("rule", "")))


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# 2. MODEL LAB — H-MODEL-01
# ═════════════════════════════════════════════════════════════════════════════════════════════════
def _ci(lo, hi, dp: int = 4) -> str:
    return "CI pending" if _none(lo) or _none(hi) else f"[{float(lo):+.{dp}f}, {float(hi):+.{dp}f}]"


def _req(ok: bool, text: str) -> str:
    return ("✓ " if ok else "○ ") + text


def hmodel_panel(meta: dict, rows: pd.DataFrame) -> None:
    ui.section_header("H-MODEL-01 — champion vs HGB challenger, per track",
                      "Paired settled first-sight rows. Negative Δ = the challenger is better. Shadow only.")
    if rows is None or rows.empty:
        ui.empty_state("NO_DATA", "output/model_lineage/shadow_experiments_eval.json has no results yet.")
        return
    for col, r in zip(st.columns(min(3, len(rows))), rows.to_dict("records")):
        with col:
            ui.metric_card("Δ log loss (HGB − champion)",
                           "—" if _none(r["delta_log_loss"]) else f"{float(r['delta_log_loss']):+.4f}",
                           title=f"{r['hypothesis']} · {r['track']}",
                           supporting=[("Log loss champion → HGB", f"{float(r['ll_champion']):.4f} → "
                                        f"{float(r['ll_challenger']):.4f}" if not _none(r["ll_champion"]) else "—"),
                                       ("Δ LL 95% CI", _ci(r["delta_ll_lo"], r["delta_ll_hi"])),
                                       ("Δ Brier", "—" if _none(r["delta_brier"]) else f"{float(r['delta_brier']):+.4f}"),
                                       ("Δ Brier 95% CI", _ci(r["delta_brier_lo"], r["delta_brier_hi"])),
                                       ("Matchdays (blocks)", "—" if _none(r["matchdays"]) else int(r["matchdays"]))],
                           status=ui.status_chip("SHADOW", label="SHADOW — NOT PROMOTED"),
                           n=int(r["settled_paired_n"]), container=st)
            st.progress(float(r["progress"]), text=f"{int(r['settled_paired_n']):,} / {int(r['required_n']):,} "
                                                    "settled paired rows")
            st.caption(" · ".join([
                _req(r["req_n_met"], f"n ≥ {int(r['required_n']):,}"),
                _req(r["req_ll_met"], f"Δ LL < {meta.get('max_ll_delta_exclusive', 0.0):g}"),
                _req(r["req_brier_met"], f"Δ Brier ≤ {meta.get('max_brier_worsening', 0.0001):g}"),
                f"window closes {r['window_expires'] or '—'} ({r['days_left'] if r['days_left'] is not None else '—'} d)"]))
            if r["early_fail_zone"]:
                st.markdown(ui.status_chip("FAIL", label="EARLY-FAIL ZONE · CHAMPION SIGNIFICANTLY AHEAD"))
    st.caption(f"Second-period requirement ({meta.get('gate_version') or 'PG-1.0'}, config/promotion_gate.json "
               f"prospective): per track n ≥ {meta.get('min_n', 1000):,} settled rows after the freeze, Δ log loss < "
               f"{meta.get('max_ll_delta_exclusive', 0.0):g}, Δ Brier ≤ {meta.get('max_brier_worsening', 0.0001):g}, "
               f"within {meta.get('max_age_days', 180)} days of registration; from n ≥ "
               f"{meta.get('early_fail_min_n', 300)} a CI wholly above zero fails early. A ✓ below the n "
               "minimum is NOT evidence. CIs are the evaluator's matchday block bootstrap (≥ 5 blocks to print; "
               f"the gate itself wants ≥ {meta.get('min_bootstrap_blocks', 20)}). Computed "
               f"{str(meta.get('computed_at') or '—')[:16]} by src/shadow_experiments.evaluate (update_results CI).")


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# 3. MARKET INTELLIGENCE — H-EDGE2 forward report
# ═════════════════════════════════════════════════════════════════════════════════════════════════
def _counts(d: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in (d or {}).items()) or "—"


def edge2_report_panel(meta: dict, items: list[dict]) -> None:
    ui.section_header("H-EDGE2 forward experiment report",
                      "One card per registered hypothesis. Status words are REGISTERED / ACCRUING / "
                      "ELIGIBLE_FOR_REVIEW only — never CONFIRMED.")
    if not items:
        ui.empty_state("NO_DATA", "Pro's H-EDGE2 forward progress has no hypotheses.")
        return
    for i in range(0, len(items), 2):
        for col, h in zip(st.columns(2), items[i:i + 2]):
            sh, de = h["shadow"], h["descriptive"]
            m, sec = h["metrics"], h["secondary"]
            eligible = h["status"] == "ELIGIBLE_FOR_REVIEW"
            sel = (sec or {}).get("selection") or {}
            pq = (sec or {}).get("probability_quality") or {}
            sup = [("Settled two-sided", f"{int(h['settled'] or 0):,} / {int(h['min_settled'] or 0):,}"),
                   ("Gameweeks · matchdays", f"{int(h['gameweeks'] or 0)} / {int(h['min_gameweeks'] or 0)} · "
                                             f"{'—' if de.get('matchdays') is None else de['matchdays']}"),
                   ("Logged since start (v9 shadow)", f"{sh['logged']:,} · two-sided {_share(sh['two_sided_share'])}"),
                   ("Bet frequency", f"{sh['selections']} of {sh['logged']} ({_share(sh['selection_rate'])})"),
                   ("Top league", "—" if not sh["top_league"] else f"{sh['top_league']} {_share(sh['top_league_share'])}"
                                                                    f" of {sh['leagues']}"),
                   ("Sides selected", _counts(sh["sides_selected"]))]
            if eligible:
                sup += [("ROI (selections)", "—" if not sel.get("n") else
                         f"{_roi(sel.get('roi'))} {_ci(*(sel.get('roi_ci95') or [None, None]), dp=3)} · n={sel['n']}"),
                        ("CLV coverage", _share(((sec or {}).get("clv") or {}).get("coverage"))),
                        ("LL model / market / market+Wowza", " / ".join(
                            "—" if (pq.get(k) or {}).get("log_loss") is None else f"{pq[k]['log_loss']:.4f}"
                            for k in ("model", "market", "market_plus_wowza"))),
                        ("Primary Δ LL · BH q", f"{m.get('ll_diff_mean', 0):+.5f} {_ci(*m.get('ll_diff_ci95', [None, None]), dp=5)}"
                                                f" · q={m.get('bh_q_family', '—')}")]
            state = h["status"] if not str(h["status"]).startswith("INVALID") else "ERROR"
            ui.metric_card("Progress", f"{float(h['progress_pct'] or 0):.1f}%", title=f"{h['id']} · {_track(h['track'])} "
                           f"{h['market']}", supporting=sup,
                           status=ui.status_chip(state, label=str(h["status"]).replace("_", " ")),
                           evidence=ui.evidence_badge("PROSPECTIVE", h["settled"]), container=col)
            if not eligible:
                col.caption("Outcome metrics (ROI and CI, CLV, model / market / market+Wowza log loss and Brier, "
                            "BH/FDR) are withheld until ELIGIBLE_FOR_REVIEW — the frozen registry's interim policy, "
                            "so the test cannot be stopped on a peek. Counts above are not evidence.")
            if sh["odds_bands_selected"]:
                col.caption("Selected odds: " + _counts(sh["odds_bands_selected"]))
    st.caption(f"Forward start {meta.get('forward_start') or '—'} · engine {meta.get('engine_version') or '—'} · "
               f"registration intact {meta.get('registration_intact')} · engine hashes match "
               f"{meta.get('engine_hashes_match_registration')} · production changed {meta.get('production_changed')}. "
               "Counts 'since start' are read live from v9 output/edge_engine_shadow.csv with the evaluator's row rules "
               "(first sight, pre-kickoff, engine version); settled n, gameweeks, status and every metric are Pro's "
               "(src/studies/edge2_forward.py). CLV: the shadow log carries no certified close, so CLV coverage is 0% "
               "until a close source is attached.")


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# 4. NF OVER 1.5 — REAL PRICE ONLY
# ═════════════════════════════════════════════════════════════════════════════════════════════════
O15_CONFIG = {"n": ui.num_col("n", fmt="%d"), "matchdays": ui.num_col("Matchdays", fmt="%d"),
              "roi": ui.pct_col("ROI"), "roi_lo95": ui.pct_col("ROI 95% lo"), "roi_hi95": ui.pct_col("ROI 95% hi"),
              "units": ui.num_col("P/L (u)"), "mean_odds": ui.num_col("Mean odds"),
              "ledger_clv_mean_pct": ui.num_col("Ledger CLV % (not certified)", fmt="%+.2f"),
              "ledger_clv_coverage": ui.pct_col("Ledger CLV coverage"),
              "certified_clv_mean_pct": ui.num_col("Certified CLV %", fmt="%+.2f"),
              "certified_clv_coverage": ui.pct_col("Certified CLV coverage")}


def over15_panel(meta: dict, rows: pd.DataFrame) -> None:
    ui.section_header("New-format Over 1.5 — REAL PRICE ONLY",
                      "Live settled tips on genuinely quoted prices, by gap band. Filled / default / fair-odds prices "
                      "are excluded.")
    if rows is None or rows.empty or not int(rows["n"].max() or 0):
        ui.empty_state("NO_DATA", "No settled new-format Over 1.5 tip on a genuine price yet.")
        return
    st.markdown(ui.status_chip("PAPER", label="EVIDENCE ONLY · NO SELECTION CHANGE") + " "
                + ui.status_chip("PROSPECTIVE", label="LIVE TIP RECORD · REAL QUOTES ONLY"))
    allr = rows[rows["gap_band"] == "ALL"].set_index("window")
    cols = st.columns(3)
    for col, w in zip(cols, ("30d", "60d", "season")):
        if w not in allr.index:
            continue
        r = allr.loc[w]
        ui.metric_card("ROI (flat 1u)", _roi(r["roi"]), title=f"All gap bands · {w}",
                       supporting=[("95% CI", _ci(r["roi_lo95"], r["roi_hi95"], dp=3)),
                                   ("n · matchdays", f"{int(r['n'])} · {int(r['matchdays'])}"),
                                   ("Mean odds", "—" if _none(r["mean_odds"]) else ui.fmt_odds(r["mean_odds"])),
                                   ("Certified CLV coverage", _share(r["certified_clv_coverage"]))],
                       n=int(r["n"]), floor=SMALL_N, container=col)
    show = rows.assign(gap_band=rows["gap_band"].map(lambda b: "All" if b == "ALL" else tier_label(b)))
    ui.compact_table(show.drop(columns=["small_sample"]), O15_CONFIG)
    small = rows[(rows["n"] > 0) & rows["small_sample"]]
    if len(small):
        st.markdown(ui.status_chip("SMALL_SAMPLE", label=f"SMALL SAMPLE · {len(small)} cells below n={SMALL_N}")
                    + " — a rate on so few bets is noise; read the CI.")
    ex = meta.get("exclusions") or {}
    st.caption(f"Season = settled from {meta.get('season_from')} (first {meta.get('first') or '—'}, last "
               f"{meta.get('last') or '—'}); windows end {meta.get('today')}. Genuine price: the ledger odds as "
               f"captured, excluding no price ({ex.get('no_price', 0)}), the {1.40:.2f} backtest fill constant "
               f"({ex.get('fill_constant_1.40', 0)}; src/backtest.py _DEFAULT_ODDS, dropped fail-closed as in "
               f"market_edge.exclude_synthetic_prices) and fair-odds settlements ({ex.get('settled_at_fair_odds', 0)}) — "
               f"{ex.get('kept', 0)} of {ex.get('settled_rows', 0)} settled rows kept. ROI CI: matchday block "
               "bootstrap, 2,000 draws. CERTIFIED CLV (close strictly pre-kickoff, within 30 min) does not exist for "
               "ledger Over 1.5 rows — certified closes are recorded only for O/U 2.5 shadow rows — so its coverage is "
               "0%; the ledger CLV column uses the ledger's close, which is NOT certified. Gap bands are the legacy "
               "tier sizes (model-vs-price gap), not evidence.")
