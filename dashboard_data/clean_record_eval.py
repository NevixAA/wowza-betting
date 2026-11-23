"""Read model of the committed post-repair evaluation (src/clean_record_eval.py, owner brief 2026-11-22 P1 item 8).

The pages never recompute it: they read output/clean_record_evaluation.json (summary, one total per segment / part /
source) and output/clean_record_evaluation_cells.csv (one row per league x market x model version) exactly as
committed. Segments are shown apart; no table here sums two segments. No price is read or shown.
"""
from __future__ import annotations

import json

import pandas as pd

SUMMARY = "clean_record_evaluation.json"
CELLS = "clean_record_evaluation_cells.csv"

SEGMENT_LABEL = {"pre_repair_historical": "Before the repairs (older models)",
                 "post_repair_forward": "Clean Record (forward)",
                 "research_only": "Research only",
                 "shadow_experiments": "Shadow experiments",
                 "real_price_backtests": "Backtests at real prices (older models)",
                 "synthetic_price_diagnostics": "Synthetic-price diagnostics (quarantined)"}
PART_LABEL = {"tips": "Tips", "top3_staked": "TOP-3 staked", "top3_unstaked": "TOP-3 no stake",
              "props_tips": "Props tips", "prediction_ledger_learning": "Prediction ledger (learning rows)",
              "first_sight": "first sight", "final": "final"}
#: keyed by the source file's stem (the pages never name a ledger file, so the data mirror is not asked for one)
_SOURCE_LABEL = {"bets_ledger": "O/U 2.5", "side_bets_ledger": "Side markets", "ht_ledger": "HT O/U",
                 "player_ledger": "Player props", "props_top3_ledger": "Props TOP-3",
                 "league_weekly_research": "Weekly league panel", "sharp_ledger": "Sharp tracker"}


class _Labels(dict):
    def get(self, k, default=None):                                    # noqa: D401 - dict.get on the file stem
        return _SOURCE_LABEL.get(str(k).rsplit(".", 1)[0], default)


SOURCE_LABEL = _Labels()
VERDICT_LABEL = {"NOT_ENOUGH_DATA": "Not enough data", "NEGATIVE_EDGE_CI_BELOW_ZERO": "Negative (CI below 0)",
                 "NO_PROVEN_EDGE": "No proven edge", "POSITIVE_ROI_UNCONFIRMED": "Positive ROI, unconfirmed",
                 "EDGE_CANDIDATE_FOR_OWNER_REVIEW": "Candidate for owner review"}
#: track-registry states as the pages word them (the retired word is never shown)
TRACK_STATE_LABEL = {"PAPER": "no automatic stake", "PAPER_LEAGUE": "no automatic stake (league)",
                     "BLOCKED": "blocked", "REAL": "automatic stake", "NOT_IN_TRACK_REGISTRY": "not registered"}


def load(out_dir=None) -> tuple[dict | None, pd.DataFrame | None, str]:
    """(summary, cells, problem). A missing or unreadable file gives None and says why; nothing is invented."""
    import config
    from pathlib import Path
    d = Path(out_dir) if out_dir else Path(config.OUTPUT_DIR)
    try:
        summary = json.loads((d / SUMMARY).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return None, None, f"output/{SUMMARY} is missing or unreadable ({type(e).__name__})"
    try:
        cells = pd.read_csv(d / CELLS, low_memory=False)
    except Exception as e:                                              # noqa: BLE001 - a viewer never dies on one file
        return summary, None, f"output/{CELLS} is missing or unreadable ({type(e).__name__})"
    return summary, cells, ""


def _pct(x, nd=1):
    return None if x is None or pd.isna(x) else round(100 * float(x), nd)


def _ci(lo, hi) -> str:
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return "—"
    return f"{100 * lo:+.1f}% … {100 * hi:+.1f}%"


def _row(seg: str, part: str, src: str | None, t: dict) -> dict:
    clv = t.get("clv") or {}
    ci = t.get("roi_ci") or [None, None]
    return {"Segment": SEGMENT_LABEL.get(seg, seg), "Record": " · ".join(x for x in (PART_LABEL.get(part, SOURCE_LABEL.get(part, part)),
                                                                                   SOURCE_LABEL.get(src or "", PART_LABEL.get(src or "", src or "")))
                                                                       if x),
            "Settled": t.get("settled"), "Matchdays": t.get("matchdays"),
            "Real odds %": _pct(t.get("real_odds_coverage")), "P/L (flat 1u)": t.get("pnl_units"),
            "ROI %": _pct(t.get("roi")), "ROI 95% CI": _ci(*ci),
            "Staked": (t.get("staked") or {}).get("n"), "Max drawdown (u)": t.get("max_drawdown_units"),
            "Log loss model": t.get("log_loss_model"), "Log loss vs market": t.get("d_log_loss_model_minus_market"),
            "Rows vs market": t.get("n_vs_market"), "Brier model": t.get("brier_model"),
            "Certified CLV %": _pct(clv.get("coverage")), "Certified CLV n": clv.get("certified_n"),
            "Verdict": VERDICT_LABEL.get(t.get("verdict"), t.get("verdict"))}


def totals_table(summary: dict) -> pd.DataFrame:
    """One row per segment x part x source total (the committed totals, never a sum across rows)."""
    rows = []
    segs = (summary or {}).get("segments") or {}
    for seg, body in segs.items():
        if not isinstance(body, dict):
            continue
        for part, v in body.items():
            if not isinstance(v, dict):
                continue
            if isinstance(v.get("total"), dict):
                rows.append(_row(seg, part, None, v["total"]))
                continue
            for src, w in v.items():
                if isinstance(w, dict) and isinstance(w.get("total"), dict):
                    rows.append(_row(seg, part, src, w["total"]))
    return pd.DataFrame(rows)


def historical_table(summary: dict) -> pd.DataFrame:
    """The owner's historical figures beside their recomputation on the committed ledgers."""
    figs = (((summary or {}).get("segments") or {}).get("pre_repair_historical") or {}).get("owner_historical_figures") or []
    rows = []
    for f in figs:
        raw, cor = f.get("raw_ledger") or {}, f.get("corrections_applied") or {}
        rows.append({"Figure": f.get("figure"), "Owner n": f.get("owner_n"), "Owner ROI %": f.get("owner_roi_pct"),
                     "Ledger n": raw.get("n"), "Ledger ROI %": raw.get("roi_pct"),
                     "Corrected n": cor.get("n"), "Corrected ROI %": cor.get("roi_pct"),
                     "Corrected 95% CI": "—" if not cor.get("roi_ci_pct") or None in cor["roi_ci_pct"]
                     else f"{cor['roi_ci_pct'][0]:+.1f}% … {cor['roi_ci_pct'][1]:+.1f}%",
                     "Matches now": "yes" if f.get("matches_owner_figure_now") else "no (later settlements)"})
    return pd.DataFrame(rows)


def cells_table(cells: pd.DataFrame, segment: str, part: str | None = None) -> pd.DataFrame:
    """The cells of ONE segment (and part), for display."""
    if cells is None or not len(cells):
        return pd.DataFrame()
    c = cells[cells["segment"] == segment]
    if part:
        c = c[c["part"] == part]
    if not len(c):
        return pd.DataFrame()
    out = pd.DataFrame({
        "Record": [" · ".join(x for x in (PART_LABEL.get(str(p), SOURCE_LABEL.get(str(p), str(p))), SOURCE_LABEL.get(str(s), "" if pd.isna(s) else
                                                                                         str(s))) if x)
                   for p, s in zip(c["part"], c["source"])],
        "League": c["league"], "Market": c["market"], "Model version": c["model_version"],
        "Settled": c["settled"], "Matchdays": c["matchdays"],
        "ROI %": [_pct(x) for x in c.get("roi", pd.Series(index=c.index, dtype=float))],
        "ROI 95% CI": [_ci(a, b) for a, b in zip(c.get("roi_ci_lo", pd.Series(index=c.index, dtype=float)),
                                                 c.get("roi_ci_hi", pd.Series(index=c.index, dtype=float)))],
        "P/L (flat 1u)": c.get("pnl_units"), "Log loss model": c.get("log_loss_model"),
        "Log loss vs market": c.get("d_log_loss_model_minus_market"),
        "Certified CLV n": c.get("clv_certified_n"),
        "Policy track": [TRACK_STATE_LABEL.get(str(x), str(x)) if pd.notna(x) else "—"
                  for x in c.get("ft_track_state", pd.Series(index=c.index, dtype=object))],
        "Verdict": [VERDICT_LABEL.get(str(v), str(v)) for v in c["verdict"]]})
    return out.reset_index(drop=True)


# ── page piece (Clean Record page and Admin Board read the same committed files) ─────────────────────
def render(st, ui, *, admin: bool = False) -> None:
    """The evaluation section: headline verdict, per-segment totals, the owner's historical figures, and (admin)
    every cell. Read-only; says plainly when the committed files are missing."""
    summary, cells, problem = load()
    ui.section_header("Post-repair evaluation (every record apart)",
                      "Reproducible evaluation (src/clean_record_eval.py): before the repairs, the Clean Record "
                      "(TOP-3 staked and no stake, tips, props, prediction-ledger learning rows), research and "
                      "shadow logs, each on its own row and never summed. Built from the committed ledgers.")
    if summary is None:
        ui.empty_state("NO_DATA", problem + ". Nothing is shown in its place.")
        return
    h = summary.get("headline") or {}
    fs = h.get("forward_settled") or {}
    team = [v for k, v in fs.items() if k.rsplit(".", 1)[0] in ("bets_ledger", "side_bets_ledger", "ht_ledger")]
    ui.stat_line([("Verdict", str(h.get("verdict", "—"))),
                  ("Clean Record settled (TOP-3 staked)", str(fs["top3_staked"]) if "top3_staked" in fs else "—"),
                  ("Clean Record settled (team tips)", str(sum(team)) if team else "—"),
                  ("Forward cells", str(h["forward_cells"]) if "forward_cells" in h else "—"),
                  ("Data as of", str(summary.get("as_of_last_settled_match") or "—"))])
    mins = h.get("minimums") or {}
    st.caption(f"Evaluation only, not a performance claim. A cell is readable from {mins.get('settled', '?')} settled "
               f"bets over {mins.get('matchdays', '?')} matchdays; CLV evidence needs {mins.get('clv', '?')} certified "
               "closes. 'No proven edge yet' is the expected result this early. Nothing here changes a bar or a stake.")
    tt = totals_table(summary)
    if len(tt):
        ui.compact_table(tt, {"ROI 95% CI": ui.text_col("ROI 95% CI", help="Matchday block bootstrap, 2,000 draws, 95%."),
                              "Log loss vs market": ui.num_col("Log loss vs market", fmt="%+.4f",
                                                               help="Model minus de-vigged bookmaker log loss on the "
                                                                    "same rows; negative = model better."),
                              "Certified CLV %": ui.num_col("Certified CLV %", fmt="%.1f",
                                                            help="Share of settled rows with a certified close "
                                                                 "(pre-kickoff, within 30 minutes).")})
    ht = historical_table(summary)
    if len(ht):
        ui.eyebrow("Before the repairs: the owner's historical figures, recomputed")
        ui.compact_table(ht)
        st.caption("Older models, the two top retired tiers (the old staked basis), matches from the performance "
                   "cutoff, settled before the Clean Record start. Kept apart; never part of a Clean Record figure.")
    if cells is None:
        st.caption(problem)
        return
    fwd = cells_table(cells, "post_repair_forward")
    fwd = fwd[~fwd["Record"].str.startswith(PART_LABEL["prediction_ledger_learning"])] if len(fwd) else fwd
    ui.eyebrow("Clean Record cells (league × market × model version)")
    if len(fwd):
        ui.compact_table(fwd)
    else:
        ui.empty_state("NO_DATA", "No settled Clean Record bet yet.")
    if admin:
        for seg in ("pre_repair_historical", "research_only"):
            with st.expander(f"{SEGMENT_LABEL[seg]}: every cell"):
                ui.compact_table(cells_table(cells, seg), height=420)
        with st.expander("Prediction ledger learning rows: every cell (scores only, no P/L)"):
            ui.compact_table(cells_table(cells, "post_repair_forward", "prediction_ledger_learning"), height=420)
    st.caption(f"Source: output/{SUMMARY} and output/{CELLS} (committed; start {summary.get('clean_record_start_utc')}).")
