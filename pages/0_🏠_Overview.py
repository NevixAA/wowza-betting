"""Overview — the whole estate on one screen.

WHY THIS PAGE EXISTS. The dashboard had nine pages and every one of them was a v9 page. Exactly
one reached Pro, and nothing at all reached v11. So the only way to answer "is Wowza healthy"
was to open three repos in three terminals. This page answers it in one screen, and it is honest
about the parts it cannot reach rather than leaving them blank.

WHAT IT REFUSES TO DO. It does not add v9's P/L to Pro's research numbers or to Fantasy's points.
Those are three different units measuring three different things, and a single "Wowza score"
would be a fabrication. Each repo keeps its own scale and its own state label.

Visual redesign 2026-10-25 (stage 2b): four sections - SYSTEM HEALTH (overall RED / AMBER / GREEN at
the top, one card per check), PRODUCTION, RESEARCH, DATA FEEDS (one card per subsystem: V9 live, Pro
research, V11 shadow, Fantasy). PRESENTATION ONLY - components in dashboard_data/overview_view.py. Reads
are cached for two minutes (they were re-run on every widget interaction) and Pro's directory listings
are fetched concurrently; the values are the same calls' values.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_access import require as _require_tier  # noqa: E402
_require_tier("admin")  # tier guard (dashboard_access.PAGES), 2026-10-18

import dashboard_ui as ui  # noqa: E402
from src.tier_labels import GAP_EXPLAINER, PAPER_TIER_LABEL, STAKED_LABEL, tier_label  # noqa: E402
from dashboard_data import fantasy, pro, v9, v11  # noqa: E402
from dashboard_data import overview_view as V  # noqa: E402
from dashboard_data.core import REPOS, repo_path  # noqa: E402

st.set_page_config(page_title="Wowza | Overview", page_icon="🏠", layout="wide")


@st.cache_data(ttl=120, show_spinner="Reading the estate…")
def _estate() -> dict:
    """Every read the page makes, once per two minutes (not once per widget interaction)."""
    out = {}
    try:
        from src.system_health import evaluate as _sh_eval
        out["checks"], out["checks_error"] = _sh_eval(), None
    except Exception as e:                                          # noqa: BLE001 - health must never break the page
        out["checks"], out["checks_error"] = None, type(e).__name__
    out["perf"] = v9.performance()
    out["by_league"] = v9.by_league(days=None, staked_only=True)
    out["v9_health"] = v9.health()
    out["pro_available"] = pro.available()
    if out["pro_available"]:
        V.prefetch_pro_listings()
        out["pro_stores"], out["pro_health"] = pro.canonical_stores(), pro.health_artifacts()
    else:
        out["pro_absent"] = pro.absent_reason()
    out["v11_available"] = v11.available()
    out["v11_health"] = v11.health() if out["v11_available"] else None
    out["v11_absent"] = None if out["v11_available"] else v11.absent_reason()
    out["fantasy"] = fantasy.health()
    out["validity"] = pro.research_validity()
    return out


E = _estate()
checks = E["checks"]
hv = E["v9_health"]
_pf = hv["predictions"]["freshness"]
ui.page_header("Wowza — estate overview",
               "Three repositories, three jobs, three scales. Nothing on this page is summed across them, because "
               "units in betting, research and Fantasy are not the same unit.",
               eyebrow_text="Admin · estate", state=V.overall(checks) if checks is not None else "NO_DATA",
               freshness=ui.freshness_badge(_pf.state, f"v9 predictions · {_pf.label}"))

# ═══ SYSTEM HEALTH ═══════════════════════════════════════════════════════════════════════════════
ui.section_header("System health", "Explicit red / amber / green rules (owner repair brief 2026-10-22). Worst first; "
                  "every card carries its rule.")
if checks is None:
    ui.empty_state("ERROR", f"System health unavailable: {E['checks_error']}")
else:
    _ov = V.overall(checks)
    st.markdown(ui.status_chip(_ov, label=f"OVERALL {_ov}") + f" &nbsp;**{V.count_line(checks)}**")
    V.health_cards(checks)

# ═══ PRODUCTION ═══════════════════════════════════════════════════════════════════════════════════
ui.section_header("Production — V9 live betting", f"Staked tiers only ({STAKED_LABEL}). {PAPER_TIER_LABEL} is paper "
                  f"and is kept out of every P/L headline. {GAP_EXPLAINER}")
perf = E["perf"]
if not perf.get("available"):
    ui.empty_state("NO_DATA", perf.get("why", "no settled bets"))
else:
    tracks = {k: val for k, val in perf["by_track"].items() if k != "unknown"}
    total_n = sum(t["n"] for t in tracks.values())
    # THE HEADLINE IS THE STAKED FIGURE, NOT THE ALL-TIPS FIGURE (see overview_view.staked_cards).
    V.staked_cards(perf)
    st.caption(
        f"Every card above is {STAKED_LABEL} only — the tiers that carry money. "
        f"Including the paper {PAPER_TIER_LABEL} tier there are {total_n:,} tips worth "
        f"{sum(t['pnl'] for t in tracks.values()):+.2f}u, but a paper tier is not a result and "
        f"is kept out of every P/L headline on this page.")

    # per-track, per-tier P/L. One chart, one scale, n in every tooltip.
    rows = [{"track_tier": f"{tr} · {tier_label(ti)}", "pnl": v["pnl"], "n": v["n"], "roi": v["roi"]}
            for tr, t in tracks.items() for ti, v in t["by_tier"].items()]
    if rows:
        d = pd.DataFrame(rows).sort_values("pnl")
        left, right = st.columns([3, 2])
        with left:
            st.altair_chart(ui.pnl_bars(d, "track_tier", "pnl", "n",
                                        title="P/L by track and tier (units, flat 1u)"), width="stretch")
        with right:
            ui.eyebrow("Does a bigger gap mean better outcomes?")
            st.caption(
                f"The tier is only the size of the gap between our probability and the price: "
                f"{tier_label('SNIPER')} is the largest gap, {tier_label('VALUABLE')} the smallest. "
                "Read the bars against the idea that a bigger gap wins more. Measured across the whole "
                "ledger the ladder carries no information: the model is overconfident by +13.64pp "
                "(claimed 0.5430, realised 0.4066, n=792), and the overconfidence is as large on "
                f"never-staked {tier_label('VALUABLE')} tips (+10.78pp) as on staked bets (+16.80pp). "
                "A tier that does not separate outcomes is a label, not a signal.")
            ui.compact_table(d.rename(columns={"track_tier": "track · tier"}), V.tier_table_config())

    # league detail, with small samples flagged rather than hidden
    ui.section_header("Where the money went, by league")
    # Staked tiers only — this section is titled "where the money went", and VALUABLE is paper.
    bl = E["by_league"]
    if bl.empty:
        ui.empty_state("NO_DATA", "No settled bets grouped by league yet.")
    else:
        bl = bl.copy()
        bl["label"] = bl["league"] + bl["reliable"].map({True: "", False: "  ⚠ thin"})
        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(ui.pnl_bars(bl, "label", "pnl", "n", h=26 * len(bl) + 60,
                                        title="P/L by league (units)"), width="stretch")
        with c2:
            # EXCESS OVER EACH LEAGUE'S OWN BREAK-EVEN, not a hit rate against an estate-wide mean.
            # A 45% hit rate is excellent at 2.40 and a disaster at 1.70.
            st.altair_chart(ui.pnl_bars(bl, "label", "excess_hit", "n", h=26 * len(bl) + 60,
                                        title="Hit rate minus that league's own break-even"), width="stretch")
        st.caption(f"Staked tiers only ({STAKED_LABEL}). Including the paper {PAPER_TIER_LABEL} tier the "
                   "same chart reads -123.16u across 1,350 tips instead of -35.66u across 561 — "
                   "three and a half times larger, and none of the difference is money.")
        thin = int((~bl["reliable"]).sum())
        if thin:
            st.markdown(ui.status_chip("SMALL_SAMPLE", label=f"{thin} THIN LEAGUE(S) · n < 30")
                        + " :gray[shown rather than hidden, because a missing league reads as 'we don't bet "
                          "there' when the truth is 'we have not measured it yet'. Do not act on a thin row.]")

# ═══ RESEARCH ═════════════════════════════════════════════════════════════════════════════════════
ui.section_header("Research", "Pro and V11 produce evidence, never bets. What is currently known to be invalid is "
                  "listed so it stops circulating — a conclusion nobody retracts keeps getting quoted.")
dead = [r for r in E["validity"] if str(r.get("status", "")).upper() == "INVALIDATED"]
if not dead:
    st.markdown(ui.status_chip("HEALTHY", label="NO INVALIDATED RESULTS LISTED"))
else:
    ui.eyebrow(f"Results known to be invalid · {len(dead)}")
    for r in dead:
        with st.expander(f"{r.get('research_id')} — {r.get('repo')} · {r.get('script', '')}"):
            st.markdown(ui.status_chip("FAIL", label="INVALIDATED") + f" &nbsp;**Why it is void:** {r.get('reason', '—')}")
            if r.get("example_invalid_numbers"):
                st.markdown(f"**Do not quote:** {r['example_invalid_numbers']}")
            if r.get("replacement"):
                st.markdown(f"**Use instead:** {r['replacement']}")

# ═══ DATA FEEDS ═══════════════════════════════════════════════════════════════════════════════════
ui.section_header("Data feeds", "Every age below is derived from the newest RECORD inside the file, or from a "
                  "timestamp the writer recorded — never from the file's modification time, which `git checkout` "
                  "resets to now on every CI run. This estate has been misled by an mtime-based age three separate "
                  "times.")
r1 = st.columns(2)
r2 = st.columns(2)

with r1[0].container(border=True):
    V.subsystem_header("v9", REPOS["v9"]["label"], repo_path("v9"))
    ui.freshness_chip(hv["predictions"]["freshness"], f"predictions · {hv['predictions']['rows']} rows")
    ui.freshness_chip(hv["ledger"]["freshness"], f"ledger · {hv['ledger']['settled']:,} settled")
    rt = hv.get("retrain") or {}
    if rt.get("latest_run"):
        st.caption(f"🔁 last retrain {rt['latest_run']}")
    ssh = hv.get("shadow_settlement") or {}
    if ssh:
        ui.freshness_chip(ssh["freshness"],
                          f"shadow log · {ssh.get('settled_rows', 0):,}/{ssh.get('shadow_rows', 0):,} settled")
        st.caption(f"close {ssh.get('valid_closing_price_pct')}% · CLV {ssh.get('clv_pct')}% "
                   f"of settled · coverage only, not evidence of edge")

with r1[1].container(border=True):
    V.subsystem_header("pro", REPOS["pro"]["label"], repo_path("pro"))
    if not E["pro_available"]:
        ui.empty_state("NOT_CONNECTED", E["pro_absent"])
    else:
        cs = E["pro_stores"]
        if cs.get("available"):
            parts = [(k, v) for k, v in cs["stores"].items() if v.get("partitioned")]
            fresh = sum(1 for _, v in parts if v["freshness"].state == "CURRENT")
            st.caption(f"📦 {cs['season']} · {len(cs['stores'])} stores · {fresh}/{len(parts)} current")
            for name, v in sorted(parts, key=lambda kv: kv[1]["freshness"].state)[:3]:
                ui.freshness_chip(v["freshness"], name)
        ha = E["pro_health"]
        for fname, meta in list((ha.get("files") or {}).items())[:2]:
            ui.freshness_chip(meta["freshness"], fname.replace(".json", ""))

with r2[0].container(border=True):
    V.subsystem_header("v11", REPOS["v11"]["label"], repo_path("v11"))
    if not E["v11_available"]:
        ui.empty_state("NOT_CONNECTED", E["v11_absent"])
    else:
        hv11 = E["v11_health"]
        ui.freshness_chip(hv11["freshness"], f"shadow log · {hv11['shadow_rows']:,} rows")
        rh = hv11.get("research_health") or {}
        raw = rh.get("raw") or {}
        if raw.get("snapshots_rows"):
            st.caption(f"📈 {raw['snapshots_rows']:,} market snapshots")

with r2[1].container(border=True):
    V.subsystem_header("fantasy", "Fantasy (inside v9)", None)
    fh = E["fantasy"]
    if not fh.get("available"):
        ui.empty_state("NO_DATA", fh.get("why", "not run"))
    else:
        ui.freshness_chip(fh["run"], "last run")
        for name, f in fh["feeds"].items():
            ui.freshness_chip(f, name)
