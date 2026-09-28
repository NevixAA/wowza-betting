"""Overview — the whole estate on one screen.

WHY THIS PAGE EXISTS. The dashboard had nine pages and every one of them was a v9 page. Exactly
one reached Pro, and nothing at all reached v11. So the only way to answer "is Wowza healthy"
was to open three repos in three terminals. This page answers it in one screen, and it is honest
about the parts it cannot reach rather than leaving them blank.

WHAT IT REFUSES TO DO. It does not add v9's P/L to Pro's research numbers or to Fantasy's points.
Those are three different units measuring three different things, and a single "Wowza score"
would be a fabrication. Each repo keeps its own scale and its own state label.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_data import fantasy, pro, ui, v9, v11
from dashboard_data.core import REPOS, repo_path

st.set_page_config(page_title="Wowza | Overview", page_icon="🏠", layout="wide")
st.title("🏠 Wowza — estate overview")
st.caption("Three repositories, three jobs, three scales. Nothing on this page is summed across "
           "them, because units in betting, research and Fantasy are not the same unit.")

# ── which repos are reachable ─────────────────────────────────────────────────────────────────
cols = st.columns(3)
_ROLE = {
    "v9": ("LIVE", "Production. Sends tips, stakes paper, learns every week."),
    "pro": ("RESEARCH", "Canonical evidence, validation, Bet Builder. Never writes into v9."),
    "v11": ("SHADOW / RESEARCH ONLY", "Market-first shadow. Logs decisions, places nothing."),
}
for c, key in zip(cols, ("v9", "pro", "v11")):
    with c:
        state, blurb = _ROLE[key]
        found = repo_path(key)
        st.markdown(
            f"<div style='border-top:3px solid {ui.repo_colour(key)};padding-top:8px'>"
            f"<b style='font-size:1.05rem'>{REPOS[key]['label']}</b></div>",
            unsafe_allow_html=True)
        ui.state_banner(state, blurb)
        if found is None:
            st.caption("⚪ not checked out beside v9")
        else:
            st.caption(f"📁 `{found.name}`")

st.divider()

# ── v9: the only track carrying money-shaped numbers ──────────────────────────────────────────
st.subheader("V9 — live betting")
perf = v9.performance()
if not perf.get("available"):
    st.warning(perf.get("why", "no settled bets"))
else:
    tracks = {k: val for k, val in perf["by_track"].items() if k != "unknown"}
    k = st.columns(len(tracks) + 1)
    total_n = sum(t["n"] for t in tracks.values())
    staked_n = sum(t["staked_only"]["n"] for t in tracks.values())
    staked_pnl = sum(t["staked_only"]["pnl"] for t in tracks.values())
    with k[0]:
        # THE HEADLINE IS THE STAKED FIGURE, NOT THE ALL-TIPS FIGURE. VALUABLE is a paper tier;
        # counting it into the number a reader reads as "how we are doing" inflates the bet count
        # by roughly 3x and mixes two different decisions into one line.
        ui.kpi("Staked P/L", ui.units(staked_pnl),
               f"{staked_n:,} staked bets · SNIPER + MARKSMAN only",
               tone=ui.polarity()[0] if staked_pnl >= 0 else ui.polarity()[1])
    for c, (name, t) in zip(k[1:], sorted(tracks.items())):
        with c:
            ui.kpi(name.replace("_", "-"), ui.units(t["pnl"]),
                   f"{t['n']:,} tips · hit {ui.pct(t['hit'], None)} · ROI {ui.pct(t['roi'], None)}",
                   tone=ui.polarity()[0] if t["pnl"] >= 0 else ui.polarity()[1])

    st.caption(f"All tips including the paper VALUABLE tier: {total_n:,}. "
               "The two are kept apart deliberately — a paper tier is not a result.")

    # per-track, per-tier P/L. One chart, one scale, n in every tooltip.
    rows = [{"track_tier": f"{tr} · {ti}", "pnl": v["pnl"], "n": v["n"], "roi": v["roi"]}
            for tr, t in tracks.items() for ti, v in t["by_tier"].items()]
    if rows:
        d = pd.DataFrame(rows).sort_values("pnl")
        left, right = st.columns([3, 2])
        with left:
            st.altair_chart(ui.pnl_bars(d, "track_tier", "pnl", "n",
                                        title="P/L by track and tier (units, flat 1u)"),
                            width='stretch')
        with right:
            st.markdown("**Does the tier ladder order the outcomes?**")
            st.caption(
                "SNIPER is supposed to be the strongest tier and VALUABLE the weakest. Read the "
                "bars against that claim. Measured across the whole ledger the ladder carries no "
                "information: the model is overconfident by +13.64pp (claimed 0.5430, realised "
                "0.4066, n=792), and the gap is as large on never-staked VALUABLE (+10.78pp) as "
                "on staked bets (+16.80pp). A tier that does not separate outcomes is a label, "
                "not a signal.")
            ui.table(d.assign(roi=d["roi"].map(lambda x: ui.pct(x, None)),
                              pnl=d["pnl"].map(ui.units))
                      .rename(columns={"track_tier": "track · tier"}))

st.divider()

# ── league detail, with small samples flagged rather than hidden ───────────────────────────────
st.subheader("Where the money went, by league")
bl = v9.by_league(days=None)
if bl.empty:
    st.caption("No settled bets grouped by league yet.")
else:
    bl = bl.copy()
    bl["label"] = bl["league"] + bl["reliable"].map({True: "", False: "  ⚠ thin"})
    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(ui.pnl_bars(bl, "label", "pnl", "n", h=26 * len(bl) + 60,
                                    title="P/L by league (units)"), width='stretch')
    with c2:
        st.altair_chart(ui.rate_bars(bl, "label", "hit", "n", ref=float(bl["hit"].mean()),
                                     h=26 * len(bl) + 60,
                                     title="Hit rate by league (dashed = estate mean)"),
                        width='stretch')
    thin = int((~bl["reliable"]).sum())
    if thin:
        st.caption(f"⚠ {thin} league(s) marked thin — fewer than 30 settled bets. They are shown "
                   "rather than hidden, because a missing league reads as 'we don't bet there' "
                   "when the truth is 'we have not measured it yet'. Do not act on a thin row.")

st.divider()

# ── health strip across all four surfaces ─────────────────────────────────────────────────────
st.subheader("Feed health")
st.caption("Every age below is derived from the newest RECORD inside the file, or from a "
           "timestamp the writer recorded — never from the file's modification time, which "
           "`git checkout` resets to now on every CI run. This estate has been misled by an "
           "mtime-based age three separate times.")

h1, h2, h3, h4 = st.columns(4)
with h1:
    st.markdown(f"**{REPOS['v9']['label']}**")
    hv = v9.health()
    ui.freshness_chip(hv["predictions"]["freshness"], f"predictions · {hv['predictions']['rows']} rows")
    ui.freshness_chip(hv["ledger"]["freshness"],
                      f"ledger · {hv['ledger']['settled']:,} settled")
    rt = hv.get("retrain") or {}
    if rt.get("latest_run"):
        st.caption(f"🔁 last retrain {rt['latest_run']}")

with h2:
    st.markdown(f"**{REPOS['pro']['label']}**")
    if not pro.available():
        ui.absent(pro.absent_reason())
    else:
        cs = pro.canonical_stores()
        if cs.get("available"):
            parts = [(k, v) for k, v in cs["stores"].items() if v.get("partitioned")]
            fresh = sum(1 for _, v in parts if v["freshness"].state == "CURRENT")
            st.caption(f"📦 {cs['season']} · {len(cs['stores'])} stores · "
                       f"{fresh}/{len(parts)} current")
            for name, v in sorted(parts, key=lambda kv: kv[1]["freshness"].state)[:3]:
                ui.freshness_chip(v["freshness"], name)
        ha = pro.health_artifacts()
        for fname, meta in list((ha.get("files") or {}).items())[:2]:
            ui.freshness_chip(meta["freshness"], fname.replace(".json", ""))

with h3:
    st.markdown(f"**{REPOS['v11']['label']}**")
    if not v11.available():
        ui.absent(v11.absent_reason())
    else:
        hv11 = v11.health()
        ui.freshness_chip(hv11["freshness"], f"shadow log · {hv11['shadow_rows']:,} rows")
        rh = hv11.get("research_health") or {}
        raw = rh.get("raw") or {}
        if raw.get("snapshots_rows"):
            st.caption(f"📈 {raw['snapshots_rows']:,} market snapshots")

with h4:
    st.markdown("**Fantasy** (inside v9)")
    fh = fantasy.health()
    if not fh.get("available"):
        st.caption(fh.get("why", "not run"))
    else:
        ui.freshness_chip(fh["run"], "last run")
        for name, f in fh["feeds"].items():
            ui.freshness_chip(f, name)

st.divider()

# ── what is currently known to be invalid ─────────────────────────────────────────────────────
rv = pro.research_validity()
dead = [r for r in rv if str(r.get("status", "")).upper() == "INVALIDATED"]
if dead:
    st.subheader("⛔ Results known to be invalid")
    st.caption("These numbers were produced by code that has since been shown to be wrong. They "
               "are listed here so they stop circulating — a conclusion nobody retracts keeps "
               "getting quoted.")
    for r in dead:
        with st.expander(f"{r.get('research_id')} — {r.get('repo')} · {r.get('script', '')}"):
            st.markdown(f"**Why it is void:** {r.get('reason', '—')}")
            if r.get("example_invalid_numbers"):
                st.markdown(f"**Do not quote:** {r['example_invalid_numbers']}")
            if r.get("replacement"):
                st.markdown(f"**Use instead:** {r['replacement']}")
