"""Non-regression for the 2026-10-25 Fantasy visual redesign (stage 2a).

The redesign re-homed every Fantasy value into task tabs (Overview / Players / Transfers / Captaincy /
Fixtures / Squad / Model lab) and restyled it as cards. It must not change a single number. Each test
renders the page headlessly (AppTest, absolute path) and compares what is displayed with (a) a FROZEN
copy of the pre-redesign selection logic and (b) the canonical data functions the page reads.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = next((ROOT / "pages").glob("13_*_Fantasy.py"))
OUT = ROOT / "output"
STAT = re.compile(r"<span class='wz-stat-l'>(.*?)</span><span class='wz-stat-v'>(.*?)</span>")
TITLE = re.compile(r"<div class='wz-title'[^>]*>(.*?)</div>")

needs_artifacts = pytest.mark.skipif(
    not (OUT / "fantasy_v2_projections.csv").exists() or not (OUT / "fantasy_v2_summary.json").exists(),
    reason="Fantasy V2 artifacts not built")


@pytest.fixture(scope="module")
def at():
    from streamlit.testing.v1 import AppTest
    mp = pytest.MonkeyPatch()
    mp.setenv("WOWZA_TIER_OVERRIDE", "admin")
    try:
        a = AppTest.from_file(str(PAGE.resolve()), default_timeout=180).run()
    finally:
        mp.undo()
    assert not a.exception, [e.value for e in a.exception]
    return a


@pytest.fixture(scope="module")
def data():
    from dashboard_data import fantasy as fdata
    return (fdata.v2_projections(), fdata.v2_summary(), fdata.v2_benchmark(), fdata.v2_transfers(),
            fdata.v2_fixture_matrix())


def _stats(at, label):
    return [html.unescape(v) for m in at.markdown for k, v in STAT.findall(m.value) if html.unescape(k) == label]


def _titles(at):
    return [html.unescape(t) for m in at.markdown for t in TITLE.findall(m.value)]


def _metrics(at, label=None, prefix=None):
    return [(m.label, m.value) for m in at.metric
            if (label is None or m.label == label) and (prefix is None or m.label.startswith(prefix))]


# ── pure helpers ─────────────────────────────────────────────────────────────────────────────────
def test_fdr_helpers():
    from dashboard_data import fantasy_view as fv
    assert fv.fdr_value("4") == 4.0 and fv.fdr_value("2+4") == 3.0 and fv.fdr_value("") is None
    assert fv.avg_fdr("4 | 2 | 3 | 4 | 3") == pytest.approx(3.2)
    # one semantic everywhere: easy green, hard red; ease is the mirror of FDR
    assert [fv.fdr_tone(v) for v in (1, 2, 3, 4, 5)] == ["POSITIVE", "POSITIVE", "NEUTRAL", "WARNING", "NEGATIVE"]
    assert [fv.ease_tone(v) for v in (5, 1)] == ["POSITIVE", "NEGATIVE"]
    chips = fv.fdr_chips("LIV(A) | HUL(H)", "4 | 2")
    assert "LIV(A) 4" in chips and "HUL(H) 2" in chips and ":green-badge" in chips
    assert "LOW RISK" in fv.risk_chip("Low") and "UNAVAILABLE" in fv.risk_chip("Unavailable")


def test_presentation_files_invent_no_colours_no_css_no_dividers():
    for p in (PAGE, ROOT / "dashboard_data" / "fantasy_view.py", ROOT / "dashboard_fantasy_tools.py"):
        src = p.read_text(encoding="utf-8")
        assert not re.search(r"#[0-9A-Fa-f]{6}\b", src), f"{p.name} hardcodes a hex colour"
        assert "<style" not in src, f"{p.name} carries its own CSS"
        assert "st.divider()" not in src and 'st.markdown("---")' not in src, f"{p.name} uses dividers"


def test_page_never_reads_player_history_at_render():
    """The page crashed the ~1 GB app by reading player_history at render. Only the on-demand club
    squads (behind a button) may touch it, through player_model.fantasy.build_squads."""
    for p in (PAGE, ROOT / "dashboard_data" / "fantasy_view.py", ROOT / "dashboard_fantasy_tools.py"):
        assert "read_parquet" not in p.read_text(encoding="utf-8"), p.name
    for p in (PAGE, ROOT / "dashboard_data" / "fantasy_view.py"):
        assert "player_history" not in p.read_text(encoding="utf-8"), p.name
    tools = (ROOT / "dashboard_fantasy_tools.py").read_text(encoding="utf-8")
    assert 'st.button("Load club squads"' in tools and "fantasy_set_pieces.csv" in tools


# ── rendered page ────────────────────────────────────────────────────────────────────────────────
@needs_artifacts
def test_task_tabs(at):
    labels = [t.label for t in at.tabs]
    for want in ("Overview", "Players", "Transfers", "Captaincy", "Fixtures", "Squad", "Model lab"):
        assert want in labels
    # model/research detail lives in the Model lab, not before the decisions
    for want in ("Benchmark", "Gameweek scoreboard", "Calibration & method", "Decomposition"):
        assert want in labels
    assert labels.index("Overview") < labels.index("Model lab")


@needs_artifacts
def test_captaincy_cards_match_summary(at, data):
    proj, summ, *_ = data
    titles, evs, p10 = _titles(at), _stats(at, "Captain EV"), _stats(at, "P(10+)")
    for key in ("SAFE", "BALANCED", "HIGH_UPSIDE"):
        for src in ("captaincy", "captaincy_shadow"):
            c = (summ.get(src) or {}).get(key)
            if not c:
                continue
            if src == "captaincy":                       # the headline set is always rendered
                assert c["web_name"] in titles
                assert f"{c['captain_ev']:.1f}" in evs
                assert f"{c['p10']:.0%}" in p10
    xp = [v for _, v in _metrics(at, "xPts")]
    for key in ("SAFE", "BALANCED", "HIGH_UPSIDE"):
        c = (summ.get("captaincy") or {}).get(key)
        if c:
            assert f"{c['final_xpts']:.2f}" in xp


def _old_top_cards(proj, pos="All"):
    """Frozen pre-redesign 'Top recommendations' rule (pages/13 at 50af39e4)."""
    avail = proj[~proj["risk"].isin(["Unavailable"])]
    cards = avail if pos == "All" else avail[avail["position"] == pos]
    return cards.sort_values("final_xpts", ascending=False).head(6)


@needs_artifacts
def test_top_projected_unchanged(at, data):
    from dashboard_data import fantasy_view as fv
    proj = data[0]
    old = _old_top_cards(proj)
    assert old[["element"]].equals(fv.top_projected(proj, "All", 6)[["element"]])
    titles, fx = _titles(at), [v for _, v in _metrics(at, "Final xPts")]
    for r in old.itertuples():
        assert r.web_name in titles and f"{r.final_xpts:.2f}" in fx


def _old_planner(proj, bank=0.5, horizon="5 GW"):
    """Frozen pre-redesign transfer-planner rule (defaults: most-owned OUT, £0.5m, 5 GW)."""
    owned = proj.sort_values("ownership", ascending=False)
    out_name = owned["player_name"].tolist()[0]
    hcol = {"next GW": "final_xpts", "3 GW": "xpts_3gw", "5 GW": "xpts_5gw"}[horizon]
    o = proj[proj["player_name"] == out_name].iloc[0]
    cand = proj[(proj["position"] == o["position"]) & (proj["element"] != o["element"])
                & (proj["price"] <= o["price"] + bank) & (proj["risk"] != "Unavailable")].copy()
    cand["gain"] = cand[hcol] - o[hcol]
    return o, cand.sort_values("gain", ascending=False).head(5), hcol


@needs_artifacts
def test_transfer_planner_unchanged(at, data):
    from dashboard_data import fantasy_view as fv
    proj = data[0]
    o, cand, hcol = _old_planner(proj)
    o2, cand2 = fv.transfer_candidates(proj, o["player_name"], 0.5, "5 GW")
    assert cand["element"].tolist() == cand2["element"].tolist()
    assert cand["gain"].round(6).tolist() == cand2["gain"].round(6).tolist()
    shown = _metrics(at, prefix="Expected gain · 5 GW")
    assert [v for _, v in shown] == [f"{g:+.2f}" for g in cand["gain"]]
    assert ("5 GW projection", f"{o[hcol]:.2f}") in _metrics(at)
    g = fv.transfer_gains(o, cand.iloc[0])
    assert g["gain_5gw"] == pytest.approx(float(cand.iloc[0]["gain"]))


@needs_artifacts
def test_benchmark_and_explorer_tables_unchanged(at, data):
    from dashboard_data import fantasy as fdata
    proj, _, bench, trans, _ = data
    frames = [d.value.reset_index(drop=True) for d in at.dataframe]
    bt = fdata.v2_benchmark_table(bench, "ACTIVE", False)
    if not bt.empty:
        assert any(bt.drop(columns=["arm"]).reset_index(drop=True).equals(f) for f in frames)
    show = ["web_name", "team", "position", "price", "final_xpts", "fpl_ep_next", "wowza_residual",
            "p_start", "exp_minutes", "goal_prob", "assist_prob", "cs_prob", "median", "p90",
            "xpts_3gw", "xpts_5gw", "value_5gw_per_m", "ownership", "next_fixture", "risk", "confidence"]
    exp = proj.sort_values("final_xpts", ascending=False)[show].reset_index(drop=True)
    assert any(exp.equals(f) for f in frames)
    if not trans.empty:
        assert any(f.shape[0] == min(40, len(trans)) and "net_gain_5gw_with_hit" in f.columns for f in frames)


def _old_fixture_agg(fxm):
    """Frozen pre-redesign fixture-matrix aggregation."""
    f = fxm.copy()
    f["GW"] = "GW" + f["gw"].astype(str)
    return f.groupby(["team", "GW"], as_index=False).agg(
        label=("label", lambda s: " + ".join(s)), attack_ease=("attack_ease", "mean"),
        defence_ease=("defence_ease", "mean"), fdr=("fdr", "mean"),
        lam_for=("lam_for", "sum"), cs_prob=("cs_prob", "mean"))


@needs_artifacts
def test_fixture_matrix_unchanged(data):
    from dashboard_data import fantasy_view as fv
    fxm = data[4]
    if fxm.empty:
        pytest.skip("no fixtures")
    old, new = _old_fixture_agg(fxm), fv.fixture_agg(fxm)
    assert old.equals(new)
    old_order = old.groupby("team")["fdr"].mean().sort_values(ascending=True).index.tolist()
    assert fv.team_order(new, "fdr") == old_order
    easy, hard = fv.fixture_swing(new, 5)
    assert easy["avg_fdr"].is_monotonic_increasing and hard["avg_fdr"].is_monotonic_decreasing
    import dashboard_ui as ui
    t = ui.tokens()
    for lens, key in fv.LENS.items():
        spec = fv.fixture_heatmap_spec(new, lens)
        vals = pd.DataFrame(spec["data"]["values"])
        assert len(vals) == len(new) and (vals[key].values == new[key].values).all()
        assert spec["encoding"]["y"]["sort"] == fv.team_order(new, key)
        rng = spec["layer"][0]["encoding"]["color"]["scale"]["range"]
        # one semantic: the easy end is green in every lens (FDR 1, ease 5)
        easy_end = rng[0] if key == "fdr" else rng[-1]
        assert easy_end == t["POSITIVE"]
    assert all(r for r in fv.team_runs(new).values())


@needs_artifacts
def test_interactive_views_render(at, data):
    """Shadow captaincy, every fixture lens, the minutes-calibration scoreboard and a different
    planner horizon all render without an exception."""
    summ = data[1]
    next(r for r in at.radio if r.label == "Mean used").set_value("Shadow (FPL + residual)")
    at.radio(key="sb_view").set_value("Minutes calibration")
    next(r for r in at.radio if r.label == "Horizon").set_value("next GW")
    for lens in ("Attack", "Defence", "Official FDR"):
        at.radio(key="fx_lens").set_value(lens).run()
        assert not at.exception, [e.value for e in at.exception]
    shadow = (summ.get("captaincy_shadow") or {}).get("BALANCED")
    if shadow:
        assert f"{shadow['captain_ev']:.1f}" in _stats(at, "Captain EV")
    assert _metrics(at, prefix="Expected gain · next GW")


@needs_artifacts
def test_recommendation_rules(data):
    from dashboard_data import fantasy_view as fv
    proj, summ, *_ = data
    recs = fv.recommendations(proj, summ)
    avail = proj[~proj["risk"].isin(["Unavailable"])]
    pool = proj[(proj["p_start"] >= 0.6) & ~proj["risk"].isin(["Unavailable", "Doubtful"])]
    assert recs["best_overall"]["element"] == avail.sort_values("final_xpts", ascending=False).iloc[0]["element"]
    assert recs["best_value"]["element"] == pool.sort_values("value_5gw_per_m", ascending=False).iloc[0]["element"]
    if "best_differential" in recs:
        assert recs["best_differential"]["ownership"] < 10
    if (summ.get("captaincy") or {}).get("BALANCED"):
        assert recs["best_captain"] is summ["captaincy"]["BALANCED"]
    assert recs["best_defender"]["position"] == "DEF"
