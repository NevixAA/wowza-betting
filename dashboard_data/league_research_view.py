"""Presentation for pages/18_League_Research.py (League Repair Lab), visual redesign 2026-10-25 (stage 2b).

PRESENTATION ONLY. Every number shown here is handed in from the page's canonical sources:
  * league_weekly_eval (through dashboard_data/league_research.py) - metric_table, history grid, overview;
  * output/league_market_health.json and output/weekly_league_report.json (written by CI);
  * config/track_registry.json (verbatim - regimes, production states, engines).

Nothing here ranks, grades, aggregates or changes a state. The two frame builders below only RESHAPE:
  * `heatmap_frame` takes the panel rows of one league x market (one row per gameweek x rank, exactly the
    rows league_weekly_eval.weekly_grid draws) and lays them out long for a chart; its tooltip text is
    league_weekly_eval's own cell text, so the heatmap and the history table cannot disagree;
  * `engine_frame` / `rank_strip_rows` pick rows and columns out of league_weekly_eval.league_table.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import dashboard_ui as ui
from src.research import league_weekly_eval as EV

REGIMES = ("REPAIR", "LEARNING", "PROTECTED")
RANKS = [f"#{i}" for i in range(1, 6)]
STRIP = (("RANK1", "TOP 1"), ("TOP2", "TOP 2"), ("TOP3", "TOP 3"), ("TOP5", "TOP 5"))
#: league_weekly_eval subsets compared in "current production vs market-first shadow"
ENGINE_SUBSETS = (("PRODUCTION_SELECTED", "Production selected"), ("TOP5", "Shadow top-5"),
                  ("QUALIFIED", "Shadow qualified"))
PROB_SOURCES = (("model", "Production model"), ("decision", "Market-first decision"), ("market", "Market (de-vigged)"))
#: panel health older than this is STALE (the system-health rule for the weekly panel: > 48 h = AMBER)
HEALTH_MAX_AGE_H = 48


# ── small text helpers (same output as the pre-redesign page, missing as an em dash) ─────────────────
def pct0(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{100 * float(x):.0f}%"


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(v) else v


def age_hours(ts) -> float | None:
    t = pd.to_datetime(ts, errors="coerce", utc=True)
    if t is None or pd.isna(t):
        return None
    return (pd.Timestamp.now(tz="UTC") - t).total_seconds() / 3600


def freshness(health: dict) -> str:
    """The page's ONE freshness statement: the panel health file's recorded generated_at."""
    gen = (health or {}).get("generated_at")
    h = age_hours(gen)
    if h is None:
        return ui.freshness_badge("UNKNOWN", "panel health has no timestamp")
    return ui.freshness_badge("FRESH" if h <= HEALTH_MAX_AGE_H else "STALE",
                              f"panel health {h:.1f} h old · {str(gen)[:16].replace('T', ' ')} UTC")


def regime_chip(regime: str) -> str:
    """PROTECTED is governance (blue shield), never an error or warning style."""
    if regime == "PROTECTED":
        return ui.status_chip("PROTECTED", label="PROTECTED · SHADOW ONLY")
    return ui.status_chip(regime)


# ── overview strip ─────────────────────────────────────────────────────────────────────────────
def regime_counts(order: list[tuple[str, str]]) -> dict:
    out = {r: 0 for r in REGIMES}
    for _, rg in order:
        out[rg] = out.get(rg, 0) + 1
    return out


def overview_strip(health: dict, order: list[tuple[str, str]]) -> None:
    h = health or {}
    rc = regime_counts(order)
    tracked = h.get("tracked_leagues", "—")
    with_rows = h.get("leagues_with_current_week_data", "—")
    a, b, c = st.columns(3)
    ui.metric_card("Tracked leagues", f"{tracked}", title="Coverage", container=a,
                   help=f"{with_rows} with rows this gameweek",
                   supporting=[("Current gameweek coverage", f"{with_rows}/{tracked} leagues"),
                               ("Missing leagues", f"{len(h.get('missing_leagues') or [])}"),
                               ("Candidates (this week)",
                                f"{h.get('actual_candidate_rows', '—')}/{h.get('expected_candidate_rows', '—')}")])
    ui.metric_card("Leagues in REPAIR", f"{rc['REPAIR']}", title="Research regimes", container=b,
                   supporting=[("LEARNING", f"{rc['LEARNING']}"), ("PROTECTED", f"{rc['PROTECTED']}")],
                   status=" ".join(regime_chip(r) for r in REGIMES if rc.get(r)))
    ptm = h.get("protected_tracks_modified")
    ui.metric_card("CLV coverage", pct0(h.get("clv_coverage")), title="Evidence", container=c,
                   help="certified close (≤30 min, pre-kickoff)",
                   supporting=[("Panel rows", f"{h.get('panel_rows', '—')}"),
                               ("Settled", f"{h.get('settled_rows', '—')}"), ("Pending", f"{h.get('pending_rows', '—')}"),
                               ("Protected tracks modified", "NO" if ptm is False else "YES" if ptm else "—")],
                   evidence=ui.evidence_badge("TRUE_FORWARD", None, state="nothing replayed"))


def attention(health: dict) -> list[dict]:
    h = health or {}
    items = []
    if h.get("protected_tracks_modified"):
        items.append({"status": "RED", "title": "protected_tracks_modified = true",
                      "detail": "production moved without re-approval", "source": "src/research_lock"})
    if h.get("missing_leagues"):
        items.append({"status": "AMBER", "title": "Fixtures but no panel rows this gameweek",
                      "detail": ", ".join(h["missing_leagues"]), "source": "league_market_health.json"})
    return items


# ── league cards ───────────────────────────────────────────────────────────────────────────────
def league_cards(ov: pd.DataFrame, per_row: int = 3) -> None:
    """One card per league, grouped REPAIR -> LEARNING -> PROTECTED (ov is already in that order)."""
    if ov is None or ov.empty:
        ui.empty_state("NO_DATA", "No tracked leagues in config/track_registry.json.")
        return
    for regime in REGIMES:
        g = ov[ov["regime"] == regime]
        if g.empty:
            continue
        ui.eyebrow(f"{regime} · {len(g)} league{'s' if len(g) != 1 else ''}")
        rows = list(g.itertuples(index=False))
        for i in range(0, len(rows), per_row):
            for col, r in zip(st.columns(per_row), rows[i:i + per_row]):
                cur = r.current_state or "—"
                ui.metric_card("Observations", f"{int(r.observations)}", title=r.league, container=col,
                               supporting=[("Gameweeks", f"{int(r.gameweeks)}"), ("Settled", f"{int(r.settled)}"),
                                           ("CLV coverage", pct0(r.clv_coverage)), ("This gameweek", cur)],
                               status=regime_chip(regime))


def overview_table_config() -> dict:
    return {"league": ui.text_col("League"), "regime": ui.text_col("Regime"),
            "observations": ui.num_col("Observations", fmt="%d"), "gameweeks": ui.num_col("Gameweeks", fmt="%d"),
            "settled": ui.num_col("Settled", fmt="%d"),
            "clv_coverage": ui.pct_col("CLV coverage", help="certified close / priced settled rows"),
            "current_week": ui.text_col("Current gameweek"), "current_state": ui.text_col("State")}


# ── league profile ─────────────────────────────────────────────────────────────────────────────
def engines(league: str, entries: list[dict]) -> dict:
    """The registry's engines for the league, verbatim (production / shadow selection, model versions)."""
    es = [e for e in entries if e.get("league") == league]
    sel = [e.get("selection_engine") or {} for e in es]
    return {"production": sorted({str(s.get("production")) for s in sel if s.get("production")}),
            "shadow": sorted({str(s.get("shadow")) for s in sel if s.get("shadow")}),
            "models": sorted({str((e.get("prediction_engine") or {}).get("model_version"))
                              for e in es if (e.get("prediction_engine") or {}).get("model_version")})}


def production_chips(tracks: pd.DataFrame) -> str:
    """One chip per market x side with its production state (registry verbatim)."""
    if tracks is None or tracks.empty:
        return ui.status_chip("NO_DATA", label="NO TRACKS")
    out = []
    for r in tracks.itertuples(index=False):
        icon, word, tone = ui.status_meta(r.production_state)
        out.append(ui.status_chip(r.production_state, label=f"{r.market} {r.side} · {word}"))
    return " ".join(out)


def profile_card(league: str, regime: str, hdr: dict, ov_row: dict, cur: dict, tracks: pd.DataFrame,
                 eng: dict) -> None:
    with st.container(border=True):
        ui.eyebrow("League profile")
        st.markdown(f"<div class='wz-title'>{_esc(league)}</div>", unsafe_allow_html=True)
        st.markdown(regime_chip(regime) + " " + ui.status_chip("RESEARCH", label="RESEARCH ONLY · NOT BETS"))
        ui.stat_line([("Research regime", regime), ("Observations", f"{hdr['observations']}"),
                      ("Gameweeks", f"{hdr['gameweeks']}"),
                      ("CLV coverage", pct0((ov_row or {}).get("clv_coverage"))),
                      ("This gameweek", cur.get("state", "—") or "—")])
        ui.stat_line([("Current engine (production)", ", ".join(eng["production"]) or "—"),
                      ("Shadow engine", ", ".join(eng["shadow"]) or "—"),
                      ("Prediction engine(s) in panel", ", ".join(hdr["prediction_engines"]) or "—"),
                      ("Selection engine(s) in panel", ", ".join(hdr["selection_engines"]) or "—")])
        st.markdown("Production state · " + production_chips(tracks))
        if cur.get("league_week_id"):
            st.caption(f"Current gameweek {cur.get('league_week_id')}")


def _esc(s) -> str:
    import html
    return html.escape(str(s))


def tracks_config() -> dict:
    return {"market": ui.text_col("Market"), "side": ui.text_col("Side"), "model_type": ui.text_col("Track"),
            "research_regime": ui.text_col("Research regime"), "production_state": ui.text_col("Production state"),
            "real_money": st.column_config.CheckboxColumn("Real money"),
            "sent": st.column_config.CheckboxColumn("Sent"),
            "owner_locked": st.column_config.CheckboxColumn("Owner locked")}


# ── current gameweek top-5 ─────────────────────────────────────────────────────────────────────
TOP_NUMERIC = ("ev_lower_bound", "ev_point", "p_model_raw", "p_market_fair", "odds", "flat_pnl", "clv_pct",
               "league_market_rank", "legacy_rank")


def top_frame(records: list[dict]) -> pd.DataFrame:
    """The report's top-5 rows, numbers typed as numbers (the JSON stores them as text). Same values."""
    top = pd.DataFrame(records or [])
    for c in TOP_NUMERIC:
        if c in top.columns:
            top[c] = pd.to_numeric(top[c].replace("", None), errors="coerce")
    return top


def top_config() -> dict:
    return {"league_market_rank": ui.num_col("#", fmt="%d"), "ev_lower_bound": ui.num_col("EV_LB", fmt="%+.4f"),
            "ev_point": ui.num_col("EV", fmt="%+.4f"), "p_model_raw": ui.num_col("p model", fmt="%.4f"),
            "p_market_fair": ui.num_col("p market", fmt="%.4f"), "odds": ui.num_col("Odds", fmt="%.2f"),
            "legacy_rank": ui.num_col("Legacy rank", fmt="%d"), "flat_pnl": ui.num_col("P/L", fmt="%+.2f"),
            "clv_pct": ui.num_col("CLV %", fmt="%+.2f")}


def _yes(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def top5_cards(top: pd.DataFrame, per_row: int = 3) -> None:
    rows = list(top.to_dict("records"))
    for i in range(0, len(rows), per_row):
        for col, r in zip(st.columns(per_row), rows[i:i + per_row]):
            with col.container(border=True):
                # three elements per card (header, number, figures + chips): 20 cards render on every rerun
                ko = str(r.get("kickoff_utc", "") or "")[:16].replace("T", " ")
                rank = (ui.rank_badge(int(r["league_market_rank"])) + " "
                        if pd.notna(r.get("league_market_rank")) else "")
                st.markdown(f"{rank}:gray[{_md(ko)} UTC]{_PARA}<div class='wz-title'>{_esc(r.get('home_team', ''))} v "
                            f"{_esc(r.get('away_team', ''))}</div>", unsafe_allow_html=True)
                st.metric(f"EV lower bound · {r.get('side', '')}", ui.fmt_roi(r.get("ev_lower_bound")),
                          help="market_edge_v2's EV lower bound per unit staked (the panel's ranking score)")
                prod = _yes(r.get("production_selected"))
                chips = " ".join(x for x in (
                    ui.status_chip(r.get("decision_state") or "NO_DATA"),
                    ui.status_chip("LIVE", label="PRODUCTION SELECTED · YES") if prod
                    else ui.status_chip("NO_DATA", label="PRODUCTION SELECTED · NO"),
                    ui.status_chip("UNKNOWN", label=str(r.get("status") or "").upper()) if r.get("status") else "",
                ) if x)
                st.markdown(ui.stat_html([("Side", r.get("side", "") or "—"), ("Odds", ui.fmt_odds(r.get("odds"))),
                                          ("p model", ui.fmt_prob(r.get("p_model_raw"))),
                                          ("p market", ui.fmt_prob(r.get("p_market_fair")))]) + _PARA + chips,
                            unsafe_allow_html=True)


_PARA = "\n\n"


def _md(s) -> str:
    return ui._md_safe(s)


# ── TOP 1 / 2 / 3 / 5 strip ────────────────────────────────────────────────────────────────────
def rank_strip_rows(t: pd.DataFrame) -> pd.DataFrame:
    """league_weekly_eval.league_table rows for the strip subsets (one market already selected)."""
    if t is None or t.empty:
        return pd.DataFrame()
    return t[t["subset"].isin([s for s, _ in STRIP])].reset_index(drop=True)


def rank_strip(t: pd.DataFrame) -> None:
    rows = rank_strip_rows(t)
    if rows.empty:
        ui.empty_state("NO_DATA", "NOT_ENOUGH_DATA — no frozen rows yet for this market.")
        return
    for (mt, side), g in rows.groupby(["model_type", "side"], sort=True):
        ui.eyebrow(f"{side} · {mt}")
        by = {r["subset"]: r for r in g.to_dict("records")}
        for col, (sub, lab) in zip(st.columns(len(STRIP)), STRIP):
            m = by.get(sub)
            if m is None:
                with col.container(border=True):
                    ui.eyebrow(lab)
                    st.markdown(ui.status_chip("NO_DATA"))
                continue
            n = m.get("n_priced_settled") or 0
            ui.metric_card("ROI (flat 1u)", ui.fmt_roi(m.get("roi")), title=lab, container=col,
                           supporting=[("CLV mean", ui.fmt_clv(m.get("clv_mean"))),
                                       ("n · GW", f"{int(n)} · {int(m.get('gameweeks') or 0)} GW"),
                                       ("Win rate", ui.pct(m.get("win_rate"), None))],
                           status=ui.status_chip("QUALIFIED", label="OK") if m.get("status") == "OK"
                           else ui.status_chip("SMALL_SAMPLE", label="NOT ENOUGH DATA"),
                           n=n, floor=EV.MIN_N)


# ── weekly rank heatmap (the signature visual) ─────────────────────────────────────────────────
OUTCOMES = ("WIN", "LOSS", "PENDING", "NO ROW")


def heatmap_frame(panel: pd.DataFrame, league: str, market: str, prepared: bool = False) -> pd.DataFrame:
    """Gameweek x rank, long. One row per cell; a rank with no panel row is an explicit NO ROW cell.

    The rows are the ones league_weekly_eval.weekly_grid draws (same filter, same prepare()); `cell` is
    league_weekly_eval's own cell text, so the heatmap and the history table say the same thing."""
    d = panel if prepared else EV.prepare(panel)
    cols = ["gameweek", "league_week_id", "rank", "outcome", "label", "fixture", "side", "result", "clv",
            "decision_state", "production", "ev_lower_bound", "odds", "cell"]
    if d.empty:
        return pd.DataFrame(columns=cols)
    x = d[(d["league"] == league) & (d["market"] == market)]
    out = []
    for wk in sorted(x["league_week_id"].unique(), reverse=True):
        g = x[x["league_week_id"] == wk]
        have = {}
        for _, r in g.iterrows():
            have[int(r["rank"])] = r
        for i in range(1, 6):
            r = have.get(i)
            gw = str(wk).split("|")[-1]
            if r is None:
                out.append({"gameweek": gw, "league_week_id": wk, "rank": f"#{i}", "outcome": "NO ROW",
                            "label": "—\nno fixture", "fixture": "", "side": "", "result": "", "clv": None,
                            "decision_state": "", "production": "", "ev_lower_bound": None, "odds": None,
                            "cell": ""})
                continue
            res = str(r.get("result", "") or "").upper()
            outcome = res if res in ("WIN", "LOSS") else "PENDING"
            clv = r.get("clv")
            clv_s = f"CLV {clv:+.1f}%" if clv is not None and not pd.isna(clv) else "CLV —"
            state = str(r.get("decision_state", "") or "").replace("_", " ")
            prod = bool(r.get("prod_sel"))
            out.append({"gameweek": gw, "league_week_id": wk, "rank": f"#{i}", "outcome": outcome,
                        "label": f"{outcome}\n{clv_s}\n{state}{' · PROD' if prod else ''}",
                        "fixture": f"{r.get('home_team', '')} v {r.get('away_team', '')}",
                        "side": str(r.get("side", "")), "result": res or "pending",
                        "clv": None if clv is None or pd.isna(clv) else float(clv),
                        "decision_state": str(r.get("decision_state", "") or ""),
                        "production": "yes" if prod else "no",
                        "ev_lower_bound": None if pd.isna(r.get("ev_lb")) else float(r.get("ev_lb")),
                        "odds": None if pd.isna(r.get("odds_f")) else float(r.get("odds_f")),
                        "cell": EV._cell(r)})
    return pd.DataFrame(out, columns=cols)


def heatmap_chart(hm: pd.DataFrame):
    """Rect + text layers: colour AND words (WIN / LOSS / PENDING / NO ROW), tooltip with every field."""
    import altair as alt
    t = ui.tokens()
    weeks = list(dict.fromkeys(hm["gameweek"]))
    fill = alt.Scale(domain=list(OUTCOMES), range=[t["POSITIVE"], t["NEGATIVE"], t["SURFACE_ELEVATED"], t["SURFACE"]])
    ink = alt.Scale(domain=list(OUTCOMES), range=[t["BACKGROUND"], t["BACKGROUND"], t["TEXT_SECONDARY"], t["TEXT_MUTED"]])
    axis = dict(labelColor=t["TEXT_MUTED"], titleColor=t["TEXT_MUTED"], domain=False, ticks=False)
    base = alt.Chart(hm).encode(
        x=alt.X("rank:O", sort=RANKS, title=None, axis=alt.Axis(orient="top", labelAngle=0, **axis)),
        y=alt.Y("gameweek:O", sort=weeks, title=None, axis=alt.Axis(**axis)))
    tips = [alt.Tooltip("league_week_id:N", title="gameweek"), alt.Tooltip("rank:N", title="rank"),
            alt.Tooltip("fixture:N"), alt.Tooltip("side:N"), alt.Tooltip("result:N"),
            alt.Tooltip("clv:Q", title="CLV %", format="+.2f"), alt.Tooltip("decision_state:N", title="decision"),
            alt.Tooltip("production:N", title="production selected"),
            alt.Tooltip("ev_lower_bound:Q", title="EV lower bound", format="+.4f"),
            alt.Tooltip("odds:Q", format=".2f")]
    rect = base.mark_rect(cornerRadius=6, stroke=t["BORDER"], strokeWidth=1).encode(
        color=alt.Color("outcome:N", scale=fill, legend=alt.Legend(title=None, orient="bottom")), tooltip=tips)
    text = base.mark_text(lineBreak="\n", fontSize=11, fontWeight=600).encode(
        text="label:N", color=alt.Color("outcome:N", scale=ink, legend=None), tooltip=tips)
    return (rect + text).properties(height=max(1, len(weeks)) * 70 + 40)


def heatmap(panel: pd.DataFrame, league: str, market: str, prepared: bool = False) -> None:
    hm = heatmap_frame(panel, league, market, prepared=prepared)
    if hm.empty:
        ui.empty_state("NO_DATA", "No frozen gameweeks yet — the heatmap fills one row per gameweek once the "
                                  "panel has rows for this league and market.")
        return
    n_pending = int((hm["outcome"] == "PENDING").sum())
    n_settled = int(hm["outcome"].isin(["WIN", "LOSS"]).sum())
    st.altair_chart(heatmap_chart(hm), width="stretch")
    st.caption(f"{n_settled} settled · {n_pending} pending · rows = gameweeks (newest first), columns = rank #1..#5. "
               "Each cell says its result, certified CLV and the market-first decision state in words; PROD = "
               "production tipped the same fixture × market × side. Hover a cell for the fixture and EV lower bound.")


# ── current production vs market-first shadow ─────────────────────────────────────────────────
def engine_frame(t: pd.DataFrame) -> pd.DataFrame:
    """Selections compared on outcomes: production-selected vs the shadow engine's top-5 / qualified rows."""
    if t is None or t.empty:
        return pd.DataFrame()
    rows = []
    for r in t.to_dict("records"):
        lab = dict(ENGINE_SUBSETS).get(r["subset"])
        if lab is None:
            continue
        rows.append({"side": r["side"], "model_type": r["model_type"], "selection": lab,
                     "n_rows": r["n_rows"], "n_settled": r["n_settled"], "n_priced_settled": r["n_priced_settled"],
                     "roi": r["roi"], "clv_mean": r["clv_mean"], "win_rate": r["win_rate"], "status": r["status"]})
    return pd.DataFrame(rows)


def prob_frame(t: pd.DataFrame) -> pd.DataFrame:
    """Brier / log loss of production's model, the market-first decision and the market, on the SAME top-5 rows."""
    if t is None or t.empty:
        return pd.DataFrame()
    rows = []
    for r in t[t["subset"] == "TOP5"].to_dict("records"):
        for key, lab in PROB_SOURCES:
            rows.append({"side": r["side"], "model_type": r["model_type"], "source": lab, "n": r["n_prob"],
                         "brier": r[f"brier_{key}"], "log_loss": r[f"logloss_{key}"]})
    return pd.DataFrame(rows)


def overlap_line(t: pd.DataFrame) -> str:
    """'OVER: production selected 2 of the shadow top-5's 10 rows' per side."""
    if t is None or t.empty:
        return ""
    bits = []
    for (mt, side), g in t.groupby(["model_type", "side"], sort=True):
        by = {r["subset"]: r for r in g.to_dict("records")}
        p, s = by.get("PRODUCTION_SELECTED"), by.get("TOP5")
        if p is not None and s is not None:
            bits.append(f"{side}: production selected {int(p['n_rows'])} of the shadow top-5's {int(s['n_rows'])} rows")
    return " · ".join(bits)


def _dots(df: pd.DataFrame, value: str, colour_field: str, title: str, fmt: str):
    """Paired dots on ONE scale per chart (never a second axis); n rides in the tooltip."""
    import altair as alt
    t = ui.tokens()
    axis = dict(labelColor=t["TEXT_MUTED"], titleColor=t["TEXT_MUTED"], gridColor=t["BORDER"])
    d = df[df[value].notna()]
    return alt.Chart(d).mark_circle(size=140, opacity=0.95).encode(
        x=alt.X(f"{value}:Q", title=title, axis=alt.Axis(format=fmt, **axis)),
        y=alt.Y("side:N", title=None, axis=alt.Axis(labelColor=t["TEXT_MUTED"])),
        color=alt.Color(f"{colour_field}:N", scale=alt.Scale(range=ui.CATEGORICAL),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("side:N"), alt.Tooltip(f"{colour_field}:N"),
                 alt.Tooltip(f"{value}:Q", format=fmt),
                 alt.Tooltip("n_priced_settled:Q" if "n_priced_settled" in d.columns else "n:Q", title="n")],
    ).properties(height=40 * max(1, d["side"].nunique()) + 50)


def engine_compare(t: pd.DataFrame) -> None:
    ef, pf = engine_frame(t), prob_frame(t)
    if ef.empty:
        ui.empty_state("NO_DATA", "No frozen rows yet for this market — nothing to compare.")
        return
    line = overlap_line(t)
    if line:
        st.markdown(ui.status_chip("LIVE", label="PRODUCTION · legacy engine") + " "
                    + ui.status_chip("SHADOW", label="SHADOW · market-first") + f" :gray[{_md(line)}]")
    have_out = ef["roi"].notna().any() or ef["clv_mean"].notna().any()
    have_prob = len(pf) and pf["brier"].notna().any()
    if have_out or have_prob:
        a, b = st.columns(2)
        if ef["roi"].notna().any():
            a.altair_chart(_dots(ef, "roi", "selection", "ROI (flat 1u, priced settled rows)", "+.1%"), width="stretch")
        if ef["clv_mean"].notna().any():
            b.altair_chart(_dots(ef, "clv_mean", "selection", "Mean certified CLV (%)", "+.2f"), width="stretch")
        if have_prob:
            a, b = st.columns(2)
            a.altair_chart(_dots(pf, "brier", "source", "Brier (lower is better)", ".4f"), width="stretch")
            b.altair_chart(_dots(pf, "log_loss", "source", "Log loss (lower is better)", ".4f"), width="stretch")
    else:
        ui.empty_state("NO_DATA", "No settled rows yet: ROI, CLV, Brier and log loss appear once the first "
                                  "gameweek settles. n settled = 0 on every selection below.")
    cfg = {"n_rows": ui.num_col("Rows", fmt="%d"), "n_settled": ui.num_col("Settled", fmt="%d"),
           "n_priced_settled": ui.num_col("n (priced settled)", fmt="%d"), "roi": ui.num_col("ROI", fmt="%+.4f"),
           "clv_mean": ui.num_col("CLV mean %", fmt="%+.3f"), "win_rate": ui.num_col("Win rate", fmt="%.4f")}
    ui.compact_table(ef, cfg)
    if len(pf):
        ui.compact_table(pf, {"n": ui.num_col("n", fmt="%d"), "brier": ui.num_col("Brier", fmt="%.5f"),
                              "log_loss": ui.num_col("Log loss", fmt="%.5f")})


def metric_config() -> dict:
    """league_weekly_eval.METRIC_COLS as numbers (no string formatting in the frame)."""
    cfg = {}
    for c in EV.METRIC_COLS:
        if c in ("status",):
            continue
        if c.startswith("n_") or c in ("gameweeks", "clv_n", "longest_losing_streak"):
            cfg[c] = ui.num_col(c, fmt="%d")
        elif c.startswith(("brier", "logloss", "dlogloss")):
            cfg[c] = ui.num_col(c, fmt="%.5f")
        else:
            cfg[c] = ui.num_col(c, fmt="%.4f")
    return cfg
