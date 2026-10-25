"""Presentation helpers for the Fantasy page (visual redesign stage 2a, 2026-10-25).

PRESENTATION ONLY. Nothing here projects, ranks a captain, optimises a squad, settles or fetches.
Every number a card shows is a column of the precomputed V2 artifacts (fantasy_v2_projections.csv,
fantasy_v2_summary.json, fantasy_v2_transfers.csv, fantasy_v2_fixture_matrix.csv), handed in by the
page. The few selections made here are display filters over those columns (a sort, a head, a
threshold) and are written as plain functions so tests/test_fantasy_page_values.py can pin them.

The transfer-planner candidate rule and the fixture-matrix aggregation were MOVED here verbatim from
pages/13_⚽_Fantasy.py so the page and the test call the same code; their behaviour is unchanged.

Colour: fixture difficulty uses the theme's named colours (native badges) and the design-system
tokens for the heatmap. Easy is always green and hard always red, in every lens, and the number is
always printed beside the colour.
"""
from __future__ import annotations

import html as _html

import pandas as pd
import streamlit as st

import dashboard_ui as ui

POS_ORDER = ["GKP", "DEF", "MID", "FWD"]
POS_LABEL = {"GKP": "Goalkeeper", "DEF": "Defender", "MID": "Midfielder", "FWD": "Forward"}
#: the captaincy pool the page has always used for its upside chart
POOL_MIN_START = 0.6
POOL_EXCLUDE_RISK = ("Unavailable", "Doubtful")
#: same default as the V1 differentials slider
DIFFERENTIAL_MAX_OWN = 10.0
HIT_COST = 4

#: risk label -> (status word, tone)
RISK_STATE = {"Low": ("LOW RISK", "POSITIVE"), "Rotation": ("ROTATION RISK", "WARNING"),
              "Volatile": ("VOLATILE", "WARNING"), "Doubtful": ("DOUBTFUL", "NEGATIVE"),
              "Unavailable": ("UNAVAILABLE", "NEGATIVE")}
_RISK_ICON = {"POSITIVE": "●", "WARNING": "▲", "NEGATIVE": "✕"}


# ═════════════════════════════════════════════════════════════════════════════════════════════
# PURE HELPERS (no Streamlit) — tested directly
# ═════════════════════════════════════════════════════════════════════════════════════════════
def _missing(x) -> bool:
    try:
        return x is None or (not isinstance(x, str) and pd.isna(x))
    except Exception:                                                 # noqa: BLE001
        return False


def fdr_value(s) -> float | None:
    """One fixture-difficulty entry: '4' -> 4.0, a double gameweek '2+4' -> 3.0, '' -> None."""
    parts = [p for p in str(s or "").split("+") if p.strip()]
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    return sum(vals) / len(vals) if vals else None


def fdr_list(fdr_run) -> list[float | None]:
    return [fdr_value(x) for x in str(fdr_run or "").split(" | ")] if not _missing(fdr_run) else []


def avg_fdr(fdr_run) -> float | None:
    v = [x for x in fdr_list(fdr_run) if x is not None]
    return sum(v) / len(v) if v else None


def fdr_tone(v) -> str:
    """Official FDR semantics: 1-2 easy (green), 3 neutral (grey), 4 hard (amber), 5 hardest (red)."""
    if v is None or _missing(v):
        return "NO_DATA"
    v = float(v)
    if v < 2.5:
        return "POSITIVE"
    if v < 3.5:
        return "NEUTRAL"
    if v < 4.5:
        return "WARNING"
    return "NEGATIVE"


def ease_tone(v) -> str:
    """Wowza ease quintile (1 hard .. 5 easy) on the SAME colour semantics as FDR."""
    return "NO_DATA" if v is None or _missing(v) else fdr_tone(6 - float(v))


def _md(s) -> str:
    return str(s).replace("[", "(").replace("]", ")").replace("*", "").replace("_", " ")


def fdr_chips(fixture_run, fdr_run) -> str:
    """Next-N fixture chips as theme-aware badges: 'LIV(A) 4' on the FDR colour (markdown)."""
    labs = [x for x in str(fixture_run or "").split(" | ") if x] if not _missing(fixture_run) else []
    fdrs = str(fdr_run or "").split(" | ") if not _missing(fdr_run) else []
    chips = []
    for lab, f in zip(labs, fdrs + [""] * (len(labs) - len(fdrs))):
        if lab == "BLANK":
            chips.append(":gray-badge[BLANK]")
            continue
        colour = ui.TONE_COLOUR.get(fdr_tone(fdr_value(f)), "gray")
        chips.append(f":{colour}-badge[{_md(lab)} {_md(f) if f else '?'}]")
    return " ".join(chips)


def run_chips(pairs) -> str:
    """Chips from (label, fdr) pairs (fixture-matrix rows)."""
    out = []
    for lab, f in pairs:
        colour = ui.TONE_COLOUR.get(fdr_tone(f), "gray")
        fv = "—" if f is None or _missing(f) else (f"{f:.0f}" if float(f).is_integer() else f"{f:.1f}")
        out.append(f":{colour}-badge[{_md(lab)} {fv}]")
    return " ".join(out)


def risk_chip(risk) -> str:
    word, tone = RISK_STATE.get(str(risk), (str(risk).upper() if not _missing(risk) else "RISK —", "NO_DATA"))
    return f":{ui.TONE_COLOUR.get(tone, 'gray')}-badge[{_RISK_ICON.get(tone, '○')} {_md(word)}]"


def pool(proj: pd.DataFrame) -> pd.DataFrame:
    """Startable players: P(start) >= 0.6 and not Unavailable/Doubtful (the captaincy-chart pool)."""
    return proj[(proj["p_start"] >= POOL_MIN_START) & ~proj["risk"].isin(list(POOL_EXCLUDE_RISK))]


def available(proj: pd.DataFrame) -> pd.DataFrame:
    """Everyone not ruled out (the old Top-recommendations filter)."""
    return proj[~proj["risk"].isin(["Unavailable"])]


def top_projected(proj: pd.DataFrame, position: str = "All", n: int = 6) -> pd.DataFrame:
    """The old 'Top recommendations' cards: available, optional position, by final_xpts."""
    a = available(proj)
    a = a if position == "All" else a[a["position"] == position]
    return a.sort_values("final_xpts", ascending=False).head(n)


def recommendations(proj: pd.DataFrame, summ: dict) -> dict:
    """The Players-tab summary. Display picks over published columns, each rule stated:

    best_overall      available, highest final_xpts (= the first Top-projected card)
    best_value        startable pool, highest value_5gw_per_m (5-GW xPts per £m)
    best_differential startable pool, ownership < 10 %, highest final_xpts
    best_captain      the published BALANCED captaincy pick (fantasy_v2_summary.json)
    best_defender     available defenders, highest final_xpts
    """
    out: dict = {}
    a, p = available(proj), pool(proj)
    if len(a):
        out["best_overall"] = a.sort_values("final_xpts", ascending=False).iloc[0]
    if len(p):
        out["best_value"] = p.sort_values("value_5gw_per_m", ascending=False).iloc[0]
        d = p[p["ownership"] < DIFFERENTIAL_MAX_OWN]
        if len(d):
            out["best_differential"] = d.sort_values("final_xpts", ascending=False).iloc[0]
    cap = (summ.get("captaincy") or {}).get("BALANCED")
    if cap:
        out["best_captain"] = cap
    dfn = a[a["position"] == "DEF"]
    if len(dfn):
        out["best_defender"] = dfn.sort_values("final_xpts", ascending=False).iloc[0]
    return out


def best_value(proj: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    return pool(proj).sort_values("value_5gw_per_m", ascending=False).head(n)


def lookup(proj: pd.DataFrame, cap: dict):
    """The projection row behind a captaincy pick (player_name + team), or None."""
    m = proj[(proj["player_name"] == cap.get("player")) & (proj["team"] == cap.get("team"))]
    if m.empty:
        m = proj[(proj["web_name"] == cap.get("web_name")) & (proj["team"] == cap.get("team"))]
    return None if m.empty else m.iloc[0]


HORIZON_COL = {"next GW": "final_xpts", "3 GW": "xpts_3gw", "5 GW": "xpts_5gw"}


def transfer_candidates(proj: pd.DataFrame, out_name: str, bank: float, horizon: str, n: int = 5):
    """MOVED VERBATIM from the page: same position, not the same player, affordable within bank,
    not Unavailable, ranked by gain on the horizon. Returns (out_row, candidates)."""
    hcol = HORIZON_COL[horizon]
    o = proj[proj["player_name"] == out_name].iloc[0]
    cand = proj[(proj["position"] == o["position"]) & (proj["element"] != o["element"])
                & (proj["price"] <= o["price"] + bank) & (proj["risk"] != "Unavailable")].copy()
    cand["gain"] = cand[hcol] - o[hcol]
    cand = cand.sort_values("gain", ascending=False).head(n)
    return o, cand


def transfer_gains(o, c) -> dict:
    """Display deltas for one OUT -> IN pair (pure subtraction of published columns)."""
    fo, fi = avg_fdr(o["fdr_run"]), avg_fdr(c["fdr_run"])
    return {"price_change": float(c["price"]) - float(o["price"]),
            "gain_next": float(c["final_xpts"]) - float(o["final_xpts"]),
            "gain_3gw": float(c["xpts_3gw"]) - float(o["xpts_3gw"]),
            "gain_5gw": float(c["xpts_5gw"]) - float(o["xpts_5gw"]),
            "fixture_swing": (None if fo is None or fi is None else fi - fo)}


def fixture_agg(fxm: pd.DataFrame) -> pd.DataFrame:
    """MOVED VERBATIM from the page: team x GW (double gameweeks joined with '+')."""
    f = fxm.copy()
    f["GW"] = "GW" + f["gw"].astype(str)
    return f.groupby(["team", "GW"], as_index=False).agg(
        label=("label", lambda s: " + ".join(s)), attack_ease=("attack_ease", "mean"),
        defence_ease=("defence_ease", "mean"), fdr=("fdr", "mean"),
        lam_for=("lam_for", "sum"), cs_prob=("cs_prob", "mean"))


LENS = {"Attack": "attack_ease", "Defence": "defence_ease", "Official FDR": "fdr"}
LENS_HELP = {"Attack": "Wowza attack ease 1–5 (team goal expectancy; 5 = easiest)",
             "Defence": "Wowza defence ease 1–5 (clean-sheet chance; 5 = easiest)",
             "Official FDR": "FPL's official difficulty 1–5 (1 = easiest)"}


def team_order(agg: pd.DataFrame, key: str) -> list:
    """Easiest run first (same rule as before: FDR ascending, ease descending)."""
    return agg.groupby("team")[key].mean().sort_values(ascending=(key == "fdr")).index.tolist()


def gw_sort(agg: pd.DataFrame) -> list:
    return sorted(agg["GW"].unique(), key=lambda g: int(g[2:]))


def fixture_swing(agg: pd.DataFrame, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Easiest and hardest runs over the matrix horizon by mean official FDR."""
    m = agg.groupby("team", as_index=False)["fdr"].mean().rename(columns={"fdr": "avg_fdr"})
    m = m.sort_values(["avg_fdr", "team"])
    return m.head(n), m.sort_values(["avg_fdr", "team"], ascending=[False, True]).head(n)


def team_runs(agg: pd.DataFrame) -> dict:
    """team -> [(label, fdr), ...] in gameweek order (one pivot, not a filter per team)."""
    gws = gw_sort(agg)
    lab = agg.pivot(index="team", columns="GW", values="label").reindex(columns=gws)
    fdr = agg.pivot(index="team", columns="GW", values="fdr").reindex(columns=gws)
    out = {}
    for team in lab.index:
        out[team] = [(a, b) for a, b in zip(lab.loc[team].tolist(), fdr.loc[team].tolist())
                     if isinstance(a, str)]
    return out


def upside_scatter_spec() -> dict:
    """Captaincy chart (expected points vs P(15+), dot size = P(start)) as a Vega-Lite dict; the
    data (EV, P(15+), p_start, p_blank, position, web_name, team, next_fixture, lab) is passed
    beside it. Rows with a non-empty `lab` (the ten highest EV) are labelled. Plain dict for the
    same reason as the heatmap."""
    t = ui.tokens()
    x = {"field": "EV", "type": "quantitative", "title": "expected points (one game)"}
    y = {"field": "P(15+)", "type": "quantitative", "title": "P(15+ points) — upside", "axis": {"format": "%"}}
    return {
        "height": 380,
        "layer": [
            {"mark": {"type": "circle", "opacity": 0.8},
             "encoding": {"x": x, "y": y,
                          "size": {"field": "p_start", "type": "quantitative", "title": "P(start)",
                                   "scale": {"range": [30, 300]}},
                          "color": {"field": "position", "type": "nominal",
                                    "legend": {"orient": "top", "title": None}},
                          "tooltip": [{"field": "web_name", "type": "nominal"}, {"field": "team", "type": "nominal"},
                                      {"field": "EV", "type": "quantitative", "format": ".2f"},
                                      {"field": "P(15+)", "type": "quantitative", "format": ".1%"},
                                      {"field": "p_start", "type": "quantitative", "format": ".0%"},
                                      {"field": "p_blank", "type": "quantitative", "format": ".0%", "title": "P(blank)"},
                                      {"field": "next_fixture", "type": "nominal"}]}},
            {"transform": [{"filter": "datum.lab != ''"}],
             "mark": {"type": "text", "dy": -10, "fontSize": 10, "color": t["TEXT_SECONDARY"]},
             "encoding": {"x": x, "y": y, "text": {"field": "lab", "type": "nominal"}}},
        ],
    }


def fixture_heatmap_spec(agg: pd.DataFrame, lens: str) -> dict:
    """Team x gameweek heatmap as a Vega-Lite dict. Green = easy, red = hard in EVERY lens; the
    value is printed in each cell.

    Written directly rather than through Altair: Altair's schema validation of this layered spec costs ~90 ms on every rerun, and Streamlit
    re-runs every tab body on each interaction; a plain dict costs ~2 ms. Same data, same encoding.
    """
    key = LENS[lens]
    t = ui.tokens()
    d = agg[["team", "GW", "label", "attack_ease", "defence_ease", "fdr", "lam_for", "cs_prob"]].copy()
    d["value"] = d[key].round(1)
    d["cell"] = d["label"] + " · " + d["value"].map(lambda v: f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}")
    rng = [t["POSITIVE"], t["BORDER_STRONG"], t["NEGATIVE"]]
    order = team_order(agg, key)
    legend_title = "difficulty (1 easy … 5 hard)" if key == "fdr" else "ease (1 hard … 5 easy)"
    enc = {
        "x": {"field": "GW", "type": "nominal", "title": None, "sort": gw_sort(agg),
              "axis": {"orient": "top", "labelAngle": 0, "labelColor": t["TEXT_MUTED"], "domain": False,
                       "ticks": False}},
        "y": {"field": "team", "type": "nominal", "title": None, "sort": order,
              "axis": {"labelColor": t["TEXT_SECONDARY"], "domain": False, "ticks": False, "labelLimit": 140}},
    }
    tooltip = [{"field": "team", "type": "nominal"}, {"field": "GW", "type": "nominal"},
               {"field": "label", "type": "nominal"},
               {"field": "lam_for", "type": "quantitative", "format": ".2f", "title": "team xG"},
               {"field": "cs_prob", "type": "quantitative", "format": ".0%", "title": "CS prob"},
               {"field": "fdr", "type": "quantitative", "format": ".0f", "title": "FDR"},
               {"field": "attack_ease", "type": "quantitative", "format": ".1f", "title": "attack ease"},
               {"field": "defence_ease", "type": "quantitative", "format": ".1f", "title": "defence ease"}]
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": d.to_dict("records")},
        "height": max(320, 30 * len(order)),
        "encoding": enc,
        "layer": [
            {"mark": {"type": "rect", "stroke": t["BACKGROUND"], "strokeWidth": 2, "cornerRadius": 3},
             "encoding": {"color": {"field": key, "type": "quantitative",
                                    "scale": {"domain": [1, 3, 5], "range": rng if key == "fdr" else rng[::-1],
                                              "clamp": True},
                                    "legend": {"title": legend_title, "orient": "top",
                                               "labelColor": t["TEXT_MUTED"], "titleColor": t["TEXT_MUTED"]}},
                          "tooltip": tooltip}},
            {"mark": {"type": "text", "fontSize": 11, "fontWeight": 600, "color": t["TEXT_PRIMARY"]},
             "encoding": {"text": {"field": "cell", "type": "nominal"}}},
        ],
    }


# ═════════════════════════════════════════════════════════════════════════════════════════════
# CARDS (Streamlit)
# ═════════════════════════════════════════════════════════════════════════════════════════════
def _get(r, k, default=None):
    """A field from a projection row given as a Series or a dict."""
    v = r.get(k, default) if isinstance(r, (dict, pd.Series)) else getattr(r, k, default)
    return default if v is None else v


def _name_html(name: str, eyebrow_text: str = "") -> str:
    eb = f"<div class='wz-eyebrow'>{_html.escape(eyebrow_text)}</div>" if eyebrow_text else ""
    # The shared .wz-title class only (stage 3: the last inline style in a view module was removed).
    return eb + f"<div class='wz-title'>{_html.escape(str(name))}</div>"


def player_card(r, *, eyebrow_text: str = "", primary_label: str = "Final xPts", primary_value=None,
                supporting=None, extra_caption: str = "", show_baseline: bool = True, container=None) -> None:
    """The standard Fantasy player card.

    NAME · team · position · price · ownership · FINAL XPTS · fixture chips · P(start), xMin,
    ceiling, P(goal), P(assist) · risk. `r` is a projection row (Series / namedtuple / dict).
    """
    c = container or st
    with c.container(border=True):
        name = _get(r, "web_name") or _get(r, "player_name", "—")
        meta = " · ".join(x for x in (
            str(_get(r, "team", "")), str(_get(r, "position", "")),
            ui.fmt_money(_get(r, "price")),
            (f"owned {_get(r, 'ownership'):.1f}%" if not _missing(_get(r, "ownership")) else "")) if x)
        st.markdown(_name_html(name, eyebrow_text) + f"<div class='wz-sub'>{_html.escape(meta)}</div>",
                    unsafe_allow_html=True)
        pv = _get(r, "final_xpts") if primary_value is None else primary_value
        st.metric(primary_label, pv if isinstance(pv, str) else ui.fmt_xpts(pv))
        nf = _get(r, "next_fixture", "")
        chips = fdr_chips(_get(r, "fixture_run"), _get(r, "fdr_run"))
        st.markdown((f"**Next {_md(nf)}** &nbsp; " if nf else "") + chips)
        stats = list(supporting) if supporting is not None else []
        stats += [("P(start)", ui.fmt_prob(_get(r, "p_start"))),
                  ("xMin", "—" if _missing(_get(r, "exp_minutes")) else f"{_get(r, 'exp_minutes'):.0f}"),
                  ("Ceiling", "—" if _missing(_get(r, "ceiling")) else f"{_get(r, 'ceiling'):.0f}"),
                  ("P(goal)", ui.fmt_prob(_get(r, "goal_prob"))),
                  ("P(assist)", ui.fmt_prob(_get(r, "assist_prob")))]
        if show_baseline and not _missing(_get(r, "fpl_ep_next")):
            applied = bool(_get(r, "residual_applied", False))
            stats += [("FPL baseline", f"{_get(r, 'fpl_ep_next'):.1f}"),
                      ("Wowza adj.", ("—" if _missing(_get(r, "wowza_residual"))
                                      else f"{_get(r, 'wowza_residual'):+.2f}")
                       + (" applied" if applied else " shadow"))]
        conf = _get(r, "confidence")
        tail = risk_chip(_get(r, "risk")) + (f" :gray-badge[confidence {_md(conf)}]" if conf else "")
        st.markdown(ui.stat_html(stats) + "\n\n" + tail, unsafe_allow_html=True)
        news = _get(r, "news")
        if isinstance(news, str) and news and str(_get(r, "risk")) != "Low":
            st.caption(news)
        if extra_caption:
            st.caption(extra_caption)


def captain_card(c: dict, row, title: str, container=None) -> None:
    """A captaincy pick (from fantasy_v2_summary.json) with the projection row for fixture/risk."""
    box = container or st
    with box.container(border=True):
        meta = " · ".join(x for x in (c.get("team", ""), c.get("position", ""),
                                      ui.fmt_money(c.get("price"))) if x)
        st.markdown(_name_html(c.get("web_name", "—"), title) + f"<div class='wz-sub'>{_html.escape(meta)}</div>",
                    unsafe_allow_html=True)
        st.metric("xPts", ui.fmt_xpts(c.get("final_xpts")))
        fd = None
        fx = f"**vs {_md(c.get('next_fixture', '—'))}**"
        if row is not None and not _missing(row.get("fdr_run")):
            fd = fdr_value(str(row.get("fdr_run")).split(" | ")[0])
            fx += " &nbsp; " + fdr_chips(row.get("fixture_run"), row.get("fdr_run"))
        st.markdown(fx)
        st.markdown(ui.stat_html([
            ("Captain EV", f"{c['captain_ev']:.1f}"), ("P(start)", f"{c['p_start']:.0%}"),
            ("Ceiling", f"{c['ceiling']:.0f}"), ("P(10+)", f"{c['p10']:.0%}"),
            ("Fixture difficulty", "—" if fd is None else f"{fd:.0f}"),
            ("P(6+)", f"{c['p6']:.0%}"), ("P(15+)", f"{c['p15']:.0%}"),
            ("xMin", f"~{c['exp_minutes']:.0f}"),
            ("Pens", "yes" if c.get("pen_first") else "no"),
            ("Set pieces", "yes" if c.get("set_piece") else "no")])
            + "\n\n" + (risk_chip(row.get("risk")) if row is not None else ":gray-badge[○ RISK —]"),
            unsafe_allow_html=True)
        if c.get("reasoning"):
            st.caption(c["reasoning"])
        if c.get("note"):
            st.caption(f"Note: {c['note']}")


def transfer_card(o, c, *, horizon: str = "5 GW", gain=None, container=None) -> None:
    """OUT -> IN as one card: the move, then the deltas, then risk. `gain` is the planner's
    horizon gain (the ranking number) when given."""
    g = transfer_gains(o, c)
    box = container or st
    with box.container(border=True):
        st.markdown(
            f"<div class='wz-eyebrow'>{_html.escape(str(c['position']))} · OUT → IN</div>"
            f"<div class='wz-sub'>OUT {_html.escape(str(o['web_name']))} · {_html.escape(str(o['team']))} · "
            f"{ui.fmt_money(o['price'])}</div>"
            + _name_html(f"→ {c['web_name']}")
            + f"<div class='wz-sub'>{_html.escape(str(c['team']))} · {ui.fmt_money(c['price'])} · "
              f"owned {float(c['ownership']):.1f}%</div>", unsafe_allow_html=True)
        main = gain if gain is not None else g["gain_5gw"]
        lab = f"Expected gain · {horizon}" if gain is not None else "Expected gain · 5 GW"
        st.metric(lab, f"{main:+.2f}", f"{main - HIT_COST:+.2f} after a -{HIT_COST} hit", delta_color="off")
        sw = g["fixture_swing"]
        st.markdown(ui.stat_html([
            ("Price change", f"{g['price_change']:+.1f}m"), ("xPts gain (next)", f"{g['gain_next']:+.2f}"),
            ("3-GW gain", f"{g['gain_3gw']:+.2f}"), ("5-GW gain", f"{g['gain_5gw']:+.2f}"),
            ("Hit-adjusted 5-GW", f"{g['gain_5gw'] - HIT_COST:+.2f}"),
            ("Fixture swing", "—" if sw is None else f"{sw:+.1f} FDR"),
            ("P(start)", ui.fmt_prob(c["p_start"])), ("Ceiling", f"{c['ceiling']:.0f}")])
            + "\n\n" + risk_chip(c["risk"]), unsafe_allow_html=True)
        st.markdown(fdr_chips(c["fixture_run"], c["fdr_run"]))


def precomputed_move_card(t, container=None) -> None:
    """A row of fantasy_v2_transfers.csv (precomputed most-owned moves) as an OUT -> IN card."""
    box = container or st
    with box.container(border=True):
        st.markdown(
            f"<div class='wz-eyebrow'>{_html.escape(str(t['position']))} · OUT → IN</div>"
            f"<div class='wz-sub'>OUT {_html.escape(str(t['out_web']))} · {_html.escape(str(t['out_team']))} · "
            f"{ui.fmt_money(t['out_price'])}</div>"
            + _name_html(f"→ {t['in_web']}")
            + f"<div class='wz-sub'>{_html.escape(str(t['in_team']))} · {ui.fmt_money(t['in_price'])}</div>",
            unsafe_allow_html=True)
        st.metric("5-GW gain", f"{t['gain_5gw']:+.2f}",
                  f"{t['net_gain_5gw_with_hit']:+.2f} after a hit", delta_color="off")
        st.markdown(ui.stat_html([
            ("Price change", f"{t['price_delta']:+.1f}m"), ("Next GW", f"{t['gain_next']:+.2f}"),
            ("3-GW", f"{t['gain_3gw']:+.2f}"), ("P(start)", ui.fmt_prob(t["in_p_start"]))])
            + "\n\n" + risk_chip(t["in_risk"]), unsafe_allow_html=True)
