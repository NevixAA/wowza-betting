"""
Live Center — in-play signals, half-time predictions, and signal history in one place.
Merged from the former Live / HalfTime / Live History pages (three tabs).

Visual redesign 2026-10-25 (stage 2b): the three hand-written HTML card blocks (a hardcoded near-black card
background, white and grey text, per-signal hex accents) are replaced by ONE shared live match card (dashboard_data/live_view.py):
match, league, minute as a progress bar, score, signal chip, fair price, live and pre-match probability.
PRESENTATION ONLY - the loaders, the results lookup and the settlement below are unchanged.
"""
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

import dashboard_ui as ui

st.set_page_config(page_title="Live Center | Wowza", page_icon="⚡", layout="wide")

import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from dashboard_access import require as _require_tier  # noqa: E402
_require_tier("pro")  # tier guard (dashboard_access.PAGES), 2026-10-18

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from utils.access import is_public  # noqa: E402
from dashboard_data import live_view as LV  # noqa: E402

TIPS_FILE    = BASE_DIR / "output" / "live_tips.csv"
GAMES_FILE   = BASE_DIR / "output" / "live_games.csv"
PREDS_FILE   = BASE_DIR / "output" / "predictions.csv"
HISTORY_FILE = BASE_DIR / "output" / "live_signals_history.csv"

# Auto-refresh every 2 minutes (in-play data moves fast)
# Was components.v1.html with a JS reload: an API whose announced removal date
# (2026-06-01) has passed, and which reloaded the whole browser tab and so DISCARDED
# every filter the user had set.
ui.autorefresh(minutes=2, key="6_live_refresh")


# ── Loaders ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=120)
def load_tips():
    return pd.read_csv(TIPS_FILE) if TIPS_FILE.exists() else pd.DataFrame()


@st.cache_data(ttl=120)
def load_games():
    return pd.read_csv(GAMES_FILE) if GAMES_FILE.exists() else pd.DataFrame()


@st.cache_data(ttl=120)
def load_ht():
    if not PREDS_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(PREDS_FILE)
    if "p_ht_over05" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"] >= pd.Timestamp.now().normalize()]
    df = df[df["p_ht_over05"].notna()].copy()
    df["p_ht_over05"] = pd.to_numeric(df["p_ht_over05"], errors="coerce")
    df["p_ht_over15"] = pd.to_numeric(df.get("p_ht_over15"), errors="coerce")
    return df.sort_values("p_ht_over05", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=120)
def load_history():
    if not HISTORY_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(HISTORY_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date", ascending=False).reset_index(drop=True)


def _norm(s: str) -> str:
    return "".join(c for c in str(s).lower() if c.isalnum())


@st.cache_data(ttl=1800, show_spinner="Loading final scores…")
def load_results_lookup():
    """Final scores, read straight from the committed history parquets.

    THIS USED TO CALL `src.data_loader.load_all_matches()`, AND THAT WAS A WRITE. That function is
    the training-data path: it runs API-Football enrichment and then banks the result by writing
    `output/af_history.parquet` and `output/training_coverage.json`. So merely OPENING this page
    mutated two files the models train on — confirmed by `git status` after a render, with both
    files modified at the exact minute the page was loaded.

    Nothing here needed any of that. The page wants one thing: full-time and half-time goals for
    finished matches, to settle a live signal. Those are already sitting in committed parquets.
    Reading them is faster, spends no API quota, and cannot change a training input.

    COVERAGE IS UNCHANGED, which was checked rather than assumed. `af_history` turns out to hold
    full-time goals on only 277 of its 89,531 rows — it is a shots and xG store, and the scores in
    the old merged frame came from `fd_history` regardless. The union here settles 62,617 matches
    from 2019-02-15 to yesterday, which is the same set the old path could settle.

    `fd_history.parquet` is refreshed and committed by `predict.yml` (every five minutes) and
    `retrain.yml`, so reading it rather than rebuilding it costs no freshness in production.
    """
    import pathlib
    out = pathlib.Path(__file__).resolve().parents[1] / "output"
    frames = []
    try:
        # COLUMN-PROJECTED ON READ, not after. Reading the frame whole and slicing costs
        # 14 MB for fd and 21 MB for af that are then thrown away — and this is a cached
        # function, so the peak is what the deployment pays for the rest of its life.
        fd = pd.read_parquet(out / "fd_history.parquet",
                             columns=["date", "home_team", "away_team",
                                      "home_goals", "away_goals"])
        frames.append(fd)
    except Exception:
        pass
    try:
        af = pd.read_parquet(out / "af_history.parquet",
                             columns=["date", "home_team", "away_team", "FTHG", "FTAG"])
        frames.append(af.rename(columns={"FTHG": "home_goals", "FTAG": "away_goals"}))
    except Exception:
        pass
    if not frames:
        return {}
    m = pd.concat(frames, ignore_index=True)
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m = m.dropna(subset=["date", "home_goals", "away_goals"])
    m = m.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    m["total_goals"] = pd.to_numeric(m["home_goals"], errors="coerce") + \
        pd.to_numeric(m["away_goals"], errors="coerce")
    try:
        ht = pd.read_parquet(out / "af_ht_history.parquet",
                             columns=["date", "home_team", "away_team", "HTHG", "HTAG"])
        ht["date"] = pd.to_datetime(ht["date"], errors="coerce")
        ht["ht_total_goals"] = pd.to_numeric(ht["HTHG"], errors="coerce") + \
            pd.to_numeric(ht["HTAG"], errors="coerce")
        m = m.merge(ht[["date", "home_team", "away_team", "ht_total_goals"]],
                    on=["date", "home_team", "away_team"], how="left")
    except Exception:
        m["ht_total_goals"] = None
    if m.empty:
        return {}
    look = {}
    for r in m.itertuples(index=False):
        d = getattr(r, "date", None)
        if pd.isna(d):
            continue
        key = (_norm(getattr(r, "home_team", "")), _norm(getattr(r, "away_team", "")),
               pd.Timestamp(d).strftime("%Y-%m-%d"))
        ht = getattr(r, "ht_total_goals", None)
        look[key] = (getattr(r, "total_goals", None),
                     ht if ht is not None and not pd.isna(ht) else None)
    return look


def _settle(bet: str, total, ht_total):
    b = str(bet).upper()
    if "HT" in b:
        if ht_total is None or pd.isna(ht_total):
            return None
        if "OVER 0.5"  in b: return "WIN" if ht_total >= 1 else "LOSS"
        if "UNDER 0.5" in b: return "WIN" if ht_total == 0 else "LOSS"
        if "OVER 1.5"  in b: return "WIN" if ht_total >= 2 else "LOSS"
        if "UNDER 1.5" in b: return "WIN" if ht_total <= 1 else "LOSS"
        return None
    if total is None or pd.isna(total):
        return None
    if "OVER 2.5"  in b: return "WIN" if total >= 3 else "LOSS"
    if "UNDER 2.5" in b: return "WIN" if total <= 2 else "LOSS"
    return None


# ── Tab 1: Live Now ───────────────────────────────────────────────────────────
def render_live(df, games):
    st.caption("In-play signals from Poisson recalculation. Check your bookmaker's live screen to compare odds.")

    col_ref, col_run = st.columns([1, 4])
    with col_ref:
        if st.button("🔄 Refresh", key="live_refresh"):
            st.cache_data.clear(); st.rerun()
    with col_run:
        # needs a full checkout: the slim deployment carries no src/live_scanner.py
        if (not is_public() and (BASE_DIR / "src" / "live_scanner.py").exists()
                and st.button("▶ Run Live Scan Now", key="live_run")):
            import subprocess
            subprocess.run([sys.executable, str(BASE_DIR / "src" / "live_scanner.py")],
                           cwd=BASE_DIR, capture_output=True, text=True, timeout=60)
            st.cache_data.clear(); st.rerun()

    with st.expander("ℹ️ How this works"):
        st.markdown("""
        **No live odds API needed.** We compute the *fair price* with Poisson statistics:
        pre-match model → expected goals (λ) → adjust for current score + time → live P(under)/P(over).
        If your bookmaker's live odds are higher than our fair price → **value**.
        """)

    if df.empty or "signal_type" not in df.columns:
        import os
        last = ""
        if TIPS_FILE.exists():
            last = f" (last scan: {pd.Timestamp.fromtimestamp(os.path.getmtime(TIPS_FILE)).strftime('%H:%M')})"
        ui.empty_state("NO_DATA", f"⏳ No live value signals right now. Scanner runs every ~10 min during match hours.{last}")
    else:
        under_tips = df[df["bet"].str.contains("UNDER", na=False)]
        over_tips = df[df["bet"].str.contains("OVER", na=False)]
        ui.metric_grid([{"label": "Live Signals", "value": len(df)}, {"label": "UNDER Value", "value": len(under_tips)},
                        {"label": "OVER Value", "value": len(over_tips)},
                        {"label": "Last Scan",
                         "value": df["updated_at"].iloc[0] if "updated_at" in df.columns and not df.empty else "—"}],
                       per_row=4)

        ui.section_header("Active live signals", "One card per signal: the minute bar runs to 90'. Fair price is "
                          "the Poisson price; value exists only if your bookmaker pays more.")
        cols = st.columns(2)
        for i, (_, row) in enumerate(df.iterrows()):
            v = LV.signal_values(row)
            LV.live_match_card(match=v["match"], league=v["league"], minute=v["minute"], score=v["score"],
                               icon=v["icon"], chips=LV.signal_chip(v["signal"]), stats=v["stats"], note=v["note"],
                               container=cols[i % 2])

        with st.expander("📊 Raw data"):
            cols_ = ["league", "match", "score", "elapsed_mins", "signal_type", "bet",
                     "fair_under_odds", "fair_over_odds", "live_p_under", "live_p_over",
                     "pre_p_over", "lam_remaining"]
            ui.compact_table(df[[c for c in cols_ if c in df.columns]], LV.raw_config())

    ui.section_header(f"All monitored games ({len(games)})")
    if games.empty:
        ui.empty_state("NO_DATA", "No live games right now in our tracked leagues.")
        return
    signal_matches = set(df["match"].tolist()) if not df.empty and "match" in df.columns else set()
    cols = st.columns(2)
    for i, (_, g) in enumerate(games.iterrows()):
        v = LV.game_values(g, signal_matches)
        LV.live_match_card(match=v["match"], league=v["league"], minute=v["minute"], score=v["score"],
                           chips=v["chips"], stats=v["stats"], note=v["note"], container=cols[i % 2])


# ── Tab 2: Half-Time ────────────────────────────────────────────────────────────
def render_ht():
    st.caption("Model predictions for HT OVER/UNDER 0.5 and 1.5 · Standard-format leagues only")
    df = load_ht()
    if df.empty:
        ui.empty_state("NO_DATA", "⏳ No HT predictions available. Only standard-format leagues have half-time data.")
        return

    strong_over05 = df[df["p_ht_over05"] >= 0.75]
    strong_under05 = df[df["p_ht_over05"] <= 0.30]
    strong_over15 = df[df["p_ht_over15"] >= 0.60] if "p_ht_over15" in df.columns else pd.DataFrame()
    ui.metric_grid([{"label": "Total Fixtures", "value": len(df)},
                    {"label": "⚡ Strong OVER 0.5", "value": len(strong_over05), "help": "P(HT OVER 0.5) ≥ 75%"},
                    {"label": "🧊 Strong UNDER 0.5", "value": len(strong_under05), "help": "P(HT OVER 0.5) ≤ 30%"},
                    {"label": "🔥 Strong OVER 1.5", "value": len(strong_over15), "help": "P(HT OVER 1.5) ≥ 60%"}],
                   per_row=4)
    ui.space("small")

    col1, col2 = st.columns([1, 2])
    with col1:
        show = st.selectbox("Show", ["All signals", "Strong OVER 0.5 only", "Strong UNDER 0.5 only",
                                     "Strong OVER 1.5 only", "All fixtures"], key="ht_show")
    with col2:
        leagues = sorted(df["league"].unique())
        league_filter = st.multiselect("League", leagues, default=leagues, key="ht_league")
    df = df[df["league"].isin(league_filter)]

    has15 = "p_ht_over15" in df.columns
    if show == "Strong OVER 0.5 only":
        df = df[df["p_ht_over05"] >= 0.75]
    elif show == "Strong UNDER 0.5 only":
        df = df[df["p_ht_over05"] <= 0.30]
    elif show == "Strong OVER 1.5 only" and has15:
        df = df[df["p_ht_over15"] >= 0.60]
    elif show == "All signals":
        mask = (df["p_ht_over05"] >= 0.75) | (df["p_ht_over05"] <= 0.30)
        if has15:
            mask = mask | (df["p_ht_over15"] >= 0.60)
        df = df[mask]
    if df.empty:
        ui.empty_state("NO_DATA", "No fixtures match your filters.")
        return

    ui.section_header(f"{len(df)} Fixture(s)")
    cols = st.columns(2)
    for i, (_, row) in enumerate(df.iterrows()):
        v = LV.ht_values(row)
        LV.live_match_card(match=v["match"], eyebrow_text=v["eyebrow"], chips=v["chips"], stats=v["stats"],
                           container=cols[i % 2])
    st.caption("AUC 0.567 (HT OVER 0.5) · AUC 0.569 (HT OVER 1.5) · standard-format leagues only")


# ── Tab 3: History ──────────────────────────────────────────────────────────────
def render_history():
    st.caption("All past in-play signals — which types fire most and whether they'd have won.")
    df = load_history()
    if df.empty:
        ui.empty_state("NO_DATA", "⏳ No live signal history yet. Signals are logged automatically when the scanner fires.")
        return

    lookup = load_results_lookup()

    def _lookup_result(row):
        m = str(row.get("match", ""))
        if " vs " not in m or pd.isna(row["date"]):
            return (None, None)
        h, a = m.split(" vs ", 1)
        for off in (0, -1, 1):
            key = (_norm(h), _norm(a), (pd.Timestamp(row["date"]) + pd.Timedelta(days=off)).strftime("%Y-%m-%d"))
            if key in lookup:
                return lookup[key]
        return (None, None)

    # Prefer scores/results written by the grader (update_results.py) — the match DB has no
    # current-season new-format scores, so the on-the-fly join below is only a fallback.
    _res = df.apply(_lookup_result, axis=1, result_type="expand")
    join_ft = pd.to_numeric(pd.Series(list(_res[0]), index=df.index), errors="coerce")
    join_ht = pd.to_numeric(pd.Series(list(_res[1]), index=df.index), errors="coerce")

    ft = pd.to_numeric(df["final_total"], errors="coerce") if "final_total" in df.columns \
        else pd.Series(pd.NA, index=df.index)
    ft = ft.where(ft.notna(), join_ft)
    ht = pd.to_numeric(df["ht_total"], errors="coerce") if "ht_total" in df.columns \
        else pd.Series(pd.NA, index=df.index)
    ht = ht.where(ht.notna(), join_ht)
    df["final_total"] = ft

    computed = [_settle(b, t, h) for b, t, h in zip(df["bet"], ft, ht)]
    if "result" in df.columns:
        stored = df["result"].astype(str).str.strip().str.upper()
        df["result"] = [s if s in ("WIN", "LOSS") else c for s, c in zip(stored, computed)]
    else:
        df["result"] = computed

    settled = df[df["result"].isin(["WIN", "LOSS"])]
    pending = int((~df["result"].isin(["WIN", "LOSS"])).sum())
    ui.section_header("Results — did the live signals win?")
    if settled.empty:
        ui.empty_state("NO_DATA", f"No signals settled yet — {pending} awaiting final scores.")
    else:
        n = len(settled); wins = int((settled["result"] == "WIN").sum())
        ui.metric_grid([{"label": "Settled", "value": n}, {"label": "Record", "value": f"{wins}W – {n - wins}L"},
                        {"label": "Hit rate", "value": f"{wins / n * 100:.0f}%"}, {"label": "Pending", "value": pending}],
                       per_row=4)
        by_type = (settled.assign(win=(settled["result"] == "WIN").astype(int))
                   .groupby("signal_type").agg(bets=("win", "size"), wins=("win", "sum")).reset_index())
        by_type["hit_rate_%"] = (by_type["wins"] / by_type["bets"] * 100).round(0)
        ui.compact_table(by_type.sort_values("bets", ascending=False),
                         {"bets": ui.num_col("bets", fmt="%d"), "wins": ui.num_col("wins", fmt="%d"),
                          "hit_rate_%": ui.num_col("hit_rate_%", fmt="%.0f")})
        st.caption("⚠️ Small sample — directional only. U/O 2.5 settled on full-time; HT bets on half-time goals.")
        if ui.sample_marker(n):
            st.markdown(ui.sample_marker(n))

    total = len(df); matches = df["match"].nunique() if "match" in df.columns else 0
    ui.space("small")
    ui.metric_grid([{"label": "Total Signals", "value": total}, {"label": "Unique Matches", "value": matches},
                    {"label": "First Signal", "value": df["date"].min().strftime("%b %d") if not df.empty else "—"},
                    {"label": "Latest Signal", "value": df["date"].max().strftime("%b %d, %Y") if not df.empty else "—"}],
                   per_row=4)

    ui.section_header("Signal type breakdown")
    if "signal_type" in df.columns:
        sig_counts = df["signal_type"].value_counts().reset_index()
        sig_counts.columns = ["Signal Type", "Count"]
        col1, col2 = st.columns([1, 2])
        with col1:
            ui.compact_table(sig_counts, {"Count": ui.num_col("Count", fmt="%d")})
        with col2:
            st.altair_chart(ui.coverage_bars(sig_counts, "Signal Type", "Count", title="Signal frequency",
                                             h=28 * len(sig_counts) + 60), width="stretch")

    ui.space("small")
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        sig_types = sorted(df["signal_type"].unique()) if "signal_type" in df.columns else []
        sig_filter = st.multiselect("Signal type", sig_types, default=sig_types, key="hist_sig")
    with col_f2:
        leagues = sorted(df["league"].unique()) if "league" in df.columns else []
        league_filter = st.multiselect("League", leagues, default=leagues, key="hist_league")
    filtered = df.copy()
    if sig_filter and "signal_type" in filtered.columns:
        filtered = filtered[filtered["signal_type"].isin(sig_filter)]
    if league_filter and "league" in filtered.columns:
        filtered = filtered[filtered["league"].isin(league_filter)]

    ui.section_header(f"{len(filtered)} Signal(s)")
    display_cols = ["date", "league", "match", "score", "elapsed_mins", "signal_type", "bet",
                    "result", "final_total", "fair_under_odds", "fair_over_odds",
                    "live_p_under", "live_p_over", "pre_p_over"]
    show_cols = [c for c in display_cols if c in filtered.columns]
    ui.compact_table(
        filtered[show_cols].rename(columns={
            "elapsed_mins": "min", "signal_type": "signal", "final_total": "FT goals",
            "fair_under_odds": "fair_U", "fair_over_odds": "fair_O",
            "live_p_under": "P(U)", "live_p_over": "P(O)", "pre_p_over": "pre P(O)",
        }), LV.history_config())


# ── Layout ──────────────────────────────────────────────────────────────────────
_tips, _games = load_tips(), load_games()
_scan = LV.last_scan(_tips) or LV.last_scan(_games)
ui.page_header("Live Center", "In-play signals, half-time predictions and the signal history. The scanner decides "
               "every signal; this page only shows it.", eyebrow_text="Live · in-play",
               freshness=ui.status_chip("UNKNOWN", label=f"LAST SCAN {_scan} (time of day recorded by the scanner)")
               if _scan else ui.status_chip("NO_DATA", label="NO SCAN RECORDED"))
tab_live, tab_ht, tab_hist = st.tabs(["⚡ Live Now", "⏱ Half-Time", "📜 History"])
with tab_live:
    render_live(_tips, _games)
with tab_ht:
    render_ht()
with tab_hist:
    render_history()


# ── Disclaimer & Terms (shown on every dashboard page) ──
from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()
