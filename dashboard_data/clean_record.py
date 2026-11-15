"""THE Clean Record split for every dashboard results figure (owner 2026-11-11).

Owner, 2026-11-11: "we need to update the dashboard with all the new models and fixes ... adjust to count
from 0 after all the fixing (to see the difference later)".

ONE DATE. config.CLEAN_RECORD_START (stored once in config/trustworthy_performance.json clean_record_start_utc,
owner decision 2026-11-11: 2026-11-21, the first round on the clean models). Nothing here has a default.

TWO RECORDS, NEVER MIXED.
  CLEAN    the Clean Record: a tip GENERATED on or after the start AND played on or after it (the rule of
           src/trustworthy_performance.assign_live, view C). Every headline counts only these rows. Before the
           start there are none, and every headline says "no settled tips yet", never an old number.
  BEFORE   "Before the repairs (pre-start, old models and tiers)": a tip generated before the start, or a tip
           without a generation time whose match was played before it. Shown only in its own section, for
           comparison.
  UNPLACED a row neither rule can place (no generation time and a match on or after the start, or a tip
           generated after the start for a match played before it). Counted and shown, never in either total.

A tip dated before the start can never count in a Clean Record total: CLEAN needs BOTH times on or after it.

FIGURES. summary() gives, for one record's settled rows: n, wins, hit rate, units (flat 1u P/L), ROI, the
matchday block-bootstrap 95% CI of the ROI (the Clean Record framework's own settings, config/
trustworthy_performance.json "bootstrap") and the owner's minimums status (NOT_ENOUGH_DATA below them).
The function refuses rows of two records (MixedRecordsError), like trustworthy_performance.metrics.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TP_CONFIG = ROOT / "config" / "trustworthy_performance.json"

CLEAN, BEFORE, UNPLACED = "CLEAN", "BEFORE", "UNPLACED"
RECORDS = (CLEAN, BEFORE)
SETTLED = ("WIN", "LOSS", "PUSH", "HALF_WIN", "HALF_LOSS")


class MixedRecordsError(ValueError):
    """A figure was asked for over rows of the Clean Record and of the pre-repair record together."""


# ── the date and the words ────────────────────────────────────────────────────────────────────────
def start_date() -> str:
    """'YYYY-MM-DD' of the owner's Clean Record start (config.CLEAN_RECORD_START)."""
    import config
    return str(config.CLEAN_RECORD_START)[:10]


def start_ts() -> pd.Timestamp:
    """The start as a UTC timestamp (00:00 UTC of the start day)."""
    return pd.Timestamp(start_date()).tz_localize("UTC")


def started(now=None) -> bool:
    """True once the start day has begun (UTC)."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    return now >= start_ts()


def clean_label() -> str:
    return f"Clean Record (from {start_date()})"


def before_label() -> str:
    return f"Before the repairs (pre-{start_date()}, old models and tiers)"


def empty_text(what: str = "tips") -> str:
    return (f"No settled {what} yet. The Clean Record counts from {start_date()}, the first round on the clean "
            "models; nothing from before it is counted here.")


BEFORE_NOTE = ("For comparison only: tips of the UNREPAIRED system (old models, global bars and the retired "
               "Big / Medium / Small gap tiers). Never part of a Clean Record total.")


# ── the split ──────────────────────────────────────────────────────────────────────────────────────
def assign(d: pd.DataFrame, gen_col: str | None, date_col: str) -> pd.Series:
    """CLEAN / BEFORE / UNPLACED per row (see the module doc). `gen_col` None or absent = no generation
    time recorded on that ledger: then only the match date can place a row, and only as BEFORE."""
    s = start_ts()
    day = pd.to_datetime(d[date_col] if date_col in d.columns else pd.Series(pd.NaT, index=d.index),
                         errors="coerce", format="mixed")
    day = day.dt.tz_localize("UTC") if day.dt.tz is None else day.dt.tz_convert("UTC")
    if gen_col and gen_col in d.columns:
        gen = pd.to_datetime(d[gen_col], errors="coerce", utc=True, format="mixed")
    else:
        gen = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns, UTC]")
    out = pd.Series(UNPLACED, index=d.index, dtype=object)
    out[gen.notna() & (gen < s)] = BEFORE
    out[gen.isna() & day.notna() & (day < s)] = BEFORE
    out[gen.notna() & (gen >= s) & day.notna() & (day >= s)] = CLEAN
    return out


def split(d: pd.DataFrame, gen_col: str | None, date_col: str) -> dict:
    """{CLEAN: rows, BEFORE: rows, UNPLACED: rows} (each a copy carrying `_record`)."""
    if d is None or not len(d):
        e = (d if d is not None else pd.DataFrame()).assign(_record=pd.Series(dtype=object))
        return {CLEAN: e, BEFORE: e.copy(), UNPLACED: e.copy()}
    r = assign(d, gen_col, date_col)
    d = d.assign(_record=r.to_numpy())
    return {k: d[d["_record"] == k] for k in (CLEAN, BEFORE, UNPLACED)}


# ── figures ────────────────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _tp_cfg() -> dict:
    return json.loads(TP_CONFIG.read_text(encoding="utf-8"))


def minimums() -> dict:
    from src import trustworthy_performance as TP
    return TP.minimums(_tp_cfg())


def summary(d: pd.DataFrame, *, pnl: str, date: str, key: str, result: str | None = None) -> dict:
    """The record's figures over its SETTLED rows (pnl recorded). `result` (optional) names a result column;
    VOID / pending rows are then left out even when a P/L is stored. Hit = share of rows with P/L > 0."""
    if d is not None and len(d) and "_record" in d.columns:
        recs = set(d["_record"].dropna().unique())
        if len(recs) > 1:
            raise MixedRecordsError(f"{key}: rows of {sorted(recs)} - the Clean Record and the pre-repair record "
                                    "are never counted together")
    out = {"n": 0, "wins": 0, "hit": None, "units": None, "roi": None, "ci": (None, None), "matchdays": 0,
           "status": "NO_SETTLED"}
    if d is None or not len(d):
        return out
    p = pd.to_numeric(d[pnl], errors="coerce")
    keep = p.notna()
    if result and result in d.columns:
        keep &= d[result].astype(str).str.strip().str.upper().isin(SETTLED)
    p = p[keep]
    n = int(len(p))
    if not n:
        return out
    from src import trustworthy_performance as TP
    from src import units as U
    days = pd.to_datetime(d.loc[p.index, date], errors="coerce").dt.strftime("%Y-%m-%d").fillna("").to_numpy()
    bs = _tp_cfg()["bootstrap"]
    lo, hi = TP.block_ci(p.to_numpy(float), np.ones(n), days, key, int(bs["draws"]), float(bs["ci"]))
    mins = minimums()
    nd = len(set(days))
    units = U.total(p)
    return {"n": n, "wins": int((p > 0).sum()), "hit": float((p > 0).mean()), "units": units,
            "roi": units / n, "ci": (lo, hi), "matchdays": nd,
            "status": "OK" if n >= mins["settled"] and nd >= mins["matchdays"] else "NOT_ENOUGH_DATA"}


def fmt_ci(ci) -> str:
    lo, hi = ci if ci else (None, None)
    if lo is None or hi is None:
        return "—"
    return f"{100 * lo:+.1f}% … {100 * hi:+.1f}%"


def row(name: str, s: dict) -> dict:
    """One display row of a summary (the same columns everywhere)."""
    return {"Record": name, "Settled": s["n"], "Won": s["wins"],
            "Hit %": None if s["hit"] is None else round(100 * s["hit"], 1),
            "Units (flat 1u)": None if s["units"] is None else round(s["units"], 2),
            "ROI %": None if s["roi"] is None else round(100 * s["roi"], 1),
            "ROI 95% CI": fmt_ci(s["ci"]), "Matchdays": s["matchdays"], "Evidence": s["status"]}


def comparison(clean: dict, before: dict) -> pd.DataFrame:
    """Clean Record vs Before the repairs, same metrics side by side (one row per record)."""
    return pd.DataFrame([row(clean_label(), clean), row(before_label(), before)])


TABLE_CONFIG_KEYS = ("Hit %", "Units (flat 1u)", "ROI %")


def table_config(ui) -> dict:
    return {"Hit %": ui.num_col("Hit %", fmt="%.1f%%"), "Units (flat 1u)": ui.num_col("Units (flat 1u)", fmt="%+.2f"),
            "ROI %": ui.num_col("ROI %", fmt="%+.1f%%"),
            "ROI 95% CI": ui.text_col("ROI 95% CI", help="Matchday block bootstrap, 2,000 draws, 95% "
                                      "(config/trustworthy_performance.json). '—' below 2 matchdays."),
            "Evidence": ui.text_col("Evidence", help="NOT_ENOUGH_DATA below the owner's minimums (30 settled, "
                                    "8 matchdays; config/trustworthy_performance.json).")}


# ── page pieces ────────────────────────────────────────────────────────────────────────────────────
def banner(st, ui) -> None:
    """One line under a page header: which record the headline figures are."""
    if started():
        st.caption(f"**{clean_label()}** — every headline figure on this page counts only tips generated and "
                   f"played on or after {start_date()}. The pre-repair record is in its own section below.")
    else:
        st.caption(f"**{clean_label()}** — starts {start_date()} (the first round on the clean models). Until "
                   "then every Clean Record figure is 0 / no settled tips yet. The pre-repair record is in its own "
                   "section below, for comparison only.")


# ── team ledgers (today_view.load_settled rows: _pnl, _dt, _gen, _staked) ──────────────────────────────
def team_split(settled: pd.DataFrame) -> dict:
    """today_view.load_settled() rows split into the two records (`_gen` = generated_at, `_dt` = match date)."""
    return split(settled, "_gen", "_dt")


def team_summaries(settled: pd.DataFrame) -> dict:
    """{record: {'staked': summary, 'all': summary}} over today_view.load_settled() rows. 'staked' = THE
    staked definition (src/staked.py, a recorded policy stake > 0); 'all' = every settled tip, paper + staked,
    flat 1u (the paper track)."""
    sp = team_split(settled)
    out = {}
    for rec in RECORDS:
        d = sp[rec]
        st_ = d[d["_staked"].astype(bool)] if len(d) and "_staked" in d.columns else d.iloc[0:0]
        out[rec] = {"staked": summary(st_, pnl="_pnl", date="_dt", key=f"{rec}|team|staked"),
                    "all": summary(d, pnl="_pnl", date="_dt", key=f"{rec}|team|all")}
    out[UNPLACED] = int(len(sp[UNPLACED]))
    return out


def group_table(d: pd.DataFrame, by: str, *, pnl: str, date: str, key: str) -> pd.DataFrame:
    """One summary row per value of `by` over ONE record's rows (n, hit, units, ROI, CI)."""
    cols = [by, "Settled", "Won", "Hit %", "Units (flat 1u)", "ROI %", "ROI 95% CI", "Matchdays", "Evidence"]
    if d is None or not len(d):
        return pd.DataFrame(columns=cols)
    rows = []
    for k, g in d.groupby(by, dropna=False, sort=True):
        s = summary(g, pnl=pnl, date=date, key=f"{key}|{k}")
        if s["n"]:
            r = row(str(k), s)
            r[by] = r.pop("Record")
            rows.append(r)
    return pd.DataFrame(rows, columns=cols).sort_values("Settled", ascending=False).reset_index(drop=True)


def with_cell(d: pd.DataFrame) -> pd.DataFrame:
    """today_view.load_settled rows + `Cell` (league · market-side · track), the bars cell a tip belongs to."""
    from dashboard_data import cells_view as CV
    if d is None or not len(d):
        return (d if d is not None else pd.DataFrame()).assign(Cell=pd.Series(dtype=object))
    keys = [CV.key_of(m, mk, sd, lg) for m, mk, sd, lg in zip(d["_model"], d["_mkt"], d["_side"], d["league"])]
    return d.assign(Cell=[f"{lg} · {CV.MSIDE_LABEL.get(ms, '—') if ms else '—'} · {tr}" for tr, ms, lg in keys])


# ── props (paper research) and TOP-3 (a named paper track) ────────────────────────────────────────────
def props_split(led: pd.DataFrame) -> dict:
    """player_ledger tip rows (props_view.tips: every non-AVOID row) split on signal_date + match_date."""
    from dashboard_data import props_view as PV
    if led is None or not len(led):
        return split(pd.DataFrame(columns=["signal_date", "match_date", "pnl", "result"]), "signal_date", "match_date")
    t = PV.tips(PV.prepare_ledger(led))
    return split(t, "signal_date", "match_date")


def props_summaries(led: pd.DataFrame) -> dict:
    sp = props_split(led)
    return {rec: summary(sp[rec], pnl="pnl", date="match_date", key=f"{rec}|props", result="result")
            for rec in RECORDS} | {UNPLACED: int(len(sp[UNPLACED]))}


def top3_split(ledger: pd.DataFrame) -> dict:
    """output/top3_league_ledger.csv picks split on issued_at + match_date (P/L = flat_pnl)."""
    if ledger is None or not len(ledger):
        return split(pd.DataFrame(columns=["issued_at", "match_date", "flat_pnl"]), "issued_at", "match_date")
    return split(ledger, "issued_at", "match_date")


def top3_summaries(ledger: pd.DataFrame) -> dict:
    sp = top3_split(ledger)
    return {rec: summary(sp[rec], pnl="flat_pnl", date="match_date", key=f"{rec}|top3") for rec in RECORDS} \
        | {UNPLACED: int(len(sp[UNPLACED]))}


# ── the daily Best bets / Safest / Value lists (src/top_picks, frozen each morning) ───────────────────
def lists_frame(graded: pd.DataFrame) -> pd.DataFrame:
    """src.top_picks.graded() rows with a flat-1u `pnl` (WIN: odds - 1, LOSS: -1; every other result has no
    P/L) and the frozen `day` as both generation day and match day for the split."""
    if graded is None or not len(graded):
        return pd.DataFrame(columns=["list", "day", "pnl", "result"])
    g = graded.copy()
    r = g["result"].astype(str).str.upper()
    o = pd.to_numeric(g.get("odds"), errors="coerce")
    g["pnl"] = np.where(r.eq("WIN") & (o > 1), o - 1.0, np.where(r.eq("LOSS"), -1.0, np.nan))
    return g


def lists_table(graded: pd.DataFrame, record: str) -> pd.DataFrame:
    """One row per list over ONE record's settled picks (n, hit, units, ROI, CI)."""
    g = lists_frame(graded)
    if not len(g):
        return pd.DataFrame(columns=["list", "Settled"])
    part = split(g, "day", "day")[record]
    return group_table(part, "list", pnl="pnl", date="day", key=f"{record}|lists")
