"""The morning picture pack — one Telegram album instead of a stream of long text messages.

    from telegram_bot.daily_pack import send_daily_pack, sharp_instant

Owner request 2026-10-13: "less message spam and more visual tables". Instant messages are kept
ONLY for new tips (every model) and really big sharp moves; everything else is one album a day:

    1. Today's tips             (telegram_bot.tips_table, unchanged)
    2. Yesterday's results      every tip settled yesterday, WON/LOST, staked P/L by market
    3. Record                   staked P/L per market, all-time and last 7 days, + paper VALUABLE
    4. Player props today       priced props first (model vs bookmaker), then top model-only
    5. Sharp movements today    money moves on today's fixtures, bad data filtered out
    6. Pro tips scoreboard      1X2 + Bet Builder hit rates (was Pro's own daily text message)

send_weekly_pack() is the Monday album (week by market, by league, props week, scoreboard) that
replaces the two weekly text summaries. Its "Leagues" and "Last week by league" pictures read
dashboard_data/leagues.py - the same functions as the app's Leagues page (owner 2026-10-25).

Same rows and same bases as the dashboard: paper leagues out, staked = SNIPER + MARKSMAN, VALUABLE
shown as paper, performance from PERFORMANCE_CUTOFF_DATE by match date, first tip per fixture.

Dedup key DIGEST|<utc-date> in telegram_bot/notified.json — the same key the text digest used, so
daily_summary.yml's morning-window gate keeps working unchanged. Every image is best effort: one
failing image is skipped and logged; the rest still go out.
"""
from __future__ import annotations

import datetime as _dt
import io
import os

import pandas as pd

import config
from telegram_bot.tips_table import _fit, _font, render_pngs, todays_rows

STAKED = ("SNIPER", "MARKSMAN")
TIER_COLOUR = {"SNIPER": (196, 43, 43), "MARKSMAN": (217, 119, 6), "VALUABLE": (37, 99, 235),
               "PAPER": (107, 114, 128), "AVOID": (156, 163, 175)}
GREEN, RED, INK, MUTED = (21, 128, 61), (185, 28, 28), (17, 24, 39), (107, 114, 128)
SIDE_LABEL = {"btts": "BTTS", "over15": "O/U 1.5", "over35": "O/U 3.5"}
PROP_LABEL = {"goals": "Anytime scorer", "goals2": "2+ goals", "assists": "Assist",
              "sot": "1+ shot on target", "sot2": "2+ SOT", "sot3": "3+ SOT", "sot4": "4+ SOT",
              "cards": "Booked"}
ROWS_PER_IMAGE = 22


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_table(title: str, columns: list[tuple[str, int]], rows: list[dict],
                 subtitle: str = "", footer: str = "") -> list[bytes]:
    """Generic PNG table. Each row: {"cells": [...], "stripe": rgb|None, "colours": {col: rgb},
    "bold": {col, ...}, "band": "text"} — a "band" row is a full-width section heading."""
    from PIL import Image, ImageDraw
    pad, row_h, head_h, foot_h = 18, 34, 34, 30
    width = pad * 2 + sum(w for _, w in columns)
    f_title, f_sub, f_head, f_cell, f_bold, f_foot = (_font(22, True), _font(14), _font(15, True),
                                                      _font(15), _font(15, True), _font(12))
    # Subtitle wrapped onto as many lines as it needs (max 3) instead of being cut off.
    sub_lines: list[str] = []
    if subtitle:
        _probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        cur = ""
        for word in subtitle.split():
            nxt = f"{cur} {word}".strip()
            if _probe.textlength(nxt, font=f_sub) <= width - 2 * pad:
                cur = nxt
            else:
                sub_lines.append(cur)
                cur = word
        sub_lines.append(cur)
        sub_lines = sub_lines[:3]
    title_h = 54 + 20 * len(sub_lines)
    pages = [rows[i:i + ROWS_PER_IMAGE] for i in range(0, len(rows), ROWS_PER_IMAGE)] or [[]]
    out = []
    for n, page in enumerate(pages, start=1):
        height = title_h + head_h + row_h * max(len(page), 1) + foot_h + pad
        img = Image.new("RGB", (width, height), (255, 255, 255))
        dr = ImageDraw.Draw(img)
        dr.text((pad, 14), title + (f"  ({n}/{len(pages)})" if len(pages) > 1 else ""),
                font=f_title, fill=INK)
        for k, line in enumerate(sub_lines):
            dr.text((pad, 46 + 20 * k), _fit(dr, line, f_sub, width - 2 * pad), font=f_sub, fill=MUTED)
        y = title_h
        dr.rectangle([0, y, width, y + head_h], fill=(31, 41, 55))
        x = pad
        for name, w in columns:
            dr.text((x + 4, y + 8), name, font=f_head, fill=(255, 255, 255))
            x += w
        y += head_h
        if not page:
            dr.text((pad + 4, y + 8), "Nothing to show", font=f_cell, fill=MUTED)
        for i, r in enumerate(page):
            if r.get("band"):
                dr.rectangle([0, y, width, y + row_h], fill=(229, 231, 235))
                dr.text((pad + 4, y + 8), r["band"], font=f_bold, fill=INK)
                y += row_h
                continue
            dr.rectangle([0, y, width, y + row_h], fill=(249, 250, 251) if i % 2 else (255, 255, 255))
            if r.get("stripe"):
                dr.rectangle([0, y, 6, y + row_h], fill=r["stripe"])
            x = pad
            for (name, w), val in zip(columns, r["cells"]):
                font = f_bold if name in r.get("bold", ()) else f_cell
                fill = r.get("colours", {}).get(name, INK)
                dr.text((x + 4, y + 8), _fit(dr, "" if val is None else str(val), font, w - 8),
                        font=font, fill=fill)
                x += w
            y += row_h
        if footer:
            dr.text((pad, y + 8), _fit(dr, footer, f_foot, width - 2 * pad), font=f_foot, fill=MUTED)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        out.append(buf.getvalue())
    return out


def send_album(token: str, chat_id: str, pngs: list[bytes], caption: str = "") -> bool:
    """sendMediaGroup in chunks of 10 (Telegram's cap); a single image uses sendPhoto.
    Never raises; the token is never logged."""
    import json

    import requests
    ok = True
    for start in range(0, len(pngs), 10):
        chunk = pngs[start:start + 10]
        try:
            if len(chunk) == 1:
                r = requests.post(f"https://api.telegram.org/bot{token}/sendPhoto",
                                  data={"chat_id": chat_id, "caption": caption[:1000] if start == 0 else "",
                                        "parse_mode": "HTML"},
                                  files={"photo": ("p.png", chunk[0], "image/png")}, timeout=60)
            else:
                media = [{"type": "photo", "media": f"attach://p{i}"} for i in range(len(chunk))]
                if start == 0 and caption:
                    media[0].update({"caption": caption[:1000], "parse_mode": "HTML"})
                r = requests.post(f"https://api.telegram.org/bot{token}/sendMediaGroup",
                                  data={"chat_id": chat_id, "media": json.dumps(media)},
                                  files={f"p{i}": (f"p{i}.png", b, "image/png") for i, b in enumerate(chunk)},
                                  timeout=90)
            if r.status_code != 200:
                print(f"[daily_pack] album HTTP {r.status_code}: {r.text[:200]}")
                ok = False
        except Exception as e:                                          # noqa: BLE001
            print(f"[daily_pack] album failed: {type(e).__name__}")
            ok = False
    return ok


# ── Data ──────────────────────────────────────────────────────────────────────

def _read(name: str, include_paper: bool = False, usecols=None) -> pd.DataFrame:
    p = config.OUTPUT_DIR / name
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p, low_memory=False, usecols=(lambda c: c in usecols) if usecols else None)
    return d if include_paper else config.drop_paper_leagues(d)


#: The only ledger columns settled_tips() uses - column-projected so the dashboard (a viewer on
#: ~1 GB) never loads the CLV / provenance columns it does not draw.
_TIP_COLS = {"source", "match_date", "league", "home_team", "away_team", "signal_tier", "side", "market",
             "odds", "result", "pnl", "edge_pct", "model_prob"}


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """A numeric column, or NaN when the ledger does not carry it (never an invented number)."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(float("nan"), index=df.index)


def settled_tips(include_paper: bool = False) -> pd.DataFrame:
    """Every live tip on the O/U 2.5 and side-market ledgers, normalised: date, league, match,
    tier, market, odds, result, pnl. EVERY SENT tip counts, exactly as on the dashboard — a first-
    tip-per-fixture filter here made the record read +27.97u against the dashboard's +36.22u on
    2026-10-13, the same two-answers problem the home page had the day before.

    include_paper=True keeps paper-league rows (config.PAPER_LEAGUES) so the Leagues page and
    picture can SHOW them, labelled as paper; every staked total still excludes them
    (dashboard_data/leagues.py). Default False = the behaviour every other image relies on.
    Also carries edge_pct (as recorded) and model_prob (side markets record it; on O/U 2.5 it is
    NaN here - see dashboard_data/leagues.py for the derivation)."""
    parts = []
    m = _read("bets_ledger.csv", include_paper, _TIP_COLS)
    if len(m):
        if "source" in m.columns:
            m = m[m["source"].fillna("live").astype(str) == "live"]
        parts.append(pd.DataFrame({
            "date": m["match_date"].astype(str).str[:10], "league": m["league"],
            "match": m["home_team"].astype(str) + " v " + m["away_team"].astype(str),
            "home_team": m["home_team"], "away_team": m["away_team"],
            "tier": m["signal_tier"], "market": "O/U 2.5",
            "pick": "O/U 2.5 " + m["side"].astype(str).str.upper(),
            "odds": pd.to_numeric(m["odds"], errors="coerce"),
            "result": m["result"], "pnl": pd.to_numeric(m["pnl"], errors="coerce"),
            "edge_pct": _col(m, "edge_pct"), "model_prob": float("nan")}))
    s = _read("side_bets_ledger.csv", include_paper, _TIP_COLS)
    if len(s):
        if "source" in s.columns:
            s = s[s["source"].fillna("live").astype(str) == "live"]
        mk = s["market"].astype(str)
        parts.append(pd.DataFrame({
            "date": s["match_date"].astype(str).str[:10], "league": s["league"],
            "match": s["home_team"].astype(str) + " v " + s["away_team"].astype(str),
            "home_team": s["home_team"], "away_team": s["away_team"],
            "tier": s["signal_tier"], "market": mk.map(SIDE_LABEL).fillna(mk.str.upper()),
            "pick": mk.map(lambda k: "BTTS YES" if k == "btts" else f"{SIDE_LABEL.get(k, k)} OVER"),
            "odds": pd.to_numeric(s["odds"], errors="coerce"),
            "result": s["result"], "pnl": pd.to_numeric(s["pnl"], errors="coerce"),
            "edge_pct": _col(s, "edge_pct"), "model_prob": _col(s, "model_prob")}))
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    d["result"] = d["result"].astype(str).str.upper().str.strip()
    from src.tip_day import add_tip_day                    # "yesterday" = the album that carried it
    d = add_tip_day(d, date_col="date")
    return d[d["tier"].isin(("SNIPER", "MARKSMAN", "VALUABLE"))]


def _res_cell(result: str, pnl) -> tuple[str, tuple]:
    if result == "WIN":
        return "WON", GREEN
    if result == "LOSS":
        return "LOST", RED
    if result in ("VOID", "PUSH"):
        return result.title(), MUTED
    return "pending", MUTED


def results_image(day: str) -> list[bytes]:
    d = settled_tips()
    if d.empty:
        return []
    y = d[d["tip_day"] == day].copy()
    if y.empty:
        return []
    y["_t"] = y["tier"].map({"SNIPER": 0, "MARKSMAN": 1, "VALUABLE": 2})
    y = y.sort_values(["_t", "market", "match"])
    st = y[y["tier"].isin(STAKED) & y["pnl"].notna()]
    pp = y[(y["tier"] == "VALUABLE") & y["pnl"].notna()]
    wins = int((st["result"] == "WIN").sum())
    by_mkt = " · ".join(f"{k} {g['pnl'].sum():+.2f}u" for k, g in st.groupby("market"))
    subtitle = (f"Staked (SNIPER+MARKSMAN): {wins}W-{len(st) - wins}L, {st['pnl'].sum():+.2f}u"
                + (f"  ({by_mkt})" if by_mkt else "")
                + (f"   ·   paper VALUABLE {pp['pnl'].sum():+.2f}u on {len(pp)}" if len(pp) else ""))
    cols = [("League", 200), ("Match", 320), ("Tier", 170), ("Pick", 150), ("Odds", 64),
            ("Result", 90), ("P/L", 76)]
    rows = []
    for _, r in y.iterrows():
        res, rc = _res_cell(r["result"], r["pnl"])
        pnl = "" if pd.isna(r["pnl"]) else f"{r['pnl']:+.2f}"
        rows.append({"cells": [r["league"], r["match"], r["tier"] + ("" if r["tier"] in STAKED else " (paper)"),
                               r["pick"], "" if pd.isna(r["odds"]) else f"{r['odds']:.2f}", res, pnl],
                     "stripe": TIER_COLOUR.get(r["tier"]), "bold": {"Result", "P/L"},
                     "colours": {"Tier": TIER_COLOUR.get(r["tier"], INK), "Result": rc,
                                 "P/L": GREEN if (r["pnl"] or 0) > 0 else RED if (r["pnl"] or 0) < 0 else MUTED}})
    label = pd.Timestamp(day).strftime("%a %d %b")
    return render_table(f"Yesterday's results — {label}", cols, rows, subtitle=subtitle,
                        footer="Flat 1u stakes · VALUABLE is paper (never staked) · paper leagues excluded")


def record_image(today: str) -> list[bytes]:
    d = settled_tips()
    if d.empty:
        return []
    cut = str(getattr(config, "PERFORMANCE_CUTOFF_DATE", "1970-01-01"))[:10]
    d = d[(d["date"] >= cut) & d["pnl"].notna()]
    week = (pd.Timestamp(today) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    cols = [("Market", 170), ("Bets", 70), ("Won-Lost", 110), ("Hit", 70), ("Break-even", 110),
            ("P/L all", 100), ("P/L 7 days", 110)]
    rows = []

    def _row(name, g, stripe=None, paper=False):
        w = int((g["pnl"] > 0).sum())
        be = 100 / g["odds"].mean() if g["odds"].notna().sum() >= 0.8 * len(g) else None
        g7 = g[g["date"] >= week]
        hit = 100 * w / len(g)
        return {"cells": [name, len(g), f"{w}-{len(g) - w}", f"{hit:.0f}%",
                          "" if be is None else f"{be:.0f}%", f"{g['pnl'].sum():+.2f}u",
                          f"{g7['pnl'].sum():+.2f}u ({len(g7)})"],
                "stripe": stripe, "bold": {"P/L all"},
                "colours": {"P/L all": GREEN if g["pnl"].sum() > 0 else RED,
                            "P/L 7 days": GREEN if g7["pnl"].sum() > 0 else RED if len(g7) else MUTED,
                            "Hit": GREEN if be is not None and hit > be else RED if be is not None else INK,
                            "Market": MUTED if paper else INK}}

    st = d[d["tier"].isin(STAKED)]
    rows.append({"band": "Staked — SNIPER + MARKSMAN"})
    for mk, g in sorted(st.groupby("market"), key=lambda kv: -len(kv[1])):
        rows.append(_row(mk, g, TIER_COLOUR["SNIPER"]))
    if len(st):
        rows.append(_row("All staked", st, (31, 41, 55)))
    pp = d[d["tier"] == "VALUABLE"]
    if len(pp):
        rows.append({"band": "Paper — VALUABLE (tracked, never staked)"})
        for mk, g in sorted(pp.groupby("market"), key=lambda kv: -len(kv[1])):
            rows.append(_row(mk, g, TIER_COLOUR["VALUABLE"], paper=True))
    return render_table("Record by market", cols, rows,
                        subtitle=f"Settled since {cut}, flat 1u. Hit is green when above the break-even "
                                 f"rate implied by the average price.",
                        footer="Same basis as the dashboard · paper leagues excluded")


PROPS_SHOWN_LOG = config.OUTPUT_DIR / "props_shown_log.csv"
_SHOWN_COLS = ["date", "shown_at", "section", "rank", "player_id", "player_name", "match", "market",
               "model_prob", "market_odds", "edge_pts"]


def _freeze_props_shown(today: str, pr: pd.DataFrame, un: pd.DataFrame) -> None:
    """Record the props the morning album showed, ONCE per day (first sight), so the next morning's
    results picture grades exactly those players at the odds that were shown. Before this the
    results re-ranked every settled prop by edge at the FINAL price: Haaland to score was #6 of 12 on
    the 2026-10-24 album at 2.14, shortened to 1.70 and dropped out of the next day's results."""
    try:
        old = pd.read_csv(PROPS_SHOWN_LOG, dtype=str) if PROPS_SHOWN_LOG.exists() else pd.DataFrame()
        if len(old) and (old["date"] == today).any():
            return
        now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
        parts = []
        for section, g in (("priced", pr), ("model_only", un)):
            if len(g):
                parts.append(pd.DataFrame({
                    "date": today, "shown_at": now, "section": section, "rank": range(1, len(g) + 1),
                    "player_id": g["player_id"].values if "player_id" in g else None,
                    "player_name": g["player_name"].values, "match": g["match"].values,
                    "market": g["market"].values, "model_prob": g["model_prob"].round(4).values,
                    "market_odds": g["market_odds"].values,
                    "edge_pts": g["_edge"].round(2).values if "_edge" in g else None}))
        if parts:
            new = pd.concat([old, *parts], ignore_index=True).reindex(columns=_SHOWN_COLS)
            new.to_csv(PROPS_SHOWN_LOG, index=False, lineterminator="\n")
            try:                                 # SHADOW contract check (src/contracts); observe only
                from src.contracts.shadow import observe
                observe("props_shown_log.csv (appended rows)",
                        pd.concat(parts, ignore_index=True).reindex(columns=_SHOWN_COLS), "PROPS_SHOWN_ROW")
            except Exception:                                         # noqa: BLE001
                pass
    except Exception as e:                                              # noqa: BLE001 - never blocks the album
        print(f"[daily_pack] could not log shown props: {e}")


def props_image(today: str, freeze: bool = False) -> list[bytes]:
    t = _read("player_tips.csv")
    if t.empty:
        return []
    from src.tip_day import tip_day_of                     # player_tips carries its own kickoff_utc
    t["_tip_day"] = [tip_day_of(k, d) for k, d in zip(t.get("kickoff_utc", pd.Series(None, index=t.index)),
                                                      t["date"].astype(str))]
    t = t[t["_tip_day"] == today].copy()
    if t.empty:
        return []
    for c in ("model_prob", "market_odds", "ev"):
        t[c] = pd.to_numeric(t.get(c), errors="coerce")
    t["_priced"] = t["market_odds"].notna() & (t["market_odds"] > 1.0)
    t["_ko"] = pd.to_datetime(t.get("kickoff_utc"), errors="coerce")
    if {"player_id", "market"} <= set(t.columns):
        t = (t.assign(_hk=t["_ko"].notna())
              .sort_values(["_priced", "_hk"], ascending=False, kind="mergesort")
              .drop_duplicates(["player_id", "market"], keep="first"))
    pr = t[t["_priced"]].assign(_edge=lambda x: (x["model_prob"] - 1 / x["market_odds"]) * 100) \
        .sort_values("_edge", ascending=False).head(12)
    un = t[~t["_priced"]].sort_values("model_prob", ascending=False).head(ROWS_PER_IMAGE - len(pr) - 2)
    if freeze:
        _freeze_props_shown(today, pr, un)
    cols = [("Time", 62), ("Player", 190), ("Match", 300), ("Market", 160), ("Model", 70),
            ("Odds", 70), ("Edge", 86)]
    rows = []

    def _time(v):
        return v.strftime("%H:%M") if pd.notna(v) else "–"

    if len(pr):
        rows.append({"band": f"Priced by bookmakers — {int(t['_priced'].sum())} today, best {len(pr)} by edge"})
        for _, r in pr.iterrows():
            e = r["_edge"]
            rows.append({"cells": [_time(r["_ko"]), r["player_name"], r["match"],
                                   PROP_LABEL.get(r["market"], r["market"]), f"{r['model_prob'] * 100:.0f}%",
                                   f"{r['market_odds']:.2f}", f"{e:+.1f} pts"],
                         "stripe": GREEN if e > 0 else RED, "bold": {"Edge"},
                         "colours": {"Edge": GREEN if e > 0 else RED}})
    if len(un):
        rows.append({"band": f"Model only — no bookmaker price ({int((~t['_priced']).sum())} today, top {len(un)})"})
        for _, r in un.iterrows():
            rows.append({"cells": [_time(r["_ko"]), r["player_name"], r["match"],
                                   PROP_LABEL.get(r["market"], r["market"]), f"{r['model_prob'] * 100:.0f}%",
                                   "–", "–"], "colours": {"Odds": MUTED, "Edge": MUTED}})
    label = pd.Timestamp(today).strftime("%a %d %b")
    return render_table(f"Player props — {label}", cols, rows,
                        subtitle="Edge = model probability minus the bookmaker's implied probability. "
                                 "Props are paper research — never staked.",
                        footer="Times UTC · paper leagues excluded · not betting advice")


def _pro_csv(rel: str) -> pd.DataFrame:
    """A Pro output, from a sibling checkout or Pro's public repo. Empty on any failure."""
    try:
        for base in (os.getenv("PRO_LOCAL"), "../pro", "../wowzaV9-Pro"):
            if base and os.path.exists(os.path.join(base, rel)):
                return pd.read_csv(os.path.join(base, rel), low_memory=False)
        return pd.read_csv("https://raw.githubusercontent.com/NevixAA/wowzaV9-Pro/main/" + rel, low_memory=False)
    except Exception as e:                                              # noqa: BLE001
        print(f"[daily_pack] {rel} unavailable: {type(e).__name__}")
        return pd.DataFrame()


def one_x_two_image(now: _dt.datetime) -> list[bytes]:
    """Pro's paper 1X2 picks in the morning album (owner 2026-10-20: "if there is 1x2 bet - add it to the
    daily"). Pro selects them (top slice by edge) and records them; it no longer sends its own message.
    Upcoming picks (next 36 h) plus yesterday's settled ones. PAPER — never staked."""
    d = _pro_csv("output/paper_1x2.csv")
    if d.empty or "notified_pick" not in d.columns:
        return []
    d = d[d["notified_pick"].notna()].copy()
    if d.empty:
        return []
    d["_ko"] = pd.to_datetime(d["kickoff_utc"], errors="coerce", utc=True)
    nowts = pd.Timestamp(now)
    up = d[(d["_ko"] > nowts) & (d["_ko"] <= nowts + pd.Timedelta(hours=36))].sort_values("_ko")
    yday = (nowts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    done = d[(d["match_date"].astype(str).str[:10] == yday) & pd.to_numeric(d["result"], errors="coerce").notna()]
    if up.empty and done.empty:
        return []
    pick_lbl = {"HOME": "Home win", "DRAW": "Draw", "AWAY": "Away win"}
    res_code = {"HOME": 0, "DRAW": 1, "AWAY": 2}
    cols = [("Time", 112), ("League", 190), ("Match", 320), ("Pick", 110), ("Odds", 70), ("Edge", 80),
            ("Result", 90)]
    rows = []
    if len(up):
        rows.append({"band": f"Upcoming 1X2 picks — {len(up)}"})
        for _, r in up.iterrows():
            ev = pd.to_numeric(r.get("notified_ev"), errors="coerce")
            rows.append({"cells": [r["_ko"].strftime("%a %H:%M"), r["league"], f"{r['home_team']} v {r['away_team']}",
                                   pick_lbl.get(str(r["notified_pick"]), r["notified_pick"]),
                                   f"{float(r['notified_odds']):.2f}", "–" if pd.isna(ev) else f"+{ev * 100:.0f}%", "–"],
                         "colours": {"Result": MUTED}})
    if len(done):
        won = sum(int(pd.to_numeric(r["result"]) == res_code.get(str(r["notified_pick"]), -1)) for _, r in done.iterrows())
        rows.append({"band": f"Yesterday — {won} of {len(done)} won"})
        for _, r in done.iterrows():
            w = int(pd.to_numeric(r["result"])) == res_code.get(str(r["notified_pick"]), -1)
            rows.append({"cells": ["", r["league"], f"{r['home_team']} v {r['away_team']}",
                                   pick_lbl.get(str(r["notified_pick"]), r["notified_pick"]),
                                   f"{float(r['notified_odds']):.2f}", "", "WON" if w else "LOST"],
                         "stripe": GREEN if w else RED, "colours": {"Result": GREEN if w else RED}})
    return render_table("1X2 picks (paper)", cols, rows,
                        subtitle="Pro's 1X2 model, top slice by edge. Paper only - the backtest says this slice "
                                 "has lost; it is tracked forward to settle that.",
                        footer="Times UTC · not betting advice")


def _shown_results(day: str, ledger_day: pd.DataFrame) -> list[bytes] | None:
    """Results of exactly the props yesterday's album showed (props_shown_log.csv), at the odds shown.
    None when nothing was logged for that day, so the older all-props ranking is used instead."""
    try:
        log = pd.read_csv(PROPS_SHOWN_LOG, dtype={"player_id": str}) if PROPS_SHOWN_LOG.exists() else pd.DataFrame()
    except Exception:                                                   # noqa: BLE001
        return None
    log = log[log["date"].astype(str) == day] if len(log) else log
    if log.empty or "player_id" not in ledger_day:
        return None
    led = ledger_day.assign(player_id=ledger_day["player_id"].astype(str).str.replace(r"\.0$", "", regex=True))
    led = led.assign(_done=led["result"].isin(["WIN", "LOSS", "VOID"])).sort_values("_done", ascending=False,
                                                                                    kind="mergesort")
    res = led.drop_duplicates(["player_id", "market"]).set_index(["player_id", "market"])["result"]
    log = log.assign(player_id=log["player_id"].astype(str).str.replace(r"\.0$", "", regex=True))
    log["result"] = [res.get((a, b)) for a, b in zip(log["player_id"], log["market"])]
    for c in ("model_prob", "market_odds"):
        log[c] = pd.to_numeric(log[c], errors="coerce")

    def _cell(r):
        if r == "WIN":
            return "WON", GREEN
        if r == "LOSS":
            return "LOST", RED
        if r == "VOID":
            return "did not play", MUTED
        return "pending", MUTED

    cols = [("Player", 200), ("Match", 300), ("Market", 160), ("Model", 70), ("Odds", 70), ("Result", 120)]
    rows = []
    for section, label in (("priced", "Priced"), ("model_only", "Model only (no price)")):
        g = log[log["section"] == section].sort_values("rank")
        if g.empty:
            continue
        played = g[g["result"].isin(["WIN", "LOSS"])]
        won = int((played["result"] == "WIN").sum())
        band = f"{label} — shown yesterday: {won}-{len(played) - won}"
        if section == "priced" and len(played):
            pl = float((played["market_odds"] - 1).where(played["result"] == "WIN", -1.0).sum())
            band += f" · paper {pl:+.2f}u at the odds shown"
        n_pend = int((~g["result"].isin(["WIN", "LOSS", "VOID"])).sum())
        rows.append({"band": band + (f" · {n_pend} pending" if n_pend else "")})
        for _, r in g.iterrows():
            txt, colour = _cell(r["result"])
            rows.append({"cells": [r["player_name"], r["match"], PROP_LABEL.get(r["market"], r["market"]),
                                   "–" if pd.isna(r["model_prob"]) else f"{r['model_prob'] * 100:.0f}%",
                                   "–" if pd.isna(r["market_odds"]) else f"{r['market_odds']:.2f}", txt],
                         "stripe": colour if r["result"] in ("WIN", "LOSS") else None, "bold": {"Result"},
                         "colours": {"Result": colour}})
    # one line for everything else that settled, so the picture still answers "how did props do overall"
    allp = ledger_day[ledger_day["result"].isin(["WIN", "LOSS"])].copy()
    allp["market_odds"] = pd.to_numeric(allp.get("market_odds"), errors="coerce")
    pr_all = allp[allp["market_odds"] > 1.0]
    if {"player_id", "market"} <= set(pr_all.columns):
        pr_all = pr_all.drop_duplicates(["player_id", "market"])
    if len(pr_all):
        w = int((pr_all["result"] == "WIN").sum())
        need = 100 * (1 / pr_all["market_odds"]).mean()
        rows.append({"band": f"All priced props that day — {w}-{len(pr_all) - w} · hit {100 * w / len(pr_all):.0f}% "
                             f"vs {need:.0f}% needed"})
    label = pd.Timestamp(day).strftime("%a %d %b")
    return render_table(f"Player props — results {label}", cols, rows,
                        subtitle="The same players yesterday's album showed, at the odds shown then. "
                                 "Paper — never staked.",
                        footer="paper leagues excluded · not betting advice")


def props_results_image(day: str) -> list[bytes]:
    """Yesterday's settled player props (owner 2026-10-20: the morning album showed today's props but
    never how yesterday's did — prop results lived only in the Monday pack).

    One row per (player, market), played only (VOID / did-not-play excluded). Priced props are scored
    against the price (hit vs the hit rate the odds needed, paper P/L at 1u); model-only props against
    the model's own probability. Paper research — never staked."""
    p = _read("player_ledger.csv")
    if p.empty:
        return []
    p = p[p["match_date"].astype(str).str[:10] == day].copy()
    shown = _shown_results(day, p)
    if shown is not None:
        return shown
    p = p[p["result"].isin(["WIN", "LOSS"])]
    if p.empty:
        return []
    for c in ("model_prob", "market_odds", "pnl"):
        p[c] = pd.to_numeric(p.get(c), errors="coerce")
    p["_priced"] = p["market_odds"].notna() & (p["market_odds"] > 1.0)
    if {"player_id", "market"} <= set(p.columns):
        p = p.sort_values("_priced", ascending=False, kind="mergesort").drop_duplicates(
            ["player_id", "market"], keep="first")
    pr, un = p[p["_priced"]], p[~p["_priced"]]
    cols = [("Player", 200), ("Match", 300), ("Market", 160), ("Model", 70), ("Odds", 70), ("Result", 90)]
    rows = []
    if len(pr):
        won = int((pr["result"] == "WIN").sum())
        need = 100 * (1 / pr["market_odds"]).mean()
        hit = 100 * won / len(pr)
        # flat 1u at the recorded price (the ledger's pnl column is not filled for every row)
        pl = float((pr["market_odds"] - 1).where(pr["result"] == "WIN", -1.0).sum())
        rows.append({"band": f"Priced — {won}-{len(pr) - won} · hit {hit:.0f}% vs {need:.0f}% needed · "
                             f"paper {pl:+.2f}u at 1u"})
        pr = pr.assign(_edge=pr["model_prob"] - 1 / pr["market_odds"]).sort_values("_edge", ascending=False)
        for _, r in pr.head(12).iterrows():
            w = r["result"] == "WIN"
            rows.append({"cells": [r["player_name"], f"{r['home_team']} v {r['away_team']}",
                                   PROP_LABEL.get(r["market"], r["market"]), f"{r['model_prob'] * 100:.0f}%",
                                   f"{r['market_odds']:.2f}", "WON" if w else "LOST"],
                         "stripe": GREEN if w else RED, "bold": {"Result"},
                         "colours": {"Result": GREEN if w else RED}})
    if len(un):
        won = int((un["result"] == "WIN").sum())
        exp = 100 * un["model_prob"].mean()
        rows.append({"band": f"Model only (no price) — {won} of {len(un)} hit ({100 * won / len(un):.0f}%) "
                             f"vs {exp:.0f}% the model expected"})
        top = un.sort_values("model_prob", ascending=False).head(max(0, ROWS_PER_IMAGE - len(rows) - 2))
        for _, r in top.iterrows():
            w = r["result"] == "WIN"
            rows.append({"cells": [r["player_name"], f"{r['home_team']} v {r['away_team']}",
                                   PROP_LABEL.get(r["market"], r["market"]), f"{r['model_prob'] * 100:.0f}%",
                                   "–", "WON" if w else "LOST"],
                         "colours": {"Odds": MUTED, "Result": GREEN if w else RED}})
    label = pd.Timestamp(day).strftime("%a %d %b")
    return render_table(f"Player props — results {label}", cols, rows,
                        subtitle="Played only (VOID / did not play excluded). Priced props vs the price; "
                                 "model-only props vs the model's own probability. Paper — never staked.",
                        footer="paper leagues excluded · not betting advice")


# ── Sharp movements ───────────────────────────────────────────────────────────

def _sharp_frame() -> pd.DataFrame:
    s = _read("sharp_tips.csv")
    if s.empty:
        return s
    for c in ("opening_odds", "current_odds", "drift_pct"):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    return s


def sharp_artifacts(s: pd.DataFrame) -> pd.Series:
    """True where a 'move' cannot be a real market move. A real move is a REPRICING: when one
    outcome shortens, another drifts. Found 2026-10-13 that most 30%+ moves were not that:
      * two outcomes of one match both shortening hard (Huracan 2.19 -> 1.10 AND Gimnasia
        3.04 -> 1.10 at once) — the prices no longer form a book;
      * every listed outcome of a match drifting together (Argentina 1X2s, all three +20-52%);
      * a price at or below 1.10, the floor such feeds use for suspended/garbage quotes."""
    bad = s["current_odds"].le(1.10) | s["opening_odds"].le(1.10)
    key = s["date"].astype(str) + "|" + s["match"].astype(str)
    g = s.groupby(key)["drift_pct"]
    n_short = g.transform(lambda x: int((x <= -15).sum()))
    n_out = g.transform(lambda x: int((x >= 15).sum()))
    n_rows = g.transform("size")
    bad |= n_short >= 2
    bad |= (n_rows >= 2) & (n_out == n_rows)
    return bad


def sharp_instant(s: pd.DataFrame, min_pct: float | None = None) -> pd.DataFrame:
    """Rows big enough for an INSTANT message: money coming IN (price shortening) by at least
    SHARP_INSTANT_MIN_PCT (owner chose 30, 2026-10-13), on a move that passes sharp_artifacts."""
    if s.empty:
        return s
    if min_pct is None:
        min_pct = float(os.getenv("SHARP_INSTANT_MIN_PCT", "30"))
    ok = ~sharp_artifacts(s)
    return s[ok & (s["drift_pct"] <= -min_pct)]


def sharp_jumps(history: dict | None = None, min_pct: float | None = None,
                max_age_hours: float = 8.0, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """SUDDEN moves for the instant alert (owner, 2026-10-13: "big jump from snapshots, 10% and
    above"): the price change between the LAST TWO captured snapshots, not since opening.

    Raw 10%+ jumps ran ~7/day, but most were the AVERAGE changing composition rather than money:
    the book count jumping 4 -> 17 moves a consensus price with nobody betting. Kept only when:
      * >= 5 books on both snapshots and the count within +-20% (same panel of books);
      * another outcome of the same market moved the OPPOSITE way by >= 2% (a real repricing);
      * no other outcome jumped >= 10% the SAME way (that is a broken book, not a move);
      * the new price is above 1.10; the snapshot is recent (default 8h) and before kickoff day;
      * direction: SHARP_JUMP_DIRECTIONS = "in" (default: price shortening, money coming in),
        "out", or "both". Measured over 30 days to 2026-10-13: in 1.5/day, both 4.4/day.
    """
    import math
    if min_pct is None:
        min_pct = float(os.getenv("SHARP_JUMP_MIN_PCT", "10"))
    dirs = os.getenv("SHARP_JUMP_DIRECTIONS", "in").strip().lower()
    if history is None:
        from src.sharp_tracker import _load_history
        history = _load_history()
    now = now or pd.Timestamp.now(tz="UTC")
    today = now.strftime("%Y-%m-%d")
    rows = []
    for key, rec in history.items():
        sn = rec.get("snapshots") or []
        if len(sn) < 2 or str(rec.get("date", ""))[:10] < today:
            continue
        a, b = sn[-2], sn[-1]
        t = pd.to_datetime(b.get("at"), utc=True, errors="coerce")
        if pd.isna(t) or (now - t).total_seconds() > max_age_hours * 3600:
            continue
        na, nb = a.get("n_books") or 0, b.get("n_books") or 0
        if na < 5 or nb < 5 or not (0.8 <= nb / na <= 1.25):
            continue
        totals = rec.get("market") == "totals"
        keys = ("odds_over", "odds_under") if totals else ("odds_home", "odds_away", "odds_draw")
        moves = {}
        for k in keys:
            p, c = a.get("odds", {}).get(k), b.get("odds", {}).get(k)
            if p and c and p > 1 and c > 1:
                moves[k] = (p, c, c / p - 1)
        for k, (p, c, m) in moves.items():
            if abs(m) * 100 < min_pct or c <= 1.10 or math.isnan(m):
                continue
            if (dirs == "in" and m > 0) or (dirs == "out" and m < 0):
                continue
            others = [v[2] for kk, v in moves.items() if kk != k]
            if not any(o * m < 0 and abs(o) >= 0.02 for o in others):
                continue
            if any(o * m > 0 and abs(o) >= 0.10 for o in others):
                continue
            side = k.split("_", 1)[1].upper()
            label = (f"O/U 2.5 {side}" if totals else
                     f"1X2 {side} ({rec['home'] if side == 'HOME' else rec['away'] if side == 'AWAY' else 'Draw'})")
            rows.append({"key": key, "sel": k, "at": str(b.get("at")), "date": str(rec["date"])[:10],
                         "league": rec.get("league", ""), "match": f"{rec['home']} vs {rec['away']}",
                         "market": label, "prev_odds": p, "now_odds": c, "jump_pct": round(m * 100, 1),
                         "books": nb, "snapshots": len(sn)})
    d = pd.DataFrame(rows)
    return config.drop_paper_leagues(d) if len(d) else d


def sharp_image(today: str) -> list[bytes]:
    s = _sharp_frame()
    if s.empty:
        return []
    s = s[s["date"].astype(str).str[:10] == today].copy()
    if s.empty:
        return []
    s["_bad"] = sharp_artifacts(s)
    good = s[~s["_bad"]].assign(_a=lambda x: x["drift_pct"].abs()).sort_values("_a", ascending=False)
    cols = [("League", 190), ("Match", 320), ("Selection", 200), ("Open -> now", 130),
            ("Move", 80), ("Signal", 130)]
    rows = []
    for _, r in good.head(ROWS_PER_IMAGE).iterrows():
        d = r["drift_pct"]
        clr = GREEN if d < 0 else RED
        rows.append({"cells": [r["league"], r["match"], r["market"],
                               f"{r['opening_odds']:.2f} -> {r['current_odds']:.2f}",
                               f"{d:+.0f}%", str(r.get("signal", ""))],
                     "stripe": clr, "bold": {"Move"}, "colours": {"Move": clr}})
    n_bad = int(s["_bad"].sum())
    label = pd.Timestamp(today).strftime("%a %d %b")
    return render_table(f"Sharp movements — {label}", cols, rows,
                        subtitle=f"{len(good)} moves on today's fixtures, biggest first. Green = price "
                                 f"shortening (money in), red = drifting (money out)."
                                 + (f" {n_bad} impossible move(s) filtered as bad data." if n_bad else ""),
                        footer="Opening price -> latest captured price · not betting advice")


# ── Pro tip scoreboard (1X2 + Bet Builder) ────────────────────────────────────
# Pro sent this as a separate daily text message. It is read here from Pro's committed JSON
# (local sibling checkout if present, else the public repo over HTTP — the same source the
# dashboard uses) and drawn as pictures, and Pro's workflow no longer sends the text.
PRO_SCOREBOARD_URL = ("https://raw.githubusercontent.com/NevixAA/wowzaV9-Pro/main/"
                      "output/tip_scoreboard.json")


def _pro_scoreboard() -> dict:
    import json
    for p in (os.getenv("PRO_LOCAL"), "../wowzaV9-Pro", "../pro", "../v10"):
        if p:
            f = os.path.join(p, "output", "tip_scoreboard.json")
            if os.path.exists(f):
                with open(f, encoding="utf-8") as fh:
                    return json.load(fh)
    try:
        import requests
        r = requests.get(PRO_SCOREBOARD_URL, timeout=20)
        return r.json() if r.status_code == 200 else {}
    except Exception:                                                   # noqa: BLE001
        return {}


def top_picks_image(today: str, freeze: bool = False) -> list[bytes]:
    """The 3 lists of 5 (src/top_picks.py) - first in the album, because people take a few tips, not 68.
    freeze=True (the real send) logs them once to output/top_picks_log.csv so they are graded later."""
    from src import top_picks as TP
    d = TP.freeze_day(today) if freeze else TP.todays_lists(today)
    if d is None or d.empty:
        return []
    sb = TP.scoreboard(30)
    desc = {"BEST_BETS": "most expected profit, at least a 45% chance", "SAFEST": "the 5 shortest prices",
            "VALUE": "most expected profit"}
    cols = [("#", 34), ("Match", 330), ("League", 180), ("Pick", 110), ("Odds", 70), ("Chance", 86)]
    rows = []

    def _ids(n):
        x = d[d["list"] == n].sort_values("rank")
        return list(zip(x["home_team"], x["away_team"], x["market"]))

    same_bv = bool(_ids("BEST_BETS")) and _ids("BEST_BETS") == _ids("VALUE")
    for name in TP.LISTS:
        x = d[d["list"] == name].sort_values("rank")
        if x.empty or (same_bv and name == "VALUE"):
            continue                        # identical to Best bets today: shown once (both still logged)
        rec = sb[sb["list"] == name] if len(sb) else sb
        rec_txt = (f" · last 30 days: {rec['hit_rate'].iloc[0]:.0%} hit, ROI {rec['roi'].iloc[0]:+.0%}"
                   if len(rec) else " · record starts today")
        title = ("Best bets = Value today — most expected profit, every pick at least a 45% chance"
                 if same_bv and name == "BEST_BETS" else f"{TP.LIST_LABEL[name]} — {desc[name]}")
        rows.append({"band": f"{title}{rec_txt}"})
        for r in x.itertuples(index=False):
            pw = "–" if pd.isna(r.p_win) else f"{float(r.p_win):.0%}"
            rows.append({"cells": [str(r.rank), f"{r.home_team} v {r.away_team}", r.league, r.pick,
                                   f"{float(r.odds):.2f}", pw]})
    label = pd.Timestamp(today).strftime("%a %d %b")
    return render_table(f"Top picks — {label}", cols, rows,
                        subtitle="Taking a few tips rather than all of them? Start here. Chance = our honest "
                                 "win estimate, learned from past results. The full list follows.",
                        footer="Frozen each morning and scored · not betting advice")


PRO_BET_BUILDER_URL = ("https://raw.githubusercontent.com/NevixAA/wowzaV9-Pro/main/"
                       "output/bet_builder_daily.csv")


def _pro_bet_builder() -> pd.DataFrame:
    """Pro's daily Bet Builder selection (Pro src/combo/notify.daily_selection): local sibling
    checkout if present, else the public repo - the same source as the scoreboard above."""
    for p in (os.getenv("PRO_LOCAL"), "../wowzaV9-Pro", "../pro", "../v10"):
        if p:
            f = os.path.join(p, "output", "bet_builder_daily.csv")
            if os.path.exists(f):
                return pd.read_csv(f)
    try:
        return pd.read_csv(PRO_BET_BUILDER_URL)
    except Exception:                                                   # noqa: BLE001
        return pd.DataFrame()


def bet_builder_image(today: str) -> list[bytes]:
    """Today's Bet Builder combos (owner 2026-10-18: moved here from Pro's separate Telegram
    messages, which were "spammy"). Pro chooses them with its own tip rules; this only draws them:
    one band per combo (match, joint probability, fair odds), one row per leg."""
    d = _pro_bet_builder()
    if d.empty or "match_date" not in d.columns:
        return []
    d = d[d["match_date"].astype(str).str[:10] == today]
    if d.empty:
        return []
    cols = [("", 34), ("Leg", 640), ("Leg chance", 120)]
    blocks = []                                   # one block = a combo's band + its legs
    for _, r in d.iterrows():
        fair = pd.to_numeric(pd.Series([r.get("fair_odds")]), errors="coerce").iloc[0]
        p = float(r.get("joint_probability") or 0)
        b = [{"band": f"{r.get('match')} · {r.get('league')} · all legs {p:.1%}"
                      + (f" · fair odds {fair:.2f}" if pd.notna(fair) else "")}]
        for i, leg in enumerate(str(r.get("legs") or "").split(" · "), start=1):
            label, _, pct = leg.rpartition(" (")
            b.append({"cells": [f"{i}.", label or leg, pct.rstrip(")") if label else ""]})
        blocks.append(b)
    # Never split a combo across two images: fill each image with whole combos only.
    pages, cur = [], []
    for b in blocks:
        if cur and len(cur) + len(b) > ROWS_PER_IMAGE:
            pages.append(cur)
            cur = []
        cur += b
    if cur:
        pages.append(cur)
    label = pd.Timestamp(today).strftime("%a %d %b")
    out = []
    for n, rows in enumerate(pages, start=1):
        out += render_table(f"Bet Builder — {label}" + (f"  ({n}/{len(pages)})" if len(pages) > 1 else ""),
                            cols, rows,
                            subtitle="Same-match combos where the legs are correlated more than a book "
                                     "pricing them separately assumes. FAIR odds - no bookmaker builder "
                                     "price is collected. Paper research, never staked.",
                            footer="One combo per match · chosen by Pro's combo rules · not betting advice")
    return out


def scoreboard_images() -> list[bytes]:
    sb = _pro_scoreboard()
    if not sb:
        return []
    cols = [("Tips sent", 300), ("Settled", 80), ("Won", 70), ("Hit", 70), ("Needed", 80),
            ("Model said", 100), ("P/L", 90)]

    def _pct(x):
        return "" if x is None else f"{100 * float(x):.0f}%"

    def _row(label, b, stripe=None, units=True):
        if not b or not b.get("settled"):
            return {"cells": [label, 0, "", "", "", "", ""], "colours": {"Tips sent": MUTED}}
        hit, need = b.get("hit_rate"), b.get("break_even")
        u = b.get("units")
        return {"cells": [label, b["settled"], b.get("won", ""), _pct(hit), _pct(need),
                          _pct(b.get("model_claimed")), "" if u is None else f"{u:+.2f}u"],
                "stripe": stripe, "bold": {"Hit", "P/L"},
                "colours": {"Hit": (GREEN if need is not None and hit is not None and hit > need
                                    else RED if need is not None else INK),
                            "P/L": GREEN if (u or 0) > 0 else RED if (u or 0) < 0 else MUTED}}

    rows = []
    x = sb.get("one_x_two", {})
    if x:
        rows.append({"band": f"1X2 tips — {x.get('sent_excl_paper', x.get('sent', 0))} sent, "
                             f"{x.get('pending', 0)} pending"})
        rows.append(_row("All 1X2 tips", x.get("headline"), (31, 41, 55)))
        for pick, b in (x.get("by_pick") or {}).items():
            rows.append(_row(f"   {pick.title()} picks", b, (156, 163, 175)))
    bb = sb.get("bet_builder", {})
    if bb:
        h = bb.get("headline") or {}
        rows.append({"band": f"Bet Builder — {bb.get('sent', 0)} sent, {bb.get('ungraded', 0)} not gradable yet"})
        if h.get("settled"):
            hit, claim = h.get("hit_rate"), h.get("model_claimed")
            rows.append({"cells": ["All legs played", h["settled"], h.get("won", ""), _pct(hit), "",
                                   _pct(claim), f"exp {h.get('expected_wins', '')} wins"],
                         "stripe": (31, 41, 55), "bold": {"Hit"},
                         "colours": {"Hit": GREEN if (hit or 0) >= (claim or 0) else RED}})
        v = bb.get("with_voided_legs") or {}
        if v.get("settled"):
            rows.append({"cells": ["A leg was voided (not comparable)", v["settled"], v.get("won", ""),
                                   _pct(v["won"] / v["settled"]), "", "", ""],
                         "colours": {"Tips sent": MUTED, "Hit": MUTED}})
    out = render_table("Pro tips scoreboard — 1X2 and Bet Builder", cols, rows,
                       subtitle="Are the tips Pro sends hitting? Hit vs the rate needed to break even "
                                "(1X2) or vs the model's own claim (Bet Builder). Paper leagues excluded.",
                       footer=f"From Pro output/tip_scoreboard.json · generated {sb.get('generated_at', '?')}")
    last = (x or {}).get("last_10") or []
    if last:
        lc = [("Date", 110), ("Match", 360), ("Pick", 90), ("Odds", 80), ("Result", 100)]
        lr = [{"cells": [r.get("date", ""), r.get("match", ""), str(r.get("pick", "")).title(),
                         f"{float(r['odds']):.2f}" if r.get("odds") else "",
                         "WON" if r.get("won") else "LOST"],
               "stripe": GREEN if r.get("won") else RED, "bold": {"Result"},
               "colours": {"Result": GREEN if r.get("won") else RED}} for r in reversed(last)]
        out += render_table("Last 10 settled 1X2 tips", lc, lr)
    return out


# ── Weekly ────────────────────────────────────────────────────────────────────

def _market_rows(d: pd.DataFrame) -> list[dict]:
    rows = []
    for band, tiers, stripe, paper in (("Staked — SNIPER + MARKSMAN", STAKED, TIER_COLOUR["SNIPER"], False),
                                       ("Paper — VALUABLE", ("VALUABLE",), TIER_COLOUR["VALUABLE"], True)):
        g0 = d[d["tier"].isin(tiers)]
        if g0.empty:
            continue
        rows.append({"band": band})
        groups = sorted(g0.groupby("market"), key=lambda kv: -len(kv[1])) + [("All", g0)]
        for mk, g in groups:
            w = int((g["pnl"] > 0).sum())
            be = 100 / g["odds"].mean() if g["odds"].notna().sum() >= 0.8 * len(g) else None
            hit = 100 * w / len(g)
            pl = g["pnl"].sum()
            rows.append({"cells": [mk, len(g), f"{w}-{len(g) - w}", f"{hit:.0f}%",
                                   "" if be is None else f"{be:.0f}%", f"{pl:+.2f}u", f"{pl / len(g):+.3f}"],
                         "stripe": (31, 41, 55) if mk == "All" else stripe, "bold": {"P/L"},
                         "colours": {"P/L": GREEN if pl > 0 else RED,
                                     "Market": MUTED if paper else INK,
                                     "Hit": GREEN if be is not None and hit > be else RED if be is not None else INK}})
    return rows


def week_images(today: str) -> list[bytes]:
    d = settled_tips()
    if d.empty:
        return []
    start = (pd.Timestamp(today) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    w = d[(d["date"] >= start) & (d["date"] < today) & d["pnl"].notna()]
    if w.empty:
        return []
    span = f"{pd.Timestamp(start):%d %b} – {pd.Timestamp(today) - pd.Timedelta(days=1):%d %b}"
    out = render_table(f"Week in review — {span}",
                       [("Market", 170), ("Bets", 70), ("Won-Lost", 110), ("Hit", 70),
                        ("Break-even", 110), ("P/L", 100), ("Per bet", 90)],
                       _market_rows(w),
                       subtitle="Every settled tip of the last 7 days, flat 1u. Hit is green when above "
                                "the break-even rate implied by the average price.",
                       footer="Same basis as the dashboard · paper leagues excluded")
    st = w[w["tier"].isin(STAKED)]
    if len(st):
        lg = (st.groupby("league")["pnl"].agg(Bets="size", PL="sum",
                                              W=lambda x: int((x > 0).sum()))
              .reset_index().sort_values("PL", ascending=False))
        rows = [{"cells": [r["league"], r["Bets"], f"{r['W']}-{r['Bets'] - r['W']}", f"{r['PL']:+.2f}u"],
                 "stripe": GREEN if r["PL"] > 0 else RED, "bold": {"P/L"},
                 "colours": {"P/L": GREEN if r["PL"] > 0 else RED}} for _, r in lg.iterrows()]
        out += render_table("Week by league — staked tips", [("League", 300), ("Bets", 70),
                                                             ("Won-Lost", 110), ("P/L", 100)], rows,
                            subtitle="SNIPER + MARKSMAN, all markets, best to worst. Small samples — "
                                     "one week is noise, not a verdict on a league.")
    return out


def props_week_image(today: str) -> list[bytes]:
    p = _read("player_ledger.csv")
    if p.empty:
        return []
    start = (pd.Timestamp(today) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    p["d"] = p["match_date"].astype(str).str[:10]
    p["pnl"] = pd.to_numeric(p["pnl"], errors="coerce")
    p["market_odds"] = pd.to_numeric(p["market_odds"], errors="coerce")
    w = p[(p["d"] >= start) & (p["d"] < today) & p["result"].isin(["WIN", "LOSS"])
          & p["market_odds"].notna()]
    if w.empty:
        return []
    rows = []
    for mk, g in sorted(w.groupby("market"), key=lambda kv: -len(kv[1])):
        n, won = len(g), int((g["result"] == "WIN").sum())
        need = 100 * (1 / g["market_odds"]).mean()
        hit = 100 * won / n
        pl = g["pnl"].sum()
        rows.append({"cells": [PROP_LABEL.get(mk, mk), n, f"{won}-{n - won}", f"{hit:.0f}%",
                               f"{need:.0f}%", f"{pl:+.2f}u"],
                     "bold": {"P/L at 1u"}, "colours": {"Hit": GREEN if hit > need else RED,
                                                       "P/L at 1u": GREEN if pl > 0 else RED}})
    return render_table("Player props — week in review (paper)",
                        [("Market", 220), ("Priced", 80), ("Won-Lost", 120), ("Hit", 70),
                         ("Needed", 90), ("P/L at 1u", 110)], rows,
                        subtitle="Every priced prop signal settled in the last 7 days, all tiers. "
                                 "Needed = average implied probability of the price. Never staked.")


def props_model_check_image(today: str, weeks: int = 8) -> list[bytes]:
    """Are the props models getting better? Week by week: how often settled prop signals hit vs
    the probability the model gave them, and which model version was live that week.

    Owner request 2026-10-13, after seeing more props "Won": that was VOLUME (the 26-league
    collection fix graded 6x more signals in the week of 10-05), not a higher hit rate — and the
    models had not been retrained between 09-20 and 10-12 at all. This picture makes the
    distinction visible every week.

    Read the GAP, not the hit rate: hit rate moves with the market mix (cards hit ~22%, 1+ SOT
    ~40%). A model that is improving keeps the gap near zero across markets. The model's figure
    includes the chance a player does not play, while only players who played are graded, so a
    positive gap is expected; what matters is whether it is stable or shrinking after a retrain.
    """
    import json
    p = _read("player_ledger.csv")
    if p.empty:
        return []
    p = p[p["result"].isin(["WIN", "LOSS"])].copy()
    p["dt"] = pd.to_datetime(p["match_date"], errors="coerce")
    start = pd.Timestamp(today) - pd.Timedelta(weeks=weeks)
    p = p[(p["dt"] >= start) & (p["dt"] < pd.Timestamp(today))]
    if p.empty:
        return []
    p["win"] = p["result"].eq("WIN")
    p["mp"] = pd.to_numeric(p["model_prob"], errors="coerce")
    p["wk"] = p["dt"].dt.to_period("W").dt.start_time

    runs = []
    try:
        log = json.loads((config.OUTPUT_DIR / "props_retrain_log.json").read_text(encoding="utf-8"))
        runs = sorted(k for k, r in log.get("runs", {}).items()
                      if any(v.get("promoted") for v in r.values()))
    except Exception:                                                   # noqa: BLE001
        pass

    # Which model made each tip is decided by WHEN THE TIP WAS MADE (signal_date), not the match
    # week: a 12 Oct match was tipped days earlier by whatever model was live then. A retrain day
    # itself counts as old (the retrain lands mid-day, after that morning's tips).
    first_new = runs[0] if runs else None
    sig = p["signal_date"].astype(str).str[:10] if "signal_date" in p.columns else p["dt"].dt.strftime("%Y-%m-%d")
    p["new_model"] = (sig > first_new) if first_new else False

    def _version(g: pd.DataFrame) -> str:
        share = float(g["new_model"].mean()) if len(g) else 0.0
        if share == 0:
            return "old models"
        return "new models" if share == 1 else f"{share:.0%} new"

    rows = []
    for wk, g in sorted(p.groupby("wk"), key=lambda kv: kv[0]):
        hit, said = 100 * g["win"].mean(), 100 * g["mp"].mean()
        gap = hit - said
        rows.append({"cells": [f"{wk:%d %b}", len(g), int(g["win"].sum()), f"{hit:.1f}%", f"{said:.1f}%",
                               f"{gap:+.1f} pts", _version(g)],
                     "stripe": (31, 41, 55) if g["new_model"].any() else (156, 163, 175),
                     "bold": {"Gap"}})
    last = p[p["wk"] == p["wk"].max()]
    if len(last):
        rows.append({"band": f"Last week by market ({last['wk'].max():%d %b})"})
        for mk, g in sorted(last.groupby("market"), key=lambda kv: -len(kv[1])):
            hit, said = 100 * g["win"].mean(), 100 * g["mp"].mean()
            rows.append({"cells": [PROP_LABEL.get(mk, mk), len(g), int(g["win"].sum()), f"{hit:.1f}%",
                                   f"{said:.1f}%", f"{hit - said:+.1f} pts", ""], "bold": {"Gap"}})
    return render_table("Props model check — hit rate vs what the model said",
                        [("Week of", 170), ("Settled", 80), ("Won", 70), ("Hit", 80), ("Model said", 110),
                         ("Gap", 100), ("Tipped by", 130)], rows,
                        subtitle="All settled prop signals (every tier, played only). Watch the GAP across "
                                 "weeks, not the hit rate: hit rate moves with the market mix. Dark rows = "
                                 "weeks with tips from the retrained models (fast learner, 10-12/10-13).")


def _pro_csv(rel: str) -> pd.DataFrame:
    """A Pro output CSV: local sibling checkout if present, else the public repo (as the scoreboard)."""
    for p in (os.getenv("PRO_LOCAL"), "../wowzaV9-Pro", "../pro", "../v10"):
        if p and os.path.exists(os.path.join(p, rel)):
            return pd.read_csv(os.path.join(p, rel))
    try:
        return pd.read_csv("https://raw.githubusercontent.com/NevixAA/wowzaV9-Pro/main/" + rel)
    except Exception:                                                   # noqa: BLE001
        return pd.DataFrame()


TP_LABEL = {"BEST_BETS": "Best bets", "SAFEST": "Safest", "VALUE": "Value", "RANDOM_5": "Random 5",
            "CURRENT": "Old order (tier, edge)"}


def top_picks_week_image(now: _dt.datetime) -> list[bytes]:
    """Weekly Top-picks table (Pro src/studies/top_picks_weekly.py): last completed week, to date, backtest."""
    h = _pro_csv("output/studies/top_picks_weekly.csv")
    if h.empty:
        return []
    done = (now - _dt.timedelta(days=7)).strftime("%G-W%V")              # the week that just finished
    latest = h[h["week"] == h["week"].max()]
    cols = [("List", 190), ("Hit", 70), ("Avg odds", 96), ("ROI", 80), ("0/5 days", 96), ("Picks", 70)]
    rows = []
    for kind, title, src in (("live_this_week", f"Live — week {done[-3:]}", h[h["week"] == done]),
                             ("live_to_date", "Live — to date", latest),
                             ("backtest", "Backtest — all days with 10+ tips", latest)):
        x = src[src["kind"] == kind]
        if x.empty:
            continue
        rows.append({"band": title})
        for name in ("BEST_BETS", "SAFEST", "VALUE", "RANDOM_5", "CURRENT"):
            r = x[x["list"] == name]
            if r.empty:
                continue
            r = r.iloc[0]
            roi = float(r["roi"])
            odds = r.get("avg_odds")
            rows.append({"cells": [TP_LABEL[name], f"{float(r['hit_rate']):.0%}",
                                   "–" if pd.isna(odds) else f"{float(odds):.2f}", f"{roi:+.0%}",
                                   f"{float(r['zero_days']):.0%}", str(int(r["picks"]))],
                         "colours": {"ROI": GREEN if roi > 0 else RED}, "bold": {"ROI"}})
    if not rows:
        return []
    return render_table("Top picks — weekly table", cols, rows,
                        subtitle="Top 5 of each list per day. Live = what you were shown, frozen each morning. "
                                 "Backtest = the same rules re-tested on all history.",
                        footer="Flat 1 unit per pick · not betting advice")


def _signed(v: float) -> tuple:
    return GREEN if v > 0 else RED if v < 0 else MUTED


def _na(v, fmt: str) -> str:
    return "–" if v is None or pd.isna(v) else fmt.format(v)


LEAGUE_COLUMNS = [("League", 250), ("Rating", 70), ("Tips", 56), ("W-L", 84), ("Hit", 64), ("Avg odds", 86),
                  ("Avg edge", 86), ("ROI", 74), ("P/L", 84), ("Last 7d", 84), ("Best market", 176),
                  ("Worst market", 176), ("State", 118)]


def league_rows(tab: pd.DataFrame) -> list[dict]:
    """Telegram rows from dashboard_data.leagues.overview_table - the page's own table, formatted."""
    rows = []
    for r in tab.to_dict("records"):
        paper = r["League"].endswith(" (paper)")
        rows.append({"cells": [r["League"], r["Rating"], str(r["Tips"]), r["W-L"], _na(r["Hit %"], "{:.0f}%"),
                               _na(r["Avg odds"], "{:.2f}"), _na(r["Avg edge %"], "{:.1f}%"),
                               _na(r["ROI %"], "{:+.0f}%"), f"{r['P/L (u)']:+.2f}u", f"{r['P/L 7d (u)']:+.2f}u",
                               r["Best market"], r["Worst market"], "Paper league" if paper else r["State"]],
                     "stripe": MUTED if paper else _signed(r["P/L (u)"]), "bold": {"P/L"},
                     "colours": {"League": MUTED if paper else INK, "ROI": _signed(r["ROI %"] or 0),
                                 "P/L": _signed(r["P/L (u)"]), "Last 7d": _signed(r["P/L 7d (u)"]),
                                 "State": MUTED if paper else INK}})
    return rows


def leagues_image(today: str | None = None) -> list[bytes]:
    """The Leagues picture (owner 2026-10-25): one row per league with a settled staked tip - rating,
    record, averages, last 7 days, best / worst market, state. Numbers from dashboard_data.leagues, the
    same functions as the app's Leagues page. Paginates past ROWS_PER_IMAGE by itself."""
    from dashboard_data import leagues as L
    today = today or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    d = L.settled()
    if d.empty:
        return []
    ov = L.overview(d, today, rat=L.ratings(), pol=L.policy(), reg=L.regimes())
    ov = ov[ov["tips"] > 0]
    if ov.empty:
        return []
    t = L.totals(d)
    sub = (f"Staked tips (SNIPER + MARKSMAN), settled since {str(config.PERFORMANCE_CUTOFF_DATE)[:10]}, flat 1u. "
           f"All leagues: {t['tips']} tips, {L.wl(t['won'], t['lost'], t['void'])}, ROI {t['roi']:+.0%}, "
           f"{t['pl']:+.2f}u (paper leagues not counted). Rating = Pro league rating 1-10.")
    return render_table("Leagues — season to date", LEAGUE_COLUMNS, league_rows(L.overview_table(ov)),
                        subtitle=sub,
                        footer="Same numbers as the app's Leagues page · best/worst market needs 5+ tips · "
                               "edge as recorded · not betting advice")


def last_week_leagues_image(today: str | None = None) -> list[bytes]:
    """Last week by league: settled tips of the 7 match days before today, leagues with tips only."""
    from dashboard_data import leagues as L
    today = today or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    lw = L.last_week(L.settled(), today)
    if lw.empty:
        return []
    rows = []
    for r in lw.to_dict("records"):
        paper = r["paper"]
        rows.append({"cells": [r["league"] + (" (paper)" if paper else ""), str(r["tips"]),
                               L.wl(r["won"], r["lost"], r["void"]) if r["tips"] else "–",
                               f"{r['pl']:+.2f}u" if r["tips"] else "–",
                               str(r["valuable_tips"]), f"{r['valuable_pl']:+.2f}u" if r["valuable_tips"] else "–"],
                     "stripe": MUTED if paper else _signed(r["pl"]), "bold": {"P/L"},
                     "colours": {"P/L": _signed(r["pl"]), "League": MUTED if paper else INK,
                                 "VALUABLE P/L": _signed(r["valuable_pl"])}})
    start = pd.Timestamp(today) - pd.Timedelta(days=7)
    span = f"{start:%d %b} – {pd.Timestamp(today) - pd.Timedelta(days=1):%d %b}"
    return render_table(f"Last week by league — {span}",
                        [("League", 300), ("Staked", 80), ("W-L", 96), ("P/L", 96), ("VALUABLE", 100),
                         ("VALUABLE P/L", 130)], rows,
                        subtitle="Settled tips by match date. Staked = SNIPER + MARKSMAN; VALUABLE is paper. "
                                 "One week is noise, not a verdict on a league.",
                        footer="Flat 1u · same basis as the app's Leagues page")


def send_weekly_pack() -> bool:
    """Monday album: week by market, week by league, props week, Pro scoreboard.
    Dedup WEEKLY|<iso-year>-W<week> in notified.json. Returns True if sent."""
    from telegram_bot.notifier import _load_config, _load_notified, _save_notified
    cfg = _load_config()
    token, chat_id = cfg.get("token", ""), cfg.get("chat_id", "")
    if not token or token == "YOUR_BOT_TOKEN":
        return False
    now = _dt.datetime.now(_dt.timezone.utc)
    y, wk, _ = now.isocalendar()
    key = f"WEEKLY|{y}-W{wk:02d}"
    notified = _load_notified()
    if key in notified:
        print("[weekly_pack] already sent this week")
        return False
    pngs, notes = [], []
    for name, fn in (("top_picks", lambda: top_picks_week_image(now)),
                     ("leagues", lambda: leagues_image(now.strftime("%Y-%m-%d"))),
                     ("leagues_week", lambda: last_week_leagues_image(now.strftime("%Y-%m-%d"))),
                     ("week", lambda: week_images(now.strftime("%Y-%m-%d"))),
                     ("props", lambda: props_week_image(now.strftime("%Y-%m-%d"))),
                     ("props_check", lambda: props_model_check_image(now.strftime("%Y-%m-%d"))),
                     ("scoreboard", scoreboard_images)):
        try:
            imgs = fn()
            pngs += imgs
            notes.append(f"{name} {len(imgs)}")
        except Exception as e:                                          # noqa: BLE001
            print(f"[weekly_pack] {name} skipped: {type(e).__name__}: {e}")
    print("[weekly_pack] " + ", ".join(notes))
    if not pngs or not send_album(token, chat_id, pngs, f"📅 <b>Wowza weekly — week {wk}</b>"):
        return False
    notified.add(key)
    _save_notified(notified)
    return True


# ── The pack ──────────────────────────────────────────────────────────────────

def build_pack(now: _dt.datetime | None = None, freeze: bool = False) -> tuple[list[bytes], str]:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    today = now.strftime("%Y-%m-%d")
    yday = (now - _dt.timedelta(days=1)).strftime("%Y-%m-%d")
    header = now.strftime("%a %d %b")
    pngs: list[bytes] = []
    notes = []
    steps = [("top_picks", lambda: top_picks_image(today, freeze=freeze)),
             ("tips", lambda: render_pngs(todays_rows(today), f"Today's tips — {header}")
              if len(todays_rows(today)) else []),
             ("results", lambda: results_image(yday)),
             ("props_results", lambda: props_results_image(yday)),
             ("record", lambda: record_image(today)),
             ("props", lambda: props_image(today, freeze=freeze)),
             ("one_x_two", lambda: one_x_two_image(now)),
             ("bet_builder", lambda: bet_builder_image(today)),
             ("sharp", lambda: sharp_image(today)),
             ("scoreboard", scoreboard_images)]
    for name, fn in steps:
        try:
            imgs = fn()
            pngs += imgs
            notes.append(f"{name} {len(imgs)}")
        except Exception as e:                                          # noqa: BLE001
            print(f"[daily_pack] {name} skipped: {type(e).__name__}: {e}")
            notes.append(f"{name} FAILED")
    n_tips = len(todays_rows(today))
    try:
        from src.top_picks import day_line
        yline = day_line(yday)
    except Exception:                                                   # noqa: BLE001
        yline = ""
    caption = (f"☀️ <b>Wowza daily — {header}</b>\n"
               f"Top picks · today's tips ({n_tips}) · yesterday's results (team + props) · record · player props · 1X2 · "
               f"Bet Builder · sharp moves · Pro scoreboard"
               + (f"\nYesterday's top picks: {yline}" if yline else ""))
    print("[daily_pack] " + ", ".join(notes))
    return pngs, caption


def send_daily_pack() -> bool:
    """Send the morning album once per UTC day (DIGEST|<date>). Returns True if sent."""
    from telegram_bot.notifier import _load_config, _load_notified, _save_notified
    cfg = _load_config()
    token, chat_id = cfg.get("token", ""), cfg.get("chat_id", "")
    if not token or token == "YOUR_BOT_TOKEN":
        return False
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    notified = _load_notified()
    key = f"DIGEST|{today}"
    if key in notified:
        print("[daily_pack] already sent today")
        return False
    pngs, caption = build_pack(freeze=True)                 # the real send freezes today's top picks
    if not pngs:
        print("[daily_pack] nothing to send")
        return False
    if not send_album(token, chat_id, pngs, caption):
        return False
    notified.add(key)
    # Every tip the album showed counts as notified under the instant sender's own keys
    # (owner 2026-10-18: no more instant tip messages - see notifier.instant_tip_allowed), so a later
    # predict run cannot send any of them again; only tips first found AFTER this album go out alone.
    try:
        rows = todays_rows(today)
        if "notify_key" in rows.columns:
            notified.update(k for k in rows["notify_key"].dropna().astype(str) if k)
    except Exception as e:                                              # noqa: BLE001
        print(f"[daily_pack] could not mark album tips as notified: {e}")
    _save_notified(notified)
    return True


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    if len(sys.argv) > 2 and sys.argv[2] == "weekly":
        _t = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
        pngs, cap = (week_images(_t) + props_week_image(_t) + props_model_check_image(_t)
                     + scoreboard_images(), "weekly")
    else:
        pngs, cap = build_pack()
    for i, b in enumerate(pngs):
        with open(os.path.join(out, f"pack_{i + 1}.png"), "wb") as fh:
            fh.write(b)
    print(cap.replace("<b>", "").replace("</b>", ""), f"\n{len(pngs)} image(s) -> {out}")
