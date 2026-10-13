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
replaces the two weekly text summaries.

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

def _read(name: str) -> pd.DataFrame:
    p = config.OUTPUT_DIR / name
    if not p.exists():
        return pd.DataFrame()
    return config.drop_paper_leagues(pd.read_csv(p, low_memory=False))


def settled_tips() -> pd.DataFrame:
    """Every live tip on the O/U 2.5 and side-market ledgers, normalised: date, league, match,
    tier, market, odds, result, pnl. EVERY SENT tip counts, exactly as on the dashboard — a first-
    tip-per-fixture filter here made the record read +27.97u against the dashboard's +36.22u on
    2026-10-13, the same two-answers problem the home page had the day before."""
    parts = []
    m = _read("bets_ledger.csv")
    if len(m):
        if "source" in m.columns:
            m = m[m["source"].fillna("live").astype(str) == "live"]
        parts.append(pd.DataFrame({
            "date": m["match_date"].astype(str).str[:10], "league": m["league"],
            "match": m["home_team"].astype(str) + " v " + m["away_team"].astype(str),
            "tier": m["signal_tier"], "market": "O/U 2.5",
            "pick": "O/U 2.5 " + m["side"].astype(str).str.upper(),
            "odds": pd.to_numeric(m["odds"], errors="coerce"),
            "result": m["result"], "pnl": pd.to_numeric(m["pnl"], errors="coerce")}))
    s = _read("side_bets_ledger.csv")
    if len(s):
        if "source" in s.columns:
            s = s[s["source"].fillna("live").astype(str) == "live"]
        mk = s["market"].astype(str)
        parts.append(pd.DataFrame({
            "date": s["match_date"].astype(str).str[:10], "league": s["league"],
            "match": s["home_team"].astype(str) + " v " + s["away_team"].astype(str),
            "tier": s["signal_tier"], "market": mk.map(SIDE_LABEL).fillna(mk.str.upper()),
            "pick": mk.map(lambda k: "BTTS YES" if k == "btts" else f"{SIDE_LABEL.get(k, k)} OVER"),
            "odds": pd.to_numeric(s["odds"], errors="coerce"),
            "result": s["result"], "pnl": pd.to_numeric(s["pnl"], errors="coerce")}))
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    d["result"] = d["result"].astype(str).str.upper().str.strip()
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
    y = d[d["date"] == day].copy()
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


def props_image(today: str) -> list[bytes]:
    t = _read("player_tips.csv")
    if t.empty:
        return []
    t = t[t["date"].astype(str).str[:10] == today].copy()
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
    for name, fn in (("week", lambda: week_images(now.strftime("%Y-%m-%d"))),
                     ("props", lambda: props_week_image(now.strftime("%Y-%m-%d"))),
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

def build_pack(now: _dt.datetime | None = None) -> tuple[list[bytes], str]:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    today = now.strftime("%Y-%m-%d")
    yday = (now - _dt.timedelta(days=1)).strftime("%Y-%m-%d")
    header = now.strftime("%a %d %b")
    pngs: list[bytes] = []
    notes = []
    steps = [("tips", lambda: render_pngs(todays_rows(today), f"Today's tips — {header}")
              if len(todays_rows(today)) else []),
             ("results", lambda: results_image(yday)),
             ("record", lambda: record_image(today)),
             ("props", lambda: props_image(today)),
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
    caption = (f"☀️ <b>Wowza daily — {header}</b>\n"
               f"Today's tips ({n_tips}) · yesterday's results · record · player props · "
               f"sharp moves · Pro scoreboard")
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
    pngs, caption = build_pack()
    if not pngs:
        print("[daily_pack] nothing to send")
        return False
    if not send_album(token, chat_id, pngs, caption):
        return False
    notified.add(key)
    _save_notified(notified)
    return True


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    if len(sys.argv) > 2 and sys.argv[2] == "weekly":
        _t = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
        pngs, cap = week_images(_t) + props_week_image(_t) + scoreboard_images(), "weekly"
    else:
        pngs, cap = build_pack()
    for i, b in enumerate(pngs):
        with open(os.path.join(out, f"pack_{i + 1}.png"), "wb") as fh:
            fh.write(b)
    print(cap.replace("<b>", "").replace("</b>", ""), f"\n{len(pngs)} image(s) -> {out}")
