"""Today's tips as a table IMAGE for Telegram — readable at a glance on a phone.

    from telegram_bot.tips_table import todays_rows, render_pngs, send_photo

Owner request 2026-10-11: "better visuality for the today tips — table or photo with model, tier,
market, edge, date". Telegram text has no tables and wraps badly on a phone, so the table is drawn
as a PNG. Pillow is used because it is already installed everywhere (Streamlit depends on it);
adding matplotlib would change requirements.txt for every workflow, predict included.

SAME ROWS AS THE TEXT DIGEST: today's (UTC) live tips from the two LEDGERS, tiers SNIPER /
MARKSMAN / VALUABLE, paper leagues excluded. The image is an addition; the text briefing is still
sent, so a rendering failure can never cost the morning's tips.

The rows keep the internal tier CODES (callers sort and filter on them); the picture shows the
owner's display words from src/tier_labels.py - Big / Medium / Small gap (2026-10-26).
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

import config
from src.tier_labels import GAP_EXPLAINER, tier_label

TIER_ORDER = ["SNIPER", "MARKSMAN", "VALUABLE"]
TIER_COLOUR = {"SNIPER": (196, 43, 43), "MARKSMAN": (217, 119, 6), "VALUABLE": (37, 99, 235)}
MARKET_LABEL = {"btts": "BTTS", "over15": "O/U 1.5", "over35": "O/U 3.5"}
ROWS_PER_IMAGE = 22
COLUMNS = [("Time", 62), ("League", 190), ("Rating", 70), ("Match", 330), ("Model", 70), ("Gap", 104),
           ("Market", 150), ("Odds", 62), ("Edge", 66)]

_RATINGS: dict | None = None


def league_ratings() -> dict:
    """Pro's league rating 1-10 (10 = best league), the same source the stake plan sizes by.
    Shown next to every tip (owner 2026-10-20). Missing or stale -> '-' on the row, never a guess."""
    global _RATINGS
    if _RATINGS is None:
        try:
            from src.stake_plan import load_league_ratings
            _RATINGS = load_league_ratings()
        except Exception:                                                 # noqa: BLE001
            _RATINGS = {}
    return _RATINGS


def _kickoffs() -> pd.DataFrame:
    f = config.OUTPUT_DIR / "predictions.csv"
    if not f.exists():
        return pd.DataFrame(columns=["league", "date", "home_team", "away_team", "kickoff_utc"])
    p = pd.read_csv(f, usecols=lambda c: c in {"league", "date", "home_team", "away_team", "kickoff_utc"})
    return p.drop_duplicates(["league", "date", "home_team", "away_team"])


def todays_rows(today_str: str) -> pd.DataFrame:
    """One row per tip, best first: tier order, then edge."""
    return tips_rows(today_str, today_str)


def tips_rows(date_from: str, date_to: str) -> pd.DataFrame:
    """Every live tip whose TIP DAY is in [date_from, date_to] (inclusive, 'YYYY-MM-DD').

    Tip day, not UTC match date (owner 2026-10-25, src/tip_day.py): a match that kicks off before the
    next Israeli-morning album (J-League ~06:00 UTC, Liga MX/MLS ~01:00-03:00 UTC) belongs to the
    PREVIOUS morning's album, which is the last one sent before it starts.

    Shared by the Telegram image and the dashboard's tips board, so both show the same rows.
    Extra columns the image does not draw: `drift` (O/U 2.5 only), `model_prob` (side markets
    only), `generated_at`.
    """
    out = []
    main = config.OUTPUT_DIR / "bets_ledger.csv"
    side = config.OUTPUT_DIR / "side_bets_ledger.csv"

    # A tip day spans two UTC dates, so read one day either side and filter on the tip day below.
    _lo = (pd.Timestamp(date_from) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    _hi = (pd.Timestamp(date_to) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    def _in_range(df):
        d = df["match_date"].astype(str).str[:10]
        return (d >= _lo) & (d <= _hi)

    if main.exists():
        m = config.drop_paper_leagues(pd.read_csv(main, low_memory=False))
        if "source" in m.columns:
            m = m[m["source"].astype(str) == "live"]
        m = m[m["signal_tier"].isin(TIER_ORDER) & _in_range(m)]
        # one side per fixture: the FIRST tip stands (flip guard, 2026-10-11)
        if "generated_at" in m.columns:
            m = m.sort_values("generated_at", kind="mergesort").drop_duplicates(
                ["match_date", "home_team", "away_team"], keep="first")
        for _, r in m.iterrows():
            out.append({"league": r["league"], "date": str(r["match_date"])[:10],
                        "home_team": r["home_team"], "away_team": r["away_team"],
                        "model": r.get("model_type") or config.model_type_for_league(r["league"]),
                        "tier": r["signal_tier"], "market": f"O/U 2.5 {str(r.get('side', '')).upper()}",
                        "odds": r.get("odds"), "edge": r.get("edge_pct"),
                        "drift": r.get("drift_signal"), "model_prob": None,
                        "generated_at": r.get("generated_at"),
                        # the instant sender's dedup key (notifier.notify_new_snipers), so the
                        # morning album can mark what it showed and nothing is sent twice
                        "notify_key": f"{str(r['match_date'])[:10]}|{r['home_team']}|{r['away_team']}"
                                      f"|{str(r.get('side', '')).upper()}"})
    if side.exists():
        s = config.drop_paper_leagues(pd.read_csv(side, low_memory=False))
        s = s[s["signal_tier"].isin(TIER_ORDER) & _in_range(s)]
        for _, r in s.iterrows():
            mk = str(r.get("market", ""))
            label = MARKET_LABEL.get(mk, mk.upper())
            label += " YES" if mk == "btts" else " OVER"
            out.append({"league": r["league"], "date": str(r["match_date"])[:10],
                        "home_team": r["home_team"], "away_team": r["away_team"],
                        "model": r.get("model_type") or config.model_type_for_league(r["league"]),
                        "tier": r["signal_tier"], "market": label, "odds": r.get("odds"),
                        "edge": r.get("edge_pct"), "drift": None,
                        "model_prob": r.get("model_prob"), "generated_at": r.get("generated_at"),
                        # notifier.notify_side_bets' dedup key (side suffix only when not YES/OVER)
                        "notify_key": (f"SIDE|{str(r['match_date'])[:10]}|{r['home_team']}|{r['away_team']}|{mk}"
                                       + (f"|{_sd}" if (_sd := str(r.get('side', '') or '')) and _sd != 'nan'
                                          and _sd.upper() not in ('YES', 'OVER') else ""))})
    d = pd.DataFrame(out)
    if d.empty:
        return d
    from src.tip_day import add_tip_day
    d = add_tip_day(d, date_col="date")
    d = d[(d["tip_day"] >= date_from) & (d["tip_day"] <= date_to)]
    if d.empty:
        return d.reset_index(drop=True)
    ko = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    # "Mon 03:00" when the kickoff falls on the next UTC date, so an after-midnight match is unmistakable
    d["time"] = [("–" if pd.isna(k) else (k.strftime("%a %H:%M") if k.strftime("%Y-%m-%d") != td
                                          else k.strftime("%H:%M"))) for k, td in zip(ko, d["tip_day"])]
    d["edge"] = pd.to_numeric(d["edge"], errors="coerce")
    d["rating"] = d["league"].map(league_ratings())
    d["_t"] = d["tier"].map({t: i for i, t in enumerate(TIER_ORDER)})
    d = d.sort_values(["_t", "edge"], ascending=[True, False]).drop(columns="_t")
    return d.reset_index(drop=True)


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    names = (["DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"] if bold
             else ["DejaVuSans.ttf", "arial.ttf", "Arial.ttf"])
    dirs = ["/usr/share/fonts/truetype/dejavu", "C:/Windows/Fonts", "/Library/Fonts", ""]
    for n in names:
        for d in dirs:
            try:
                return ImageFont.truetype(str(Path(d) / n) if d else n, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                   # Pillow < 10.1
        return ImageFont.load_default()


def _fit(draw, text: str, font, width: int) -> str:
    text = str(text)
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text + "…"


def render_pngs(rows: pd.DataFrame, title: str) -> list[bytes]:
    """One PNG per ROWS_PER_IMAGE tips."""
    from PIL import Image, ImageDraw
    pad, row_h, head_h, title_h, foot_h = 18, 34, 34, 54, 48
    width = pad * 2 + sum(w for _, w in COLUMNS)
    f_title, f_head, f_cell, f_foot = _font(22, True), _font(15, True), _font(15), _font(12)
    pages = [rows.iloc[i:i + ROWS_PER_IMAGE] for i in range(0, len(rows), ROWS_PER_IMAGE)] or [rows]
    out = []
    for n, page in enumerate(pages, start=1):
        height = title_h + head_h + row_h * max(len(page), 1) + foot_h + pad
        img = Image.new("RGB", (width, height), (255, 255, 255))
        dr = ImageDraw.Draw(img)
        t = title + (f"  ({n}/{len(pages)})" if len(pages) > 1 else "")
        dr.text((pad, 14), t, font=f_title, fill=(17, 24, 39))
        y = title_h
        dr.rectangle([0, y, width, y + head_h], fill=(31, 41, 55))
        x = pad
        for name, w in COLUMNS:
            dr.text((x + 4, y + 8), name, font=f_head, fill=(255, 255, 255))
            x += w
        y += head_h
        if page.empty:
            dr.text((pad + 4, y + 8), "No tips today", font=f_cell, fill=(75, 85, 99))
        for i, (_, r) in enumerate(page.iterrows()):
            bg = (249, 250, 251) if i % 2 else (255, 255, 255)
            dr.rectangle([0, y, width, y + row_h], fill=bg)
            colour = TIER_COLOUR.get(r["tier"], (107, 114, 128))
            dr.rectangle([0, y, 6, y + row_h], fill=colour)             # tier stripe
            model = "Standard" if str(r["model"]) == "standard" else "New-fmt"
            edge = "–" if pd.isna(r["edge"]) else f"+{r['edge']:.0f}%"
            odds = "–" if pd.isna(pd.to_numeric(r["odds"], errors="coerce")) else f"{float(r['odds']):.2f}"
            rt = r.get("rating")
            rating = "–" if rt is None or pd.isna(rt) else f"{int(rt)}/10"
            cells = [r["time"], r["league"], rating, f"{r['home_team']} v {r['away_team']}", model,
                     tier_label(r["tier"]), r["market"], odds, edge]
            x = pad
            for (name, w), val in zip(COLUMNS, cells):
                font = f_head if name in ("Gap", "Edge", "Rating") else f_cell
                fill = colour if name == "Gap" else (17, 24, 39)
                if name == "Rating" and val != "–":
                    v = int(val.split("/")[0])
                    fill = (21, 128, 61) if v >= 8 else (185, 28, 28) if v <= 3 else (17, 24, 39)
                dr.text((x + 4, y + 8), _fit(dr, val, font, w - 8), font=font, fill=fill)
                x += w
            y += row_h
        dr.text((pad, y + 8), "Times UTC · Rating = league rating 1-10 (10 best) · edge as recorded by v9 · paper leagues excluded",
                font=f_foot, fill=(107, 114, 128))
        dr.text((pad, y + 26), GAP_EXPLAINER, font=f_foot, fill=(107, 114, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        out.append(buf.getvalue())
    return out


def send_photo(token: str, chat_id: str, png: bytes, caption: str = "") -> bool:
    """Telegram sendPhoto. Never raises; the token is never logged."""
    import requests
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendPhoto",
                          data={"chat_id": chat_id, "caption": caption[:1000], "parse_mode": "HTML"},
                          files={"photo": ("tips.png", png, "image/png")}, timeout=30)
        if r.status_code != 200:
            print(f"[tips_table] sendPhoto HTTP {r.status_code}")
            return False
        return True
    except Exception as e:                                                # noqa: BLE001
        print(f"[tips_table] sendPhoto failed: {type(e).__name__}")
        return False
