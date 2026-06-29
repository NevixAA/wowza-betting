"""
Telegram Notifier — sends SNIPER alerts after each predict run.

Setup:
  1. Message @BotFather on Telegram → /newbot → copy token
  2. Message your bot once, then run:
       python notifier.py --get-chat-id
  3. Edit bot_config.json with your token and chat_id
  4. Run: python notifier.py  (called automatically by scheduler)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import requests

BASE_DIR   = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "bot_config.json"
NOTIFIED_FILE = BASE_DIR / "notified.json"
SHARP_NOTIFIED_FILE = BASE_DIR / "sharp_notified.json"
WC_NOTIFIED_FILE = BASE_DIR / "wc_notified.json"
PLAYER_NOTIFIED_FILE   = BASE_DIR / "player_notified.json"
PLAYER_KICKOFF_FILE   = BASE_DIR / "player_kickoff_cache.json"

sys.path.insert(0, str(BASE_DIR.parent))
import config as app_config


def _load_config() -> dict:
    import os
    # GitHub Actions / CI: read from environment variables
    env_token   = os.getenv("TELEGRAM_TOKEN", "")
    env_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if env_token and env_chat_id:
        return {"token": env_token, "chat_id": env_chat_id}
    # Local: read from bot_config.json
    if not CONFIG_FILE.exists():
        default = {"token": "YOUR_BOT_TOKEN", "chat_id": "YOUR_CHAT_ID"}
        CONFIG_FILE.write_text(json.dumps(default, indent=2))
        print(f"Created {CONFIG_FILE} — fill in your token and chat_id.")
        sys.exit(1)
    return json.loads(CONFIG_FILE.read_text())


def _send(token: str, chat_id: str, text: str) -> bool:
    import time
    for attempt in range(10):
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
        except Exception as e:
            print(f"Telegram error: {e}")
            return False
        if r.status_code == 429:
            retry_after = r.json().get("parameters", {}).get("retry_after", 5)
            print(f"Telegram rate limit — sleeping {retry_after}s (attempt {attempt + 1})")
            time.sleep(retry_after + 1)
            continue
        if r.status_code != 200:
            print(f"Telegram API error {r.status_code}: {r.text[:400]}")
            return False
        return True
    return False


def _load_notified(path: "Path | None" = None) -> set:
    f = path or NOTIFIED_FILE
    if f.exists():
        text = f.read_text(encoding="utf-8-sig").strip()
        if not text:
            return set()
        data = json.loads(text)
        keys = set(data.get("keys", []))
        # Migrate old PLAYER keys that included match string (5th segment).
        # New format is PLAYER|date|name|market (4 segments).
        normalized: set[str] = set()
        for k in keys:
            parts = k.split("|")
            if parts[0] == "PLAYER" and len(parts) == 5:
                k = "|".join(parts[:4])  # drop the match segment
            normalized.add(k)
        return normalized
    return set()


def _save_notified(keys: set, path: "Path | None" = None) -> None:
    f = path or NOTIFIED_FILE
    # Merge with whatever is currently on disk — prevents concurrent CI runs from
    # overwriting each other's notifications.
    if f.exists():
        try:
            existing = json.loads(f.read_text(encoding="utf-8-sig"))
            keys = keys | set(existing.get("keys", []))
        except Exception:
            pass
    f.write_text(json.dumps({"keys": sorted(keys)}, indent=2), encoding="utf-8")


def _escape_html(text: str) -> str:
    """Escape characters that break Telegram HTML parse mode."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _drift_emoji(signal: str) -> str:
    return {"Confirmed": "✅", "Conflicted": "⚠️", "Neutral": "➡️", "New": "🆕"}.get(str(signal), "")


def notify_new_snipers() -> int:
    """Check bets.csv for new SNIPER and VALUE tips and send Telegram alerts. Returns count sent."""
    cfg = _load_config()
    token   = cfg.get("token", "")
    chat_id = cfg.get("chat_id", "")

    if not token or token == "YOUR_BOT_TOKEN":
        print("Telegram not configured — skipping notifications.")
        return 0

    bets_file = app_config.OUTPUT_DIR / "bets.csv"
    if not bets_file.exists():
        return 0

    # Load league ROI config — only alert on leagues with proven backtest edge
    _roi_cfg_path = app_config.OUTPUT_DIR / "league_roi_config.json"
    _approved_leagues: set[str] = set()
    if _roi_cfg_path.exists():
        try:
            import json as _json
            _roi_data = _json.loads(_roi_cfg_path.read_text())
            for league, markets in _roi_data.get("approved_markets_by_league", {}).items():
                if markets:  # at least one market with positive ROI
                    _approved_leagues.add(league)
        except Exception:
            pass

    df = pd.read_csv(bets_file)
    # VALUABLE = info only (backtest shows 4-8% edge is -8% ROI).
    # Only send SNIPER and MARKSMAN as actual tips.
    tips = df[
        df["signal_tier"].isin(["SNIPER", "MARKSMAN"]) &
        df["bet"].isin(["UNDER", "OVER"])
    ].copy()

    # Filter to approved leagues only (when ROI config exists).
    # SNIPER tips always go through — ROI config gates MARKSMAN only.
    if _approved_leagues:
        tips = tips[
            (tips["signal_tier"] == "SNIPER") |
            (tips.get("league", pd.Series(dtype=str)).isin(_approved_leagues))
        ]

    # Only upcoming games — never send finished matches
    today_str = datetime.now().strftime("%Y-%m-%d")
    tips = tips[tips["date"].astype(str).str[:10] >= today_str]

    if tips.empty:
        return 0

    notified = _load_notified()
    sent = 0

    for _, row in tips.iterrows():
        key = f"{str(row['date'])[:10]}|{row['home_team']}|{row['away_team']}|{row.get('best_side','')}"
        if key in notified:
            continue

        tier  = row["signal_tier"]
        side  = row.get("best_side") or row.get("bet", "")
        odds  = row["odds_under25"] if side == "UNDER" else row["odds_over25"]
        edge  = float(row.get("best_edge", 0)) * 100
        drift = _drift_emoji(row.get("drift_signal", "New"))
        model = "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Standard" if row.get("model_type") == "standard" else "🌍 New-Format"

        if tier == "SNIPER":
            header = f"🎯 <b>SNIPER TIP</b> {drift}"
        elif tier == "MARKSMAN":
            header = f"🔫 <b>MARKSMAN TIP</b> {drift}"
        else:
            header = f"🔍 <b>EDGE WATCH</b> (small edge, not a tip) {drift}"

        msg = (
            f"{header}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📅 {str(row['date'])[:10]}\n"
            f"🏆 {row.get('league','')}\n"
            f"⚽ {row['home_team']} vs {row['away_team']}\n"
            f"📌 <b>{side} 2.5</b>  @ {odds:.2f}\n"
            f"📊 Edge: <b>{edge:.1f}%</b>  |  {model}\n"
            f"🔀 Drift: {row.get('drift_signal','New')} {drift}"
        )

        if _send(token, chat_id, msg):
            notified.add(key)
            sent += 1
            _save_notified(notified)  # save after each send so crashes don't cause duplicates
            print(f"  Sent [{tier}]: {row['home_team']} vs {row['away_team']} — {side}")

    if sent:
        _send(token, chat_id,
              f"📋 <b>{sent} new tip(s)</b> sent at {datetime.now().strftime('%H:%M')}")
    return sent


def notify_live_signals() -> int:
    """Send Telegram alerts for live value signals. Returns count sent."""
    cfg = _load_config()
    token   = cfg.get("token", "")
    chat_id = cfg.get("chat_id", "")
    if not token or token == "YOUR_BOT_TOKEN":
        return 0

    live_file = app_config.OUTPUT_DIR / "live_tips.csv"
    if not live_file.exists():
        return 0

    df = pd.read_csv(live_file)
    if df.empty or "signal_type" not in df.columns:
        return 0

    notified = _load_notified()

    # Load live-specific notified (keyed by match+minute window to re-alert if situation changes)
    live_notified_file = Path(__file__).resolve().parent / "live_notified.json"
    live_notified = set()
    if live_notified_file.exists():
        try:
            text = live_notified_file.read_text(encoding="utf-8-sig").strip()
            live_notified = set(json.loads(text).get("keys", []))
        except Exception:
            pass

    SIGNAL_EMOJI = {
        "UNDER_HOLD":     "🔒",
        "SLEEPING_GAME":  "😴",
        "UNDER_RECOVERY": "📉",
        "STRONG_STUCK":   "💪",
        "COMEBACK":       "🔥",
    }

    sent = 0
    new_live_keys = set(live_notified)

    for _, row in df.iterrows():
        # Key includes 10-minute window so re-alerts if match progresses significantly
        minute_bucket = (int(row["elapsed_mins"]) // 10) * 10
        key = f"LIVE|{row['match']}|{row['signal_type']}|{minute_bucket}"
        if key in live_notified:
            continue

        emoji   = SIGNAL_EMOJI.get(row["signal_type"], "📌")
        is_under = "UNDER" in str(row["bet"])
        fair    = row["fair_under_odds"] if is_under else row["fair_over_odds"]
        live_p  = row["live_p_under"] if is_under else row["live_p_over"]

        msg = (
            f"{emoji} <b>LIVE VALUE — {row['signal_type']}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"🏆 {row['league']}\n"
            f"⚽ {row['match']}\n"
            f"⏱ {row['elapsed_mins']}' | Score: <b>{row['score']}</b>\n"
            f"📌 Bet: <b>{row['bet']}</b>\n"
            f"💰 Fair price: <b>{fair}</b> | P={live_p*100:.0f}%\n"
            f"📋 {row['reason'][:120]}\n"
            f"⚠️ Check your bookmaker live screen!"
        )

        if _send(token, chat_id, msg):
            new_live_keys.add(key)
            sent += 1
            print(f"  Live: {row['match']} [{row['signal_type']}] {row['score']} @{row['elapsed_mins']}'")

    live_notified_file.write_text(
        json.dumps({"keys": list(new_live_keys)}, indent=2), encoding="utf-8"
    )
    return sent


def notify_wc_strong() -> int:
    """Check worldcup_tips.csv for new STRONG drift signals and send Telegram alerts."""
    cfg = _load_config()
    token   = cfg.get("token", "")
    chat_id = cfg.get("chat_id", "")

    if not token or token == "YOUR_BOT_TOKEN":
        return 0

    wc_file = app_config.OUTPUT_DIR / "worldcup_tips.csv"
    if not wc_file.exists():
        return 0

    df = pd.read_csv(wc_file)
    strong = df[df["signal"].isin(["STRONG", "SHARP", "FADING"])].copy()

    # Only alert for matches in the next 3 days — future signals fire when relevant
    today = datetime.utcnow().date()
    cutoff = today + __import__("datetime").timedelta(days=3)
    strong["_date"] = pd.to_datetime(strong["date"], errors="coerce").dt.date
    strong = strong[strong["_date"].between(today, cutoff)]

    # Priority: STRONG first, then SHARP, then FADING; within each by absolute drift
    _sig_order = {"STRONG": 0, "SHARP": 1, "FADING": 2}
    strong["_sig_rank"] = strong["signal"].map(_sig_order).fillna(9)
    strong = strong.sort_values(["_sig_rank", "drift_pct"], key=lambda s: s if s.name != "drift_pct" else s.abs(), ascending=[True, False])

    if strong.empty:
        return 0

    MAX_PER_RUN = 50

    notified = _load_notified(WC_NOTIFIED_FILE)
    sent = 0

    _WC_SIG = {
        "STRONG": ("🔴", "STEAM — sharp money IN"),
        "SHARP":  ("🟡", "SHARP — money moving in"),
        "FADING": ("🔵", "FADING — sharp against this outcome"),
    }

    for _, row in strong.iterrows():
        if sent >= MAX_PER_RUN:
            break

        key = f"WC|{str(row['date'])[:10]}|{row['match']}|{row['market']}"
        if key in notified:
            continue

        sig = row["signal"]
        emoji, label = _WC_SIG.get(sig, ("⚪", sig))
        direction = "▼ Shortening" if row["drift_pct"] < 0 else "▲ Lengthening"
        n_books = int(row["n_books"]) if "n_books" in row and not pd.isna(row.get("n_books")) else None
        books_line = f" across {n_books} bookmakers" if n_books and n_books > 1 else ""
        msg = (
            f"{emoji} <b>WC {label}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📅 {str(row['date'])[:10]}\n"
            f"⚽ {row['match']}\n"
            f"📌 <b>{row['market']}</b>\n"
            f"💰 Opening: {row['opening_odds']} → Now: {row['current_odds']}\n"
            f"📉 Drift: <b>{row['drift_pct']:+.1f}%</b>  {direction}\n"
            f"🔍 {row['snapshots']} snapshots{books_line}"
        )

        if _send(token, chat_id, msg):
            notified.add(key)
            sent += 1
            print(f"  WC Sent: {row['match']} — {row['market']} {row['drift_pct']:+.1f}%")

    _save_notified(notified, WC_NOTIFIED_FILE)
    return sent


def notify_sharp_strong() -> int:
    """Send Telegram alerts for STRONG sharp money signals across all leagues."""
    cfg = _load_config()
    token   = cfg.get("token", "")
    chat_id = cfg.get("chat_id", "")
    if not token or token == "YOUR_BOT_TOKEN":
        return 0

    sharp_file = app_config.OUTPUT_DIR / "sharp_tips.csv"
    if not sharp_file.exists():
        return 0

    df = pd.read_csv(sharp_file)
    strong = df[df["signal"].isin(["STRONG", "SHARP"])].copy()
    if strong.empty:
        return 0

    notified = _load_notified(SHARP_NOTIFIED_FILE)
    sent = 0

    for _, row in strong.iterrows():
        key = f"SHARP|{str(row['date'])[:10]}|{row['match']}|{row['market']}"
        if key in notified:
            continue

        sig       = row["signal"]
        emoji     = "🔴" if sig == "STRONG" else "🟡"
        direction = "▼ Sharp money IN" if row["drift_pct"] < 0 else "▲ Money moving OUT"
        msg = (
            f"{emoji} <b>SHARP MONEY — {sig}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📅 {str(row['date'])[:10]}\n"
            f"🏆 {row.get('league','')}\n"
            f"⚽ {row['match']}\n"
            f"📌 <b>{row['market']}</b>\n"
            f"💰 Opening: {row['opening_odds']} → Now: {row['current_odds']}\n"
            f"📉 Drift: <b>{row['drift_pct']:+.1f}%</b>  {direction}\n"
            f"🔍 Based on {row['snapshots']} snapshots"
        )

        if _send(token, chat_id, msg):
            notified.add(key)
            sent += 1
            _save_notified(notified, SHARP_NOTIFIED_FILE)
            print(f"  Sharp [{sig}]: {row['match']} — {row['market']} {row['drift_pct']:+.1f}%")

    return sent


def notify_ht_tips() -> int:
    """Send Telegram alerts for strong HT O/U model predictions. Returns count sent."""
    cfg = _load_config()
    token   = cfg.get("token", "")
    chat_id = cfg.get("chat_id", "")
    if not token or token == "YOUR_BOT_TOKEN":
        return 0

    preds_file = app_config.OUTPUT_DIR / "predictions.csv"
    if not preds_file.exists():
        return 0

    df = pd.read_csv(preds_file)
    if "p_ht_over05" not in df.columns:
        return 0

    today_str = datetime.now().strftime("%Y-%m-%d")
    df = df[df["date"].astype(str).str[:10] >= today_str]
    df = df[df["p_ht_over05"].notna()]

    if df.empty:
        return 0

    notified = _load_notified()
    sent = 0

    for _, row in df.iterrows():
        p05  = float(row["p_ht_over05"])
        p15  = float(row["p_ht_over15"]) if pd.notna(row.get("p_ht_over15")) else None
        date = str(row["date"])[:10]

        # Only alert on strong signals
        if p05 >= 0.75:
            side, prob, line = "OVER", p05, "0.5"
            fair = round(1 / max(p05, 0.01), 2)
            emoji = "⚡"
        elif p05 <= 0.30:
            side, prob, line = "UNDER", 1 - p05, "0.5"
            fair = round(1 / max(1 - p05, 0.01), 2)
            emoji = "🧊"
        elif p15 is not None and p15 >= 0.60:
            side, prob, line = "OVER", p15, "1.5"
            fair = round(1 / max(p15, 0.01), 2)
            emoji = "🔥"
        elif p15 is not None and p15 <= 0.25:
            side, prob, line = "UNDER", 1 - p15, "1.5"
            fair = round(1 / max(1 - p15, 0.01), 2)
            emoji = "🔒"
        else:
            continue

        key = f"HT|{date}|{row['home_team']}|{row['away_team']}|HT{side}{line}"
        if key in notified:
            continue

        msg = (
            f"{emoji} <b>HT {side} {line} — MODEL TIP</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📅 {date}\n"
            f"🏆 {row.get('league', '')}\n"
            f"⚽ {row['home_team']} vs {row['away_team']}\n"
            f"📌 <b>HT {side} {line}</b>\n"
            f"📊 P(HT {side} {line}) = <b>{prob*100:.0f}%</b>\n"
            f"💰 Fair price: <b>{fair}</b>\n"
            f"⚠️ Check your bookmaker's HT market"
        )

        if _send(token, chat_id, msg):
            notified.add(key)
            sent += 1
            _save_notified(notified)
            print(f"  HT tip: {row['home_team']} vs {row['away_team']} — HT {side} {line} ({prob*100:.0f}%)")

    return sent


def notify_weekly_summary() -> bool:
    """
    Send weekly performance summary grouped by signal family:
    Prediction Model / Sharp Tracker / Player Props / WC Drift.
    Returns True if message was sent.
    """
    import datetime as _dt
    cfg = _load_config()
    token   = cfg.get("token", "")
    chat_id = cfg.get("chat_id", "")
    if not token or token == "YOUR_BOT_TOKEN":
        return False

    week_ago = datetime.now() - _dt.timedelta(days=7)
    week_start = week_ago.strftime("%b %d")
    week_end   = datetime.now().strftime("%b %d, %Y")

    def _stats(data: "pd.DataFrame") -> dict | None:
        d = data[data["pnl"].notna()] if "pnl" in data.columns else data
        if d.empty:
            return None
        n = len(d); w = (d["pnl"] > 0).sum(); pnl = d["pnl"].sum()
        return {"n": n, "win": float(w) / n, "roi": pnl / n * 100, "pnl": pnl}

    def _fmt(label: str, emoji: str, stats: dict | None, pending_n: int = 0) -> str:
        if stats and stats["n"] > 0:
            roi_s = "+" if stats["roi"] >= 0 else ""
            pnl_s = "+" if stats["pnl"] >= 0 else ""
            return (f"  {emoji} <b>{label}</b>: {stats['n']} bets | "
                    f"{stats['win']:.0%} win | "
                    f"ROI <b>{roi_s}{stats['roi']:.1f}%</b> | "
                    f"PnL {pnl_s}{stats['pnl']:.2f}u")
        if pending_n:
            return f"  {emoji} <b>{label}</b>: {pending_n} signals sent | ⏳ awaiting results"
        return f"  {emoji} <b>{label}</b>: no settled bets this week"

    # ── 1. Prediction model (bets_ledger.csv, source=live) ──────────────────
    pred_week: dict | None = None
    pred_all:  dict | None = None
    pred_tier_week: dict = {}
    pred_tier_all:  dict = {}
    ledger_file = app_config.OUTPUT_DIR / "bets_ledger.csv"
    if ledger_file.exists():
        bl = pd.read_csv(ledger_file)
        bl["pnl"]        = pd.to_numeric(bl["pnl"], errors="coerce")
        bl["match_date"] = pd.to_datetime(bl["match_date"], errors="coerce")
        live = bl[bl["source"] == "live"].copy()
        lw = live[live["match_date"] >= week_ago]
        pred_week = _stats(lw)
        pred_all  = _stats(live)
        for tier, emj in [("SNIPER", "🎯"), ("MARKSMAN", "🔫"), ("VALUABLE", "💎")]:
            pred_tier_week[tier] = _stats(lw[lw["signal_tier"] == tier]) if "signal_tier" in lw.columns else None
            pred_tier_all[tier]  = _stats(live[live["signal_tier"] == tier]) if "signal_tier" in live.columns else None

    # ── 2. Sharp tracker (sharp_ledger.csv) ─────────────────────────────────
    sharp_week: dict | None = None; sharp_week_pending = 0
    sharp_all:  dict | None = None
    sharp_file = app_config.OUTPUT_DIR / "sharp_ledger.csv"
    if sharp_file.exists():
        sl = pd.read_csv(sharp_file)
        sl["pnl"]        = pd.to_numeric(sl["pnl"], errors="coerce")
        sl["signal_date"] = pd.to_datetime(sl["signal_date"], errors="coerce")
        sw = sl[sl["signal_date"] >= week_ago]
        sharp_week         = _stats(sw)
        sharp_week_pending = int((sw["pnl"].isna()).sum()) if not sw.empty else 0
        sharp_all          = _stats(sl)

    # ── 3. Player props (player_ledger.csv) ──────────────────────────────────
    props_week: dict | None = None; props_week_pending = 0
    props_all:  dict | None = None
    props_file = app_config.OUTPUT_DIR / "player_ledger.csv"
    if props_file.exists():
        pl = pd.read_csv(props_file)
        pl["pnl"]         = pd.to_numeric(pl["pnl"], errors="coerce")
        pl["signal_date"] = pd.to_datetime(pl["signal_date"], errors="coerce")
        pw = pl[pl["signal_date"] >= week_ago]
        props_week         = _stats(pw)
        props_week_pending = int((pw["pnl"].isna()).sum()) if not pw.empty else 0
        props_all          = _stats(pl)

    # ── 4. WC drift signals (worldcup_tips.csv — no PnL yet) ─────────────────
    wc_pending = 0
    wc_file = app_config.OUTPUT_DIR / "worldcup_tips.csv"
    if wc_file.exists():
        wc = pd.read_csv(wc_file)
        wc["date"] = pd.to_datetime(wc["date"], errors="coerce")
        wc_pending = int(len(wc[wc["date"] >= week_ago]))

    # ── Build message — by market then tier ──────────────────────────────────
    TIER_SYM   = {"SNIPER": "🎯", "MARKSMAN": "🔫", "VALUABLE": "💎"}
    TIER_ORDER = ["SNIPER", "MARKSMAN", "VALUABLE"]
    SIDE_LABELS = {"btts": "BTTS", "over15": "Over 1.5", "over35": "Over 3.5"}

    lines = [
        "📊 <b>WEEKLY SUMMARY</b>",
        "━━━━━━━━━━━━━━━━",
        f"📅 {week_start} → {week_end}",
        "",
        "<b>This week by market</b>",
        "",
    ]

    # ── Over 2.5 — this week by tier ─────────────────────────────────────────
    if pred_week and pred_week["n"] > 0:
        pnl_s = "+" if pred_week["pnl"] >= 0 else ""
        lines.append(f"⚽ <b>Over 2.5</b> — {pred_week['n']} bets | {pred_week['win']:.0%} win | PnL {pnl_s}{pred_week['pnl']:.2f}u")
        for tier in TIER_ORDER:
            s = pred_tier_week.get(tier)
            if s and s["n"] > 0:
                roi_s = "+" if s["roi"] >= 0 else ""
                pnl_t = "+" if s["pnl"] >= 0 else ""
                lines.append(f"  {TIER_SYM[tier]} {tier}: {s['n']} bets | {s['win']:.0%} win | ROI <b>{roi_s}{s['roi']:.1f}%</b> | PnL {pnl_t}{s['pnl']:.2f}u")
    else:
        lines.append("⚽ <b>Over 2.5</b>: no settled bets this week")
    lines.append("")

    # ── Side markets (BTTS / O1.5 / O3.5) — this week ────────────────────────
    side_led = app_config.OUTPUT_DIR / "side_bets_ledger.csv"
    if side_led.exists():
        try:
            sl = pd.read_csv(side_led)
            sl["pnl"]         = pd.to_numeric(sl["pnl"], errors="coerce")
            sl["signal_date"] = pd.to_datetime(sl.get("signal_date", sl.get("match_date", "")), errors="coerce")
            sl_week = sl[sl["signal_date"] >= week_ago] if "signal_date" in sl.columns else sl
            for mkt, mkt_label in SIDE_LABELS.items():
                ms = sl_week[sl_week["market"] == mkt] if "market" in sl_week.columns else pd.DataFrame()
                ms = ms[ms["pnl"].notna()]
                if ms.empty:
                    continue
                n = len(ms); mw2 = int((ms["pnl"] > 0).sum()); pnl = ms["pnl"].sum(); roi = pnl / n * 100
                lines.append(f"📌 <b>{mkt_label}</b> — {mw2}W/{n-mw2}L | ROI {roi:+.1f}% | PnL {pnl:+.2f}u")
                for tier in TIER_ORDER:
                    ts = ms[ms["signal_tier"] == tier] if "signal_tier" in ms.columns else pd.DataFrame()
                    if ts.empty:
                        continue
                    tn = len(ts); w = int((ts["pnl"] > 0).sum()); troi = ts["pnl"].sum() / tn * 100
                    lines.append(f"  {TIER_SYM[tier]} {tier}: {w}W/{tn-w}L | ROI {troi:+.1f}%")
                lines.append("")
        except Exception:
            pass

    lines.append(_fmt("Sharp Tracker", "💰", sharp_week, sharp_week_pending))

    if wc_pending:
        lines.append(f"  🌍 <b>WC Drift</b>: {wc_pending} signals active | ⏳ no PnL tracking yet")
    else:
        lines.append("  🌍 <b>WC Drift</b>: no signals this week")

    lines.append("  👤 <b>Player Props</b>: see Player Props Weekly for details")

    # ── All-time by market then tier ──────────────────────────────────────────
    sharp_all_total = int(len(pd.read_csv(sharp_file))) if sharp_file.exists() else 0
    wc_all_total    = int(len(pd.read_csv(wc_file)))    if wc_file.exists() else 0

    lines += ["", "━━━━━━━━━━━━━━━━", "<b>All-time by market</b>", ""]

    if pred_all and pred_all["n"] > 0:
        pnl_s = "+" if pred_all["pnl"] >= 0 else ""
        lines.append(f"⚽ <b>Over 2.5</b> — {pred_all['n']} settled | {pred_all['win']:.0%} win | PnL {pnl_s}{pred_all['pnl']:.2f}u")
        for tier in TIER_ORDER:
            s = pred_tier_all.get(tier)
            if s and s["n"] > 0:
                roi_s = "+" if s["roi"] >= 0 else ""
                pnl_t = "+" if s["pnl"] >= 0 else ""
                lines.append(f"  {TIER_SYM[tier]} {tier}: {s['n']} bets | {s['win']:.0%} win | ROI <b>{roi_s}{s['roi']:.1f}%</b> | PnL {pnl_t}{s['pnl']:.2f}u")
    else:
        lines.append("⚽ <b>Over 2.5</b>: no settled bets")
    lines.append("")

    # Side markets all-time
    if side_led.exists():
        try:
            sl = pd.read_csv(side_led)
            sl["pnl"] = pd.to_numeric(sl["pnl"], errors="coerce")
            sl_all = sl[sl["pnl"].notna()]
            for mkt, mkt_label in SIDE_LABELS.items():
                ms = sl_all[sl_all["market"] == mkt] if "market" in sl_all.columns else pd.DataFrame()
                if ms.empty:
                    continue
                n = len(ms); mw3 = int((ms["pnl"] > 0).sum()); pnl = ms["pnl"].sum(); roi = pnl / n * 100
                lines.append(f"📌 <b>{mkt_label}</b> — {mw3}W/{n-mw3}L | ROI {roi:+.1f}% | PnL {pnl:+.2f}u")
                for tier in TIER_ORDER:
                    ts = ms[ms["signal_tier"] == tier] if "signal_tier" in ms.columns else pd.DataFrame()
                    if ts.empty:
                        continue
                    tn = len(ts); w = int((ts["pnl"] > 0).sum()); troi = ts["pnl"].sum() / tn * 100
                    lines.append(f"  {TIER_SYM[tier]} {tier}: {w}W/{tn-w}L | ROI {troi:+.1f}%")
                lines.append("")
        except Exception:
            pass

    if sharp_all and sharp_all["n"] > 0:
        pnl_s = "+" if sharp_all["pnl"] >= 0 else ""
        lines.append(f"  💰 <b>Sharp Tracker</b>: {sharp_all['n']} settled | {sharp_all['win']:.0%} win | PnL {pnl_s}{sharp_all['pnl']:.2f}u")
    elif sharp_all_total:
        lines.append(f"  💰 <b>Sharp Tracker</b>: {sharp_all_total} signals total | ⏳ awaiting results")
    else:
        lines.append("  💰 <b>Sharp Tracker</b>: no signals yet")

    if wc_all_total:
        lines.append(f"  🌍 <b>WC Drift</b>: {wc_all_total} signals total | ⏳ no PnL tracking yet")

    msg = "\n".join(lines)
    if _send(token, chat_id, msg):
        print("Weekly summary sent.")
        return True
    return False


def notify_agent_analysis() -> int:
    """
    For each new SNIPER pick, run the agent and send a follow-up Telegram message
    with the Strongest Signals shortlist extracted from the analysis.
    Requires GOOGLE_API_KEY (or ANTHROPIC_API_KEY) to be set — skips silently if not.
    Returns count of agent analyses sent.
    """
    import os
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        return 0

    cfg = _load_config()
    token   = cfg.get("token", "")
    chat_id = cfg.get("chat_id", "")
    if not token or token == "YOUR_BOT_TOKEN":
        return 0

    bets_file = app_config.OUTPUT_DIR / "bets.csv"
    if not bets_file.exists():
        return 0

    df = pd.read_csv(bets_file)
    snipers = df[
        (df["signal_tier"] == "SNIPER") &
        (df["bet"].isin(["OVER", "UNDER"]))
    ].copy()

    today_str = datetime.now().strftime("%Y-%m-%d")
    snipers = snipers[snipers["date"].astype(str).str[:10] >= today_str]

    if snipers.empty:
        return 0

    # Agent-notified tracking (separate from regular notified to avoid blocking base alerts)
    agent_notified_file = BASE_DIR / "agent_notified.json"
    agent_notified: set = set()
    if agent_notified_file.exists():
        try:
            agent_notified = set(json.loads(agent_notified_file.read_text(encoding="utf-8")).get("keys", []))
        except Exception:
            pass

    sys.path.insert(0, str(BASE_DIR.parent))
    from agent.agent_runner import run_agent  # type: ignore

    sent = 0
    for _, row in snipers.iterrows():
        key = f"AGENT|{str(row['date'])[:10]}|{row['home_team']}|{row['away_team']}"
        if key in agent_notified:
            continue

        try:
            result = run_agent(row)
        except Exception as e:
            print(f"Agent error for {row['home_team']} vs {row['away_team']}: {e}")
            continue

        if result["mode"] == "manual":
            continue

        # Extract only the STRONGEST SIGNALS section to keep Telegram message short
        import re as _re
        analysis = result["response"]
        m = _re.search(
            r"###?\s*1\.?\s*STRONGEST SIGNALS(.*?)(?=###|\Z)", analysis,
            _re.DOTALL | _re.IGNORECASE,
        )
        strongest = _escape_html((m.group(1).strip() if m else analysis)[:1200])

        side = row.get("best_side") or row.get("bet", "")
        edge = float(row.get("best_edge", 0)) * 100

        msg = (
            f"🤖 <b>AGENT ANALYSIS</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"⚽ {row['home_team']} vs {row['away_team']}\n"
            f"🏆 {row.get('league', '')}  |  📅 {str(row['date'])[:10]}\n"
            f"🎯 ML Signal: <b>{side} 2.5</b>  Edge: <b>{edge:.1f}%</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"<b>Strongest Signals:</b>\n"
            f"{strongest}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Powered by {'Gemini' if result['mode'] == 'gemini' else 'Claude'} · Full analysis in dashboard"
        )

        if _send(token, chat_id, msg):
            agent_notified.add(key)
            sent += 1
            print(f"  Agent analysis sent: {row['home_team']} vs {row['away_team']}")

    agent_notified_file.write_text(
        json.dumps({"keys": list(agent_notified)}, indent=2), encoding="utf-8"
    )
    return sent


def notify_player_props() -> int:
    """Send Telegram alerts for top player prop tips. Returns count sent."""
    cfg = _load_config()
    token   = cfg.get("token", "")
    chat_id = cfg.get("chat_id", "")
    if not token or token == "YOUR_BOT_TOKEN":
        return 0

    tips_file = app_config.OUTPUT_DIR / "player_tips.csv"
    if not tips_file.exists():
        return 0

    df = pd.read_csv(tips_file)
    if df.empty:
        return 0

    today_str = datetime.now().strftime("%Y-%m-%d")
    df = df[df["date"].astype(str).str[:10] >= today_str]

    # Pre-populate kickoff cache from ALL current tips (solves cold-start for cashout gate)
    if "kickoff_utc" in df.columns:
        try:
            _kc = json.loads(PLAYER_KICKOFF_FILE.read_text()) if PLAYER_KICKOFF_FILE.exists() else {}
            for _, _row in df.iterrows():
                _ku = str(_row.get("kickoff_utc", ""))
                if _ku and _ku not in ("nan", "None", ""):
                    _k = f"PLAYER|{str(_row['date'])[:10]}|{_row['player_name']}|{_row['market']}"
                    _kc[_k] = _ku
            PLAYER_KICKOFF_FILE.write_text(json.dumps(_kc))
        except Exception:
            pass

    # Only leagues with actual bookmaker player prop markets
    from player_model.config import PROP_LEAGUES as _PROP_LEAGUES
    df = df[df["league"].isin(_PROP_LEAGUES.keys())].copy()

    # Only SNIPER and MARKSMAN — VALUABLE and WATCH are not sent
    df = df[df["tier"].isin(["SNIPER", "MARKSMAN"])].copy()

    # Safety net (defense-in-depth; predict gates WC cards by real booking history and
    # excludes GKs from scoring markets at source):
    #  • goalkeepers have no goals/SOT/assist props — but CAN have card props (bunker GKs),
    #    so only drop GKs for non-card markets.
    if "position" in df.columns:
        _gk = df["position"].astype(str).str.strip().str.upper().str.startswith("G")
        df = df[~(_gk & (df["market"] != "cards"))].copy()

    df = df.sort_values(["tier", "ev", "model_prob"], ascending=[True, False, False])

    # Flood circuit-breaker — a healthy slate is a handful of tips, not dozens.
    _MAX_PROP_ALERTS = 25
    if len(df) > _MAX_PROP_ALERTS:
        print(f"[notify_player_props] {len(df)} tips after filters — capping to top {_MAX_PROP_ALERTS}.")
        df = df.head(_MAX_PROP_ALERTS)

    if df.empty:
        return 0

    # Load per-league player props ROI config — filter MARKSMAN to approved combos only
    _pp_roi_path = app_config.OUTPUT_DIR / "player_props_league_roi.json"
    _approved_pp: dict[str, list] = {}
    if _pp_roi_path.exists():
        try:
            import json as _json
            _pp_approved = _json.loads(_pp_roi_path.read_text()).get("approved", {})
            _approved_pp = _pp_approved
        except Exception:
            pass

    if _approved_pp:
        def _pp_allowed(row) -> bool:
            if row["tier"] == "SNIPER":
                return True
            approved_markets = _approved_pp.get(row.get("league", ""), [])
            return row["market"] in approved_markets
        df = df[df.apply(_pp_allowed, axis=1)].copy()

    if df.empty:
        return 0

    notified = _load_notified(PLAYER_NOTIFIED_FILE)
    sent = 0

    MARKET_EMOJI = {
        "goals": "⚽", "goals2": "⚽⚽",
        "assists": "🎯",
        "sot": "🔫", "sot2": "🔫", "sot3": "🔫",
        "cards": "🟨",
    }

    for _, row in df.iterrows():
        # Key intentionally excludes match string — match formatting varies between runs
        # and excluding tier means SNIPER→MARKSMAN changes don't re-trigger an alert.
        key = f"PLAYER|{str(row['date'])[:10]}|{row['player_name']}|{row['market']}"
        if key in notified:
            continue

        emoji  = MARKET_EMOJI.get(row["market"], "📌")
        p      = float(row["model_prob"])
        fair   = float(row["fair_odds"])
        tier   = row.get("tier", "WATCH")
        mkt_odds = row.get("market_odds")
        ev_val   = row.get("ev")
        tier_emoji = {"SNIPER": "🎯", "MARKSMAN": "🔫", "VALUABLE": "💎"}.get(tier, "👁")
        market_label = {
            "goals": "Anytime Goalscorer", "goals2": "Score 2+",
            "assists": "Assist",
            "sot": "SOT 1+", "sot2": "SOT 2+", "sot3": "SOT 3+",
            "cards": "Yellow Card",
        }.get(row["market"], row["market"])

        odds_line = f"📈 Odds: <b>{mkt_odds:.2f}</b>  |  Fair: <b>{fair:.2f}</b>  |  EV: <b>{ev_val:+.1%}</b>" \
                    if mkt_odds and not (isinstance(mkt_odds, float) and mkt_odds != mkt_odds) \
                       and ev_val and not (isinstance(ev_val, float) and ev_val != ev_val) \
                    else f"📊 Model P: <b>{p*100:.0f}%</b>  |  Fair Odds: <b>{fair:.2f}</b>"

        msg = (
            f"{tier_emoji} <b>{tier} — {market_label.upper()}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"👤 <b>{row['player_name']}</b> ({row.get('position', '')} · {row['team']})\n"
            f"⚽ {row['match']}\n"
            f"🏆 {row.get('league', '')}  |  📅 {str(row['date'])[:10]}\n"
            f"{odds_line}\n"
            f"⚠️ Check your bookmaker's player props market"
        )

        if _send(token, chat_id, msg):
            notified.add(key)
            # Cache kickoff time so cashout alert can gate on the 60-min window
            _ku = str(row.get("kickoff_utc", ""))
            if _ku and _ku not in ("nan", "None", ""):
                try:
                    _kc = json.loads(PLAYER_KICKOFF_FILE.read_text()) if PLAYER_KICKOFF_FILE.exists() else {}
                    _kc[key] = _ku
                    PLAYER_KICKOFF_FILE.write_text(json.dumps(_kc))
                except Exception:
                    pass
            sent += 1
            _save_notified(notified, PLAYER_NOTIFIED_FILE)
            print(f"  Player prop: {row['player_name']} — {market_label} ({p*100:.0f}%)")

    return sent


def notify_lineup_cashout() -> int:
    """
    Send CASHOUT alerts for players we previously tipped who are NOT in today's
    player_tips.csv anymore — meaning confirmed lineups show they're not starting.

    Logic:
      1. Read player_notified.json — get all keys for today's date.
      2. Read player_tips.csv — current tips after lineup filter.
      3. Any notified key not present in current tips → player benched → send CASHOUT.
      4. Track sent cashout alerts in cashout_notified.json (no repeat cashout per player/market).

    Returns count of cashout alerts sent.
    """
    cfg = _load_config()
    token   = cfg.get("token", "")
    chat_id = cfg.get("chat_id", "")
    if not token or token == "YOUR_BOT_TOKEN":
        return 0

    tips_file = app_config.OUTPUT_DIR / "player_tips.csv"
    if not tips_file.exists():
        return 0

    today_str = datetime.now().strftime("%Y-%m-%d")

    # Build the set of keys that are CURRENTLY active in player_tips.csv
    df = pd.read_csv(tips_file)
    df = df[df["date"].astype(str).str[:10] == today_str]
    df = df[df["tier"].isin(["SNIPER", "MARKSMAN"])]
    active_keys = {
        f"PLAYER|{today_str}|{r['player_name']}|{r['market']}"
        for _, r in df.iterrows()
    }

    # All keys we notified today
    notified_today = {
        k for k in _load_notified(PLAYER_NOTIFIED_FILE)
        if k.startswith(f"PLAYER|{today_str}|")
    }

    # Players whose tips disappeared (not active, not already cashed out)
    cashout_file   = BASE_DIR / "cashout_notified.json"
    already_cashed = _load_notified(cashout_file)
    benched_keys   = notified_today - active_keys - already_cashed

    if not benched_keys:
        return 0

    # Build lookup for player info from today's tips (including all tiers for match context)
    df_all = pd.read_csv(tips_file)
    df_all = df_all[df_all["date"].astype(str).str[:10] == today_str]
    tip_lookup: dict[str, dict] = {}
    for _, row in df_all.iterrows():
        k = f"PLAYER|{today_str}|{row['player_name']}|{row['market']}"
        tip_lookup[k] = row.to_dict()

    MARKET_LABEL = {
        "goals": "Anytime Goalscorer", "goals2": "Score 2+",
        "assists": "Assist",
        "sot": "SOT 1+", "sot2": "SOT 2+", "sot3": "SOT 3+",
        "cards": "Yellow Card",
    }

    # Load kickoff cache so we can gate cashout on the 60-min window
    from datetime import timezone as _tz
    _now_utc = datetime.now(_tz.utc)
    _kickoff_cache: dict[str, str] = {}
    if PLAYER_KICKOFF_FILE.exists():
        try:
            _kickoff_cache = json.loads(PLAYER_KICKOFF_FILE.read_text())
        except Exception:
            pass

    sent = 0
    for key in sorted(benched_keys):
        parts = key.split("|")
        if len(parts) < 4:
            continue
        _, date_str, player_name, market = parts[0], parts[1], parts[2], parts[3]

        # Rule: fire ONLY when kickoff is within the next 60 min (lineups are out)
        # AND the game has not started yet — i.e. 0 < minutes_to_kickoff <= 60.
        # If the kickoff time is unknown or unparseable we CANNOT confirm the
        # window, so we DO NOT fire. (The bug: a missing/bad kickoff used to fall
        # through and alert at any time — hours early or after kickoff.)
        _ku = _kickoff_cache.get(key, "")
        _mins_until = None
        if _ku and _ku not in ("nan", "None"):
            try:
                _mins_until = (pd.Timestamp(_ku, tz="UTC") - _now_utc).total_seconds() / 60
            except Exception:
                _mins_until = None
        if _mins_until is None or not (0 < _mins_until <= 60):
            continue  # unknown timing, >60 min early, or already kicked off — skip

        mkt_label = MARKET_LABEL.get(market, market)
        row = tip_lookup.get(key, {})
        match_str = row.get("match", "")
        team_str  = row.get("team", "")

        msg = (
            f"🚨 <b>LINEUP ALERT — CASHOUT RECOMMENDED</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"👤 <b>{player_name}</b> is NOT in the confirmed starting XI\n"
            f"📌 Market: <b>{mkt_label}</b>\n"
            f"⚽ {match_str or 'Unknown match'}  |  {team_str}\n"
            f"📅 {date_str}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"⚠️ If you placed this bet, consider cashing out now."
        )
        if _send(token, chat_id, msg):
            already_cashed.add(key)
            sent += 1
            _save_notified(already_cashed, cashout_file)
            print(f"  CASHOUT alert: {player_name} — {mkt_label}")

    return sent


def notify_props_daily_digest() -> bool:
    """
    Player props daily briefing structured by MARKET then SIGNAL TIER.
    Msg 1: Tips grouped by market (all 9) then tier. Watch section same way.
    Msg 2: Yesterday's results + all-time ledger by market then tier.
    Returns True if sent.
    """
    import datetime as _dt
    cfg = _load_config()
    token   = cfg.get("token", "")
    chat_id = cfg.get("chat_id", "")
    if not token or token == "YOUR_BOT_TOKEN":
        return False

    today     = datetime.now()
    today_str = today.strftime("%Y-%m-%d")
    yest_str  = (today - _dt.timedelta(days=1)).strftime("%Y-%m-%d")
    hdr_date  = today.strftime("%a, %b %d").replace(" 0", " ")

    notified = _load_notified(PLAYER_NOTIFIED_FILE)
    props_digest_key = f"PROPS_DIGEST|{today_str}"
    if props_digest_key in notified:
        print("Player props digest already sent today — skipping.")
        return False

    # (key, display label, emoji)
    PLAYER_MARKETS = [
        ("goals",   "Anytime Scorer", "⚽"),
        ("goals2",  "Score 2+",       "⚽⚽"),
        ("assists", "Assist",         "🎯"),
        ("sot",     "SOT 1+",         "🔫"),
        ("sot2",    "SOT 2+",         "🔫"),
        ("sot3",    "SOT 3+",         "🔫"),
        ("cards",   "Carded",         "🟨"),
    ]
    TIER_SYM   = {"SNIPER": "🎯", "MARKSMAN": "🔫"}
    TIER_ORDER = ["SNIPER", "MARKSMAN"]

    lines = [
        f"👤 <b>PLAYER PROPS BRIEFING</b> — {hdr_date}",
        "━━━━━━━━━━━━━━━━",
        "",
    ]

    tips_data = pd.DataFrame()
    player_file = app_config.OUTPUT_DIR / "player_tips.csv"
    if player_file.exists():
        try:
            from player_model.config import PROP_LEAGUES as _PROP_LEAGUES
            all_props = pd.read_csv(player_file)
            all_props = all_props[all_props["date"].astype(str).str[:10] >= today_str].copy()
            all_props = all_props[all_props["league"].isin(_PROP_LEAGUES.keys())].copy()
            tips_data = all_props
        except Exception as e:
            lines.append(f"  error loading player_tips.csv: {e}")

    # SNIPER and MARKSMAN only — same rule as individual alerts
    tips = tips_data[tips_data["tier"].isin(TIER_ORDER)].copy() if not tips_data.empty else pd.DataFrame()

    total_tips = len(tips)
    lines.append(f"🎯 <b>TIPS</b>  ({total_tips} signal{'s' if total_tips != 1 else ''})")
    lines.append("")

    if tips.empty:
        lines.append("  No tips today")
    else:
        for mkt_key, mkt_label, mkt_emoji in PLAYER_MARKETS:
            mkt_tips = tips[tips["market"] == mkt_key] if "market" in tips.columns else pd.DataFrame()
            if mkt_tips.empty:
                continue
            lines.append(f"── {mkt_emoji} <b>{mkt_label}</b>  ({len(mkt_tips)})")
            for tier in TIER_ORDER:
                tb = mkt_tips[mkt_tips["tier"] == tier]
                if tb.empty:
                    continue
                sym = TIER_SYM[tier]
                for _, r in tb.iterrows():
                    ev = r.get("ev"); odds = r.get("market_odds")
                    ev_str = f" EV {float(ev):+.0%}" if pd.notna(ev) else ""
                    od_str = (f" @ {float(odds):.2f}" if pd.notna(odds)
                              else f" {float(r.get('model_prob', 0))*100:.0f}%")
                    lines.append(
                        f"  {sym} <b>{r['player_name']}</b> ({r.get('team','')})"
                        f"{od_str}{ev_str}  [{r.get('league','')}]"
                    )
            lines.append("")

    if not player_file.exists():
        lines.append("  No player_tips.csv yet")

    msg1 = "\n".join(lines)
    sent1 = _send(token, chat_id, msg1)
    if not sent1:
        return False

    # ── Message 2: Yesterday's results + all-time by market then tier ─────────
    MKT_LABEL_MAP = {k: lbl for k, lbl, _ in PLAYER_MARKETS}

    lines2 = [
        f"📋 <b>PLAYER PROPS RESULTS</b> — {hdr_date}",
        "━━━━━━━━━━━━━━━━",
        "",
        "<b>Yesterday's Results</b>",
    ]
    player_led = app_config.OUTPUT_DIR / "player_ledger.csv"
    if player_led.exists():
        try:
            pled = pd.read_csv(player_led)
            pled["pnl"] = pd.to_numeric(pled["pnl"], errors="coerce")
            yest = pled[
                (pled["match_date"].astype(str).str[:10] == yest_str) &
                pled["pnl"].notna()
            ]
            if yest.empty:
                lines2.append("  No settled results yesterday")
            else:
                wins   = int((yest["pnl"] > 0).sum())
                losses = int((yest["pnl"] < 0).sum())
                voids  = int(yest["result"].str.upper().eq("VOID").sum()) if "result" in yest.columns else 0
                pnl    = yest["pnl"].sum()
                lines2.append(f"  {wins}W / {losses}L / {voids} VOID  |  PnL <b>{pnl:+.2f}u</b>")
                # Results by market
                for mkt_key, mkt_label, _ in PLAYER_MARKETS:
                    ms = yest[yest["market"] == mkt_key] if "market" in yest.columns else pd.DataFrame()
                    if ms.empty:
                        continue
                    for _, r in ms.iterrows():
                        res   = str(r.get("result", "")).upper()
                        emoji = {"WIN": "✅", "LOSS": "❌", "VOID": "⬜"}.get(res, "⬜")
                        pnl_r = float(r["pnl"])
                        tier_s = f"[{r['tier']}] " if "tier" in r and pd.notna(r.get("tier")) else ""
                        note   = f" ({r['notes']})" if pd.notna(r.get("notes")) and str(r.get("notes")).strip() else ""
                        lines2.append(
                            f"  {emoji} {tier_s}{r['player_name']} — {mkt_label}{note}  <b>{pnl_r:+.2f}u</b>"
                        )
        except Exception as e:
            lines2.append(f"  error: {e}")
    else:
        lines2.append("  No player_ledger.csv yet")

    # ── All-time player ledger by market then tier ─────────────────────────────
    lines2 += ["", "📈 <b>All-time Player Ledger</b>"]
    if player_led.exists():
        try:
            pled = pd.read_csv(player_led)
            pled["pnl"] = pd.to_numeric(pled["pnl"], errors="coerce")
            all_p = pled[pled["pnl"].notna()]
            if not all_p.empty:
                n = len(all_p); w_all = int((all_p["pnl"] > 0).sum()); pnl = all_p["pnl"].sum(); roi = pnl / n * 100
                lines2.append(f"  Total: {w_all}W/{n-w_all}L | PnL <b>{pnl:+.2f}u</b> | ROI {roi:+.1f}%")
                lines2.append("")
                for mkt_key, mkt_label, _ in PLAYER_MARKETS:
                    ms = all_p[all_p["market"] == mkt_key] if "market" in all_p.columns else pd.DataFrame()
                    if ms.empty:
                        continue
                    mn = len(ms); mw = int((ms["pnl"] > 0).sum()); mp = ms["pnl"].sum(); mroi = mp / mn * 100
                    lines2.append(f"  📌 <b>{mkt_label}</b>  {mw}W/{mn-mw}L | ROI {mroi:+.1f}% | PnL {mp:+.2f}u")
                    if "tier" in ms.columns:
                        for tier in TIER_ORDER:
                            ts = ms[ms["tier"] == tier]
                            if ts.empty:
                                continue
                            tn = len(ts); w = int((ts["pnl"] > 0).sum()); troi = ts["pnl"].sum() / tn * 100
                            lines2.append(f"    {TIER_SYM[tier]} {tier}: {w}W/{tn-w}L | ROI {troi:+.1f}%")
        except Exception:
            pass

    msg2 = "\n".join(lines2)
    _send(token, chat_id, msg2)

    notified.add(props_digest_key)
    _save_notified(notified, PLAYER_NOTIFIED_FILE)
    print("Player props daily digest sent (2 messages).")
    return True


def notify_props_weekly_summary() -> bool:
    """
    Send a player-props-only weekly performance summary.
    Groups settled bets by market (SOT/Goals/Assists/Cards) and tier.
    Returns True if message sent.
    """
    import datetime as _dt
    cfg = _load_config()
    token   = cfg.get("token", "")
    chat_id = cfg.get("chat_id", "")
    if not token or token == "YOUR_BOT_TOKEN":
        return False

    week_ago   = datetime.now() - _dt.timedelta(days=7)
    week_start = week_ago.strftime("%b %d")
    week_end   = datetime.now().strftime("%b %d, %Y")

    player_led = app_config.OUTPUT_DIR / "player_ledger.csv"
    if not player_led.exists():
        return False

    pled = pd.read_csv(player_led)
    pled["pnl"]         = pd.to_numeric(pled["pnl"], errors="coerce")
    pled["signal_date"] = pd.to_datetime(pled.get("signal_date", pled.get("match_date")), errors="coerce")

    settled_all  = pled[pled["pnl"].notna()].copy()
    settled_week = settled_all[settled_all["signal_date"] >= week_ago].copy()

    # Active player markets (goals3/sot4 removed — base rate too low)
    PLAYER_MARKETS = [
        ("goals",   "Anytime Scorer", "⚽"),
        ("goals2",  "Score 2+",       "⚽⚽"),
        ("assists", "Assist",         "🎯"),
        ("sot",     "SOT 1+",         "🔫"),
        ("sot2",    "SOT 2+",         "🔫"),
        ("sot3",    "SOT 3+",         "🔫"),
        ("cards",   "Carded",         "🟨"),
    ]
    MKT_LABEL = {k: lbl for k, lbl, _ in PLAYER_MARKETS}
    MKT_EMOJI = {k: emj for k, _, emj in PLAYER_MARKETS}
    TIER_SYM  = {"SNIPER": "🎯", "MARKSMAN": "🔫", "VALUABLE": "💎", "WATCH": "👁"}
    MARKETS   = [k for k, _, _ in PLAYER_MARKETS]
    TIERS     = ["SNIPER", "MARKSMAN", "VALUABLE", "WATCH"]

    def _row(data: "pd.DataFrame") -> tuple | None:
        d = data[data["result"].str.upper().isin(["WIN", "LOSS"])] if "result" in data.columns else data
        if d.empty:
            return None
        n   = len(d)
        w   = int((d["pnl"] > 0).sum())
        pnl = d["pnl"].sum()
        return n, w, pnl

    lines = [
        "👤 <b>PLAYER PROPS WEEKLY</b>",
        "━━━━━━━━━━━━━━━━",
        f"📅 {week_start} → {week_end}",
        "",
    ]

    # ── This week ────────────────────────────────────────────────────────────
    r = _row(settled_week)
    if r:
        n, w, pnl = r
        lines.append(f"<b>This week — {n} settled | {w}W/{n-w}L | PnL {pnl:+.2f}u</b>")
        lines.append("")
        # By market
        for mkt in MARKETS:
            sub = settled_week[settled_week["market"] == mkt] if "market" in settled_week.columns else pd.DataFrame()
            mr = _row(sub)
            if not mr:
                continue
            mn, mw, mp = mr
            roi = mp / mn * 100
            roi_s = "+" if roi >= 0 else ""
            pnl_s = "+" if mp >= 0 else ""
            lines.append(
                f"  {MKT_EMOJI.get(mkt, '📌')} <b>{MKT_LABEL.get(mkt, mkt)}</b>: "
                f"{mn} bets | {mw}W/{mn-mw}L | ROI <b>{roi_s}{roi:.1f}%</b> | PnL {pnl_s}{mp:.2f}u"
            )
        lines.append("")
        # By tier
        lines.append("  By tier:")
        for tier in TIERS:
            sub = settled_week[settled_week["tier"] == tier] if "tier" in settled_week.columns else pd.DataFrame()
            tr = _row(sub)
            if not tr:
                continue
            tn, tw, tp = tr
            roi = tp / tn * 100
            lines.append(
                f"  {TIER_SYM.get(tier, '📌')} {tier}: {tw}W/{tn-tw}L | ROI {roi:+.1f}% | PnL {tp:+.2f}u"
            )
    else:
        pending = int(settled_week["pnl"].isna().sum()) if not settled_week.empty else 0
        total   = len(settled_week)
        if total:
            lines.append(f"  {total} signals ({pending} awaiting results)")
        else:
            lines.append("  No signals this week")

    # ── All-time ─────────────────────────────────────────────────────────────
    lines += ["", "━━━━━━━━━━━━━━━━", "<b>All-time</b>", ""]
    r_all = _row(settled_all)
    if r_all:
        n, w, pnl = r_all
        roi = pnl / n * 100
        lines.append(f"  Total: {n} bets | {w}W/{n-w}L | ROI {roi:+.1f}% | PnL {pnl:+.2f}u")
        lines.append("")
        for mkt in MARKETS:
            sub = settled_all[settled_all["market"] == mkt] if "market" in settled_all.columns else pd.DataFrame()
            mr = _row(sub)
            if not mr:
                continue
            mn, mw, mp = mr
            roi_m = mp / mn * 100
            lines.append(
                f"  {MKT_EMOJI.get(mkt, '📌')} {MKT_LABEL.get(mkt, mkt)}: "
                f"{mw}W/{mn-mw}L | ROI {roi_m:+.1f}% | PnL {mp:+.2f}u"
            )
    else:
        total_all = len(pled)
        lines.append(f"  {total_all} signals tracked — no settled results yet")

    msg = "\n".join(lines)
    if _send(token, chat_id, msg):
        print("Player props weekly summary sent.")
        return True
    return False


def notify_daily_digest() -> bool:
    """
    Daily morning briefing structured by market AND signal tier.
    Msg 1: Over 2.5 (by tier) + BTTS/O1.5/O3.5 (by market then tier) + Sharp + WC.
    Msg 2: Yesterday's results + all-time ledger (same market+tier structure).
    Returns True if sent.
    """
    import datetime as _dt

    cfg = _load_config()
    token   = cfg.get("token", "")
    chat_id = cfg.get("chat_id", "")
    if not token or token == "YOUR_BOT_TOKEN":
        return False

    today         = datetime.now()
    today_str     = today.strftime("%Y-%m-%d")
    yesterday_str = (today - _dt.timedelta(days=1)).strftime("%Y-%m-%d")
    header_date   = today.strftime("%a, %b %d").replace(" 0", " ")

    notified = _load_notified()
    digest_key = f"DIGEST|{today_str}"
    if digest_key in notified:
        print("Daily digest already sent today — skipping.")
        return False

    TIER_SYM   = {"SNIPER": "🎯", "MARKSMAN": "🔫", "VALUABLE": "💎"}
    TIER_ORDER = ["SNIPER", "MARKSMAN", "VALUABLE"]
    SIDE_LABELS = {"btts": "BTTS", "over15": "Over 1.5", "over35": "Over 3.5"}
    SIG_SYM     = {"STEAM_STRONG": "🔴", "STEAM_SHARP": "🟠", "STRONG": "🟡"}

    lines = [
        f"📊 <b>DAILY BRIEFING</b> — {header_date}",
        "━━━━━━━━━━━━━━━━",
        "",
    ]

    # ── Over 2.5 by tier ───────────────────────────────────────────────────────
    bets_file = app_config.OUTPUT_DIR / "bets.csv"
    ou_total = 0
    if bets_file.exists():
        try:
            bets = pd.read_csv(bets_file)
            bets = bets[
                bets["signal_tier"].isin(TIER_ORDER) &
                bets["bet"].isin(["OVER", "UNDER"]) &
                (bets["date"].astype(str).str[:10] >= today_str)
            ].copy()
            ou_total = len(bets)
            lines.append(f"⚽ <b>Over 2.5</b>  ({ou_total} tip{'s' if ou_total != 1 else ''})")
            if bets.empty:
                lines.append("  No tips today")
            else:
                for tier in TIER_ORDER:
                    tb = bets[bets["signal_tier"] == tier]
                    if tb.empty:
                        continue
                    lines.append(f"  {TIER_SYM[tier]} <b>{tier}</b> ({len(tb)})")
                    for _, r in tb.iterrows():
                        side = r.get("best_side") or r.get("bet", "")
                        odds = r["odds_under25"] if side == "UNDER" else r["odds_over25"]
                        edge = float(r.get("best_edge", 0)) * 100
                        lines.append(
                            f"    {r['home_team']} vs {r['away_team']} — "
                            f"<b>{side} 2.5</b> @ {float(odds):.2f} (+{edge:.0f}%)"
                            f"  [{r.get('league','')}]"
                        )
        except Exception as e:
            lines.append(f"  error: {e}")
    else:
        lines.append("⚽ <b>Over 2.5</b> — no data yet")
    lines.append("")

    # ── Side markets: BTTS / Over 1.5 / Over 3.5 — by market then tier ────────
    side_file = app_config.OUTPUT_DIR / "side_bets.csv"
    if side_file.exists():
        try:
            side = pd.read_csv(side_file)
            side = side[
                side["signal_tier"].isin(TIER_ORDER) &
                (side["date"].astype(str).str[:10] >= today_str)
            ].copy()
            for mkt, mkt_label in SIDE_LABELS.items():
                ms = side[side["market"] == mkt] if "market" in side.columns else pd.DataFrame()
                if ms.empty:
                    continue
                lines.append(f"📌 <b>{mkt_label}</b>  ({len(ms)} tip{'s' if len(ms) != 1 else ''})")
                for tier in TIER_ORDER:
                    tb = ms[ms["signal_tier"] == tier]
                    if tb.empty:
                        continue
                    lines.append(f"  {TIER_SYM[tier]} <b>{tier}</b> ({len(tb)})")
                    for _, r in tb.iterrows():
                        ev_str = f" EV {float(r['ev']):+.0%}" if pd.notna(r.get("ev")) else ""
                        lines.append(
                            f"    {r['home_team']} vs {r['away_team']} — "
                            f"@ {float(r['market_odds']):.2f}{ev_str}"
                            f"  [{r.get('league','')}]"
                        )
                lines.append("")
        except Exception as e:
            lines.append(f"  Side markets error: {e}")
            lines.append("")

    # ── Player props summary line ───────────────────────────────────────────────
    player_file = app_config.OUTPUT_DIR / "player_tips.csv"
    if player_file.exists():
        try:
            all_props = pd.read_csv(player_file)
            all_props = all_props[all_props["date"].astype(str).str[:10] >= today_str]
            tips_count  = int(all_props["tier"].isin(["SNIPER", "MARKSMAN", "VALUABLE"]).sum())
            watch_count = int((all_props["tier"] == "WATCH").sum())
            lines.append(
                f"👤 <b>Player Props</b> — {tips_count} tip{'s' if tips_count != 1 else ''}, "
                f"{watch_count} on watch  →  see Player Props Briefing"
            )
        except Exception as e:
            lines.append(f"👤 <b>Player Props</b> — error: {e}")
    else:
        lines.append("👤 <b>Player Props</b> — no data yet")
    lines.append("")

    # ── Sharp signals ───────────────────────────────────────────────────────────
    sharp_file = app_config.OUTPUT_DIR / "sharp_tips.csv"
    if sharp_file.exists():
        try:
            sharp = pd.read_csv(sharp_file)
            sharp = sharp[
                sharp["signal"].isin(["STEAM_STRONG", "STEAM_SHARP", "STRONG"]) &
                (sharp["date"].astype(str).str[:10] >= today_str)
            ].copy()
            lines.append(f"💰 <b>Sharp Signals</b>  ({len(sharp)} signal{'s' if len(sharp) != 1 else ''})")
            if sharp.empty:
                lines.append("  No signals today")
            else:
                for _, r in sharp.head(5).iterrows():
                    sym = SIG_SYM.get(r["signal"], "📌")
                    lines.append(
                        f"  {sym} {r['match']} — {r['market']} ({float(r['drift_pct']):+.1f}%)  [{r.get('league','')}]"
                    )
        except Exception as e:
            lines.append(f"💰 <b>Sharp Signals</b> — error: {e}")
    else:
        lines.append("💰 <b>Sharp Signals</b> — no data yet")
    lines.append("")

    # ── WC signals ──────────────────────────────────────────────────────────────
    wc_file = app_config.OUTPUT_DIR / "worldcup_tips.csv"
    if wc_file.exists():
        try:
            wc = pd.read_csv(wc_file)
            wc = wc[
                wc["signal"].isin(["STEAM_STRONG", "STEAM_SHARP", "STRONG"]) &
                (wc["date"].astype(str).str[:10] >= today_str)
            ].copy()
            if not wc.empty:
                lines.append(f"🌍 <b>WC Signals</b>  ({len(wc)} signal{'s' if len(wc) != 1 else ''})")
                for _, r in wc.iterrows():
                    sym = SIG_SYM.get(r["signal"], "📌")
                    lines.append(
                        f"  {sym} {r['match']} — {r['market']} ({float(r['drift_pct']):+.1f}%)"
                    )
                lines.append("")
        except Exception:
            pass

    msg1 = "\n".join(lines)
    sent1 = _send(token, chat_id, msg1)
    if not sent1:
        return False

    # ── Message 2: Yesterday's results + all-time by market then tier ─────────
    lines2 = [
        f"📋 <b>DAILY RESULTS</b> — {header_date}",
        "━━━━━━━━━━━━━━━━",
        "",
        "<b>Yesterday's Results</b>",
        "",
    ]

    ledger_file = app_config.OUTPUT_DIR / "bets_ledger.csv"
    sharp_led   = app_config.OUTPUT_DIR / "sharp_ledger.csv"

    def _stats_sub(df, mask):
        sub = df[mask & df["pnl"].notna()] if not df.empty else pd.DataFrame()
        if sub.empty:
            return None
        n = len(sub); w = int((sub["pnl"] > 0).sum()); p = sub["pnl"].sum()
        return n, w, p

    # Over 2.5 yesterday by tier
    if ledger_file.exists():
        try:
            led = pd.read_csv(ledger_file)
            led["pnl"] = pd.to_numeric(led["pnl"], errors="coerce")
            yest_led = led[led["match_date"].astype(str).str[:10] == yesterday_str].copy()
            live_y = yest_led[yest_led["source"] == "live"] if "source" in yest_led.columns else yest_led
            r_all = _stats_sub(live_y, pd.Series([True] * len(live_y), index=live_y.index))
            if r_all:
                n, w, pnl = r_all
                lines2.append(f"⚽ <b>Over 2.5</b>  (PnL {pnl:+.2f}u)")
                for tier in TIER_ORDER:
                    if "signal_tier" not in live_y.columns:
                        break
                    r = _stats_sub(live_y, live_y["signal_tier"] == tier)
                    if r:
                        tn, tw, tp = r
                        lines2.append(f"  {TIER_SYM[tier]} {tier}: {tw}W/{tn-tw}L  PnL {tp:+.2f}u")
            else:
                lines2.append("  ⚽ Over 2.5: no settled results")
        except Exception:
            pass

    # Sharp/WC yesterday
    if sharp_led.exists():
        try:
            sled = pd.read_csv(sharp_led)
            sled["pnl"] = pd.to_numeric(sled["pnl"], errors="coerce")
            yest_s = sled[
                (sled["match_date"].astype(str).str[:10] == yesterday_str) &
                sled["pnl"].notna() &
                (sled["result"] != "VOID")
            ]
            if not yest_s.empty:
                n = len(yest_s)
                affected   = int((yest_s["pnl"] > 0).sum())
                inaffected = n - affected
                pnl = yest_s["pnl"].sum()
                lines2.append(
                    f"  💰 Sharp/WC: {n} settled | {affected} affected / {inaffected} inaffected | PnL <b>{pnl:+.2f}u</b>"
                )
            else:
                lines2.append("  💰 Sharp/WC: no settled results")
        except Exception:
            pass

    lines2 += ["", "━━━━━━━━━━━━━━━━", "📈 <b>All-time Ledger</b>", ""]

    # O/U 2.5 all-time — by model type, then tier
    if ledger_file.exists():
        try:
            led = pd.read_csv(ledger_file)
            led["pnl"] = pd.to_numeric(led["pnl"], errors="coerce")
            live = led[(led["source"] == "live") & led["pnl"].notna()] \
                   if "source" in led.columns else led[led["pnl"].notna()]
            has_mt = "model_type" in live.columns
            for fmt, emoji, label in [
                ("standard",   "⚽", "Over 2.5 — Standard"),
                ("new_format", "🌍", "Over 2.5 — New-Format"),
            ]:
                if not has_mt:
                    sub = live if fmt == "standard" else pd.DataFrame()
                elif fmt == "standard":
                    sub = live[(live["model_type"] == "standard") | live["model_type"].isna()]
                else:
                    sub = live[live["model_type"] == fmt]
                if sub.empty:
                    continue
                n = len(sub); w_s = int((sub["pnl"] > 0).sum()); pnl = sub["pnl"].sum(); roi = pnl / n * 100
                lines2.append(f"  {emoji} <b>{label}</b>  {w_s}W/{n-w_s}L | PnL {pnl:+.2f}u | ROI {roi:+.1f}%")
                for tier in TIER_ORDER:
                    ts = sub[sub["signal_tier"] == tier] if "signal_tier" in sub.columns else pd.DataFrame()
                    if ts.empty:
                        continue
                    tn = len(ts); w = int((ts["pnl"] > 0).sum()); troi = ts["pnl"].sum() / tn * 100
                    lines2.append(f"    {TIER_SYM[tier]} {tier}: {w}W/{tn-w}L | ROI {troi:+.1f}%")
        except Exception:
            pass

    # Side markets all-time (btts/over15/over35)
    side_led = app_config.OUTPUT_DIR / "side_bets_ledger.csv"
    if side_led.exists():
        try:
            sl = pd.read_csv(side_led)
            sl["pnl"] = pd.to_numeric(sl["pnl"], errors="coerce")
            sl_s = sl[sl["pnl"].notna()]
            if not sl_s.empty:
                lines2.append("")
                for mkt, mkt_label in SIDE_LABELS.items():
                    ms = sl_s[sl_s["market"] == mkt] if "market" in sl_s.columns else pd.DataFrame()
                    if ms.empty:
                        continue
                    mn = len(ms); mw = int((ms["pnl"] > 0).sum()); mp = ms["pnl"].sum(); mroi = mp / mn * 100
                    lines2.append(f"  📌 <b>{mkt_label}</b>  {mw}W/{mn-mw}L | PnL {mp:+.2f}u | ROI {mroi:+.1f}%")
                    for tier in TIER_ORDER:
                        ts = ms[ms["signal_tier"] == tier] if "signal_tier" in ms.columns else pd.DataFrame()
                        if ts.empty:
                            continue
                        tn = len(ts); w = int((ts["pnl"] > 0).sum()); troi = ts["pnl"].sum() / tn * 100
                        lines2.append(f"    {TIER_SYM[tier]} {tier}: {w}W/{tn-w}L | ROI {troi:+.1f}%")
        except Exception:
            pass

    # Sharp all-time
    if sharp_led.exists():
        try:
            sled = pd.read_csv(sharp_led)
            sled["pnl"] = pd.to_numeric(sled["pnl"], errors="coerce")
            all_s = sled[sled["pnl"].notna() & (sled["result"] != "VOID")]
            if not all_s.empty:
                n = len(all_s); pnl = all_s["pnl"].sum(); roi = pnl / n * 100
                lines2.append(f"  💰 <b>Sharp/WC</b>  ({n} signals | PnL {pnl:+.2f}u | ROI {roi:+.1f}%)")
        except Exception:
            pass

    msg2 = "\n".join(lines2)
    _send(token, chat_id, msg2)

    notified.add(digest_key)
    _save_notified(notified)
    print("Daily digest sent (2 messages).")
    return True


def notify_side_bets() -> int:
    """Send Telegram alerts for SNIPER/MARKSMAN BTTS / Over 1.5 / Over 3.5 tips. Returns count sent."""
    cfg = _load_config()
    token   = cfg.get("token", "")
    chat_id = cfg.get("chat_id", "")
    if not token or token == "YOUR_BOT_TOKEN":
        return 0

    side_file = app_config.OUTPUT_DIR / "side_bets.csv"
    if not side_file.exists():
        return 0

    df = pd.read_csv(side_file)
    df = df[df["signal_tier"].isin(["SNIPER", "MARKSMAN"])].copy()

    today_str = datetime.now().strftime("%Y-%m-%d")
    df = df[df["date"].astype(str).str[:10] >= today_str]

    if df.empty:
        return 0

    df = df.sort_values(["signal_tier", "ev"], ascending=[True, False])

    MARKET_LABEL = {"btts": "BTTS", "over15": "Over 1.5", "over35": "Over 3.5"}
    MARKET_EMOJI = {"btts": "🔁", "over15": "📈", "over35": "🚀"}

    notified = _load_notified()
    sent = 0

    for _, row in df.iterrows():
        key = f"SIDE|{str(row['date'])[:10]}|{row['home_team']}|{row['away_team']}|{row['market']}"
        if key in notified:
            continue

        tier  = row["signal_tier"]
        mkt   = row["market"]
        edge  = float(row["edge"]) * 100
        ev    = float(row["ev"]) * 100
        label = MARKET_LABEL.get(mkt, mkt)
        emoji = MARKET_EMOJI.get(mkt, "📌")
        tier_header = f"🎯 <b>SNIPER — {label}</b>" if tier == "SNIPER" else f"🔫 <b>MARKSMAN — {label}</b>"

        msg = (
            f"{emoji} {tier_header}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📅 {str(row['date'])[:10]}\n"
            f"🏆 {row.get('league', '')}\n"
            f"⚽ {row['home_team']} vs {row['away_team']}\n"
            f"📌 <b>{label}</b>  @ {float(row['market_odds']):.2f}\n"
            f"📊 Edge: <b>{edge:.1f}%</b>  |  EV: <b>{ev:+.1f}%</b>  |  P={float(row['model_prob'])*100:.0f}%\n"
            f"⚠️ Check your bookmaker's {label} market"
        )

        if _send(token, chat_id, msg):
            notified.add(key)
            sent += 1
            _save_notified(notified)
            print(f"  Side [{tier}]: {row['home_team']} vs {row['away_team']} — {label} @ {row['market_odds']:.2f}")

    if sent:
        _send(token, chat_id, f"📋 <b>{sent} side market tip(s)</b> sent at {datetime.now().strftime('%H:%M')}")
    return sent


def get_chat_id(token: str) -> None:
    """Print the chat_id of the last user who messaged the bot."""
    r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=10)
    updates = r.json().get("result", [])
    if not updates:
        print("No messages found. Send any message to your bot first, then run again.")
        return
    chat = updates[-1]["message"]["chat"]
    print(f"Your chat_id: {chat['id']}  (username: {chat.get('username', 'N/A')})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--get-chat-id", action="store_true")
    args = parser.parse_args()

    if args.get_chat_id:
        cfg = _load_config()
        get_chat_id(cfg["token"])
    else:
        # Only predict-specific notifications — sharp/WC handled by sharp_tracker,
        # player props by player_props workflow, digests by daily_summary.
        n    = notify_new_snipers()
        side = notify_side_bets()
        ht   = notify_ht_tips()
        live = notify_live_signals()
        print(f"Notifications sent: {n} O/U SNIPER + {side} side market + {ht} HT + {live} Live")
