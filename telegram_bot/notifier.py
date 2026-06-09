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
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        return r.status_code == 200
    except Exception as e:
        print(f"Telegram error: {e}")
        return False


def _load_notified() -> set:
    if NOTIFIED_FILE.exists():
        text = NOTIFIED_FILE.read_text(encoding="utf-8-sig").strip()
        if not text:
            return set()
        data = json.loads(text)
        return set(data.get("keys", []))
    return set()


def _save_notified(keys: set) -> None:
    NOTIFIED_FILE.write_text(json.dumps({"keys": list(keys)}, indent=2), encoding="utf-8")


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

    df = pd.read_csv(bets_file)
    tips = df[
        df["signal_tier"].isin(["SNIPER", "MARKSMAN", "VALUABLE"]) &
        df["bet"].isin(["UNDER", "OVER"])
    ].copy()

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
            header = f"💎 <b>VALUABLE TIP</b> {drift}"

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
    strong = df[df["signal"] == "STRONG"].copy()
    if strong.empty:
        return 0

    notified = _load_notified()
    sent = 0

    for _, row in strong.iterrows():
        key = f"WC|{str(row['date'])[:10]}|{row['match']}|{row['market']}"
        if key in notified:
            continue

        direction = "▼ Shortening" if row["drift_pct"] < 0 else "▲ Lengthening"
        msg = (
            f"🌍 <b>WORLD CUP SHARP MONEY</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📅 {str(row['date'])[:10]}\n"
            f"⚽ {row['match']}\n"
            f"📌 <b>{row['market']}</b>\n"
            f"💰 Opening: {row['opening_odds']} → Now: {row['current_odds']}\n"
            f"📉 Drift: <b>{row['drift_pct']:+.1f}%</b>  {direction}\n"
            f"🔍 Based on {row['snapshots']} snapshots"
        )

        if _send(token, chat_id, msg):
            notified.add(key)
            sent += 1
            print(f"  WC Sent: {row['match']} — {row['market']} {row['drift_pct']:+.1f}%")

    _save_notified(notified)
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

    notified = _load_notified()
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
            _save_notified(notified)
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
    Send weekly performance summary every Monday.
    Groups results by model format (Standard / New-Format) and signal tier (SNIPER / VALUE).
    Returns True if message was sent.
    """
    cfg = _load_config()
    token   = cfg.get("token", "")
    chat_id = cfg.get("chat_id", "")
    if not token or token == "YOUR_BOT_TOKEN":
        return False

    ledger_file = app_config.OUTPUT_DIR / "bets_ledger.csv"
    if not ledger_file.exists():
        return False

    df = pd.read_csv(ledger_file)
    df["pnl"]        = pd.to_numeric(df["pnl"],        errors="coerce")
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df["generated_at"] = pd.to_datetime(df["generated_at"], errors="coerce")

    # Last 7 days of settled bets
    week_ago = datetime.now() - __import__("datetime").timedelta(days=7)
    week = df[
        (df["source"] == "live") &
        df["pnl"].notna() &
        (df["match_date"] >= week_ago)
    ].copy()

    # All-time live settled
    alltime = df[(df["source"] == "live") & df["pnl"].notna()].copy()

    def tier_stats(data, tier):
        t = data[data["signal_tier"] == tier]
        if t.empty:
            return None
        n = len(t); w = (t["pnl"] > 0).sum(); pnl = t["pnl"].sum()
        return {"n": n, "win": w/n, "roi": pnl/n*100, "pnl": pnl}

    def format_tier(label, stats, emoji):
        if not stats or stats["n"] == 0:
            return f"  {emoji} {label}: no bets this week"
        roi_sign = "+" if stats["roi"] >= 0 else ""
        pnl_sign = "+" if stats["pnl"] >= 0 else ""
        return (f"  {emoji} {label}: <b>{stats['n']}</b> bets | "
                f"{stats['win']:.0%} win | "
                f"ROI <b>{roi_sign}{stats['roi']:.1f}%</b> | "
                f"PnL {pnl_sign}{stats['pnl']:.2f}u")

    week_start = (datetime.now() - __import__("datetime").timedelta(days=7)).strftime("%b %d")
    week_end   = datetime.now().strftime("%b %d, %Y")

    # Weekly section
    std_week  = week[week["model_type"] == "standard"]    if "model_type" in week.columns else week
    nf_week   = week[week["model_type"] == "new_format"]  if "model_type" in week.columns else pd.DataFrame()

    # All-time section
    std_all   = alltime[alltime["model_type"] == "standard"]   if "model_type" in alltime.columns else alltime
    nf_all    = alltime[alltime["model_type"] == "new_format"] if "model_type" in alltime.columns else pd.DataFrame()

    # Build message
    lines = [
        f"📊 <b>WEEKLY SUMMARY</b>",
        f"━━━━━━━━━━━━━━━━",
        f"📅 {week_start} → {week_end}",
        f"",
    ]

    # This week
    if week.empty:
        lines.append("  No settled bets this week.")
    else:
        n_w = len(week); w_w = (week["pnl"]>0).sum(); pnl_w = week["pnl"].sum()
        lines += [
            f"<b>This week ({n_w} bets | {w_w/n_w:.0%} win | PnL {pnl_w:+.2f}u)</b>",
            f"",
            f"🏴 <b>Standard Model</b>",
            format_tier("SNIPER",   tier_stats(std_week, "SNIPER"),   "🎯"),
            format_tier("MARKSMAN", tier_stats(std_week, "MARKSMAN"), "🔫"),
            format_tier("VALUABLE", tier_stats(std_week, "VALUABLE"), "💎"),
        ]
        if not nf_week.empty:
            lines += [
                f"",
                f"🌍 <b>New-Format Model</b>",
                format_tier("SNIPER",   tier_stats(nf_week, "SNIPER"),   "🎯"),
                format_tier("MARKSMAN", tier_stats(nf_week, "MARKSMAN"), "🔫"),
                format_tier("VALUABLE", tier_stats(nf_week, "VALUABLE"), "💎"),
            ]

    # All-time
    lines += ["", "━━━━━━━━━━━━━━━━"]
    if not alltime.empty:
        n_a = len(alltime); w_a = (alltime["pnl"]>0).sum(); pnl_a = alltime["pnl"].sum()
        lines += [
            f"📈 <b>All-time live ({n_a} bets)</b>",
            format_tier("SNIPER",   tier_stats(alltime, "SNIPER"),   "🎯"),
            format_tier("MARKSMAN", tier_stats(alltime, "MARKSMAN"), "🔫"),
            format_tier("VALUABLE", tier_stats(alltime, "VALUABLE"), "💎"),
            f"  Total PnL: <b>{pnl_a:+.2f}u</b>",
        ]

    msg = "\n".join(lines)
    if _send(token, chat_id, msg):
        print(f"Weekly summary sent.")
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

    # Only SNIPER-match players with high model probability
    df = df[
        (df["match_tier"] == "SNIPER") &
        (df["model_prob"] >= 0.60)
    ].sort_values("model_prob", ascending=False).head(10)

    if df.empty:
        return 0

    notified = _load_notified()
    sent = 0

    MARKET_EMOJI = {"goals": "⚽", "assists": "🎯", "sot": "🔫", "cards": "🟨"}

    for _, row in df.iterrows():
        key = f"PLAYER|{str(row['date'])[:10]}|{row['player_name']}|{row['market']}"
        if key in notified:
            continue

        emoji  = MARKET_EMOJI.get(row["market"], "📌")
        p      = float(row["model_prob"])
        fair   = float(row["fair_odds"])
        market_label = {"goals": "Anytime Goalscorer", "assists": "Assist",
                        "sot": "Shot on Target", "cards": "Yellow Card"}.get(row["market"], row["market"])

        msg = (
            f"{emoji} <b>PLAYER PROP — {market_label.upper()}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"👤 <b>{row['player_name']}</b> ({row.get('position', '')} · {row['team']})\n"
            f"⚽ {row['match']}\n"
            f"🏆 {row.get('league', '')}  |  📅 {str(row['date'])[:10]}\n"
            f"📊 Model P: <b>{p*100:.0f}%</b>  |  Fair Odds: <b>{fair:.2f}</b>\n"
            f"🎯 Match signal: {row.get('match_tier', '')}\n"
            f"⚠️ Check your bookmaker's player props market"
        )

        if _send(token, chat_id, msg):
            notified.add(key)
            sent += 1
            _save_notified(notified)
            print(f"  Player prop: {row['player_name']} — {market_label} ({p*100:.0f}%)")

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
        n      = notify_new_snipers()
        ht     = notify_ht_tips()
        live   = notify_live_signals()
        wc     = notify_wc_strong()
        sharp  = notify_sharp_strong()
        props  = notify_player_props()
        # agent = notify_agent_analysis()  # disabled — rebuilding with team-specific λ
        print(f"Notifications sent: {n} SNIPER + {ht} HT + {live} Live + {wc} WC + {sharp} Sharp + {props} Props")
