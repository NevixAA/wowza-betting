"""
Telegram Notifier — sends tip alerts after each predict run.

Tier WORDS in every message come from src/tier_labels.py (Big / Medium / Small gap, owner
2026-10-26). The codes SNIPER / MARKSMAN / VALUABLE stay the internal names: ledgers, filters
and the dedup keys in notified.json are unchanged.

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
import os
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
from src import clv_corrections as _CC
from src import league_registry as _LRG
from src import units as U
from src.utc_clock import utc_now as _utc_now, utc_today_str as _utc_today_str
from src.tier_labels import GAP_EXPLAINER, TIER_ICON, tier_label, tier_tag
from src.calibrated_edge import edge_text   # owner 2026-11-11: the only edge shown on a future tip
from player_model.fair_basis import label as _fair_basis_label


def _read_csv(*args, **kwargs):
    """Every CSV this module reads, with paper leagues removed.

    PAPER LEAGUES ARE COLLECTED, NEVER SENT AND NEVER COUNTED (config.PAPER_LEAGUES). The
    files this module reads -- predictions, the ledgers, live and sharp tips -- all still
    contain paper-league rows, because they are the collection record. So the filter has to
    happen here, at the point of reading, not at the point of writing.

    ONE READER RATHER THAN ~37 SEPARATE FILTERS. Filtering inside each send and KPI function
    would mean 37 places to get right and one forgotten path leaking a paper league into a
    Telegram tip or a digest total. Routing every read through one function makes the rule
    impossible to forget. A frame with no `league` column passes through unchanged.
    """
    return app_config.drop_paper_leagues(pd.read_csv(*args, **kwargs))


def _fmt_kickoff(row) -> str:
    """Display a fixture's date/time in the LEAGUE'S LOCAL timezone.

    OddsAPI `commence_time` is UTC, so a match's local calendar day can differ from
    its UTC day (Japan, Argentina, Brazil, …). We convert `kickoff_utc` to the league's
    local tz for what the user reads. The stored `date` stays UTC (join/dedup key) and
    is NOT touched. Falls back to the UTC date string when `kickoff_utc` is missing or
    unparseable. When the registry holds no zone for the league (multi-country cups, unknown
    names) the time is shown in UTC and LABELLED 'UTC' - never passed off as local time.
    """
    fallback = str(row.get("date", ""))[:10]
    ko = row.get("kickoff_utc", None)
    if ko is None or str(ko) in ("", "nan", "NaT", "None"):
        return fallback
    try:
        ts = pd.to_datetime(ko, utc=True, errors="coerce")
        if pd.isna(ts):
            return fallback
        tz = _LRG.timezone(row.get("league"))
        if tz is None:
            return f"{ts.strftime('%Y-%m-%d %H:%M')} UTC"
        local = ts.tz_convert(tz)
        abbr = local.strftime("%Z")
        suffix = f" {abbr}" if abbr else f" ({tz})"
        return f"{local.strftime('%Y-%m-%d %H:%M')}{suffix}"
    except Exception as e:
        print(f"[kickoff] could not localise {ko!r} for {row.get('league')!r}: {e} - showing UTC date")
        return fallback


def _fixture_bucket(league, date) -> tuple:
    """Index key for per-fixture lookups: (canonical league, YYYY-MM-DD). An unregistered
    league keeps its raw name, so it can still only meet rows spelled the same way."""
    return (_LRG.canonical_name(league) or str(league), str(date)[:10])


def _fixture_lookup(index: dict, league, date, home, away):
    """Value stored for one fixture in `index` {(league, date): {(home, away): value}}.

    Clubs are matched by src/team_names.resolve inside ONE league and ONE date - never by a
    lower-cased/punctuation-stripped string - so differently spelled sources still meet and an
    ambiguous name finds nothing (None) instead of a neighbour's fixture."""
    from src.team_names import resolve as _resolve_club
    pairs = index.get(_fixture_bucket(league, date))
    if not pairs:
        return None
    h = _resolve_club(str(home), sorted({p[0] for p in pairs}))
    if h is None:
        return None
    a = _resolve_club(str(away), sorted({p[1] for p in pairs if p[0] == h}))
    return None if a is None else pairs.get((h, a))


def _model_type_of(league) -> str:
    """config.model_type_for_league on the CANONICAL league name, so a row written under a
    registered alias is not reported as 'unknown'."""
    return app_config.model_type_for_league(_LRG.canonical_name(league) or league)


def _is_prop_league(leagues: "pd.Series") -> "pd.Series":
    """Boolean mask: the row's league (canonical or alias) is flagged `props` in the registry.
    Rows of other / unregistered leagues are dropped and counted."""
    props = set(_LRG.prop_leagues())
    mask = leagues.map(lambda lg: _LRG.canonical_name(lg) in props).astype(bool)
    if (~mask).any():
        print(f"[props] {int((~mask).sum())} row(s) outside registry prop leagues dropped")
    return mask


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


def _settled_only(df):
    """Rows that truly settled WIN/LOSS. VOID (stake returned, pnl=0) and pending are
    excluded from ALL ROI / win-rate maths — a void is not a loss and not a data point.
    (Voids store pnl=0, so a plain pnl.notna() filter wrongly keeps them.)"""
    if df is None or len(df) == 0:
        return df
    if "result" in df.columns:
        return df[df["result"].astype(str).str.upper().isin(["WIN", "LOSS"])].copy()
    if "pnl" in df.columns:
        return df[df["pnl"].notna()].copy()
    return df


# Prop tiers that are NOT picks. We deliberately TRACK every tier in player_ledger.csv +
# clv_records.csv (the 2026-08-13 "track everything" change), but an AVOID is the model
# saying "no bet here" — counting it as a settled bet is a category error.
NON_BET_TIERS = {"AVOID", "WATCH"}

# Tiers that are actually STAKED. VALUABLE is sent live on purpose — as a "flash" that
# accumulates outcome data for the model to learn from — but NO MONEY goes on it. Folding it
# into headline P/L inverts the conclusion: weekend 2026-08-14..16 read as 15W/30L −12.31u
# across all tiers, and 14W/15L +0.47u on staked tiers alone. Summaries must therefore lead
# with staked P/L and show VALUABLE separately, labelled as collection.
STAKED_TIERS = ["SNIPER", "MARKSMAN"]
COLLECT_TIERS = ["VALUABLE"]
#: tiers in display order, for a breakdown of STAKED rows (an O/U 2.5 VALUABLE tip can be staked)
ALL_TIERS = STAKED_TIERS + COLLECT_TIERS


def _staked_rows(df, ledger: str):
    """THE staked rows of a ledger frame (src/staked.py, owner 2026-11-04): the bankroll policy
    recorded a stake > 0 on a real-money track. Until then every digest headline here kept the
    SNIPER + MARKSMAN tiers, PAPER tracks included. A frame without the label columns, or with no
    stake record, proves no stake (empty), never a tier guess."""
    if df is None or len(df) == 0:
        return df
    from src import staked as _S
    lab = _S.label_ledger(df, ledger, _S.load_records(app_config.OUTPUT_DIR))
    return df[_S.is_staked(lab).to_numpy()]


def _unstaked_rows(df, staked):
    """Every row of `df` that is not in `staked` (same index): tracked, never money."""
    if df is None or len(df) == 0:
        return df
    return df.drop(index=staked.index, errors="ignore")


def _since_cutoff(df):
    """Rows whose MATCH DATE is on/after PERFORMANCE_CUTOFF_DATE - the cutoff clause of THE staked
    definition (dashboard_data.today_view). Until 2026-11-02 the digests cut on generated_at, which
    counted 764 staked rows (-3.06u) where Home/Tips/Leagues/Overview count 805 (-8.45u). A row
    without a readable match date is not counted (it cannot be placed after the cutoff)."""
    if df is None or len(df) == 0 or "match_date" not in df.columns:
        return df
    day = df["match_date"].astype(str).str[:10]
    ok = day.str.fullmatch(r"\d{4}-\d{2}-\d{2}") & (day >= str(app_config.PERFORMANCE_CUTOFF_DATE)[:10])
    return df[ok.to_numpy()]


def _bet_tiers_only(df):
    """Drop non-bet prop rows (AVOID/WATCH) before ANY count / P&L / digest line.

    Tracking and reporting are different jobs: the ledger keeps everything so the model can
    be evaluated on its full output, while the digests must only ever report things we
    actually called a pick. Without this, 2026-08-14 reported 101 'results' of which all 101
    were AVOID. Keyed on the props `tier` column; the team ledgers use `signal_tier` and are
    already tier-filtered at write time by generate_bets(), so they don't need this.
    """
    if df is None or len(df) == 0 or "tier" not in df.columns:
        return df
    keep = ~df["tier"].astype(str).str.strip().str.upper().isin(NON_BET_TIERS)
    return df[keep].copy()


_TEAM_LEAGUES = None


def _team_league_map() -> dict:
    """club -> set of competitions we hold player history for (cached per process)."""
    global _TEAM_LEAGUES
    if _TEAM_LEAGUES is None:
        try:
            pq = pd.read_parquet(app_config.BASE_DIR / "player_history.parquet",
                                 columns=["team", "league"])
            _TEAM_LEAGUES = (pq.assign(_t=pq["team"].astype(str))
                             .groupby("_t")["league"].agg(set).to_dict())
        except Exception:
            _TEAM_LEAGUES = {}
    return _TEAM_LEAGUES


def _valid_fixture_rows(df):
    """Drop prop rows whose player belongs to NEITHER team in the fixture.

    Until 2026-08-15 the squad lookup fell back to a 5-character substring match, so players
    were attached to unrelated matches - Real Madrid names in an MLS fixture (Real Salt
    Lake), Messi in Sarmiento vs Argentinos Juniors, Monchengladbach in a China Super League
    game. 207 of 1983 player_ledger rows (10%) are wrong that way. predict.py no longer
    creates them, but they are already written into the ledger and clv_records, so every
    performance number computed from that history would stay polluted. Filtering here cleans
    the reporting without deleting the tracking rows.
    """
    if df is None or len(df) == 0:
        return df
    if not {"team", "home_team", "away_team"}.issubset(df.columns):
        return df
    # Club identity goes through the ONE resolver (src/team_names), not the player model's own
    # fuzzy helpers. The ledger has no team id, so names are all there is.
    from src.team_names import resolve as _resolve_club, _tokens as _club_tokens
    tl = _team_league_map()

    def _ok(team, home, away, league) -> bool:
        team, home, away = str(team), str(home), str(away)
        hit = _resolve_club(team, [home, away])
        if hit is None:
            return False
        if _club_tokens(team) == _club_tokens(hit):
            return True
        # Subset/alias match ("Plymouth" vs "Plymouth Argyle" = same club, "Inter" vs
        # "Inter Club d'Escaldes" = not): trust it only where we hold history for that
        # club in THIS competition. Same rule predict.py applies.
        lg = _LRG.canonical_name(league) or str(league)
        return lg in tl.get(team, ()) or str(league) in tl.get(team, ())

    if "league" not in df.columns:
        # Without the competition the subset rule cannot be corroborated; those rows are dropped.
        print("[props] ledger frame has no league column - subset club matches cannot be corroborated")
        leagues = pd.Series([None] * len(df), index=df.index)
    else:
        leagues = df["league"]
    keep = pd.Series([_ok(t, h, a, lg) for t, h, a, lg in
                      zip(df["team"], df["home_team"], df["away_team"], leagues)], index=df.index)
    if (~keep).any():
        print(f"[props] {int((~keep).sum())} prop row(s) dropped: player's club is neither side of the fixture")
    return df[keep].copy()


def _countable_props(df):
    """Prop ledger rows that may enter a count, a P/L figure, or a digest line:
    a real pick (not AVOID/WATCH) on a fixture the player was actually in."""
    return _valid_fixture_rows(_bet_tiers_only(df))


# -- TIP TRUTH GATE (owner 2026-11-24 "no duplicates, no unreal data, not unreal markets") --------------------------
# Every tip sender below asks src/tip_truth before it sends: ONE real API-Football fixture, a real named-book
# two-sided bet-time price captured before the send, a market with its league's own applied bar quoted by that book,
# and no second tip on the same canonical match x market x side (across senders: the TOP-3 ledger + this run).
# FAIL CLOSED: the gate cannot load -> the tip is not sent. Market-movement alerts (sharp jumps, WC drift) carry no
# pick to bet, so they are information, not tips, and are not gated.
_TRUTH = None
_TRUTH_SENT: set = set()


def _truth_gate():
    global _TRUTH, _TRUTH_SENT
    if _TRUTH is None:
        try:
            from src import tip_truth as TT
            from src import top3_league as TL
            ctx = TL.bars_context()
            team_ok = TL.truth_bars_ok(ctx)

            def bars_ok(league, market, side):
                m = str(market)
                if m.startswith("HT_"):
                    return ht_bar(str(league), m.split("_", 1)[1], str(side))[0] is not None
                return team_ok(league, m, side)
            _TRUTH = TT.default(bars_ok=bars_ok)
            _TRUTH_SENT = TT.prior_keys(_TRUTH, TL.picks_store(app_config.OUTPUT_DIR).read(), "team")
        except Exception as e:                                          # noqa: BLE001
            print(f"  TRUTH GATE unavailable ({type(e).__name__}: {e}) - no tip is sent")
            _TRUTH = False
    return _TRUTH


def _truth_check(row, market: str, side: str, odds=None, opposite_odds=None, book="", captured_at="",
                 home=None, away=None) -> tuple:
    """('' or the blocking reason, canonical key). Never prints a price."""
    t = _truth_gate()
    if not t:
        return "TRUTH_GATE_UNAVAILABLE", None
    from datetime import timezone as _tz
    from src import tip_truth as TT
    r = {"league": row.get("league"), "kickoff_utc": row.get("kickoff_utc") or row.get("date"),
         "match_date": str(row.get("date") or row.get("match_date") or "")[:10],
         "home_team": home if home is not None else row.get("home_team"),
         "away_team": away if away is not None else row.get("away_team"),
         "market": market, "side": side, "odds": odds, "opposite_side_odds": opposite_odds,
         "entry_bookmaker": book, "entry_captured_at": captured_at}
    why, info = t.check(r, datetime.now(_tz.utc))
    key = (f"af:{info['af_fixture_id']}", market, side) if info.get("af_fixture_id") is not None else None
    if not why and key in _TRUTH_SENT:
        why = TT.DUPLICATE
    return why, key


def _truth_commit(key) -> None:
    if key is not None:
        _TRUTH_SENT.add(key)


def send_fantasy_tips(max_per_pos: int = 3) -> int:
    """FANTASY (FPL) family — captaincy + top picks per position. PREDICTIONS, not bets;
    kept visibly separate from the SNIPER/MARKSMAN betting tips.
    GUARDED: fires only when there ARE upcoming fixtures in the registry's FPL league — stays silent
    off-season so it never posts empty projections. Returns 1 if sent, else 0."""
    from pathlib import Path as _P
    import pandas as _pd
    # in-season guard: no upcoming PL GW -> stay silent
    # The FPL competition and its API-Football id come from the registry (fpl flag), not a literal.
    fpl_lg = _LRG.fpl_league()
    fpl_id = _LRG.prop_leagues().get(fpl_lg) if fpl_lg else None
    if fpl_lg is None or fpl_id is None:
        print(f"[fantasy] registry has no FPL league with an API-Football id ({fpl_lg!r}) — skipping")
        return 0
    try:
        from player_model.api_football import get_upcoming_fixtures
        from src.season_resolver import resolve_api_season, is_known
        season = resolve_api_season(fpl_lg)
        if not is_known(season):
            print(f"[fantasy] season unknown for {fpl_lg} — skipping")
            return 0
        fx = get_upcoming_fixtures(fpl_id, season, next_n=1)
        if not fx:
            print(f"[fantasy] no upcoming {fpl_lg} fixtures — skipping (off-season)")
            return 0
    except Exception as e:
        print(f"[fantasy] fixture guard failed ({e}) — skipping")
        return 0
    f = _P(__file__).resolve().parents[1] / "output" / "fantasy_tips.csv"
    if not f.exists():
        print(f"[fantasy] {f.name} missing — skipping")
        return 0
    try:
        # was `__read_csv` (NameError swallowed below), so this family never sent.
        df = _read_csv(f)
    except Exception as e:
        print(f"[fantasy] could not read {f.name} ({e}) — skipping")
        return 0
    if df.empty:
        return 0
    cfg = _load_config()
    token, chat_id = cfg.get("token", ""), cfg.get("chat_id", "")
    if not token or token == "YOUR_BOT_TOKEN":
        return 0
    POS = {"F": "Forwards", "M": "Midfielders", "D": "Defenders", "G": "Keepers"}
    cap = df.iloc[0]
    lines = ["⚽ <b>FANTASY / FPL TIPS</b>",
             "<i>Model predictions — not betting tips</i>", "",
             f"👑 <b>Captain:</b> {cap['player_name']} ({cap['position']}) — {cap['fantasy_pts']:.1f} pts", ""]
    for pos in ["F", "M", "D"]:
        sub = df[df["position"] == pos].head(max_per_pos)
        if sub.empty:
            continue
        picks = ", ".join(f"{r.player_name} ({r.fantasy_pts:.1f})" for r in sub.itertuples())
        lines.append(f"<b>{POS[pos]}:</b> {picks}")
    return 1 if _send(token, chat_id, "\n".join(lines)) else 0


def _split_for_telegram(text: str, max_len: int = 4000) -> list:
    """Split a message on line boundaries to stay under Telegram's 4096-char limit.
    Each digest line closes its own HTML tags, so per-chunk HTML stays balanced.
    (Fix for 'Bad Request: message is too long' on the daily digest.)"""
    if len(text) <= max_len:
        return [text]
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(line) > max_len:                      # pathological single long line
            if cur:
                chunks.append(cur); cur = ""
            for i in range(0, len(line), max_len):
                chunks.append(line[i:i + max_len])
            continue
        if cur and len(cur) + len(line) + 1 > max_len:
            chunks.append(cur); cur = line
        else:
            cur = line if not cur else cur + "\n" + line
    if cur:
        chunks.append(cur)
    return chunks


def _send(token: str, chat_id: str, text: str) -> bool:
    """Send a message, auto-splitting anything over the Telegram length limit."""
    from dashboard_words import relabel_text     # owner 2026-11-17: no "paper" in anything a user reads
    text = relabel_text(text)
    ok = True
    for chunk in _split_for_telegram(text):
        ok = _send_one(token, chat_id, chunk) and ok
    return ok


def _send_one(token: str, chat_id: str, text: str) -> bool:
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


def _fmt_num(v, spec: str, scale: float = 1.0) -> str:
    """`v` formatted with `spec`, or '–' when it is missing/unparseable. A missing price or edge
    is shown as missing, never as 0.00 / +0%."""
    x = pd.to_numeric(v, errors="coerce")
    return "–" if pd.isna(x) else format(float(x) * scale, spec)


_NO_DRIFT = "no drift data"


def _drift_text(signal) -> str:
    """The recorded drift signal, or 'no drift data' when none was recorded. A blank is not
    'New' - 'New' is a real signal (first snapshot) and must not be invented."""
    if signal is None or (isinstance(signal, float) and pd.isna(signal)):
        return _NO_DRIFT
    s = str(signal).strip()
    return s if s and s.lower() not in ("nan", "none") else _NO_DRIFT


def instant_tip_allowed(match_date, notified: set, now: "datetime | None" = None, row=None) -> bool:
    """Should a NEW tip go out as its own message right now? (owner 2026-10-18: "too spammy").

    No by default: every tip appears in its match day's morning album (telegram_bot/daily_pack),
    which also marks each tip it shows as notified, so nothing is sent twice. Yes only when the
    album cannot carry it any more - the match is TODAY and either today's album has already gone
    (DIGEST|<today> recorded: a tip first found after it, ~3% of tips measured since 2026-09-15) or
    it is past 12:00 UTC and no album went out at all (the scheduler failed; tips must not vanish).
    TIPS_INSTANT_ALERTS=1 restores the old every-tip behaviour."""
    if os.getenv("TIPS_INSTANT_ALERTS", "").strip() == "1":
        return True
    from datetime import timezone as _tz
    now = now or datetime.now(_tz.utc)
    today = now.strftime("%Y-%m-%d")
    # judged by TIP DAY (src/tip_day.py, 2026-10-25): tonight's 03:00 UTC Liga MX match has tomorrow's
    # UTC date but belongs to today's album, so a tip found after that album must still go out alone.
    tday = str(match_date)[:10]
    if row is not None:
        try:
            from src.tip_day import add_tip_day
            r = {k: row.get(k) for k in ("league", "home_team", "away_team", "kickoff_utc")}
            r["match_date"] = tday
            tday = add_tip_day(pd.DataFrame([r]))["tip_day"].iloc[0]
        except Exception:                                               # noqa: BLE001 - keep the match date
            pass
    if tday != today:
        return False
    return f"DIGEST|{today}" in notified or now.hour >= 12


def _paper_track_note(model_type: str, market: str, side: str, league=None) -> str:
    """Always '' (owner 2026-11-17, "Show stake on TOP-3"): an ordinary prediction carries NO stake label at all.

    From 2026-11-21 the only real stake is the flat TOP-3 stake (config/bankroll_policy.json top3_league.stake,
    placed manually); the TOP-3 message (src/top3_league.message) shows "0.5% stake" on each pick. Every cell of
    the policy is PAPER, so the per-cell line this used to add (go-live rehearsal 2026-11-15) would now sit on
    every tip; the owner asked for no label instead. The name is kept for the callers."""
    return ""


def _coherence_gate(gate, notified: set | None = None):
    """The fixture coherence gate (src/tip_coherence, owner 2026-11-19): `gate` when a caller planned one for the
    whole run, else a fresh one over every tip already sent (this module's notified store + sent TOP-3 picks)."""
    if gate is not None:
        return gate
    from src.tip_coherence import CoherenceGate
    return CoherenceGate.load(notified_keys=notified if notified is not None else _load_notified())


def _ou_key(row) -> str:
    return f"{str(row['date'])[:10]}|{row['home_team']}|{row['away_team']}|{row.get('best_side','')}"


def _ou_candidate(row):
    from src.tip_coherence import Candidate, Event, calibrated_ev, fixture_id
    side = str(row.get("best_side") or row.get("bet") or "").upper()
    odds = row.get("odds_under25") if side == "UNDER" else row.get("odds_over25")
    return Candidate(fixture_id(row["date"], row["home_team"], row["away_team"]),
                     Event("FT", "TOTAL", side, 2.5), "O/U 2.5",
                     rank=calibrated_ev(row.get("edge_calibrated"), odds, row.get("best_edge")), uid=_ou_key(row))


def _ou_pending() -> "pd.DataFrame":
    """bets.csv tips notify_new_snipers would consider (tiered, O/U, match day today or later)."""
    bets_file = app_config.OUTPUT_DIR / "bets.csv"
    if not bets_file.exists():
        return pd.DataFrame()
    df = _read_csv(bets_file)
    from src.tier_labels import TIPPED
    tips = df[df["signal_tier"].isin(TIPPED) & df["bet"].isin(["UNDER", "OVER"])].copy()
    return tips[tips["date"].astype(str).str[:10] >= _utc_today_str()]


def plan_run_gate(notified: set | None = None):
    """ONE gate for a predict run's three senders (O/U 2.5, side markets, HT): every candidate they could send
    is planned together, best calibrated EV first, so between two mismatched tips of the same run the
    higher-ranked one stands whichever sender carries it (src/tip_coherence.CoherenceGate.plan)."""
    notified = notified if notified is not None else _load_notified()
    gate = _coherence_gate(None, notified)
    cands = []
    try:
        cands += [_ou_candidate(r) for _, r in _ou_pending().iterrows() if _ou_key(r) not in notified]
    except Exception as e:                                              # noqa: BLE001
        print(f"  coherence plan: O/U candidates unreadable ({type(e).__name__}: {e})")
    try:
        cands += [_side_candidate(r) for _, r in _side_pending().iterrows() if _side_key(r) not in notified]
    except Exception as e:                                              # noqa: BLE001
        print(f"  coherence plan: side candidates unreadable ({type(e).__name__}: {e})")
    try:
        b = _ht_board()
        if not b.empty:
            cands += [_ht_candidate(c) for c in ht_tip_candidates(b) if not c["reason"]
                      and _ht_key(c) not in notified]
    except Exception as e:                                              # noqa: BLE001
        print(f"  coherence plan: HT candidates unreadable ({type(e).__name__}: {e})")
    losers = gate.plan(cands)
    print(f"  coherence gate: {len(cands)} candidate tip(s) planned, {len(losers)} lose to a mismatch")
    return gate


def notify_new_snipers(gate=None) -> int:
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

    df = _read_csv(bets_file)
    # Classify model (blanks -> by league) so VALUABLE can be gated to the standard model only.
    if "model_type" not in df.columns:
        df["model_type"] = ""
    _blk = df["model_type"].isna() | (df["model_type"].astype(str).str.strip().isin(["", "nan"]))
    df["model_type"] = df["model_type"].astype(object)   # an all-blank column reads as float
    df.loc[_blk, "model_type"] = df.loc[_blk, "league"].map(_model_type_of)
    # Send SNIPER + MARKSMAN + VALUABLE for ALL models (Nevo 2026-08-09: enable new-format
    # VALUABLE live too, now that the NF model is recalibrated). Dedup (notified.json) sends
    # each fixture+side once, so the higher NF volume won't re-flood.
    from src.tier_labels import TIPPED              # QUALIFIED (owner 2026-11-11) + legacy codes of history rows
    tips = df[
        df["signal_tier"].isin(TIPPED) &
        df["bet"].isin(["UNDER", "OVER"])
    ].copy()

    # Live-test policy (2026/27): send SNIPER + MARKSMAN for ALL leagues — including
    # negative-ROI ("losing") ones — so every league is tracked live this season. The
    # league-ROI approval gate that previously muted MARKSMAN on unapproved leagues is
    # intentionally disabled; approval now informs real-money sizing only, not sending.
    # (_approved_leagues stays available above for labelling if needed.)

    # Only upcoming games — never send finished matches
    today_str = _utc_today_str()   # UTC day: stored dates are UTC (src/utc_clock.py)
    tips = tips[tips["date"].astype(str).str[:10] >= today_str]

    if tips.empty:
        return 0

    notified = _load_notified()
    gate = _coherence_gate(gate, notified)
    sent = 0

    for _, row in tips.iterrows():
        key = _ou_key(row)
        if key in notified:
            continue

        # NEVER TIP BOTH SIDES OF THE SAME FIXTURE.
        #
        # The dedup key above ends in the side, so a fixture already tipped OVER does not
        # match when the model later says UNDER -- different key, straight through the gate.
        # 2026-09-26 Shrewsbury Town v Colchester United went out as OVER MARKSMAN on 09-22
        # and UNDER VALUABLE on 09-23, after a retrain moved p(over) from 0.557 to 0.360.
        #
        # Two tips that contradict each other are worse than either one alone: backing both
        # sides of the same match is a guaranteed loss of the vig, and it destroys any
        # reading of the ledger, because the fixture now appears as two bets that cannot
        # both be right. This is not a model question -- it is wrong under whichever model
        # turns out to be better.
        #
        # The FIRST opinion stands. A model that changes its mind after a fixture has been
        # tipped is telling us something about the model, not offering a second bet. The
        # place to act on that is the retrain canary, not another Telegram message.
        _side_now = str(row.get("best_side", "") or "")
        _opposite = {"OVER": "UNDER", "UNDER": "OVER"}.get(_side_now)
        if _opposite:
            _opp_key = (f"{str(row['date'])[:10]}|{row['home_team']}|"
                        f"{row['away_team']}|{_opposite}")
            if _opp_key in notified:
                print(f"  SKIP both-sides: {row['home_team']} vs {row['away_team']} — "
                      f"already tipped {_opposite}, model now says {_side_now}")
                continue

        # Goes out in its day's morning album instead (not marked notified here: the album marks it).
        if not instant_tip_allowed(row["date"], notified, row=row):
            continue

        # ONE MATCH, NO MISMATCHED TIPS (owner 2026-11-19, src/tip_coherence): not BTTS Yes + Under 2.5, not an
        # HT Over + FT Under, from any sender. The first tip sent stands; within a run the higher-ranked one.
        _cand = _ou_candidate(row)
        _why = gate.check(_cand)
        if _why:
            print(f"  SKIP {row['home_team']} vs {row['away_team']} O/U 2.5 {_cand.event.side}: {_why}")
            continue

        tier  = row["signal_tier"]
        side  = row.get("best_side") or row.get("bet", "")
        odds  = row["odds_under25"] if side == "UNDER" else row["odds_over25"]
        _col = "odds_under25" if side == "UNDER" else "odds_over25"
        _twhy, _tkey = _truth_check(row, "OU25", str(side), odds,
                                    row.get("odds_over25" if side == "UNDER" else "odds_under25"),
                                    row.get(f"{_col}_book", ""), row.get(f"{_col}_captured_at", ""))
        if _twhy:
            print(f"  TRUTH BLOCK {row['home_team']} vs {row['away_team']} O/U 2.5 {side}: {_twhy}")
            continue
        # A tip goes out only with its recorded edge, price and model; a missing one is a
        # skipped send (logged), never a 0% edge or a guessed model label.
        edge  = pd.to_numeric(row.get("best_edge"), errors="coerce") * 100
        odds  = pd.to_numeric(odds, errors="coerce")
        _mt   = str(row.get("model_type") or "").strip()
        if pd.isna(edge) or pd.isna(odds) or _mt not in ("standard", "new_format"):
            print(f"  SKIP unsendable tip: {row['home_team']} vs {row['away_team']} [{row.get('league')}] — "
                  f"edge={row.get('best_edge')!r} odds={odds!r} model_type={_mt!r}")
            continue
        _drift_sig = _drift_text(row.get("drift_signal"))
        drift = _drift_emoji(_drift_sig) if _drift_sig != _NO_DRIFT else ""
        _is_std = _mt == "standard"
        model = "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Standard" if _is_std else "🌍 New-Format"
        mtag  = "standard" if _is_std else "new-format"

        # Owner 2026-11-11 ("Clean rebuild"): no tier word on a future tip. The header names the tip and its
        # CELL STATUS (PROMISING / UNPROVEN) - the status of the cell's one approved bar.
        _status = str(row.get("cell_status") or "").strip().upper()
        if tier == "QUALIFIED":
            header = f"▸ <b>TIP</b> · {_status or 'UNPROVEN'} ({mtag}) {drift}"
        elif tier in ("SNIPER", "MARKSMAN"):
            header = f"{TIER_ICON[tier]} <b>{tier_label(tier).upper()} TIP</b> {drift}"
        elif tier == "VALUABLE":
            header = f"{TIER_ICON[tier]} <b>{tier_label(tier).upper()} TIP</b> ({mtag}) {drift}"
        else:
            header = f"🔍 <b>EDGE WATCH</b> (small edge, not a tip) {drift}"

        msg = (
            f"{header}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📅 {_fmt_kickoff(row)}\n"
            f"🏆 {row.get('league','')}\n"
            f"⚽ {row['home_team']} vs {row['away_team']}\n"
            f"📌 <b>{side} 2.5</b>  @ {odds:.2f}\n"
            + (f"📊 {edge_text(pd.to_numeric(row.get('edge_calibrated'), errors='coerce') * 100)}  |  {model}\n"
               if tier == "QUALIFIED" else f"📊 Edge: <b>{edge:.1f}%</b>  |  {model}\n") +
            f""
            # Drift is information only since 2026-10-26: it never changes the tier above.
            f"🔀 Drift: {_drift_sig} {drift} (info only)"
        )
        _paper = _paper_track_note(_mt, "OU25", side, row.get("league"))
        if _paper:
            msg += f"\n{_paper}"

        if _send(token, chat_id, msg):
            notified.add(key)
            gate.commit(_cand)
            _truth_commit(_tkey)
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

    df = _read_csv(live_file)
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
        # A tip needs a captured market price (owner 2026-11-19). live_tips.csv carries only the model's FAIR
        # price (and src/live_scanner's signal rules are one global set, not per-league bars), so a live
        # signal is never sent as a tip until the scanner records a real in-play price; it stays collected.
        if pd.isna(pd.to_numeric(row.get("market_odds"), errors="coerce")):
            print(f"  SKIP live {row['match']} [{row['signal_type']}]: NO_MARKET_PRICE (fair price only)")
            continue
        _h, _, _a = str(row.get("match") or "").partition(" vs ")
        _lside = "UNDER" if "UNDER" in str(row.get("bet")) else "OVER"
        _twhy, _tkey = _truth_check(row, "OU25", _lside, row.get("market_odds"), None,
                                    row.get("market_odds_book", ""), row.get("market_odds_captured_at", ""),
                                    home=_h.strip(), away=_a.strip())
        if _twhy:
            print(f"  TRUTH BLOCK live {row['match']} [{row['signal_type']}]: {_twhy}")
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

    df = _read_csv(wc_file)
    strong = df[df["signal"].isin(["STRONG", "SHARP", "FADING"])].copy()

    # Only alert for matches in the next 3 days — future signals fire when relevant
    today = datetime.utcnow().date()
    cutoff = today + __import__("datetime").timedelta(days=3)
    strong["_date"] = pd.to_datetime(strong["date"], errors="coerce").dt.date
    strong = strong[strong["_date"].between(today, cutoff)]

    # Priority: STRONG first, then SHARP, then FADING; within each by absolute drift
    # STEAM_STRONG / STEAM_SHARP rank with STRONG / SHARP (they used to fall to a sentinel 9, last)
    _sig_order = {"STRONG": 0, "STEAM_STRONG": 0, "SHARP": 1, "STEAM_SHARP": 1, "FADING": 2}
    strong["_sig_rank"] = strong["signal"].astype(str).str.upper().map(_sig_order)
    strong = strong.sort_values(["_sig_rank", "drift_pct"], key=lambda s: s if s.name != "drift_pct" else s.abs(),
                                ascending=[True, False], na_position="last")

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
    """INSTANT sharp alert — a SUDDEN jump between two consecutive odds snapshots.

    Owner, 2026-10-13: "big jump from snapshots, 10% and above". Not the drift since opening
    (that is in the morning picture pack). The rule and its bad-data guards live in
    telegram_bot.daily_pack.sharp_jumps; thresholds via SHARP_JUMP_MIN_PCT (10) and
    SHARP_JUMP_DIRECTIONS ("in" = price shortening only, ~1.5/day measured over 30 days).
    Was: every STRONG/SHARP drift row, ~8 messages a day.
    Dedup per selection per snapshot, so a second big jump later on the same match alerts again.
    """
    cfg = _load_config()
    token   = cfg.get("token", "")
    chat_id = cfg.get("chat_id", "")
    if not token or token == "YOUR_BOT_TOKEN":
        return 0
    from telegram_bot.daily_pack import sharp_jumps
    try:
        jumps = sharp_jumps()
    except Exception as e:                                              # noqa: BLE001
        print(f"  Sharp jumps unavailable: {type(e).__name__}: {e}")
        return 0
    if jumps.empty:
        return 0

    notified = _load_notified(SHARP_NOTIFIED_FILE)
    sent = 0
    for _, row in jumps.sort_values("jump_pct").iterrows():
        key = f"JUMP|{row['key']}|{row['sel']}|{row['at']}"
        if key in notified:
            continue
        money_in = row["jump_pct"] < 0
        at = pd.to_datetime(row["at"], utc=True, errors="coerce")
        msg = (
            f"{'🔴' if money_in else '🔵'} <b>BIG ODDS JUMP — money {'IN' if money_in else 'OUT'}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"⚽ {row['match']}\n"
            f"🏆 {row['league']} · 📅 {row['date']}\n"
            f"📌 <b>{row['market']}</b>\n"
            f"💰 {row['prev_odds']:.2f} → <b>{row['now_odds']:.2f}</b>  "
            f"(<b>{row['jump_pct']:+.1f}%</b> in one snapshot)\n"
            f"🔍 {int(row['books'])} books"
            + (f" · seen {at:%H:%M} UTC" if pd.notna(at) else "")
        )
        if _send(token, chat_id, msg):
            notified.add(key)
            sent += 1
            _save_notified(notified, SHARP_NOTIFIED_FILE)
            print(f"  Sharp jump: {row['match']} — {row['market']} {row['jump_pct']:+.1f}%")
    return sent


#: the four HT market-sides the board prices: (price market name, side, line, model column of P(HT over line))
HT_MARKETS = (("ht_over05", "OVER", "0.5", "p_ht_over05"), ("ht_under05", "UNDER", "0.5", "p_ht_over05"),
              ("ht_over15", "OVER", "1.5", "p_ht_over15"), ("ht_under15", "UNDER", "1.5", "p_ht_over15"))
#: HT bars would live in models/best_params_side_markets.json under these keys, one bars_opt cell per side
#: (src/bars_opt/serve.served_cell), applied by src/threshold_apply like every other cell. None exists today.
HT_PARAMS_KEY = {"0.5": "ht_ou05", "1.5": "ht_ou15"}
HT_SKIP_NO_PRICE = "NO_MARKET_PRICE"
HT_SKIP_NO_BAR = "NO_LEAGUE_BAR"


def _ht_price_fn():
    """src.ledger._ht_price_lookup (the captured bet-time HT prices), or None. Cached per process."""
    global _HT_PRICE_FN
    if "_HT_PRICE_FN" not in globals() or _HT_PRICE_FN is None:
        try:
            from src.ledger import _ht_price_lookup as _hpl
            _HT_PRICE_FN = _hpl()
        except Exception as e:                                          # noqa: BLE001
            print(f"  HT prices unavailable ({type(e).__name__}: {e})")
            _HT_PRICE_FN = None
    return _HT_PRICE_FN


def ht_bar(league: str, line: str, side: str, ctx: dict | None = None) -> tuple:
    """(cell, "") when league x HT line x side has its OWN owner-applied bar, else (None, reason). Read the way
    every other tip reads its bar (src/bars_opt/serve.served_cell over the production params file + the
    threshold_apply ledger). Never a global value: no cell -> no HT tip."""
    from src.bars_opt import serve as BS
    try:
        if ctx is None:
            ctx = {"side": json.loads((app_config.MODELS_DIR / "best_params_side_markets.json")
                                      .read_text(encoding="utf-8"))}
        key = HT_PARAMS_KEY[line]
        lp = (ctx.get("side") or {}).get(key, {}).get(league)
        cell, why = BS.served_cell(lp, key, league, side.lower(), ctx.get("applied"))
    except Exception as e:                                              # noqa: BLE001
        return None, f"{HT_SKIP_NO_BAR}: HT bars unreadable ({type(e).__name__})"
    if cell is None:
        return None, why or f"{HT_SKIP_NO_BAR}: no approved HT bar"
    return cell, ""


def ht_tip_candidates(df: "pd.DataFrame", price_fn=None, bar_fn=None) -> list:
    """Every fixture x HT market-side of the board, each with its decision. A candidate is SENDABLE only when
    (a) a captured bet-time price exists for that exact market-side (src.ledger._ht_price_lookup) and (b) the
    league's own approved HT bar for that cell is cleared (ht_bar + src/bars_opt/serve.score on the two-sided
    de-vigged price). There is NO global probability threshold here any more (owner rule 2026-11-08: every
    league its own bars, never global): until HT bars are approved nothing is sendable and every candidate
    says why. The prediction ledgers (ht_ledger / ht_observations / prediction_ledger) keep recording every
    HT prediction; this only decides what is SENT.
    Returns dicts: row, market, side, line, prob, odds, reason ("" = sendable), measure."""
    from src.bars_opt import serve as BS
    price_fn = price_fn if price_fn is not None else _ht_price_fn()
    bar_fn = bar_fn or ht_bar
    out = []
    for _, row in df.iterrows():
        date = str(row["date"])[:10]
        lg = str(row.get("league", ""))
        for mkt, side, line, col in HT_MARKETS:
            p_over = pd.to_numeric(row.get(col), errors="coerce")
            if pd.isna(p_over):
                continue
            prob = float(p_over) if side == "OVER" else 1.0 - float(p_over)
            c = {"row": row, "market": mkt, "side": side, "line": line, "prob": prob, "odds": None,
                 "reason": "", "measure": None}
            odds = price_fn(date, lg, row["home_team"], row["away_team"], mkt) if price_fn else None
            if not odds:
                c["reason"] = HT_SKIP_NO_PRICE
                out.append(c)
                continue
            c["odds"] = float(odds)
            cell, why = bar_fn(lg, line, side)
            if cell is None:
                c["reason"] = why
                out.append(c)
                continue
            other = mkt.replace("over", "under") if side == "OVER" else mkt.replace("under", "over")
            o2 = price_fn(date, lg, row["home_team"], row["away_team"], other) if price_fn else None
            fair = BS.fair_yes(odds, o2)
            try:
                res = BS.score(pd.DataFrame([{"league": lg, "p_side": prob, "fair": fair, "odds": float(odds)}]),
                               cell)
                if not str(res["tier"].iloc[0] or ""):
                    c["reason"] = str(res["reason"].iloc[0] or BS.R_BELOW)
                else:
                    c["measure"] = pd.to_numeric(res["measure"].iloc[0], errors="coerce")
            except Exception as e:                                      # noqa: BLE001
                c["reason"] = f"{HT_SKIP_NO_BAR}: HT cell not servable ({type(e).__name__})"
            out.append(c)
    return out


def _ht_key(c: dict) -> str:
    r = c["row"]
    return f"HT|{str(r['date'])[:10]}|{r['home_team']}|{r['away_team']}|HT{c['side']}{c['line']}"


def _ht_candidate(c: dict):
    from src.tip_coherence import Candidate, Event, calibrated_ev, fixture_id
    r = c["row"]
    return Candidate(fixture_id(r["date"], r["home_team"], r["away_team"]),
                     Event("HT", "TOTAL", c["side"], float(c["line"])), "HT",
                     rank=calibrated_ev(c.get("measure"), c.get("odds")), uid=_ht_key(c))


def _ht_board() -> "pd.DataFrame":
    preds_file = app_config.OUTPUT_DIR / "predictions.csv"
    if not preds_file.exists():
        return pd.DataFrame()
    df = _read_csv(preds_file)
    if "p_ht_over05" not in df.columns:
        return pd.DataFrame()
    df = df[df["date"].astype(str).str[:10] >= _utc_today_str()]   # UTC day: stored dates are UTC
    return df[df["p_ht_over05"].notna()]


def notify_ht_tips(gate=None) -> int:
    """Send HT O/U tips that have a captured market price AND clear their league's own approved HT bar
    (ht_tip_candidates), through the fixture coherence gate (src/tip_coherence). Returns count sent.
    Every skipped candidate is logged with its reason, one line per fixture."""
    cfg = _load_config()
    token   = cfg.get("token", "")
    chat_id = cfg.get("chat_id", "")
    if not token or token == "YOUR_BOT_TOKEN":
        return 0

    df = _ht_board()
    if df.empty:
        return 0

    from collections import Counter
    notified = _load_notified()
    cands = ht_tip_candidates(df)
    gate = _coherence_gate(gate, notified)
    sent = 0
    skipped: dict = {}
    counts: Counter = Counter()
    for c in cands:
        key = _ht_key(c)
        if key in notified:
            continue
        why = c["reason"] or gate.check(_ht_candidate(c))
        r = c["row"]
        _tkey = None
        if not why:
            # the HT price lookup carries no book / capture time: the truth gate blocks it until it does
            why, _tkey = _truth_check(r, f"HT_{c['line']}", c["side"], c["odds"], None, "", "")
            why = f"TRUTH {why}" if why else ""
        if why:
            counts[why.split(":")[0].split(" (")[0]] += 1
            skipped.setdefault(f"{r['home_team']} v {r['away_team']}", []).append(f"{c['market']} {why}")
            continue

        side, line, prob = c["side"], c["line"], c["prob"]
        msg = (
            f"{'⚡' if side == 'OVER' else '🧊'} <b>HT {side} {line} — TIP</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📅 {_fmt_kickoff(r)}\n"
            f"🏆 {r.get('league', '')}\n"
            f"⚽ {r['home_team']} vs {r['away_team']}\n"
            f"📌 <b>HT {side} {line}</b>  @ {c['odds']:.2f}\n"
            f"📊 P(HT {side} {line}) = <b>{prob*100:.0f}%</b>  |  {edge_text(float('nan'))}\n"
            f"📏 Cleared the league's own approved HT bar"
        )
        if _send(token, chat_id, msg):
            notified.add(key)
            gate.commit(_ht_candidate(c))
            _truth_commit(_tkey)
            sent += 1
            _save_notified(notified)
            print(f"  HT tip: {r['home_team']} vs {r['away_team']} — HT {side} {line} ({prob*100:.0f}%)")

    for fx, why in skipped.items():
        print(f"  HT skip {fx}: " + "; ".join(why))
    if counts:
        print(f"  HT candidates: {len(cands)}, sent {sent}, skipped by reason {dict(counts)}")
    return sent


class _SharpUnderRestatement(Exception):
    """The restated sharp record (output/corrections/sharp_ledger_restated.csv) is absent."""


def _sharp_split_lines(ts, out_lines, indent="      ") -> None:
    """Under a model×tier, split settled bets by SHARP MOVEMENT and append confirm/disagree
    lines. clv_pct > 0 = our side's odds SHORTENED (sharp confirmed the tip); < 0 = drifted
    (sharp disagreed). Reporting only — skips silently until CLV is populated."""
    if ts is None or ts.empty or "clv_pct" not in ts.columns:
        return
    # A first-half price stored as the close (2026-07/08, corrections sidecar) is no CLV at all:
    # the row is left out of both splits rather than counted as a "sharp confirm".
    ts = _CC.exclude_contaminated_clv(ts)
    clv = pd.to_numeric(ts["clv_pct"], errors="coerce")
    pnl = pd.to_numeric(ts["pnl"], errors="coerce")
    for tag, sym, mask in (("confirm", "✅", clv > 0), ("disagree", "⚠️", clv < 0)):
        seg = pnl[mask & pnl.notna()]
        if seg.empty:
            continue
        n = len(seg); w = int((seg > 0).sum())
        out_lines.append(f"{indent}{sym} sharp {tag}: {w}W/{n-w}L | P/L {U.fmt(U.total(seg))}")


_SHARP_MOVE_FILE = app_config.OUTPUT_DIR / "sharp_move_notified.json"


def _load_sharp_move_notified() -> set:
    if _SHARP_MOVE_FILE.exists():
        try:
            return set(json.loads(_SHARP_MOVE_FILE.read_text(encoding="utf-8")).get("keys", []))
        except Exception:
            return set()
    return set()


def _save_sharp_move_notified(keys: set) -> None:
    try:
        _SHARP_MOVE_FILE.write_text(json.dumps({"keys": sorted(keys)}, indent=2), encoding="utf-8")
    except Exception:
        pass


def notify_sharp_movement(move_thresh: float = 0.03) -> int:
    """LIVE sharp-movement alert. For tips already sent (bets_ledger, source=live, not yet
    kicked off), compare the CURRENT captured odds to our ENTRY (opening) price. If our side's
    line has moved >= move_thresh since we tipped, ping it:
        line moved TOWARD our side (odds shortened) -> sharp CONFIRM
        line moved AGAINST our side (odds drifted)  -> sharp DISAGREE
    Dedup per (fixture, side, direction) so each move alerts once (and again only if it flips).
    Reads the odds curves we already capture — no API calls, no model/betting logic touched.

    MUTED BY DEFAULT since 2026-10-13 (owner: instant messages only for new tips and really big
    sharp moves). This alert's moves are capped at 20% by design (larger is a data artifact on an
    O/U 2.5 line), so it can never meet the 30% instant bar. SHARP_TIP_MOVE_ALERTS=1 re-enables."""
    if os.getenv("SHARP_TIP_MOVE_ALERTS", "").strip() != "1":
        return 0
    import re
    cfg = _load_config()
    token = cfg.get("token", ""); chat_id = cfg.get("chat_id", "")
    if not token or token == "YOUR_BOT_TOKEN":
        return 0
    led = app_config.OUTPUT_DIR / "bets_ledger.csv"
    if not led.exists():
        return 0
    bl = _read_csv(led)
    if "source" not in bl.columns:
        return 0
    live = bl[bl["source"] == "live"].copy()
    today = _utc_today_str()   # UTC day: stored dates are UTC (src/utc_clock.py)
    live = live[live["match_date"].astype(str).str[:10] >= today]
    # A sharp-move alert is only actionable BEFORE kickoff — never alert on a match that has
    # already started. The ledger stores only the date, so: (1) drop already-settled rows, and
    # (2) use kickoff_utc from bets.csv / predictions.csv to keep only fixtures whose kickoff
    # is still in the future (this is what stopped alerts firing on matches played earlier today).
    if "result" in live.columns:
        live = live[live["result"].isna() | (live["result"].astype(str).str.strip().isin(["", "nan"]))]
    _ko: dict = {}
    for _src in ("bets.csv", "predictions.csv"):
        _p = app_config.OUTPUT_DIR / _src
        if not _p.exists():
            continue
        try:
            _b = _read_csv(_p)
        except Exception:
            continue
        if not {"kickoff_utc", "league"}.issubset(_b.columns):
            continue
        for _r in _b.itertuples(index=False):
            _ko.setdefault(_fixture_bucket(_r.league, getattr(_r, "date", "")), {}).setdefault(
                (str(_r.home_team), str(_r.away_team)), str(_r.kickoff_utc))
    _now_ts = pd.Timestamp.utcnow()

    def _upcoming(row) -> bool:
        kt = _fixture_lookup(_ko, row["league"], row["match_date"], row["home_team"], row["away_team"])
        if kt:
            t = pd.to_datetime(kt, utc=True, errors="coerce")
            if pd.notna(t):
                return bool(t > _now_ts)
        # kickoff unknown -> conservative: only keep matches dated strictly after today
        return str(row["match_date"])[:10] > today

    live = live[live.apply(_upcoming, axis=1)]
    # Keep only the LATEST tip per fixture. The model can flip a side (e.g. UNDER -> OVER after
    # the 2026-08-09 recalibration), leaving a stale opposite-side row in the ledger — without
    # this we'd alert on BOTH sides of the same match (the Palmeiras UNDER+OVER case).
    if "generated_at" in live.columns and not live.empty:
        live = live.sort_values("generated_at").drop_duplicates(
            subset=["home_team", "away_team", "match_date"], keep="last")
    if live.empty:
        return 0
    # SRC in the key + priority lookup: over25/under25 for standard fixtures live in BOTH
    # standard_odds_history (OddsAPI) AND standard_sidemarket (Bet365). Keying without the
    # source would pair a first-seen from one book with a latest from the other -> a fake
    # cross-source "move" (the Botafogo 1.86 vs 1.17 class of bug). Keep each curve isolated.
    _SRC_PRIORITY = ("standard_odds_history.csv", "newformat_odds_history.csv",
                     "standard_sidemarket_odds_history.csv")
    snaps: dict = {}
    for f in _SRC_PRIORITY:
        p = app_config.OUTPUT_DIR / f
        if not p.exists():
            continue
        try:
            oh = _read_csv(p).sort_values("snapshot_ts")
        except Exception as e:
            print(f"[sharp-move] could not read {f}: {e}")
            continue
        if "league" not in oh.columns:
            print(f"[sharp-move] {f} has no league column — skipped")
            continue
        for r in oh.itertuples(index=False):
            m = str(getattr(r, "match", ""))
            if " vs " not in m:
                continue
            h, a = m.split(" vs ", 1)
            try:
                od = float(r.odds)
            except (TypeError, ValueError):
                continue
            # {(source, market): {(league, date): {(home, away): [first, latest]}}}
            pairs = snaps.setdefault((f, str(r.market)), {}).setdefault(
                _fixture_bucket(r.league, r.match_date), {})
            # store (first-seen, latest) from the SAME source -> a valid open->close move.
            if (h, a) not in pairs:
                pairs[(h, a)] = [od, od]
            else:
                pairs[(h, a)][1] = od

    notified = _load_sharp_move_notified()
    sent = 0
    for r in live.itertuples(index=False):
        side = str(r.side).upper()
        if side not in ("OVER", "UNDER"):
            continue
        mkt = "under25" if side == "UNDER" else "over25"
        pair = None
        for f in _SRC_PRIORITY:                       # first source with a same-source curve wins
            cand = _fixture_lookup(snaps.get((f, mkt), {}), r.league, r.match_date,
                                   r.home_team, r.away_team)
            if cand:
                pair = cand
                break
        if not pair:
            continue
        # SAME source, open->close (NOT the OddsAPI ledger entry vs Bet365 current — that
        # cross-source mismatch produced fake "moves", e.g. Botafogo 1.86 vs 1.17).
        entry, cur = pair
        if not entry or not cur or entry <= 1.0:
            continue
        move = cur / entry - 1.0                 # <0 = our side shortened (confirm); >0 = drifted
        # Real O/U 2.5 line moves are small (a few %). Anything beyond ~20% is a data artifact
        # (stale/outlier first snapshot, or a residual cross-source pairing), not a sharp move.
        if abs(move) < move_thresh or abs(move) > 0.20:
            continue
        direction = "confirm" if move < 0 else "disagree"
        dkey = f"{r.home_team}|{r.away_team}|{str(r.match_date)[:10]}|{side}|{direction}"
        if dkey in notified:
            continue
        sym = "✅" if direction == "confirm" else "⚠️"
        verb = ("TOWARD your tip — sharp CONFIRM" if direction == "confirm"
                else "AGAINST your tip — sharp DISAGREE")
        tier = getattr(r, "signal_tier", "")
        msg = (f"⚡ <b>Sharp movement on your tip</b>\n"
               f"{r.home_team} vs {r.away_team}\n"
               f"<b>{side} 2.5</b> ({tier})\n"
               f"market opened {entry:.2f}, now {cur:.2f}\n"
               f"{sym} line moved {verb} ({abs(move)*100:.1f}%)")
        if _send(token, chat_id, msg):
            notified.add(dkey); sent += 1
    _save_sharp_move_notified(notified)
    if sent:
        print(f"Sharp-movement alerts sent: {sent}")
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

    week_ago = _utc_now() - _dt.timedelta(days=7)   # naive UTC, like the stored dates
    week_start = week_ago.strftime("%b %d")
    week_end   = _utc_now().strftime("%b %d, %Y")

    def _stats(data: "pd.DataFrame") -> dict | None:
        d = _settled_only(data)
        if d is None or d.empty:
            return None
        n = len(d); w = (d["pnl"] > 0).sum(); pnl = U.total(d["pnl"])
        return {"n": n, "win": float(w) / n, "roi": pnl / n * 100, "pnl": pnl}

    def _fmt(label: str, emoji: str, stats: dict | None, pending_n: int = 0) -> str:
        if stats and stats["n"] > 0:
            pnl_s = "+" if stats["pnl"] >= 0 else ""
            return (f"  {emoji} <b>{label}</b>: {stats['n']} bets | "
                    f"{stats['win']:.0%} win | "
                    f"P/L <b>{U.fmt(stats['pnl'])}</b>")
        if pending_n:
            return f"  {emoji} <b>{label}</b>: {pending_n} signals sent | ⏳ awaiting results"
        return f"  {emoji} <b>{label}</b>: no settled bets this week"

    # ── 1. Prediction model (bets_ledger.csv, source=live) ──────────────────
    live = pd.DataFrame(columns=["pnl", "signal_tier", "model_type", "match_date"])
    lw = live.copy()
    ledger_file = app_config.OUTPUT_DIR / "bets_ledger.csv"
    if ledger_file.exists():
        bl = _read_csv(ledger_file)
        bl["pnl"]        = pd.to_numeric(bl["pnl"], errors="coerce")
        bl["match_date"] = pd.to_datetime(bl["match_date"], errors="coerce")
        live = bl[bl["source"] == "live"].copy()
        # Split standard vs new-format (same as the daily digest); a blank tag is classified
        # by league so new-format bets never leak into the Standard column.
        if "model_type" not in live.columns:
            live["model_type"] = ""
        _blk = live["model_type"].isna() | (live["model_type"].astype(str).str.strip().isin(["", "nan"]))
        live["model_type"] = live["model_type"].astype(object)   # an all-blank column reads as float
        live.loc[_blk, "model_type"] = live.loc[_blk, "league"].map(_model_type_of)
        live = _since_cutoff(live)   # THE staked basis: match date on/after the cutoff
        lw = live[live["match_date"] >= week_ago]

    # ── 2. Sharp tracker (sharp_ledger.csv) ─────────────────────────────────
    # The booked P/L is priced at opening_odds, a price already gone when the signal fired; only the
    # RESTATED record (output/corrections/sharp_ledger_restated.csv) is ever shown. Without it the
    # record reads UNDER RESTATEMENT - never the booked figure (src/clv_corrections.sharp_record).
    sharp_week: dict | None = None; sharp_week_pending = 0
    sharp_all:  dict | None = None
    sharp_state = {"state": _CC.UNDER_RESTATEMENT, "unrestated": 0}
    sharp_file = app_config.OUTPUT_DIR / "sharp_ledger.csv"
    if sharp_file.exists():
        sl, sharp_state = _CC.sharp_record(_read_csv(sharp_file))
        if sl is not None:
            sl["pnl"]        = pd.to_numeric(sl["pnl"], errors="coerce")
            sl["signal_date"] = pd.to_datetime(sl["signal_date"], errors="coerce")
            sw = sl[sl["signal_date"] >= week_ago]
            sharp_week         = _stats(sw)
            sharp_week_pending = int((sw["pnl"].isna() & sw["restatement"].ne("UNRESTATED")).sum()) \
                if not sw.empty else 0
            sharp_all          = _stats(sl)

    # ── 3. Player props (player_ledger.csv) ──────────────────────────────────
    props_week: dict | None = None; props_week_pending = 0
    props_all:  dict | None = None
    props_file = app_config.OUTPUT_DIR / "player_ledger.csv"
    if props_file.exists():
        pl = _countable_props(_read_csv(props_file))  # AVOID/WATCH + wrong-fixture: never counted
        pl["pnl"]         = pd.to_numeric(pl["pnl"], errors="coerce")
        pl["signal_date"] = pd.to_datetime(pl["signal_date"], errors="coerce")
        pw = pl[pl["signal_date"] >= week_ago]
        props_week         = _stats(pw)
        props_week_pending = int((pw["pnl"].isna()).sum()) if not pw.empty else 0
        props_all          = _stats(pl)

    # ── 4. WC drift signals (worldcup_tips.csv — no P/L yet) ─────────────────
    wc_pending = 0
    wc_file = app_config.OUTPUT_DIR / "worldcup_tips.csv"
    if wc_file.exists():
        wc = _read_csv(wc_file)
        wc["date"] = pd.to_datetime(wc["date"], errors="coerce")
        wc_pending = int(len(wc[wc["date"] >= week_ago]))

    # ── Build message — by market then tier ──────────────────────────────────
    TIER_SYM   = TIER_ICON
    TIER_ORDER = ["QUALIFIED", "SNIPER", "MARKSMAN", "VALUABLE"]   # QUALIFIED = tips from 2026-11-21
    # BTTS alone does not say which side; this path only ever produces YES.
    SIDE_LABELS = {"btts": "BTTS — YES", "over15": "Over 1.5", "over35": "Over 3.5"}

    lines = [
        "📊 <b>WEEKLY SUMMARY</b>",
        "━━━━━━━━━━━━━━━━",
        f"<i>{GAP_EXPLAINER}</i>",
        f"📅 {week_start} → {week_end}",
        "",
        "<b>This week by market</b>",
        "",
    ]

    # ── Over 2.5 — this week, by MODEL then tier (matches the daily digest) ──
    MODELS = [("standard", "⚽", "Over 2.5 — Standard"),
              ("new_format", "🌍", "Over 2.5 — New-Format")]

    def _pred_block(frame, settled_word):
        any_shown = False
        for fmt, emoji, label in MODELS:
            sub_all = _settled_only(frame[frame["model_type"] == fmt]) if "model_type" in frame.columns else None
            if sub_all is None or sub_all.empty:
                continue
            any_shown = True
            # Headline = STAKED rows only (THE definition, src/staked.py: a recorded policy stake).
            sub = _staked_rows(sub_all, "bets_ledger.csv")
            if not sub.empty:
                n = len(sub); w = int((sub["pnl"] > 0).sum()); pnl = U.total(sub["pnl"])
                lines.append(f"{emoji} <b>{label}</b> — 💰 {n} {settled_word} | {w/n:.0%} win | "
                             f"P/L {U.fmt(pnl)}")
            else:
                lines.append(f"{emoji} <b>{label}</b> — no staked bets settled")
            for tier in ALL_TIERS:
                ts = sub[sub["signal_tier"] == tier] if "signal_tier" in sub.columns else pd.DataFrame()
                if ts.empty:
                    continue
                tn = len(ts); tw = int((ts["pnl"] > 0).sum())
                lines.append(f"  {TIER_SYM[tier]} {tier_label(tier)}: {tn} bets | {tw/tn:.0%} win | "
                             f"P/L <b>{U.fmt(U.total(ts['pnl']))}</b>")
                _sharp_split_lines(ts, lines, "      ")
            _ns = _unstaked_rows(sub_all, sub)
            for tier in ALL_TIERS:
                cs = _ns[_ns["signal_tier"] == tier] if "signal_tier" in _ns.columns else pd.DataFrame()
                if cs.empty:
                    continue
                cn = len(cs); cw = int((cs["pnl"] > 0).sum())
                lines.append(f"  {TIER_SYM.get(tier, '▰▱▱')} {tier_label(tier)}: {cn} signals | {cw/cn:.0%} win | "
                             f"({U.fmt(U.total(cs['pnl']))}) — 📊 data only, not staked")
        if not any_shown:
            lines.append("⚽ <b>Over 2.5</b>: no settled bets")

    _pred_block(lw, "bets")
    lines.append("")

    # ── Side markets (BTTS / O1.5 / O3.5) — this week ────────────────────────
    side_led = app_config.OUTPUT_DIR / "side_bets_ledger.csv"
    if side_led.exists():
        try:
            sl = _read_csv(side_led)
            sl["pnl"]         = pd.to_numeric(sl["pnl"], errors="coerce")
            # The week is the SIGNAL week; a row without a signal_date is not placed in it by
            # borrowing match_date - it is left out and counted.
            if "signal_date" not in sl.columns:
                print("[weekly] side_bets_ledger.csv has no signal_date column — side markets skipped")
                sl = sl.iloc[0:0].assign(signal_date=pd.NaT)
            sl["signal_date"] = pd.to_datetime(sl["signal_date"], errors="coerce")
            if sl["signal_date"].isna().any():
                print(f"[weekly] {int(sl['signal_date'].isna().sum())} side-market row(s) without signal_date excluded")
            if "source" in sl.columns:
                sl = sl[sl["source"].astype(str) == "live"]
            sl_week = _since_cutoff(sl[sl["signal_date"] >= week_ago])
            for mkt, mkt_label in SIDE_LABELS.items():
                ms = sl_week[sl_week["market"] == mkt] if "market" in sl_week.columns else pd.DataFrame()
                ms = _settled_only(ms)
                if ms.empty:
                    continue
                # Headline = STAKED rows only (THE staked definition, src/staked.py).
                stk = _staked_rows(ms, "side_bets_ledger.csv")
                n = len(stk); mw2 = int((stk["pnl"] > 0).sum()); pnl = U.total(stk["pnl"])
                lines.append(f"📌 <b>{mkt_label}</b> — 💰 staked {mw2}W/{n-mw2}L | P/L {U.fmt(pnl)}")
                for tier in TIER_ORDER:
                    ts = ms[ms["signal_tier"] == tier] if "signal_tier" in ms.columns else pd.DataFrame()
                    if ts.empty:
                        continue
                    tn = len(ts); w = int((ts["pnl"] > 0).sum())
                    lines.append(f"  {TIER_SYM[tier]} {tier_label(tier)}: {w}W/{tn-w}L | P/L {U.fmt(U.total(ts['pnl']))}")
                lines.append("")
        except Exception:
            pass

    # ── Half-time O/U (ht_ledger.csv) — this week ────────────────────────────
    # HT WAS THE ONLY TIPPING TRACK ABSENT FROM THIS DIGEST. It has sent 80 Telegram tips
    # across all four lines while reporting into nothing, so its results were invisible unless
    # somebody opened the CSV. Added 2026-09-29.
    #
    # It is labelled PAPER on purpose. Measured against captured closing prices the half-time
    # market beats the model (market Brier 0.1972 vs 0.1994, AUC 0.568 vs 0.533) and the vig is
    # ~7%, so these tips are informational. Rows settled without a captured price are excluded
    # from the P/L line rather than averaged in: they settle at the model's OWN fair odds, which
    # pays back exactly what the model claimed and measures nothing about the market.
    ht_led = app_config.OUTPUT_DIR / "ht_ledger.csv"
    if ht_led.exists():
        try:
            hl = _read_csv(ht_led)
            hl["pnl"] = pd.to_numeric(hl["pnl"], errors="coerce")
            hl["_d"] = pd.to_datetime(hl.get("match_date"), errors="coerce")
            hw = _settled_only(hl[hl["_d"] >= week_ago]) if "_d" in hl.columns else pd.DataFrame()
            pend = int(((hl["result"].isna() | (hl["result"].astype(str).str.strip() == ""))
                        & (hl["_d"] >= week_ago)).sum())
            if not hw.empty or pend:
                lines.append("⏱️ <b>Half-time O/U</b> <i>(no stake)</i>")
                if hw.empty:
                    lines.append(f"  ⏳ {pend} tip(s) awaiting results")
                else:
                    # `odds` is recorded at TIP time (new); `entry_odds` is filled at settlement
                    # from the captured history. Either is a real market price, so either counts
                    # — otherwise every row logged before `odds` existed reads as unpriced when
                    # we do in fact know what it went off at.
                    # THE rule (src/clv_corrections.ht_priced_mask) also drops the rows listed in
                    # output/corrections/ht_ledger_unpriced.csv.
                    real = hw[_CC.ht_priced_mask(hw)]
                    n = len(hw); w = int((hw["pnl"] > 0).sum())
                    lines.append(f"  {w}W/{n-w}L across {hw['market'].nunique()} line(s)"
                                 + (f" | ⏳ {pend} pending" if pend else ""))
                    if not real.empty:
                        lines.append(f"  💵 priced only: {len(real)} bet(s) | "
                                     f"P/L {U.fmt(U.total(real['pnl']))}")
                    unp = n - len(real)
                    if unp:
                        lines.append(f"  ⚠️ {unp} settled with no market price — excluded from P/L")
                    for mkt in sorted(hw["market"].dropna().unique()):
                        ms = hw[hw["market"] == mkt]
                        mw = int((ms["pnl"] > 0).sum())
                        lines.append(f"    · {mkt}: {mw}W/{len(ms)-mw}L")
                lines.append("")
        except Exception:
            pass

    if sharp_state["state"] == _CC.UNDER_RESTATEMENT:
        lines.append(f"  💰 <b>Sharp Tracker</b>: ⚠️ {_CC.UNDER_RESTATEMENT} — no P/L shown")
    else:
        lines.append(_fmt("Sharp Tracker", "💰", sharp_week, sharp_week_pending))

    if wc_pending:
        lines.append(f"  🌍 <b>WC Drift</b>: {wc_pending} signals active | ⏳ no P/L tracking yet")
    else:
        lines.append("  🌍 <b>WC Drift</b>: no signals this week")

    lines.append("  👤 <b>Player Props</b>: see Player Props Weekly for details")

    # ── All-time by market then tier ──────────────────────────────────────────
    sharp_all_total = int(len(_read_csv(sharp_file))) if sharp_file.exists() else 0
    wc_all_total    = int(len(_read_csv(wc_file)))    if wc_file.exists() else 0

    lines += ["", "━━━━━━━━━━━━━━━━", "<b>All-time by market</b>", ""]

    _pred_block(live, "settled")
    lines.append("")

    # Side markets all-time
    if side_led.exists():
        try:
            sl = _read_csv(side_led)
            sl["pnl"] = pd.to_numeric(sl["pnl"], errors="coerce")
            if "source" in sl.columns:
                sl = sl[sl["source"].astype(str) == "live"]
            sl_all = _settled_only(_since_cutoff(sl))
            for mkt, mkt_label in SIDE_LABELS.items():
                ms = sl_all[sl_all["market"] == mkt] if "market" in sl_all.columns else pd.DataFrame()
                if ms.empty:
                    continue
                stk = _staked_rows(ms, "side_bets_ledger.csv")       # THE staked definition
                n = len(stk); mw3 = int((stk["pnl"] > 0).sum()); pnl = U.total(stk["pnl"])
                lines.append(f"📌 <b>{mkt_label}</b> — 💰 staked {mw3}W/{n-mw3}L | P/L {U.fmt(pnl)}")
                for tier in TIER_ORDER:
                    ts = ms[ms["signal_tier"] == tier] if "signal_tier" in ms.columns else pd.DataFrame()
                    if ts.empty:
                        continue
                    tn = len(ts); w = int((ts["pnl"] > 0).sum())
                    lines.append(f"  {TIER_SYM[tier]} {tier_label(tier)}: {w}W/{tn-w}L | P/L {U.fmt(U.total(ts['pnl']))}")
                lines.append("")
        except Exception:
            pass

    if sharp_all_total and sharp_state["state"] == _CC.UNDER_RESTATEMENT:
        lines.append(f"  💰 <b>Sharp Tracker</b>: {sharp_all_total} signals total | ⚠️ {_CC.UNDER_RESTATEMENT} — no P/L shown")
    elif sharp_all and sharp_all["n"] > 0:
        _unr = sharp_state.get("unrestated", 0)
        lines.append(f"  💰 <b>Sharp Tracker</b>: {sharp_all['n']} settled | {sharp_all['win']:.0%} win | P/L {U.fmt(sharp_all['pnl'])}"
                     + (f" | {_unr} not yet restated — excluded" if _unr else ""))
    elif sharp_all_total:
        lines.append(f"  💰 <b>Sharp Tracker</b>: {sharp_all_total} signals total | ⏳ awaiting results")
    else:
        lines.append("  💰 <b>Sharp Tracker</b>: no signals yet")

    if wc_all_total:
        lines.append(f"  🌍 <b>WC Drift</b>: {wc_all_total} signals total | ⏳ no P/L tracking yet")

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

    df = _read_csv(bets_file)
    snipers = df[
        (df["signal_tier"] == "SNIPER") &
        (df["bet"].isin(["OVER", "UNDER"]))
    ].copy()

    today_str = _utc_today_str()   # UTC day: stored dates are UTC (src/utc_clock.py)
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
        edge = pd.to_numeric(row.get("best_edge"), errors="coerce") * 100
        edge_str = "edge not recorded" if pd.isna(edge) else f"Edge: <b>{edge:.1f}%</b>"

        msg = (
            f"🤖 <b>AGENT ANALYSIS</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"⚽ {row['home_team']} vs {row['away_team']}\n"
            f"🏆 {row.get('league', '')}  |  📅 {_fmt_kickoff(row)}\n"
            f"🎯 ML Signal: <b>{side} 2.5</b>  {edge_str}\n"
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
    """Send the props TOP-3 per league per gameweek (src/props_top3.py). Returns the number of league-week messages.

    Owner 2026-11-22 ("Like TOP-3: 3 per league, stake by proof"): the old daily feed - the top 20 priced props by
    model EV, bucketed into the RETIRED SNIPER / MARKSMAN / VALUABLE names - is gone. Each prop league gets ONE
    message per gameweek with T1 best value / T2 safest / T3 risky value, frozen 120 min (TOP3_FREEZE_LEAD_MIN)
    before the league-week's first kickoff, through the shared fixture coherence gate, never after kickoff, and
    labelled "No stake: props not proven yet". Same switch as the team TOP-3 (TOP3_LEAGUE_ENABLED)."""
    from src import props_top3
    res = props_top3.notify()
    return int(res["sent"]) if "sent" in res else 0     # flag off / no credentials: nothing was sent


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

    MUTED BY DEFAULT since 2026-10-13 together with the props alerts it follows up on: with no
    instant prop tip sent, there is nothing to cash out of. PROPS_INSTANT_ALERTS=1 re-enables.
    """
    if os.getenv("PROPS_INSTANT_ALERTS", "").strip() != "1":
        return 0
    cfg = _load_config()
    token   = cfg.get("token", "")
    chat_id = cfg.get("chat_id", "")
    if not token or token == "YOUR_BOT_TOKEN":
        return 0

    tips_file = app_config.OUTPUT_DIR / "player_tips.csv"
    if not tips_file.exists():
        return 0

    today_str = _utc_today_str()   # UTC day: stored dates are UTC (src/utc_clock.py)

    # Build the set of keys that are CURRENTLY active in player_tips.csv.
    #
    # DELIBERATELY NOT tier-filtered (fixed 2026-08-15). This used to keep only
    # SNIPER/MARKSMAN rows, which made the cashout alert fire on a TIER CHANGE rather than
    # on a lineup change — and it claimed "X is NOT in the confirmed starting XI" while X
    # was in the XI and still sitting in this very file.
    #
    # It became routine once the weekend feed (c622d60) made those tiers pure EV RANKS:
    # top 5 SNIPER, next 5 MARKSMAN, next 10 VALUABLE, re-ranked every hourly run. A player
    # slipping from rank 5 to rank 11 left the SNIPER/MARKSMAN set and was reported benched.
    # A run where the odds fetch returned nothing had the same effect.
    #
    # Presence in player_tips.csv at ANY tier is the real signal: player_model.predict
    # already drops non-starters when a confirmed lineup is available (>=18 starters
    # posted), so a player who is genuinely benched disappears from the file entirely.
    # Anything still in the file is still in the XI as far as we know, and we must not
    # assert otherwise.
    df = _read_csv(tips_file)
    df = df[df["date"].astype(str).str[:10] == today_str]
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
    df_all = _read_csv(tips_file)
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
        match_str = str(row.get("match") or "").strip()
        team_str  = str(row.get("team") or "").strip()
        # A cash-out call without the fixture is unactionable (which bet?) - skip it, logged.
        if not match_str or match_str.lower() == "nan":
            print(f"  SKIP cashout: {player_name} — {mkt_label}: fixture not found in today's tips")
            continue
        if team_str.lower() == "nan":
            team_str = "—"

        msg = (
            f"🚨 <b>LINEUP ALERT — CASHOUT RECOMMENDED</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"👤 <b>{player_name}</b> is NOT in the confirmed starting XI\n"
            f"📌 Market: <b>{mkt_label}</b>\n"
            f"⚽ {match_str}  |  {team_str or '—'}\n"
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
    """RETIRED (owner 2026-11-22, "Like TOP-3: 3 per league, stake by proof"). The props briefing grouped today's
    props by the retired SNIPER / MARKSMAN / VALUABLE / PAPER labels. The props TOP-3 per league-week
    (src/props_top3.py, notify_player_props) is the only props tip message now. Kept as a no-op so old callers
    (test_telegram.py) keep working; nothing is sent."""
    return False


def _props_top3_week_lines(week_ago) -> list[str]:
    """Lines for the props TOP-3 picks settled this week, by role (T1 / T2 / T3); [] when none."""
    try:
        from dashboard_data import props_top3_view as P3
        d = P3.load_picks(app_config.OUTPUT_DIR)
    except Exception:                                                  # noqa: BLE001
        return []
    if d.empty:
        return []
    when = pd.to_datetime(d["settled_at"], errors="coerce", utc=True).dt.tz_convert(None)
    d = d[d["R"].isin(("WIN", "LOSS")) & (when >= pd.Timestamp(week_ago))]
    if d.empty:
        return []
    out = ["", "━━━━━━━━━━━━━━━━", "<b>Props TOP-3 this week</b> (no stake: props not proven yet)", ""]
    for role in P3.ROLE_ORDER:
        g = d[d["role"] == role]
        if g.empty:
            continue
        w = int((g["R"] == "WIN").sum())
        out.append(f"  {P3.ROLE_TEXT[role]}: {w}W/{len(g) - w}L | P/L {U.fmt(U.total(g['flat_pnl']))}")
    return out


def notify_props_weekly_summary() -> bool:
    """
    Send a player-props-only weekly performance summary.
    Groups settled signals by market (SOT/Goals/Assists/Cards) and the props TOP-3 picks by role (T1 / T2 / T3);
    never by the retired tier labels (owner 2026-11-22).
    Returns True if message sent.
    """
    import datetime as _dt
    cfg = _load_config()
    token   = cfg.get("token", "")
    chat_id = cfg.get("chat_id", "")
    if not token or token == "YOUR_BOT_TOKEN":
        return False

    week_ago   = _utc_now() - _dt.timedelta(days=7)   # naive UTC, like the stored dates
    week_start = week_ago.strftime("%b %d")
    week_end   = _utc_now().strftime("%b %d, %Y")

    player_led = app_config.OUTPUT_DIR / "player_ledger.csv"
    if not player_led.exists():
        return False

    pled = _countable_props(_read_csv(player_led))  # AVOID/WATCH + wrong-fixture: never counted
    pled["pnl"]         = pd.to_numeric(pled["pnl"], errors="coerce")
    # Windowed on signal_date only; rows without one are excluded (logged), not re-dated.
    if "signal_date" not in pled.columns:
        print("[props weekly] player_ledger.csv has no signal_date column — no weekly window")
        pled = pled.assign(signal_date=pd.NaT)
    pled["signal_date"] = pd.to_datetime(pled["signal_date"], errors="coerce")
    if pled["signal_date"].isna().any():
        print(f"[props weekly] {int(pled['signal_date'].isna().sum())} prop row(s) without signal_date excluded from the week")

    settled_all  = _settled_only(pled)
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
    MARKETS   = [k for k, _, _ in PLAYER_MARKETS]

    def _row(data: "pd.DataFrame") -> tuple | None:
        d = data[data["result"].str.upper().isin(["WIN", "LOSS"])] if "result" in data.columns else data
        if d.empty:
            return None
        n   = len(d)
        w   = int((d["pnl"] > 0).sum())
        pnl = U.total(d["pnl"])
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
        lines.append(f"<b>This week — {n} settled | {w}W/{n-w}L | P/L {U.fmt(pnl)}</b>")
        lines.append("")
        # By market
        for mkt in MARKETS:
            sub = settled_week[settled_week["market"] == mkt] if "market" in settled_week.columns else pd.DataFrame()
            mr = _row(sub)
            if not mr:
                continue
            mn, mw, mp = mr
            pnl_s = "+" if mp >= 0 else ""
            lines.append(
                f"  {MKT_EMOJI.get(mkt, '📌')} <b>{MKT_LABEL.get(mkt, mkt)}</b>: "
                f"{mn} bets | {mw}W/{mn-mw}L | P/L <b>{U.fmt(mp)}</b>"
            )
        lines.append("")
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
        lines.append(f"  Total: {n} bets | {w}W/{n-w}L | P/L {U.fmt(pnl)}")
        lines.append("")
        for mkt in MARKETS:
            sub = settled_all[settled_all["market"] == mkt] if "market" in settled_all.columns else pd.DataFrame()
            mr = _row(sub)
            if not mr:
                continue
            mn, mw, mp = mr
            lines.append(
                f"  {MKT_EMOJI.get(mkt, '📌')} {MKT_LABEL.get(mkt, mkt)}: "
                f"{mw}W/{mn-mw}L | P/L {U.fmt(mp)}"
            )
    else:
        total_all = len(pled)
        lines.append(f"  {total_all} signals tracked — no settled results yet")

    # ── Props TOP-3 by role (owner 2026-11-22: T1 / T2 / T3, never the retired tier labels) ─────
    lines += _props_top3_week_lines(week_ago)

    msg = "\n".join(lines)
    if _send(token, chat_id, msg):
        print("Player props weekly summary sent.")
        return True
    return False


def notify_bug_announcement() -> bool:
    """One-time notice: the New-Format O/U model had a calibration bug (fixed 2026-08-09) that
    under-predicted goals and produced one-sided UNDER tips. Performance counting restarts at
    PERFORMANCE_CUTOFF_DATE for BOTH models; historical rows are kept for CLV. A dedup key makes
    this send exactly once, ever (safe to wire into a recurring workflow)."""
    cfg = _load_config()
    token = cfg.get("token", ""); chat_id = cfg.get("chat_id", "")
    if not token or token == "YOUR_BOT_TOKEN":
        return False
    notified = _load_notified()
    key = "ANNOUNCE|nf_calibration_bug_v1"
    if key in notified:
        print("Bug announcement already sent — skipping.")
        return False
    cutoff = getattr(app_config, "PERFORMANCE_CUTOFF_DATE", "")
    msg = (
        "📢 <b>System Notice — Model Fix</b>\n"
        "━━━━━━━━━━━━━━━━\n"
        "We found and fixed a calibration bug in the <b>New-Format</b> Over/Under model. It was "
        "under-predicting goals, so it sent one-sided UNDER tips that are not reliable.\n\n"
        f"✅ <b>Fixed.</b> From <b>{cutoff}</b>, only tips generated after the fix count towards "
        "results — for <b>both</b> the Standard and New-Format models.\n"
        "🗂 Earlier tips are <b>not deleted</b> — we keep them for closing-line-value (CLV) "
        "analysis; they're just excluded from win/loss and P/L totals.\n\n"
        "Thanks for your patience."
    )
    if _send(token, chat_id, msg):
        notified.add(key)
        _save_notified(notified)
        print("Bug announcement sent.")
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

    # TODAY ONLY. Every upcoming-tip section below filters `== today_str`, not `>= today_str`.
    # The old `>=` made this a "everything on the board" dump rather than a daily briefing: on
    # 2026-09-10 it listed 225 items spanning six days (112 O/U + 73 side + 3 props + 37 sharp)
    # when only 13 were actually today's. At 4,000 chars per Telegram message that is several
    # screens of fixtures you cannot bet yet, which buries the ones you can.
    #
    # UTC, explicitly, and that matters more than it looks. `match_date` / `date` are UTC
    # calendar days BY DESIGN — see _fmt_kickoff: display is converted to the league's local
    # tz, the stored join/dedup key deliberately is not. A naive datetime.now() happens to be
    # UTC on a GitHub runner but is Israel time when run by hand, and under `>=` that mismatch
    # was harmless (you just got extra rows). Under `==` an off-by-one day silently matches
    # NOTHING, so the basis has to be pinned. UTC is also the better betting-day boundary here:
    # UTC midnight is 03:00 in Israel, so tonight's late kickoffs stay on today's list instead
    # of jumping to tomorrow's digest.
    today         = datetime.now(_dt.timezone.utc)
    today_str     = today.strftime("%Y-%m-%d")
    yesterday_str = (today - _dt.timedelta(days=1)).strftime("%Y-%m-%d")
    header_date   = today.strftime("%a, %b %d").replace(" 0", " ")
    # What each section dropped as future-dated, for the one-line footer. A today-only digest
    # makes "no tips" ambiguous — quiet day, or broken date filter? This distinguishes them.
    upcoming: dict[str, int] = {}

    notified = _load_notified()
    digest_key = f"DIGEST|{today_str}"
    if digest_key in notified:
        print("Daily digest already sent today — skipping.")
        return False

    TIER_SYM   = TIER_ICON
    TIER_ORDER = ["QUALIFIED", "SNIPER", "MARKSMAN", "VALUABLE"]   # QUALIFIED = tips from 2026-11-21
    # BTTS alone does not say which side; this path only ever produces YES.
    SIDE_LABELS = {"btts": "BTTS — YES", "over15": "Over 1.5", "over35": "Over 3.5"}
    SIG_SYM     = {"STEAM_STRONG": "🔴", "STEAM_SHARP": "🟠", "STRONG": "🟡"}

    lines = [
        f"📊 <b>DAILY BRIEFING</b> — {header_date}",
        "━━━━━━━━━━━━━━━━",
        f"<i>{GAP_EXPLAINER}</i>",
        "",
    ]

    # ── Over 2.5 by tier — read from the LEDGER (best-tier == what was pinged + weekly) ──
    ledger_file = app_config.OUTPUT_DIR / "bets_ledger.csv"
    ou_total = 0
    if ledger_file.exists():
        try:
            bl = _read_csv(ledger_file)
            if "source" in bl.columns:
                bl = bl[bl["source"].astype(str) == "live"]
            _tier_ok = bl["signal_tier"].isin(TIER_ORDER)
            _dates   = bl["match_date"].astype(str).str[:10]
            upcoming["O/U 2.5"] = int((_tier_ok & (_dates > today_str)).sum())
            bl = bl[_tier_ok & (_dates == today_str)].copy()
            # One side per fixture: a fixture tipped on both sides before the flip guard existed
            # shows only its FIRST tip (Mito HollyHock v Shimizu, 2026-10-11).
            if "generated_at" in bl.columns:
                bl = bl.sort_values("generated_at", kind="mergesort").drop_duplicates(
                    ["match_date", "home_team", "away_team"], keep="first")
            ou_total = len(bl)
            # derive model_type from league for blanks so new-format tips can't show as standard
            if "model_type" not in bl.columns:
                bl["model_type"] = ""
            _blk = bl["model_type"].isna() | (bl["model_type"].astype(str).str.strip().isin(["", "nan"]))
            bl["model_type"] = bl["model_type"].astype(object)   # an all-blank column reads as float
            bl.loc[_blk, "model_type"] = bl.loc[_blk, "league"].map(_model_type_of)
            # Market is O/U 2.5 (tips can be OVER or UNDER) — NOT "Over 2.5". Group by model
            # then tier; the per-tip line shows the actual side.
            lines.append(f"⚽ <b>O/U 2.5</b>  ({ou_total} tip{'s' if ou_total != 1 else ''})")
            if bl.empty:
                lines.append("  No tips today")
            else:
                for fmt, femoji, flabel in [("standard", "⚽", "Standard"),
                                            ("new_format", "🌍", "New-Format")]:
                    fb = bl[bl["model_type"] == fmt]
                    if fb.empty:
                        continue
                    lines.append(f"  {femoji} <b>{flabel}</b> ({len(fb)})")
                    for tier in TIER_ORDER:
                        tb = fb[fb["signal_tier"] == tier]
                        if tb.empty:
                            continue
                        lines.append(f"    {TIER_SYM[tier]} <b>{tier_label(tier)}</b> ({len(tb)})")
                        for _, r in tb.iterrows():
                            side = str(r.get("side", ""))
                            # unpriced / no recorded edge -> '–', never @ 0.00 (+0%)
                            odds = _fmt_num(r.get("odds"), ".2f")
                            edge = _fmt_num(r.get("edge_pct"), "+.0f")
                            lines.append(
                                f"      {r['home_team']} vs {r['away_team']} — "
                                f"<b>{side} 2.5</b> @ {odds} ({edge}%)"
                                f"  [{r.get('league','')}]"
                            )
                _unk = bl[~bl["model_type"].isin(["standard", "new_format"])]
                if not _unk.empty:
                    lines.append(f"  ❔ {len(_unk)} tip(s) with no registered model — not listed")
                    print(f"[digest] {len(_unk)} O/U tip(s) without model_type: "
                          f"{sorted(_unk['league'].astype(str).unique())}")
        except Exception as e:
            lines.append(f"  error: {e}")
    else:
        lines.append("⚽ <b>O/U 2.5</b> — no data yet")
    lines.append("")

    # ── Side markets: BTTS / Over 1.5 / Over 3.5 — from the LEDGER (best-tier) ──
    side_file = app_config.OUTPUT_DIR / "side_bets_ledger.csv"
    if side_file.exists():
        try:
            side = _read_csv(side_file)
            _tier_ok = side["signal_tier"].isin(TIER_ORDER)
            _dates   = side["match_date"].astype(str).str[:10]
            upcoming["side markets"] = int((_tier_ok & (_dates > today_str)).sum())
            side = side[_tier_ok & (_dates == today_str)].copy()
            for mkt, mkt_label in SIDE_LABELS.items():
                ms = side[side["market"] == mkt] if "market" in side.columns else pd.DataFrame()
                if ms.empty:
                    continue
                lines.append(f"📌 <b>{mkt_label}</b>  ({len(ms)} tip{'s' if len(ms) != 1 else ''})")
                for tier in TIER_ORDER:
                    tb = ms[ms["signal_tier"] == tier]
                    if tb.empty:
                        continue
                    lines.append(f"  {TIER_SYM[tier]} <b>{tier_label(tier)}</b> ({len(tb)})")
                    for _, r in tb.iterrows():
                        _ev = _fmt_num(r.get("ev_pct"), "+.0f")
                        ev_str = f" EV {_ev}%" if _ev != "–" else " EV –"
                        odds = _fmt_num(r.get("odds"), ".2f")
                        lines.append(
                            f"    {r['home_team']} vs {r['away_team']} — "
                            f"@ {odds}{ev_str}"
                            f"  [{r.get('league','')}]"
                        )
                lines.append("")
        except Exception as e:
            lines.append(f"  Side markets error: {e}")
            lines.append("")

    # ── Player props summary line (owner 2026-11-22: the props TOP-3 per league, never the retired tiers) ──
    try:
        from dashboard_data import props_top3_view as _P3
        _p3 = _P3.load_picks(app_config.OUTPUT_DIR)
        if _p3.empty:
            lines.append("👤 <b>Player Props</b> — no props TOP-3 pick yet")
        else:
            _d3 = _p3["kickoff"].dt.strftime("%Y-%m-%d")
            upcoming["props TOP-3 picks"] = int((_d3 > today_str).sum())
            _n3 = int((_d3 == today_str).sum())
            lines.append(f"👤 <b>Player Props</b> — {_n3} props TOP-3 pick{'s' if _n3 != 1 else ''} today "
                         "(no stake: props not proven yet)")
    except Exception as e:                                             # noqa: BLE001
        lines.append(f"👤 <b>Player Props</b> — error: {e}")
    lines.append("")

    # ── Sharp signals ───────────────────────────────────────────────────────────
    sharp_file = app_config.OUTPUT_DIR / "sharp_tips.csv"
    if sharp_file.exists():
        try:
            sharp = _read_csv(sharp_file)
            _sig_ok = sharp["signal"].isin(["STEAM_STRONG", "STEAM_SHARP", "STRONG"])
            _dates  = sharp["date"].astype(str).str[:10]
            upcoming["sharp signals"] = int((_sig_ok & (_dates > today_str)).sum())
            sharp = sharp[_sig_ok & (_dates == today_str)].copy()
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
            wc = _read_csv(wc_file)
            _sig_ok = wc["signal"].isin(["STEAM_STRONG", "STEAM_SHARP", "STRONG"])
            _dates  = wc["date"].astype(str).str[:10]
            upcoming["WC signals"] = int((_sig_ok & (_dates > today_str)).sum())
            wc = wc[_sig_ok & (_dates == today_str)].copy()
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

    # One line for what is on the board beyond today. This exists so that "No tips today" can
    # never be confused with a date filter that has quietly stopped matching — under `==` those
    # two failure modes render identically, and that class of silent failure has burned this
    # repo repeatedly. If this line disappears while fixtures obviously exist, suspect the date
    # basis (UTC vs local) before suspecting the model.
    _later = ", ".join(f"{n} {k}" for k, n in upcoming.items() if n)
    if _later:
        lines.append(f"🗓 <b>Beyond today</b> — {_later} on the board (not bettable yet)")
        lines.append("")

    # Today's tips as a TABLE IMAGE first (owner request 2026-10-11). Best effort: a rendering or
    # upload failure is logged and the text briefing below still goes out.
    try:
        from telegram_bot.tips_table import render_pngs, send_photo, todays_rows
        _rows = todays_rows(today_str)
        if not _rows.empty:
            _pngs = render_pngs(_rows, f"Today's tips — {header_date}")
            for _k, _png in enumerate(_pngs):
                send_photo(token, chat_id, _png,
                           caption=(f"📊 <b>Today's tips</b> — {len(_rows)} tip(s)" if _k == 0 else ""))
    except Exception as _e:
        print(f"[digest] tips table skipped: {type(_e).__name__}: {_e}")

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
        sub = _settled_only(df[mask]) if not df.empty else pd.DataFrame()
        if sub is None or sub.empty:
            return None
        n = len(sub); w = int((sub["pnl"] > 0).sum()); p = U.total(sub["pnl"])
        return n, w, p

    # Over 2.5 yesterday by tier
    if ledger_file.exists():
        try:
            led = _read_csv(ledger_file)
            led["pnl"] = pd.to_numeric(led["pnl"], errors="coerce")
            yest_led = led[led["match_date"].astype(str).str[:10] == yesterday_str].copy()
            live_y = yest_led[yest_led["source"] == "live"] if "source" in yest_led.columns else yest_led
            live_y = _since_cutoff(live_y)   # THE staked basis: match date on/after the cutoff
            if "model_type" not in live_y.columns:
                live_y["model_type"] = ""
            _blk = live_y["model_type"].isna() | (live_y["model_type"].astype(str).str.strip().isin(["", "nan"]))
            live_y["model_type"] = live_y["model_type"].astype(object)   # an all-blank column reads as float
            live_y.loc[_blk, "model_type"] = live_y.loc[_blk, "league"].map(_model_type_of)
            # STAKED only in every headline. VALUABLE is shown, but never counted as money.
            _stk = _staked_rows(live_y, "bets_ledger.csv")       # THE staked definition (src/staked.py)
            r_all = _stats_sub(_stk, pd.Series([True] * len(_stk), index=_stk.index))
            if r_all:
                n, w, pnl = r_all
                lines2.append(f"⚽ <b>O/U 2.5</b>  💰 staked {w}W/{n-w}L  P/L {U.fmt(pnl)}")
                for fmt, femoji, flabel in [("standard", "⚽", "Standard"),
                                            ("new_format", "🌍", "New-Format")]:
                    fb_all = live_y[live_y["model_type"] == fmt]
                    fb = _staked_rows(fb_all, "bets_ledger.csv")
                    if fb_all.empty:
                        continue
                    rf = _stats_sub(fb, pd.Series([True] * len(fb), index=fb.index))
                    if rf:
                        fn, fw, fp = rf
                        lines2.append(f"  {femoji} <b>{flabel}</b>: {fw}W/{fn-fw}L  P/L {U.fmt(fp)}")
                        if "side" in fb.columns:
                            for sd, ssym in (("OVER", "▲"), ("UNDER", "▼")):
                                r = _stats_sub(fb, fb["side"].astype(str).str.upper() == sd)
                                if r:
                                    sn, sw, sp = r
                                    lines2.append(f"    {ssym} {sd}: {sw}W/{sn-sw}L  P/L {U.fmt(sp)}")
                        for tier in ALL_TIERS:
                            r = _stats_sub(fb, fb["signal_tier"] == tier)
                            if r:
                                tn, tw, tp = r
                                lines2.append(f"    {TIER_SYM[tier]} {tier_label(tier)}: {tw}W/{tn-tw}L  P/L {U.fmt(tp)}")
                    # Collection tiers: visible, but explicitly not money.
                    _fns = _unstaked_rows(fb_all, fb)
                    for tier in ALL_TIERS:
                        r = _stats_sub(_fns, _fns["signal_tier"] == tier) if len(_fns) else None
                        if r:
                            tn, tw, tp = r
                            lines2.append(f"    {TIER_SYM.get(tier, '▰▱▱')} {tier_label(tier)}: {tw}W/{tn-tw}L "
                                          f"({U.fmt(tp)}) — 📊 data only, not staked")
            else:
                lines2.append("  ⚽ O/U 2.5: no settled staked bets")
                _lns = _unstaked_rows(live_y, _stk)
                for tier in ALL_TIERS:
                    r = _stats_sub(_lns, _lns["signal_tier"] == tier) if len(_lns) and "signal_tier" in _lns.columns else None
                    if r:
                        tn, tw, tp = r
                        lines2.append(f"    {TIER_SYM.get(tier, '▰▱▱')} {tier_label(tier)}: {tw}W/{tn-tw}L "
                                      f"({U.fmt(tp)}) — 📊 data only, not staked")
        except Exception:
            pass

    # BTTS / O/U 1.5 / O/U 3.5 yesterday. This block did not exist: "Yesterday's Results" read
    # only the O/U 2.5 ledger and the sharp ledger, so side-market results never appeared (owner
    # report 2026-10-11). Same rules as O/U 2.5: live tips generated on/after the cutoff, STAKED
    # tiers in the headline, VALUABLE shown as data only.
    side_led_y = app_config.OUTPUT_DIR / "side_bets_ledger.csv"
    if side_led_y.exists():
        try:
            sy = _read_csv(side_led_y)
            sy["pnl"] = pd.to_numeric(sy["pnl"], errors="coerce")
            sy = sy[sy["match_date"].astype(str).str[:10] == yesterday_str]
            if "source" in sy.columns:
                sy = sy[sy["source"].astype(str) == "live"]
            sy = _since_cutoff(sy)
            for mkt, mkt_label in SIDE_LABELS.items():
                ms = sy[sy["market"] == mkt] if "market" in sy.columns else pd.DataFrame()
                if ms.empty:
                    continue
                stk = _staked_rows(ms, "side_bets_ledger.csv")       # THE staked definition
                r = _stats_sub(stk, pd.Series([True] * len(stk), index=stk.index)) if not stk.empty else None
                if r:
                    n, w, pnl = r
                    lines2.append(f"📌 <b>{mkt_label}</b>  💰 staked {w}W/{n-w}L  P/L {U.fmt(pnl)}")
                    for tier in ALL_TIERS:
                        rt = _stats_sub(stk, stk["signal_tier"] == tier)
                        if rt:
                            tn, tw, tp = rt
                            lines2.append(f"    {TIER_SYM[tier]} {tier_label(tier)}: {tw}W/{tn-tw}L  P/L {U.fmt(tp)}")
                else:
                    lines2.append(f"📌 <b>{mkt_label}</b>: no settled staked bets")
                _mns = _unstaked_rows(ms, stk)
                for tier in ALL_TIERS:
                    rt = _stats_sub(_mns, _mns["signal_tier"] == tier) if len(_mns) else None
                    if rt:
                        tn, tw, tp = rt
                        lines2.append(f"    {TIER_SYM.get(tier, '▰▱▱')} {tier_label(tier)}: {tw}W/{tn-tw}L "
                                      f"({U.fmt(tp)}) — 📊 data only, not staked")
        except Exception as e:
            lines2.append(f"  Side markets error: {e}")

    # Sharp/WC yesterday — RESTATED record only (src/clv_corrections.sharp_record); the booked P/L is
    # priced at opening_odds and is never shown.
    if sharp_led.exists():
        try:
            sled, _sst = _CC.sharp_record(_read_csv(sharp_led))
            if sled is None:
                raise _SharpUnderRestatement
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
                pnl = U.total(yest_s["pnl"])
                lines2.append(
                    f"  💰 Sharp/WC: {n} settled | {affected} affected / {inaffected} inaffected | P/L <b>{U.fmt(pnl)}</b>"
                )
            else:
                lines2.append("  💰 Sharp/WC: no settled results")
        except _SharpUnderRestatement:
            lines2.append(f"  💰 Sharp/WC: ⚠️ {_CC.UNDER_RESTATEMENT} — no P/L shown")
        except Exception:
            pass

    lines2 += ["", "━━━━━━━━━━━━━━━━",
               f"📈 <b>Ledger — since {app_config.PERFORMANCE_CUTOFF_DATE}</b>", ""]

    # O/U 2.5 — by model type, then tier (counts tips generated on/after the cutoff only)
    if ledger_file.exists():
        try:
            led = _read_csv(ledger_file)
            led["pnl"] = pd.to_numeric(led["pnl"], errors="coerce")
            live = led[(led["source"] == "live") & led["pnl"].notna()] \
                   if "source" in led.columns else led[led["pnl"].notna()]
            live = _since_cutoff(live)
            # Never treat a blank tag as 'standard' — derive it from the league so new-format
            # bets can't leak into the standard column (canonical map in app_config).
            live = live.copy()
            if "model_type" not in live.columns:
                live["model_type"] = ""
            _blank = live["model_type"].isna() | (live["model_type"].astype(str).str.strip().isin(["", "nan"]))
            live["model_type"] = live["model_type"].astype(object)   # an all-blank column reads as float
            live.loc[_blank, "model_type"] = live.loc[_blank, "league"].map(_model_type_of)
            for fmt, emoji, label in [
                ("standard",   "⚽", "O/U 2.5 — Standard"),
                ("new_format", "🌍", "O/U 2.5 — New-Format"),
            ]:
                sub_all = live[live["model_type"] == fmt]
                if sub_all.empty:
                    continue
                # Headline + side splits use STAKED rows only (THE definition, src/staked.py).
                sub = _staked_rows(sub_all, "bets_ledger.csv")
                if not sub.empty:
                    n = len(sub); w_s = int((sub["pnl"] > 0).sum()); pnl = U.total(sub["pnl"])
                    lines2.append(f"  {emoji} <b>{label}</b>  💰 {w_s}W/{n-w_s}L | P/L {U.fmt(pnl)}")
                    if "side" in sub.columns:
                        for sd, ssym in (("OVER", "▲"), ("UNDER", "▼")):
                            ss = sub[sub["side"].astype(str).str.upper() == sd]
                            if ss.empty:
                                continue
                            sn = len(ss); sw = int((ss["pnl"] > 0).sum()); sp = U.total(ss["pnl"])
                            lines2.append(f"    {ssym} {sd}: {sw}W/{sn-sw}L | P/L {U.fmt(sp)}")
                else:
                    lines2.append(f"  {emoji} <b>{label}</b>  no staked bets settled")
                # Collection tiers listed after, clearly not money.
                _sns = _unstaked_rows(sub_all, sub)
                for tier in ALL_TIERS:
                    cs = _sns[_sns["signal_tier"] == tier] if "signal_tier" in _sns.columns else pd.DataFrame()
                    if cs.empty:
                        continue
                    cn = len(cs); cw = int((cs["pnl"] > 0).sum())
                    lines2.append(f"    {TIER_SYM.get(tier, '▰▱▱')} {tier_label(tier)}: {cw}W/{cn-cw}L "
                                  f"({U.fmt(U.total(cs['pnl']))}) — 📊 data only, not staked")
                for tier in ALL_TIERS:
                    ts = sub[sub["signal_tier"] == tier] if "signal_tier" in sub.columns else pd.DataFrame()
                    if ts.empty:
                        continue
                    tn = len(ts); w = int((ts["pnl"] > 0).sum())
                    lines2.append(f"    {TIER_SYM[tier]} {tier_label(tier)}: {w}W/{tn-w}L | P/L {U.fmt(U.total(ts['pnl']))}")
                    _sharp_split_lines(ts, lines2, "        ")
        except Exception:
            pass

    # Side markets all-time (btts/over15/over35)
    side_led = app_config.OUTPUT_DIR / "side_bets_ledger.csv"
    if side_led.exists():
        try:
            sl = _read_csv(side_led)
            sl["pnl"] = pd.to_numeric(sl["pnl"], errors="coerce")
            # Same basis as the O/U 2.5 ledger lines above: live tips generated on/after the cutoff.
            if "source" in sl.columns:
                sl = sl[sl["source"].astype(str) == "live"]
            sl = _since_cutoff(sl)
            sl_s = _settled_only(sl)
            if not sl_s.empty:
                lines2.append("")
                for mkt, mkt_label in SIDE_LABELS.items():
                    ms = sl_s[sl_s["market"] == mkt] if "market" in sl_s.columns else pd.DataFrame()
                    if ms.empty:
                        continue
                    # Headline = STAKED rows only (THE definition, src/staked.py).
                    mstk = _staked_rows(ms, "side_bets_ledger.csv")
                    if not mstk.empty:
                        mn = len(mstk); mw = int((mstk["pnl"] > 0).sum()); mp = U.total(mstk["pnl"])
                        lines2.append(f"  📌 <b>{mkt_label}</b>  💰 {mw}W/{mn-mw}L | P/L {U.fmt(mp)}")
                    else:
                        lines2.append(f"  📌 <b>{mkt_label}</b>  no staked bets settled")
                    for tier in TIER_ORDER:
                        ts = ms[ms["signal_tier"] == tier] if "signal_tier" in ms.columns else pd.DataFrame()
                        if ts.empty:
                            continue
                        tn = len(ts); w = int((ts["pnl"] > 0).sum())
                        tail = ""                                   # all tips of the tier, staked or not
                        lines2.append(f"    {TIER_SYM[tier]} {tier_label(tier)}: {w}W/{tn-w}L | P/L {U.fmt(U.total(ts['pnl']))}{tail}")
        except Exception:
            pass

    # Sharp all-time — RESTATED record only
    if sharp_led.exists():
        try:
            sled, _sst = _CC.sharp_record(_read_csv(sharp_led))
            if sled is None:
                raise _SharpUnderRestatement
            sled["pnl"] = pd.to_numeric(sled["pnl"], errors="coerce")
            all_s = sled[sled["pnl"].notna() & (sled["result"] != "VOID")]
            if not all_s.empty:
                n = len(all_s); pnl = U.total(all_s["pnl"])
                _unr = _sst.get("unrestated", 0)
                lines2.append(f"  💰 <b>Sharp/WC</b>  ({n} signals | P/L {U.fmt(pnl)})"
                              + (f" | {_unr} not yet restated — excluded" if _unr else ""))
        except _SharpUnderRestatement:
            lines2.append(f"  💰 <b>Sharp/WC</b>  ⚠️ {_CC.UNDER_RESTATEMENT} — no P/L shown")
        except Exception:
            pass

    msg2 = "\n".join(lines2)
    _send(token, chat_id, msg2)

    notified.add(digest_key)
    _save_notified(notified)
    print("Daily digest sent (2 messages).")
    return True


def notify_retrain_result() -> bool:
    """Tell Nevo whether the models actually got better this retrain. Returns True if sent.

    Requested 2026-09-22: "i will want to get notice if the models become better and better from
    the retrain and backtesting". Before this, a retrain's outcome existed only in a CI log
    nobody reads, so "are the models improving" was unanswerable in practice even when the data
    to answer it was being produced.

    Reads `output/retrain_log.json`, which `pipeline._train_one` writes per model per run: the
    gate decision, and the held-out mean log loss before and after.

    LOG LOSS, AND LOWER IS BETTER. It is a proper scoring rule and it punishes overconfidence,
    which is the measured defect on the O/U track. A model can raise its accuracy while getting
    worse at this, which is exactly the kind of "improvement" not worth being told about. The
    message says "sharper"/"duller" rather than the metric name, and prints the number anyway.
    """
    cfg = _load_config()
    token, chat_id = cfg.get("token", ""), cfg.get("chat_id", "")
    if not token or token == "YOUR_BOT_TOKEN":
        return False

    path = app_config.OUTPUT_DIR / "retrain_log.json"
    if not path.exists():
        print("No retrain_log.json — nothing to report.")
        return False
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"retrain_log.json unreadable: {e}")
        return False

    run = blob.get("latest_run")
    models = (blob.get("runs") or {}).get(run) or {}
    if not models:
        print("retrain_log.json has no models for the latest run.")
        return False

    notified = _load_notified()
    key = f"RETRAIN|{run}"
    if key in notified:
        print(f"Retrain digest for {run} already sent — skipping.")
        return False

    better = [m for m, v in models.items()
              if v.get("promoted") and isinstance(v.get("logloss_old"), (int, float))
              and isinstance(v.get("logloss_new"), (int, float))
              and v["logloss_new"] < v["logloss_old"]]
    blocked = [m for m, v in models.items() if not v.get("promoted")]

    lines = [f"🔁 <b>RETRAIN</b> — {run}", "━━━━━━━━━━━━━━━━", ""]
    for m, v in sorted(models.items()):
        old, new = v.get("logloss_old"), v.get("logloss_new")
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            d = new - old
            sym = "✅" if d < -1e-6 else ("➖" if abs(d) <= 1e-6 else "⚠️")
            word = "sharper" if d < -1e-6 else ("unchanged" if abs(d) <= 1e-6 else "duller")
            score = f"{old:.5f} → {new:.5f} ({d:+.5f}, {word})"
        else:
            sym, score = "🆕", "no previous score on record"
        if not v.get("promoted"):
            sym = "⛔"
        lines.append(f"{sym} <b>{_escape_html(m)}</b>  {score}")
        # Champion high-water mark (2026-10-18): a non-promotion is either KEEP_CHAMPION
        # (worse/tied) or SHADOW_MORE (possibly better, now measured head-to-head on live
        # fixtures). Older log entries have no verdict and keep the old wording.
        _verdict = {"KEEP_CHAMPION": "KEEP — champion stays",
                    "SHADOW_MORE": "SHADOW — challenger now tested live, champion stays"}.get(
                        str(v.get("verdict", "")), "BLOCKED — incumbent kept")
        lines.append(f"     {_escape_html(str(v.get('rows', '?')))} rows · "
                     f"{'PROMOTED' if v.get('promoted') else _verdict}")
        if not v.get("promoted"):
            lines.append(f"     <i>{_escape_html(str(v.get('why', ''))[:140])}</i>")
    lines += ["", f"📈 <b>{len(better)} of {len(models)}</b> model(s) got sharper"
                  + (f" · ⛔ {len(blocked)} blocked" if blocked else ""),
              "", "<i>Log loss on a held-out chronological slice — lower is better. It punishes "
                  "overconfidence, which raw accuracy does not.</i>"]

    if not _send(token, chat_id, "\n".join(lines)):
        return False
    notified.add(key)
    _save_notified(notified)
    print(f"Retrain digest sent ({len(better)}/{len(models)} sharper, {len(blocked)} blocked).")
    return True


def threshold_source_line(src) -> str:
    """The alert line that says a side tip was tiered on the GLOBAL bar (fix9/defaults 2026-11-04).
    A tier on the historical global bar (config.HISTORICAL_SIDE_GLOBAL_BARS, served until 2026-11-09) must never read like a
    per-league tier; a LEAGUE tier, or a row that does not record its bar, adds no line."""
    s = str(src).strip().upper() if src is not None and not (isinstance(src, float) and src != src) else ""
    if s == app_config.THRESHOLD_SOURCE_GLOBAL:
        return "📏 Tier on the GLOBAL bar - no approved per-league threshold\n"
    return ""


def _side_key(row) -> str:
    # Side is part of the dedup key so a future BTTS-NO signal cannot be suppressed by an already-sent BTTS-YES
    # on the same fixture. Falls back to "" for existing keys, so no previously notified tip is re-sent.
    _sd = str(row.get("side", "") or "")
    return (f"SIDE|{str(row['date'])[:10]}|{row['home_team']}|{row['away_team']}"
            f"|{row['market']}" + (f"|{_sd}" if _sd and _sd.upper() not in ("YES", "OVER", "NAN") else ""))


def _side_candidate(row):
    from src.tip_coherence import Candidate, Event, calibrated_ev, fixture_id, parse_event
    _sd = str(row.get("side", "") or "").upper()
    ev = parse_event(row.get("market"), _sd if _sd and _sd != "NAN" else "YES")
    return Candidate(fixture_id(row["date"], row["home_team"], row["away_team"]),
                     ev or Event("FT", "OTHER", "OVER"), f"side {row.get('market')}",
                     rank=calibrated_ev(row.get("edge_calibrated"), row.get("market_odds"), row.get("edge")),
                     uid=_side_key(row))


def _side_pending() -> "pd.DataFrame":
    """side_bets.csv tips notify_side_bets would consider (tiered, match day today or later)."""
    side_file = app_config.OUTPUT_DIR / "side_bets.csv"
    if not side_file.exists():
        return pd.DataFrame()
    df = _read_csv(side_file)
    from src.tier_labels import TIPPED
    df = df[df["signal_tier"].isin(TIPPED)].copy()
    return df[df["date"].astype(str).str[:10] >= _utc_today_str()]


def notify_side_bets(gate=None) -> int:
    """Send Telegram alerts for BTTS / Over 1.5 / Over 3.5 tips. Returns count sent."""
    cfg = _load_config()
    token   = cfg.get("token", "")
    chat_id = cfg.get("chat_id", "")
    if not token or token == "YOUR_BOT_TOKEN":
        return 0

    side_file = app_config.OUTPUT_DIR / "side_bets.csv"
    if not side_file.exists():
        return 0

    df = _read_csv(side_file)
    # VALUABLE now sends too (Nevo, 2026-09-22: "the tips should be valueables as well, i will
    # choose if i want to bet or not"). The main O/U path has sent VALUABLE since 2026-08-09;
    # side markets never did, and that asymmetry was hiding the single best-performing cell in
    # the estate: BTTS VALUABLE is +18.39u over 76 settled bets at +24.2% ROI with a bootstrap
    # CI of [+0.033, +0.437] — significantly winning, and never once sent.
    #
    # Sending is not staking. These arrive labelled VALUABLE for a human to judge; real money
    # stays where the season-start plan put it. Volume is bounded by notified.json, which sends
    # each fixture+market once.
    from src.tier_labels import TIPPED              # QUALIFIED (owner 2026-11-11) + legacy codes of history rows
    df = df[df["signal_tier"].isin(TIPPED)].copy()

    today_str = _utc_today_str()   # UTC day: stored dates are UTC (src/utc_clock.py)
    df = df[df["date"].astype(str).str[:10] >= today_str]

    if df.empty:
        return 0

    # kickoff order (owner 2026-11-11: no tier and no EV ranking of tips)
    df = df.sort_values(["date", "league", "home_team"])

    # NAME THE SIDE. "BTTS" alone is ambiguous — a reader cannot tell whether the tip is that
    # both teams DO score or that they do not. "Over 1.5" and "Over 3.5" already state their side
    # in the name; BTTS did not. The side-market pipeline is one-sided by construction (it uses
    # p_{market} and odds_{market} with no side concept), so every BTTS signal is implicitly YES.
    # Saying so is the difference between a tip and a guess about what the tip meant.
    MARKET_LABEL = {"btts": "BTTS — YES", "over15": "Over 1.5", "over35": "Over 3.5"}
    MARKET_EMOJI = {"btts": "🔁", "over15": "📈", "over35": "🚀"}

    notified = _load_notified()
    gate = _coherence_gate(gate, notified)
    sent = 0

    for _, row in df.iterrows():
        key = _side_key(row)
        if key in notified:
            continue
        if not instant_tip_allowed(row["date"], notified, row=row):      # morning album carries it
            continue
        # A tip goes out only with its bet-time price and on its league's OWN bar (owner 2026-11-19): a row
        # without a market price, or tiered on the historical GLOBAL bar, is skipped and logged.
        if pd.isna(pd.to_numeric(row.get("market_odds"), errors="coerce")):
            print(f"  SKIP side {row['home_team']} vs {row['away_team']} {row['market']}: NO_MARKET_PRICE")
            continue
        _src = str(row.get("threshold_source") or "").strip().upper()
        if _src == app_config.THRESHOLD_SOURCE_GLOBAL:
            print(f"  SKIP side {row['home_team']} vs {row['away_team']} {row['market']}: GLOBAL_BAR "
                  f"(no approved per-league bar)")
            continue
        _cand = _side_candidate(row)
        _why = gate.check(_cand)
        if _why:
            print(f"  SKIP side {row['home_team']} vs {row['away_team']} {row['market']}: {_why}")
            continue
        _tm = {"btts": ("BTTS", "YES"), "over15": ("OVER15", "OVER"), "over35": ("OVER35", "OVER")}.get(
            str(row["market"]), (str(row["market"]).upper(), str(row.get("side") or "")))
        _twhy, _tkey = _truth_check(row, _tm[0], _tm[1], row.get("market_odds"), row.get("opposite_odds"),
                                    row.get("market_odds_book", ""), row.get("market_odds_captured_at", ""))
        if _twhy:
            print(f"  TRUTH BLOCK side {row['home_team']} vs {row['away_team']} {row['market']}: {_twhy}")
            continue

        tier  = row["signal_tier"]
        mkt   = row["market"]
        edge  = float(row["edge"]) * 100
        ev    = float(row["ev"]) * 100
        label = MARKET_LABEL.get(mkt, mkt)
        emoji = MARKET_EMOJI.get(mkt, "📌")
        # LOOK UP the tier, never infer it. This was
        #     "SNIPER" if tier == "SNIPER" else "MARKSMAN"
        # a binary that was correct only while exactly two tiers could reach here. Adding
        # VALUABLE to the filter above would have silently relabelled every VALUABLE side-market
        # tip as MARKSMAN — a wrong tier on a live alert, which is precisely the number the
        # reader uses to size. Same shape as the tier symbols used everywhere else in this file.
        if tier == "QUALIFIED":       # owner 2026-11-11: no tier word; the cell status of its one bar
            _st = str(row.get("cell_status") or "").strip().upper() or "UNPROVEN"
            tier_header = f"▸ TIP · {_st} — <b>{label}</b>"
        else:
            tier_header = f"{tier_tag(tier)} — <b>{label}</b>"   # icon + display word (src/tier_labels)

        msg = (
            f"{emoji} {tier_header}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📅 {str(row['date'])[:10]}\n"
            f"🏆 {row.get('league', '')}\n"
            f"⚽ {row['home_team']} vs {row['away_team']}\n"
            f"📌 <b>{label}</b>  @ {float(row['market_odds']):.2f}\n"
            + (f"📊 {edge_text(pd.to_numeric(row.get('edge_calibrated'), errors='coerce') * 100)}\n"
               if tier == "QUALIFIED" else
               f"📊 Edge: <b>{edge:.1f}%</b>  |  EV: <b>{ev:+.1f}%</b>  |  Model P={float(row['model_prob'])*100:.0f}%\n") +
            f""
            + threshold_source_line(row.get("threshold_source")) +
            f"⚠️ Check your bookmaker's {label} market"
        )
        # Same per-cell PAPER line as the O/U tips (go-live rehearsal 2026-11-15): a side tip whose cell
        # has no staking entry must not read as a staked bet.
        from src.stake_plan import SIDE_MARKET as _SIDE_TRACK_MARKET
        _paper = _paper_track_note(str(row.get("model_type") or "").strip(),
                                   _SIDE_TRACK_MARKET.get(str(mkt), str(mkt).upper()),
                                   row.get("side"), row.get("league"))
        if _paper:
            msg += f"\n{_paper}"

        if _send(token, chat_id, msg):
            notified.add(key)
            gate.commit(_cand)
            _truth_commit(_tkey)
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


def notify_predict_health(stale_hours: float = 36.0, cooldown_hours: float = 6.0) -> bool:
    """Alert if predict is producing 0 fixtures because of an OUTAGE (not a quiet night).

    Reads output/predict_health.json (written by src.health_check.record_fetch).
    Fires when EITHER:
      * hard failure — every queried league returned a non-200 (the btts-422 signature
        that silently killed all tips for ~2 weeks in Jun 2026), OR
      * staleness   — no fixtures fetched for >= stale_hours.
    Deduped by cooldown_hours so a persistent outage alerts at most every few hours,
    not every 5-minute run. Returns True if an alert was sent.
    """
    from datetime import datetime, timezone

    hf = Path(__file__).resolve().parents[1] / "output" / "predict_health.json"
    if not hf.exists():
        return False
    try:
        state = json.loads(hf.read_text(encoding="utf-8"))
    except Exception:
        return False

    if int(state.get("last_n_fixtures", 0)) > 0:
        return False  # healthy — fixtures found

    now   = datetime.now(timezone.utc)
    stats = state.get("last_stats", {}) or {}
    queried, ok, err = stats.get("queried", 0), stats.get("ok", 0), stats.get("err", 0)
    hard_fail = queried > 0 and ok == 0 and err > 0

    hours_stale = None
    lf = state.get("last_fixtures_iso")
    if lf:
        try:
            hours_stale = (now - datetime.fromisoformat(lf)).total_seconds() / 3600.0
        except Exception:
            hours_stale = None
    stale = hours_stale is not None and hours_stale >= stale_hours

    if not (hard_fail or stale):
        return False

    la = state.get("last_alert_iso")  # cooldown dedup
    if la:
        try:
            if (now - datetime.fromisoformat(la)).total_seconds() / 3600.0 < cooldown_hours:
                return False
        except Exception:
            pass

    if hard_fail:
        msg = ("⚠️ <b>Wowza health alert</b>\n"
               f"Predict fetched <b>0 fixtures</b> and <b>every league returned an API error</b> "
               f"({err}/{queried} failed). Signature of an OddsAPI outage "
               f"(unsupported market / bad key / quota). <b>Tips are DOWN.</b>")
    else:
        stale_txt = f"{hours_stale:.0f}h" if hours_stale is not None else "a long time"
        msg = ("⚠️ <b>Wowza health alert</b>\n"
               f"No fixtures fetched for <b>{stale_txt}</b> (>= {stale_hours:.0f}h). "
               f"Predict may be down (or a genuine quiet period). Last good: "
               f"{str(lf)[:16] if lf else 'unknown'}.")

    cfg = _load_config()
    if _send(cfg["token"], cfg["chat_id"], msg):
        state["last_alert_iso"] = now.isoformat()
        try:
            hf.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception:
            pass
        return True
    return False


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
        # ONE coherence gate for the run (owner 2026-11-19): every candidate of the three senders is planned
        # together, so of two mismatched tips on one match only the higher-ranked one can go out.
        try:
            gate = plan_run_gate()
        except Exception as e:                                          # noqa: BLE001 - senders build their own
            print(f"  coherence plan failed ({type(e).__name__}: {e}) - each sender checks on its own")
            gate = None
        n    = notify_new_snipers(gate)
        side = notify_side_bets(gate)
        ht   = notify_ht_tips(gate)
        # notify_live_signals() is owned by live_scanner.py + its own workflow. Calling it here
        # too (predict runs every 5 min) raced live_scanner's dedup -> DUPLICATE live alerts.
        # Removed 2026-08-09 so live signals have exactly ONE sender.
        health = notify_predict_health()
        print(f"Notifications sent: {n} O/U 2.5 + {side} side market + {ht} HT"
              + (" + HEALTH ALERT" if health else ""))
