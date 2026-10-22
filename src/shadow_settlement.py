"""P0 2026-10-20 — settle the prospective shadow log (output/v91_shadow.csv).

    from src.shadow_settlement import settle_frame, apply_updates, health

WHY THIS EXISTS. `_log_shadow` (pipeline.py) has written one frozen row per fixture since
2026-10-04, with `result`, `closing_odds` and `clv` left empty "to be filled at settlement".
Nothing ever filled them: no code anywhere wrote those columns, and the rows carried only an
OddsAPI event id — no team names — so no results source could have been joined to them even if
something had tried. `prospective_window()` returns settled rows only, so the forward period the
promotion gate needs was permanently empty.

WHAT THIS MODULE DOES, AND WHAT IT REFUSES TO DO.

* It is PURE: no network, no file paths. `update_results.settle_shadow_predictions()` supplies
  the score sources (the same football-data -> OddsAPI -> API-Football hierarchy the ledgers use)
  and the odds archives (the same `_load_archives()` the ledgers use).
* It settles a row only when the match is finished, the row has no result yet, and the row was
  logged BEFORE kickoff. A row logged at/after kickoff is not prospective and is left alone.
* It writes GRADING-ONLY columns (shadow_compare.GRADING_ONLY) plus, once, the identity columns
  when they are blank. Every first-sight column (shadow_compare.FROZEN) is hashed before and after
  and the write is refused if a single byte differs.
* The close is the LAST archived price strictly BEFORE kickoff. A post-kickoff quote is never a
  close; a missing close is NaN with a reason, never an earlier price relabelled and never a
  default. The archives are change-logs (consecutive-distinct prices), so `close_ts` is when that
  price was FIRST seen; it stayed in force until the next change or kickoff. `close_band` makes a
  far-from-kickoff close visible instead of silently calling it a closing line.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from src import team_names
from src.book_quotes import ladder_band
from src.shadow_compare import FROZEN, GRADING_ONLY, IDENTITY

log = logging.getLogger(__name__)

#: A match is treated as finished only this long after kickoff (90' + HT + stoppage + margin).
#: Results sources list only completed matches anyway; this stops a live score being read.
FINISH_GRACE_MIN = 135

#: market -> (kind, line, side -> archive market key). Archive keys follow drift.py /
#: the capture scripts: over25/under25, over15/under15, over35/under35, btts_yes/btts_no.
MARKETS = {
    "ou25":   ("total", 2.5, {"OVER": "over25", "UNDER": "under25"}),
    "over25": ("total", 2.5, {"OVER": "over25", "UNDER": "under25"}),
    "ou15":   ("total", 1.5, {"OVER": "over15", "UNDER": "under15"}),
    "over15": ("total", 1.5, {"OVER": "over15", "UNDER": "under15"}),
    "ou35":   ("total", 3.5, {"OVER": "over35", "UNDER": "under35"}),
    "over35": ("total", 3.5, {"OVER": "over35", "UNDER": "under35"}),
    "btts":   ("btts", None, {"YES": "btts_yes", "NO": "btts_no"}),
}
_SIDE_ALIASES = {"BTTS_YES": "YES", "BTTS_NO": "NO", "Y": "YES", "N": "NO"}

# close statuses (mirroring src/clv_types vocabulary where it applies)
CLOSE_OK = "OK"
NO_CLOSE = "NO_CLOSE"
POST_KICKOFF_ONLY = "POST_KICKOFF_ONLY"
NO_KICKOFF = "NO_KICKOFF"
STALE_CLOSE = "STALE_CLOSE_GT_30MIN"

#: OWNER RULE 2026-10-22: a closing price counts only if it was observed within this many minutes of
#: kickoff. The archives are change-logs, so an earlier first-seen price MAY still have been in force at
#: kickoff - but nothing proves it, and an unproven close is not a close. It is kept in
#: `last_pre_kickoff_odds`; closing_odds and clv stay NaN with close_status STALE_CLOSE_GT_30MIN.
CLOSE_MAX_MIN = 30.0

#: P1 2026-10-22: a row is settled only this long after kickoff. The NEAR capture loops now stay
#: alive up to 170 minutes (src/capture_schedule.CAP_MIN) and commit once at the end, so a close
#: observed at T-5m can reach main up to ~3h after kickoff. Settling at FINISH_GRACE_MIN (135)
#: would freeze such a row as STALE before its close landed — settled rows are never re-graded.
CLOSE_SETTLE_DELAY_MIN = 240


# ── small helpers ─────────────────────────────────────────────────────────────────────────────
def _blank(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return str(v).strip() in ("", "nan", "NaN", "None", "<NA>")


def _ts(v) -> pd.Timestamp | None:
    """UTC timestamp. Naive values are UTC (CI runners write datetime.now() in UTC)."""
    if _blank(v):
        return None
    try:
        t = pd.Timestamp(str(v).strip())
    except Exception:                                                 # noqa: BLE001
        return None
    if pd.isna(t):
        return None
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def norm_market(v) -> str:
    return "ou25" if _blank(v) else str(v).strip().lower()


def norm_side(v) -> str | None:
    if _blank(v):
        return None
    s = str(v).strip().upper()
    return _SIDE_ALIASES.get(s, s)


def row_key(fixture_id, market, match_date) -> tuple[str, str, str]:
    """Same identity append_shadow dedups on."""
    return (str(fixture_id).strip(), norm_market(market), str(match_date).strip()[:10])


def frozen_digest(df: pd.DataFrame) -> str:
    """sha256 over every first-sight column present, in a fixed order, as stored text."""
    h = hashlib.sha256()
    for c in FROZEN:
        if c in df.columns:
            h.update(c.encode())
            h.update("\x1f".join(df[c].astype(str).tolist()).encode("utf-8"))
    return h.hexdigest()


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return str(int(v))
    if isinstance(v, (np.floating, float)):
        return repr(float(v))
    return str(v)


# ── grading ───────────────────────────────────────────────────────────────────────────────────
def grade(market: str, side: str | None, home_goals: int, away_goals: int
          ) -> tuple[str | None, str | None]:
    """(outcome, result). outcome is the side that won the market; result is WIN/LOSS for `side`.

    Returns (None, None) for an unsupported market. result is None when there is no side.
    """
    spec = MARKETS.get(market)
    if spec is None:
        return None, None
    kind, line, sides = spec
    if kind == "total":
        outcome = "OVER" if (home_goals + away_goals) > line else "UNDER"
    else:
        outcome = "YES" if (home_goals > 0 and away_goals > 0) else "NO"
    if side is None or side not in sides:
        return outcome, None
    return outcome, ("WIN" if side == outcome else "LOSS")


# ── result matching ───────────────────────────────────────────────────────────────────────────
def _date_tiers(match_date: str, kickoff: pd.Timestamp | None) -> list[set[str]]:
    d = str(match_date)[:10]
    tiers = [{d}]
    if kickoff is not None and str(kickoff.date()) != d:
        tiers.append({str(kickoff.date())})
    try:
        base = pd.Timestamp(d)
        tiers.append({str((base - pd.Timedelta(days=1)).date()),
                      str((base + pd.Timedelta(days=1)).date())})
    except Exception:                                                 # noqa: BLE001
        pass
    return tiers


_DESC_CANON = {"utd": "united", "rvs": "rovers"}


def _descriptors(name: str) -> set[str]:
    return {_DESC_CANON.get(t, t) for t in team_names._tokens(name) if t in team_names._KEEP}


def guarded(matcher: Callable[[str, str], bool] | None) -> Callable[[str, str], bool] | None:
    """Wrap the ledgers' loose `_names_match` so it can never pair two clubs whose
    distinguishing descriptors differ. Its first-word rule otherwise matches "Manchester City"
    to "Manchester United" — the exact error src/team_names exists to prevent."""
    if matcher is None:
        return None

    def ok(a: str, b: str) -> bool:
        da, db = _descriptors(a), _descriptors(b)
        if da and db and da != db:
            return False
        return bool(matcher(a, b))
    return ok


def match_event(home, away, match_date, kickoff, events: list[dict], *,
                event_id: str | None = None,
                fallback_matcher: Callable[[str, str], bool] | None = None) -> dict | None:
    """Find THE completed event for a fixture, or None. Never guesses between two candidates.

    1. OddsAPI event id, when the source carries one — exact identity, no name matching needed.
    2. Per date tier (match_date, kickoff date, +-1 day): `team_names.resolve` on both sides,
       then the ledgers' own `_names_match` (passed in as `fallback_matcher`). A tier counts
       only when EXACTLY ONE event matches.
    """
    fallback_matcher = guarded(fallback_matcher)
    if event_id:
        hits = [e for e in events if str(e.get("event_id") or "") == str(event_id)]
        if len(hits) == 1:
            return hits[0]
    if _blank(home) or _blank(away):
        return None
    for dates in _date_tiers(match_date, kickoff):
        cands = [e for e in events if str(e.get("date_str", ""))[:10] in dates]
        if not cands:
            continue
        rh = team_names.resolve_result(str(home), sorted({str(e["home_team"]) for e in cands}))
        ra = team_names.resolve_result(str(away), sorted({str(e["away_team"]) for e in cands}))
        if rh and ra:
            hits = [e for e in cands if e["home_team"] == rh and e["away_team"] == ra]
            if len(hits) == 1:
                return hits[0]
        if fallback_matcher is not None:
            hits = [e for e in cands if fallback_matcher(str(home), str(e["home_team"]))
                    and fallback_matcher(str(away), str(e["away_team"]))]
            if len(hits) == 1:
                return hits[0]
    return None


# ── closing price ─────────────────────────────────────────────────────────────────────────────
def _split_match(m: str) -> tuple[str, str]:
    m = str(m)
    if " vs " not in m:
        return "", ""
    h, a = m.split(" vs ", 1)
    return h.strip(), a.strip()


def index_archives(archives: Iterable[tuple[str, pd.DataFrame]]) -> list[tuple[str, dict]]:
    """[(name, {(date, market): frame})] with parsed teams and UTC timestamps, built once."""
    out = []
    for name, d in archives:
        if d is None or d.empty or not {"snapshot_ts", "match", "market", "odds",
                                        "match_date"} <= set(d.columns):
            continue
        d = d.copy()
        teams = d["match"].map(_split_match)
        d["_home"] = [t[0] for t in teams]
        d["_away"] = [t[1] for t in teams]
        d["_date"] = d["match_date"].astype(str).str[:10]
        d["_ts"] = pd.to_datetime(d["snapshot_ts"].astype(str), errors="coerce", utc=True)
        d["_odds"] = pd.to_numeric(d["odds"], errors="coerce")
        d = d[d["_ts"].notna() & d["_odds"].gt(1.0)]
        out.append((name, {k: g for k, g in d.groupby(["_date", "market"])}))
    return out


def closing_price(indexed: list[tuple[str, dict]], home, away, match_date, kickoff,
                  archive_market: str, league: str | None = None,
                  fallback_matcher: Callable[[str, str], bool] | None = None,
                  max_min: float | None = None) -> dict:
    """The final valid PRE-KICKOFF observation for one side of one market.

    Archives are tried in the ledgers' order (densest first). Without `max_min` the first archive
    holding a pre-kickoff price wins (the original rule). With `max_min` (the settlement passes
    CLOSE_MAX_MIN) the first archive whose last pre-kickoff observation lies within `max_min`
    minutes of kickoff wins; only when NO archive can certify a close is the first archive's
    (stale) price returned, for the caller to label STALE. This stops a stale predict-archive price
    hiding a certified capture-archive close for the same fixture — the definition of a close is
    unchanged (strictly pre-kickoff, within 30 minutes), only the search is no longer first-hit.
    Returns a dict with odds (NaN when none), ts, minutes_to_kickoff, band, source and status.
    """
    none = {"odds": np.nan, "ts": None, "minutes": np.nan, "band": "", "source": ""}
    fallback_matcher = guarded(fallback_matcher)
    if kickoff is None:
        return {**none, "status": NO_KICKOFF}
    if _blank(home) or _blank(away):
        return {**none, "status": NO_CLOSE}
    status = NO_CLOSE
    first = None
    d = str(match_date)[:10]
    for name, groups in indexed:
        g = groups.get((d, archive_market))
        if g is None or g.empty:
            continue
        if league and "league" in g.columns:
            gl = g[g["league"].astype(str) == str(league)]
            g = gl if len(gl) else g
        rh = team_names.resolve_result(str(home), sorted(set(g["_home"])))
        ra = team_names.resolve_result(str(away), sorted(set(g["_away"])))
        m = g[(g["_home"] == rh) & (g["_away"] == ra)] if (rh and ra) else g.iloc[0:0]
        if m.empty and fallback_matcher is not None:
            m = g[[fallback_matcher(str(home), h) and fallback_matcher(str(away), a)
                   for h, a in zip(g["_home"], g["_away"])]]
            if m[["_home", "_away"]].drop_duplicates().shape[0] > 1:
                m = m.iloc[0:0]                                      # ambiguous -> refuse
        if m.empty:
            continue
        pre = m[m["_ts"] < kickoff]
        if pre.empty:
            status = POST_KICKOFF_ONLY        # quotes exist, every one at/after kickoff
            continue
        r = pre.sort_values("_ts").iloc[-1]
        mins = float((kickoff - r["_ts"]).total_seconds() / 60.0)
        cand = {"odds": float(r["_odds"]), "ts": r["_ts"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "minutes": round(mins, 1), "band": ladder_band(mins), "source": name,
                "status": CLOSE_OK}
        if max_min is None or mins <= max_min:
            return cand
        if first is None:
            first = cand                      # stale; keep looking for a certifiable close
    return first if first is not None else {**none, "status": status}


# ── the settlement pass ───────────────────────────────────────────────────────────────────────
def settle_frame(df: pd.DataFrame, *,
                 providers: list[tuple],
                 archives: list[tuple[str, pd.DataFrame]] | None = None,
                 identity: dict[str, tuple[str, str]] | None = None,
                 now: datetime | None = None,
                 fallback_matcher: Callable[[str, str], bool] | None = None,
                 ) -> tuple[list[dict], dict]:
    """Work out what to write. Returns (updates, pending_reasons). Does NOT modify `df`.

    `df` is the shadow file read with dtype=str. `providers` is the results hierarchy in priority
    order, each `(name, fn)` or `(name, fn, max_age_days)` with `fn(league) -> [event dicts]`
    (cached here per league, called lazily, so a lower source is only asked when a higher one
    could not resolve a row; a source with `max_age_days` is skipped for older kickoffs). `identity` maps fixture_id
    to (home, away) for rows written before the identity columns existed.
    """
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    now_ts = now_ts.tz_localize("UTC") if now_ts.tzinfo is None else now_ts.tz_convert("UTC")
    stamp = now_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    indexed = index_archives(archives or [])
    identity = identity or {}
    cache: dict[tuple[str, str], list[dict]] = {}

    def events(src_name, fn, league):
        k = (src_name, league)
        if k not in cache:
            try:
                cache[k] = list(fn(league) or [])
            except Exception as e:                                    # noqa: BLE001
                log.warning(f"shadow settle: {src_name} {league} failed ({e})")
                cache[k] = []
        return cache[k]

    updates: list[dict] = []
    pending: dict[str, int] = {}

    def skip(reason):
        pending[reason] = pending.get(reason, 0) + 1

    for _, row in df.iterrows():
        if not _blank(row.get("result")) or not _blank(row.get("settle_status")):
            continue                                                  # already settled
        market = norm_market(row.get("market"))
        if market not in MARKETS:
            skip("UNSUPPORTED_MARKET"); continue
        kickoff = _ts(row.get("kickoff_utc"))
        logged = _ts(row.get("logged_at"))
        if kickoff is None:
            skip("NO_KICKOFF"); continue
        if logged is None or logged >= kickoff:
            skip("NOT_PROSPECTIVE"); continue                         # logged at/after kickoff
        if now_ts < kickoff + pd.Timedelta(minutes=FINISH_GRACE_MIN):
            skip("NOT_FINISHED"); continue
        if now_ts < kickoff + pd.Timedelta(minutes=CLOSE_SETTLE_DELAY_MIN):
            skip("AWAITING_CLOSE_CAPTURE"); continue

        fid = str(row.get("fixture_id", "")).strip()
        home, away = row.get("home_team"), row.get("away_team")
        fill_identity = {}
        if (_blank(home) or _blank(away)) and fid in identity:
            home, away = identity[fid]
            fill_identity = {"home_team": home, "away_team": away}
        league = str(row.get("league", ""))

        ev, src = None, ""
        for prov in providers:
            src_name, fn = prov[0], prov[1]
            max_age = prov[2] if len(prov) > 2 else None
            if max_age is not None and now_ts - kickoff > pd.Timedelta(days=max_age):
                continue
            ev = match_event(home, away, row.get("match_date"), kickoff,
                             events(src_name, fn, league), event_id=fid,
                             fallback_matcher=fallback_matcher)
            if ev is not None:
                src = src_name
                break
        if ev is None:
            skip("NO_IDENTITY" if (_blank(home) or _blank(away)) else "NO_RESULT_FOUND")
            continue
        if _blank(home) or _blank(away):                              # identified by event id
            home, away = ev["home_team"], ev["away_team"]
            fill_identity = {"home_team": home, "away_team": away}

        hg, ag = int(ev["home_score"]), int(ev["away_score"])
        side = norm_side(row.get("v9_side"))
        outcome, result = grade(market, side, hg, ag)

        arch_key = MARKETS[market][2].get(side) if side else None
        close = (closing_price(indexed, home, away, row.get("match_date"), kickoff, arch_key,
                               league=league, fallback_matcher=fallback_matcher,
                               max_min=CLOSE_MAX_MIN)
                 if arch_key else {"odds": np.nan, "ts": None, "minutes": np.nan, "band": "",
                                   "source": "", "status": NO_CLOSE})
        try:
            entry = float(row.get("entry_odds"))
        except (TypeError, ValueError):
            entry = np.nan
        last_pre = close["odds"]
        if close["status"] == CLOSE_OK and np.isfinite(close["minutes"]) and close["minutes"] > CLOSE_MAX_MIN:
            close = {**close, "status": STALE_CLOSE}
        cl = close["odds"] if close["status"] == CLOSE_OK else np.nan
        clv = (round((entry / cl - 1.0) * 100.0, 2)
               if (np.isfinite(entry) and entry > 1.0 and np.isfinite(cl) and cl > 1.0)
               else np.nan)

        fields = {
            **fill_identity,
            "outcome": outcome, "home_goals": hg, "away_goals": ag, "result_source": src,
            "result": result,
            "closing_odds": cl, "clv": clv,
            "close_ts": close["ts"], "close_minutes_to_kickoff": close["minutes"],
            "close_band": close["band"], "close_source": close["source"],
            "close_status": close["status"], "last_pre_kickoff_odds": last_pre,
            "settle_status": "SETTLED" if result else "SETTLED_NO_SIDE",
            "settled_at": stamp,
        }
        updates.append({"fixture_id": fid, "market": market,
                        "match_date": str(row.get("match_date", ""))[:10],
                        "fields": {k: _fmt(v) for k, v in fields.items()}})
    return updates, pending


def apply_updates(df: pd.DataFrame, updates: list[dict]) -> tuple[pd.DataFrame, int]:
    """Write `updates` into a dtype=str frame. Idempotent and conservative:

    * only rows whose result AND settle_status are blank are touched (an already-settled row is
      never re-graded, even by a different update);
    * identity columns are filled only when blank;
    * every FROZEN column is hashed before and after — a mismatch raises and nothing is returned.
    """
    out = df.copy()
    for c in IDENTITY + GRADING_ONLY:
        if c not in out.columns:
            out[c] = ""
    out = out.fillna("")
    before = frozen_digest(out)
    keys = [row_key(f, m, d) for f, m, d in zip(out["fixture_id"], out["market"],
                                                out["match_date"])]
    pos: dict[tuple, list[int]] = {}
    for i, k in enumerate(keys):
        pos.setdefault(k, []).append(i)
    n = 0
    cols = {c: out.columns.get_loc(c) for c in out.columns}
    for u in updates:
        for i in pos.get(row_key(u["fixture_id"], u["market"], u["match_date"]), []):
            if not _blank(out.iat[i, cols["result"]]) or not _blank(out.iat[i, cols["settle_status"]]):
                continue
            for k, v in u["fields"].items():
                if k in IDENTITY:
                    if _blank(out.iat[i, cols[k]]):
                        out.iat[i, cols[k]] = v
                elif k in GRADING_ONLY:
                    out.iat[i, cols[k]] = v
            n += 1
    if frozen_digest(out) != before:
        raise RuntimeError("shadow settlement would alter a first-sight column — refused")
    return out, n


def enforce_close_rule(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Apply the 30-minute close rule to rows settled BEFORE it existed (idempotent, grading-only).

    A row with close_status OK and a close more than CLOSE_MAX_MIN before kickoff keeps that price in
    `last_pre_kickoff_odds`; closing_odds and clv are cleared and close_status becomes STALE. Once
    re-labelled a row is never touched again. Every FROZEN column is hashed before and after."""
    out = df.copy()
    for c in GRADING_ONLY:
        if c not in out.columns:
            out[c] = ""
    out = out.fillna("")
    before = frozen_digest(out)
    mins = pd.to_numeric(out["close_minutes_to_kickoff"], errors="coerce")
    hit = (out["close_status"].astype(str) == CLOSE_OK) & (mins > CLOSE_MAX_MIN)
    out.loc[hit, "last_pre_kickoff_odds"] = out.loc[hit, "closing_odds"]
    out.loc[hit, ["closing_odds", "clv"]] = ""
    out.loc[hit, "close_status"] = STALE_CLOSE
    if frozen_digest(out) != before:
        raise RuntimeError("close rule would alter a first-sight column — refused")
    return out, int(hit.sum())


def health(df: pd.DataFrame, now: datetime | None = None, pending: dict | None = None) -> dict:
    """Coverage numbers for output/shadow_settlement_health.json. Counts, not judgements."""
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    now_ts = now_ts.tz_localize("UTC") if now_ts.tzinfo is None else now_ts.tz_convert("UTC")
    n = int(len(df))
    if n == 0:
        return {"generated_at": now_ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "shadow_rows": 0}
    res = df.get("result", pd.Series([""] * n)).fillna("").astype(str).str.upper()
    settled_mask = res.isin(["WIN", "LOSS"])
    ko = pd.to_datetime(df["kickoff_utc"], errors="coerce", utc=True)
    lg = pd.to_datetime(df["logged_at"], errors="coerce", utc=True)
    finished = (ko + pd.Timedelta(minutes=FINISH_GRACE_MIN)) <= now_ts
    eligible = finished & (lg < ko)
    close = pd.to_numeric(df.get("closing_odds", pd.Series([np.nan] * n)), errors="coerce")
    clv = pd.to_numeric(df.get("clv", pd.Series([np.nan] * n)), errors="coerce")
    s = int(settled_mask.sum())

    def pct(a, b):
        return round(100.0 * a / b, 1) if b else None

    sa = df.get("settled_at", pd.Series([""] * n)).fillna("").astype(str)
    sa = sa[sa.str.len() > 0]
    by_market = {}
    for mk, g in df.assign(_m=df.get("market", pd.Series(["ou25"] * n)).fillna("ou25")
                           ).groupby("_m"):
        gm = g.get("result", pd.Series(dtype=str)).fillna("").astype(str).str.upper()
        by_market[str(mk)] = {"rows": int(len(g)), "settled": int(gm.isin(["WIN", "LOSS"]).sum())}
    return {
        "generated_at": now_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "shadow_rows": n,
        "finished_prospective_rows": int(eligible.sum()),
        "settled_rows": s,
        "settlement_pct_of_all": pct(s, n),
        "settlement_pct_of_finished": pct(s, int(eligible.sum())),
        "valid_closing_price_pct": pct(int((settled_mask & close.gt(1.0)).sum()), s),
        "clv_pct": pct(int((settled_mask & clv.notna()).sum()), s),
        "last_settlement_ts": (sa.max() if len(sa) else None),
        "pending_reasons_this_run": dict(sorted((pending or {}).items())),
        "by_market": by_market,
        "note": ("Coverage only. A settled forward window is NOT evidence of edge; the "
                 "prospective sample is small and is not a validated experiment."),
    }
