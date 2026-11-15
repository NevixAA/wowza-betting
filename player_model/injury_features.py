"""
injury_features.py - chronic_injury_risk / return_from_injury_flag / days_since_last_injury from the
collected fixture-level injury lists (owner 2026-11-15, props feature audit fix 3).

WHY. The three features were read from /sidelined spells (output/collect/sidelined_*.parquet). That
source holds almost no spell after 2025-09 (about 1 spell starting per month 2025-10..2026-05 against
450-900 per month before), so every row since Oct 2025 read "no recorded spell" (return flag 0.000
every month since 2025-11, chronic mean 0.96 -> 0.01), and its spells mix suspensions in. The
collected API-Football /injuries lists (output/collect/injuries.parquet, scripts/collect_injuries:
one row per player listed absent for a fixture, 180,886 rows, league-season answers with
coverage in injuries_seasons.parquet) replace it.

WHAT IS KNOWN BEFORE THE TIP. A fixture's list can be edited after kickoff (Pro measured about 15% of
entries added after kickoff), so a row NEVER reads its own fixture's list. It reads the lists of
fixtures played on EARLIER days: an absence listed for an earlier fixture is known by the time of a
later match. Only type 'Missing Fixture' (the player did not play) counts - 'Questionable' may have
played. Only an injury or illness counts: suspensions (yellow / red cards, suspended, doping) and
other absences (international duty, coach's decision, rest, loan, transfer, ...) are left out
(INJURY_EXCLUDED, a fixed list of the provider's reasons; a reason that is neither listed nor
medical is 'unclassified' and does not count - props_feature_truth reports it).

COVERAGE (missing, never 0, where unknown). A row's league-season (the fixture's API-Football
league_id, the row's season start year) must have been answered by /injuries with coverage
(injuries_seasons status 'ok', coverage_injuries True) and FETCHED on or after the match day (so
every earlier fixture of that league-season was in the answer); chronic_injury_risk also needs the
previous season of the same league (its 365-day window reaches into it). Otherwise all three (or
chronic alone) are missing. `injury_source` says which rule applied.

THE FEATURES (same meaning as before):
  days_since_last_injury   days from the latest earlier day the player was listed injured to the
                           match day; missing when he never was (no gap to measure) or not covered
  return_from_injury_flag  1 when that gap is 1..RETURN_WINDOW_DAYS days, else 0 (covered rows)
  chronic_injury_risk      injury episodes STARTED in [match - CHRONIC_LOOKBACK_DAYS, match): a new
                           episode starts at a listing with no earlier listing, or with a match the
                           player PLAYED between it and his previous listing (played_days)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config
from .history_integrity import CHRONIC_LOOKBACK_DAYS, INJURY_COLUMNS, RETURN_WINDOW_DAYS

COLLECT_DIR = config.OUTPUT_DIR / "collect"
SOURCE_COL = "injury_source"
FROM_LISTS = "injury_lists"                      # all three known
FROM_LISTS_NO_CHRONIC = "injury_lists_no_prev_season"   # chronic missing: previous season not covered
NOT_COVERED = "not_covered"                      # league-season not answered with coverage
STALE = "fetched_before_match"                   # covered, but fetched before the match day
ABSENT_TYPE = "Missing Fixture"

#: provider reasons that are NOT an injury (suspensions first). Never counted as injury.
SUSPENSION_REASONS = frozenset({
    "Yellow Cards", "Red Card", "Suspended", "Red card Suspended", "Suspension", "Yellow card suspension",
    "Doping",
})
OTHER_ABSENCE_REASONS = frozenset({
    "Inactive", "International duty", "National selection", "Coach's decision", "Off the roster",
    "Personal Reasons", "Loan agreement", "Rest", "Transfer negotiations", "ineligible", "Quarantine",
    "Lacking Match Fitness", "Fitness", "fitness", "unfit", "other",
})
INJURY_EXCLUDED = SUSPENSION_REASONS | OTHER_ABSENCE_REASONS
#: a reason is medical when it names an injury, an illness or a treatment
_MEDICAL = re.compile(
    r"injur|knock|ill(ness)?\b|^ill$|broken|fractur|surgery|operation|arthroscop|ligament|tendon|tendin|"
    r"muscle|muscular|strain|sprain|bruise|contusion|hernia|concussion|wound|health|heart|problem|"
    r"trouble|knee|ankle|pain|inflam|infection|virus|flu\b|influenza|cold\b|stomach|append|cartilage|"
    r"menisc|cruciate|pubalgia|pubitis|pubis|overload|fatigue|contracture|convalescence|distortion|"
    r"disloc|lung|bursitis|impingement|stress re|spur|laceration|dead leg|hyaline|torn|ruptur|"
    r"ripture|tear\b|disorder|complaint|intestial|overstretch|stretch|sideband|flexor|dental|tooth|"
    r"collarbone|bone|rib\b|wrist|wirst|finger|toe\b|heel|hip\b|groin|calf|thigh|hamstring|achilles|"
    r"adductor|shin|foot|hand\b|arm\b|elbow|neck|back\b|chest|face|facial|jaw|nose|eye|head|pelvis|"
    r"buttock|abdominal|metatars|metacarp|navicular|tibia|fibula|kneecap|patella|plantar|lumbar|"
    r"vertebra|tailbone|cheekbone|forearm|shoulder|leg\b|body injury",
    re.IGNORECASE)


def reason_class(reason) -> str:
    """'injury' | 'suspension' | 'other' | 'unclassified' for one provider reason."""
    if not isinstance(reason, str) or not reason.strip():
        return "unclassified"
    r = reason.strip()
    if r in SUSPENSION_REASONS:
        return "suspension"
    if r in OTHER_ABSENCE_REASONS:
        return "other"
    return "injury" if _MEDICAL.search(r) else "unclassified"


def _read(name: str) -> pd.DataFrame:
    from .feature_engineering import _collected
    return _collected(name)


def injury_listings(inj: pd.DataFrame | None = None) -> pd.DataFrame:
    """(player_id, _day) of every day a player was listed 'Missing Fixture' with an injury reason,
    one row per player and day, sorted."""
    inj = _read("injuries") if inj is None else inj
    cols = ["player_id", "_day"]
    if inj is None or inj.empty or not {"player_id", "date", "type", "reason"} <= set(inj.columns):
        return pd.DataFrame(columns=cols)
    cls = inj["reason"].map(reason_class)
    m = (inj["type"] == ABSENT_TYPE) & (cls == "injury")
    out = pd.DataFrame({"player_id": pd.to_numeric(inj.loc[m, "player_id"], errors="coerce").to_numpy(dtype=float),
                        "_day": pd.to_datetime(inj.loc[m, "date"], errors="coerce").dt.normalize().to_numpy()})
    return out.dropna().drop_duplicates().sort_values(["_day", "player_id"], kind="stable").reset_index(drop=True)


def episode_starts(listings: pd.DataFrame, played: pd.DataFrame) -> pd.DataFrame:
    """The listings that START an injury episode: the player's first listing, or one with a match he
    PLAYED (played_days) after his previous listing and before it."""
    if listings.empty:
        return listings.copy()
    ls = listings.sort_values(["player_id", "_day"], kind="stable").copy()
    ls["_prev"] = ls.groupby("player_id")["_day"].shift(1)
    if played is not None and not played.empty:
        p = played.assign(_lastplay=played["_day"]).sort_values("_day", kind="stable")
        p = p.assign(player_id=pd.to_numeric(p["player_id"], errors="coerce").astype(float))
        hit = pd.merge_asof(ls.sort_values("_day", kind="stable"), p[["player_id", "_day", "_lastplay"]],
                            on="_day", by="player_id", allow_exact_matches=False)
        ls = hit
    else:
        ls["_lastplay"] = pd.NaT
    new = ls["_prev"].isna() | (ls["_lastplay"].notna() & (ls["_lastplay"] > ls["_prev"]))
    return ls.loc[new.to_numpy(), ["player_id", "_day"]].sort_values("_day", kind="stable").reset_index(drop=True)


def coverage_table(seasons: pd.DataFrame | None = None) -> pd.DataFrame:
    """(league_id, _y, covered, fetched_day) per /injuries league-season answer."""
    s = _read("injuries_seasons") if seasons is None else seasons
    cols = ["league_id", "_y", "covered", "fetched_day"]
    if s is None or s.empty:
        return pd.DataFrame(columns=cols)
    from .feature_engineering import _season_year
    out = pd.DataFrame({
        "league_id": pd.to_numeric(s["league_id"], errors="coerce").to_numpy(dtype=float),
        "_y": _season_year(s["season"]).to_numpy(dtype=float),
        "covered": (s["status"].eq("ok") & s["coverage_injuries"].astype(bool)).to_numpy(),
        "fetched_day": pd.to_datetime(s["fetched_at"], errors="coerce", utc=True).dt.tz_localize(None)
                         .dt.normalize().to_numpy()})
    return out.dropna(subset=["league_id", "_y"]).drop_duplicates(["league_id", "_y"], keep="last")


def row_league_ids(df: pd.DataFrame, fx: pd.DataFrame | None = None) -> pd.Series:
    """API-Football league_id per row: the fixture's own (team_identity.fixture_results), else the
    league_id that the row's competition (registry canonical name) has in every collected fixture."""
    from . import team_identity as TI
    fx = TI.fixture_results() if fx is None else fx
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if fx is None or fx.empty:
        return out
    by_fx = fx.drop_duplicates("fixture_id").set_index("fixture_id")["league_id"]
    if "fixture_id" in df.columns:
        out = pd.to_numeric(df["fixture_id"], errors="coerce").map(by_fx).astype(float)
    comp = fx.dropna(subset=["league_id", "competition"]).groupby("competition")["league_id"].agg(
        lambda s: s.iloc[0] if s.nunique() == 1 else np.nan)
    if "league" in df.columns and out.isna().any():
        names = {lg: (TI._canon(lg) or lg) for lg in pd.unique(df["league"].dropna()) if isinstance(lg, str)}
        canon = df["league"].map(names)                    # one registry lookup per league, not per row
        out = out.where(out.notna(), canon.map(comp).astype(float))
    return out


def injury_features(q: pd.DataFrame, played: pd.DataFrame | None = None, inj: pd.DataFrame | None = None,
                    seasons: pd.DataFrame | None = None, fx: pd.DataFrame | None = None) -> pd.DataFrame:
    """The three injury features + injury_source for the query rows q (player_id, date, fixture_id,
    league, season), aligned with q's index. Training and serving both call this."""
    from .feature_engineering import _season_year
    out = pd.DataFrame(np.nan, index=q.index, columns=list(INJURY_COLUMNS))
    out[SOURCE_COL] = NOT_COVERED
    if q.empty:
        return out
    day = pd.to_datetime(q["date"], errors="coerce").dt.normalize()
    lid = row_league_ids(q, fx)
    y = (pd.Series(_season_year(q["season"]).to_numpy(dtype=float), index=q.index) if "season" in q.columns
         else pd.Series(np.nan, index=q.index))
    cov = coverage_table(seasons).set_index(["league_id", "_y"])
    k_cur = pd.MultiIndex.from_arrays([lid.to_numpy(), y.to_numpy()])
    k_prev = pd.MultiIndex.from_arrays([lid.to_numpy(), (y - 1).to_numpy()])
    cur = cov.reindex(k_cur)
    prev = cov.reindex(k_prev)
    covered = cur["covered"].fillna(False).astype(bool).to_numpy()
    fresh = (pd.to_datetime(cur["fetched_day"]).to_numpy() >= day.to_numpy())
    known = covered & fresh
    known_prev = known & prev["covered"].fillna(False).astype(bool).to_numpy()
    out.loc[covered & ~fresh, SOURCE_COL] = STALE
    out.loc[known, SOURCE_COL] = FROM_LISTS_NO_CHRONIC
    out.loc[known_prev, SOURCE_COL] = FROM_LISTS
    if not known.any():
        return out

    listings = injury_listings(inj)
    qq = pd.DataFrame({"player_id": pd.to_numeric(q["player_id"], errors="coerce").to_numpy(dtype=float),
                       "_day": day.to_numpy(), "_i": np.arange(len(q))})[known]
    qq = qq.dropna(subset=["player_id", "_day"]).sort_values("_day", kind="stable")
    if listings.empty or qq.empty:
        days = pd.Series(np.nan, index=qq["_i"])
    else:
        last = pd.merge_asof(qq, listings.assign(_last=listings["_day"]), on="_day", by="player_id",
                             allow_exact_matches=False)
        days = pd.Series((last["_day"] - last["_last"]).dt.days.to_numpy(dtype=float), index=last["_i"].to_numpy())
    pos = days.index.to_numpy()
    d = days.to_numpy()
    out.iloc[pos, out.columns.get_loc("days_since_last_injury")] = d
    flag = ((d >= 1) & (d <= RETURN_WINDOW_DAYS)).astype(float)
    out.iloc[pos, out.columns.get_loc("return_from_injury_flag")] = flag

    # chronic: episode starts in [day - lookback, day), rows whose previous season is covered too
    starts = episode_starts(listings, played)
    qc = qq[known_prev[qq["_i"].to_numpy()]]
    if len(qc):
        if starts.empty:
            n = np.zeros(len(qc))
        else:
            st = starts.sort_values(["player_id", "_day"], kind="stable").copy()
            st["_n"] = st.groupby("player_id").cumcount() + 1.0
            st = st.sort_values("_day", kind="stable")
            upto = pd.merge_asof(qc, st, on="_day", by="player_id", allow_exact_matches=False)["_n"]
            lo = qc.assign(_day=qc["_day"] - pd.Timedelta(days=CHRONIC_LOOKBACK_DAYS))
            before = pd.merge_asof(lo, st, on="_day", by="player_id", allow_exact_matches=False)["_n"]
            n = upto.fillna(0.0).to_numpy() - before.fillna(0.0).to_numpy()
        out.iloc[qc["_i"].to_numpy(), out.columns.get_loc("chronic_injury_risk")] = n
    return out
