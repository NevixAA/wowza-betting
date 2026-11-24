"""STAKE DROUGHT MONITOR (owner rule 2026-11-24, chat, verbatim):
    "if some family (league x market x tier) does not reach staked level for 1 month -> repair and research needed."

HARD RULE. This monitor NEVER changes bars, stakes, tiers or models. It only FLAGS. It reads the TOP-3 ledgers,
weeks and candidates, the serving config (read-only) and the fixture calendar, and writes exactly two files:
output/stake_drought.json and output/stake_drought.csv. Nothing under config/ or models/ is ever written, no
ledger is touched, no Telegram is sent. tests/test_stake_drought.py pins this.

FAMILY = product x league x market (with side) x tier.
  * TOP3        team TOP-3 (src/top3_league.py, output/top3_league_ledger.csv), tiers T1_BEST_VALUE / T2_SAFEST /
                T3_RISKY_VALUE (top3_league.TIER_NAMES).
  * PROPS_TOP3  props TOP-3 (src/props_top3.py, output/props_top3_ledger.csv), roles T1/T2/T3 (props_top3.ROLES).
                A prop market has no side: the side column is blank.

UNIVERSE (how the family list is derived; nothing outside the serving scope is invented):
  * TOP3: the leagues of config/track_registry.json entries that are enabled and not a paper league
    (config.PAPER_LEAGUES: ledgered, never sent, never staked by the owner's own rule) x the market/side pairs that
    occur in output/top3_league_candidates.csv, are a TOP-3 panel market (top3_league.MARKETS) and have a registry
    entry for that league x market x side x the three tiers.
  * PROPS_TOP3: the registry prop leagues (src/league_registry.prop_leagues(), minus paper leagues) x the prop
    markets that occur in output/props_top3_candidates.csv x the three roles.

STAKED = a TOP-3 pick (selection_mode of the TOP-3, a tier slot) with stake_reason STAKED and track_is_real True.
A props ledger without a stake_reason column counts a pick as staked only when its frozen stake is > 0.

WINDOW. Rolling 30 days ending on the as-of day (now floored to 00:00Z, so a re-run on the same day with the same
inputs writes the same bytes). Never counted before the observation start = max(Clean Record start for this rule,
2026-11-21T00:00Z, the product's own TOP-3 start). A family whose observed window is under 30 days is PENDING,
never flagged: the first possible flag is 2026-12-21.

STATES (first that applies):
  OK                      staked at least once in the window;
  PENDING                 observed window under 30 days;
  OFF_SEASON              no fixture of the league in the window in the fixture calendar AND no TOP-3 candidate;
  REPAIR_RESEARCH_NEEDED  everything else.

FIXTURE CALENDAR. Per league, the number of distinct fixtures kicking off inside the window, taken as the MAX over
independent sources (never summed: they spell teams differently): the v9 prediction ledger (every match of every
tracked league is predicted, owner rule 2026-11-17: v9_standard / v9_new_format ou25 partitions; v9_props for prop
leagues), output/af_results_recent.parquet, the TOP-3 candidate stores and output/predictions.csv.

CAUSES of a flagged family (counts over the window; all reported, the dominant = the largest, ties in this order):
  NO_FIXTURES               calendar fixtures of the league that never reached a TOP-3 candidate of this market
                            (team; fixtures). A calendar fixture counts here when it kicked off inside the window
                            before the as-of day and its league-week was not frozen before the window opened.
  NO_MODEL_PREDICTION       candidates without a model probability (team; candidates) / calendar fixtures without
                            a v9_props prediction of this market (props; fixtures)
  MODEL_TOOK_OTHER_SIDE     candidates of this league x market whose engine side was the other side (team O/U 2.5;
                            candidates)
  NO_PRICE_UNDER_3_BOOKS    candidates without a two-sided price or quoted by fewer than 3 books (team;
                            candidates) / predicted fixtures that never became a priced prop candidate (props;
                            fixtures - prop candidates are stored only when priced)
  NO_STAKE_NO_APPROVED_BAR  picks in this tier with that frozen stake_reason, plus unpicked candidates whose cell has
                            no owner-applied bar (cell OFF; src.top3_league.qualify under the serving bars, read-only).
                            Props: picks of an UNPROVEN cell (no prop cell is approved)
  NO_STAKE_BELOW_BAR        picks with that stake_reason, plus unpicked candidates below their cell's bar or outside
                            its odds range
  NO_STAKE_NO_VALUE         picks with that stake_reason (the approved P11 tier gate) / props candidates with no
                            positive value
  BLOCKED_REAL_PICK_GATE    picks / week exclusions carrying a real-pick gate code (duplicate / unreal), once that
                            gate records them (league-week level for week exclusions)
  PICKED_OTHER_TIER         candidates of this market that were picked, but in another tier
  NOT_PICKED                qualified candidates that ranked out (or were late / started / coherence-blocked)
  NO_STAKE_OTHER            picks with any other no-stake reason (e.g. not configured)

    python -m src.stake_drought            # write output/stake_drought.json + .csv (update_results.yml)
    python -m src.stake_drought --print    # counts per state, writes nothing

No price is read into the outputs or printed.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "output"
JSON_NAME = "stake_drought.json"
CSV_NAME = "stake_drought.csv"
VERSION = "STAKE-DROUGHT-1"
RULE = ('owner 2026-11-24: "if some family (league x market x tier) does not reach staked level for 1 month -> '
        'repair and research needed."')
#: the owner's start for this rule: never count before it
RULE_START = pd.Timestamp("2026-11-21T00:00:00Z")
WINDOW_DAYS = 30

OK, PENDING, OFF_SEASON, FLAG = "OK", "PENDING", "OFF_SEASON", "REPAIR_RESEARCH_NEEDED"
STATES = (FLAG, PENDING, OFF_SEASON, OK)
TEAM, PROPS = "TOP3", "PROPS_TOP3"
TIERS = ("T1_BEST_VALUE", "T2_SAFEST", "T3_RISKY_VALUE")

C_FIX, C_MODEL, C_PRICE = "NO_FIXTURES", "NO_MODEL_PREDICTION", "NO_PRICE_UNDER_3_BOOKS"
C_NOBAR, C_BELOW, C_NOVAL = "NO_STAKE_NO_APPROVED_BAR", "NO_STAKE_BELOW_BAR", "NO_STAKE_NO_VALUE"
C_SIDE = "MODEL_TOOK_OTHER_SIDE"
C_GATE, C_OTHER_TIER, C_NOT_PICKED, C_NS_OTHER = ("BLOCKED_REAL_PICK_GATE", "PICKED_OTHER_TIER", "NOT_PICKED",
                                                  "NO_STAKE_OTHER")
CAUSES = (C_FIX, C_MODEL, C_SIDE, C_PRICE, C_NOBAR, C_BELOW, C_NOVAL, C_GATE, C_OTHER_TIER, C_NOT_PICKED, C_NS_OTHER)
MIN_BOOKS = 3
#: a stake_reason / exclusion code that comes from the real-pick gate (owner 2026-11-24 "no duplicates, no unreal")
GATE_TOKENS = ("REAL_PICK", "DUPLICATE", "UNREAL", "NOT_REAL", "UNCONFIRMED_FIXTURE")
_QUAL_NOBAR = ("NO_LEAGUE_BAR", "BAR_NOT_OWNER_APPLIED", "LEAGUE_BAR_INVALID")
_QUAL_BELOW = ("BELOW_LEAGUE_BAR", "OUTSIDE_ODDS_RANGE")

CSV_COLS = ["product", "league", "market", "side", "tier", "state", "dominant_cause", "days_since_last_stake",
            "last_stake_at", "n_staked_window", "n_picks_window", "n_candidates_window", "calendar_fixtures_window",
            "window_start", "window_end"] + [f"n_{c.lower()}" for c in CAUSES]


# ── readers (missing = empty, never a guess) ────────────────────────────────────────────────────────
def _csv(p: Path, **kw) -> pd.DataFrame:
    try:
        return pd.read_csv(p, low_memory=False, **kw)
    except Exception:                                                  # noqa: BLE001
        return pd.DataFrame()


def _json(p: Path) -> dict:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:                                                  # noqa: BLE001
        return {}


def _ts(s) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def _tsv(v) -> pd.Timestamp | None:
    try:
        t = pd.Timestamp(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(t):
        return None
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def _is_gate(code) -> bool:
    c = str(code or "").upper()
    return any(t in c for t in GATE_TOKENS)


def _calendar_frames(out_dir: Path, props: bool) -> list[pd.DataFrame]:
    """[DataFrame(league, kickoff, fixture)] per source; read-only."""
    frames = []
    pl = out_dir / "prediction_ledger" / "v9"
    models = (("v9_props", None),) if props else (("v9_standard", "ou25"), ("v9_new_format", "ou25"))
    for model, mk in models:
        base = pl / model
        parts = []
        dirs = [base / mk] if mk else ([base / "goals"] if (base / "goals").exists() else [])
        for d in dirs:
            for f in sorted(d.glob("*/*.csv")) if d.exists() else []:
                x = _csv(f, usecols=["league", "kickoff_utc", "fixture_ref"])
                x = x.drop_duplicates() if len(x) else x
                if len(x):
                    parts.append(x.rename(columns={"kickoff_utc": "kickoff", "fixture_ref": "fixture"}))
        if parts:
            frames.append(pd.concat(parts, ignore_index=True))
    try:
        r = pd.read_parquet(out_dir / "af_results_recent.parquet", columns=["league", "kickoff_utc", "fixture_id"])
        frames.append(r.rename(columns={"kickoff_utc": "kickoff", "fixture_id": "fixture"}))
    except Exception:                                                  # noqa: BLE001
        pass
    c = _csv(out_dir / ("props_top3_candidates.csv" if props else "top3_league_candidates.csv"),
             usecols=["league", "kickoff_utc", "fixture_id"])
    if len(c):
        frames.append(c.rename(columns={"kickoff_utc": "kickoff", "fixture_id": "fixture"}))
    b = _csv(out_dir / "predictions.csv", usecols=["league", "kickoff_utc", "home_team", "away_team"])
    if len(b):
        frames.append(pd.DataFrame({"league": b["league"], "kickoff": b["kickoff_utc"],
                                    "fixture": b["home_team"].astype(str) + "|" + b["away_team"].astype(str)
                                    + "|" + b["kickoff_utc"].astype(str)}))
    out = []
    for f in frames:
        f = f.assign(kickoff=_ts(f["kickoff"]), league=f["league"].astype(str).str.strip(),
                     fixture=f["fixture"].astype(str))
        out.append(f.drop_duplicates(["league", "fixture", "kickoff"]).reset_index(drop=True))
    return out


def _props_predictions(out_dir: Path) -> pd.DataFrame:
    """(league, market, kickoff, fixture) of every v9_props prediction (market = the partition directory)."""
    base = out_dir / "prediction_ledger" / "v9" / "v9_props"
    parts = []
    for f in sorted(base.glob("*/*/*.csv")) if base.exists() else []:
        x = _csv(f, usecols=["league", "kickoff_utc", "fixture_ref"])
        if len(x):
            x = x.drop_duplicates(["league", "kickoff_utc", "fixture_ref"])
            parts.append(pd.DataFrame({"league": x["league"].astype(str).str.strip(), "market": f.parent.parent.name,
                                       "kickoff": _ts(x["kickoff_utc"]), "fixture": x["fixture_ref"].astype(str)}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["league", "market", "kickoff",
                                                                                     "fixture"])


def load_inputs(out_dir: Path | None = None, root: Path | None = None) -> dict:
    """Everything evaluate() needs, read once. Read-only."""
    od = Path(out_dir) if out_dir else OUT
    rt = Path(root) if root else ROOT
    from src import top3_league as T3
    from src import props_top3 as P3
    import config as CFG
    try:
        from src import league_registry as LRG
        prop_leagues = sorted(LRG.prop_leagues())
    except Exception:                                                  # noqa: BLE001
        prop_leagues = []
    try:
        ctx = T3.bars_context(rt)
    except Exception:                                                  # noqa: BLE001
        ctx = None
    return {
        "registry": _json(rt / "config" / "track_registry.json").get("entries") or [],
        "paper": sorted(getattr(CFG, "PAPER_LEAGUES", set()) or set()),
        "prop_leagues": prop_leagues,
        "team_start": T3.top3_start(rt), "props_start": P3.start(rt),
        "team_ledger": _csv(od / T3.PICKS_NAME), "team_weeks": _csv(od / T3.WEEKS_NAME),
        "team_cands": _csv(od / T3.CANDIDATES_NAME),
        "props_ledger": _csv(od / P3.PICKS_NAME), "props_weeks": _csv(od / P3.WEEKS_NAME),
        "props_cands": _csv(od / P3.CANDIDATES_NAME),
        "team_calendar": _calendar_frames(od, props=False), "props_calendar": _calendar_frames(od, props=True),
        "props_predictions": _props_predictions(od),
        "bars_ctx": ctx,
    }


# ── pure logic ───────────────────────────────────────────────────────────────────────────────────
def as_of_day(now) -> pd.Timestamp:
    t = _tsv(now) if now is not None else pd.Timestamp.now(tz="UTC")
    return t.floor("D")


def window(now, product_start: pd.Timestamp | None) -> dict:
    """{start, end, observed_start, observed_days, pending}. end = the as-of day + 1 day (exclusive)."""
    day = as_of_day(now)
    obs = max([RULE_START] + ([product_start] if product_start is not None else []))
    end = day + pd.Timedelta(days=1)
    start = max(obs, day - pd.Timedelta(days=WINDOW_DAYS))
    days = (day - obs.floor("D")).days
    return {"start": start, "end": end, "observed_start": obs, "observed_days": days,
            "pending": days < WINDOW_DAYS}


def staked_mask(p: pd.DataFrame) -> pd.Series:
    if p is None or p.empty:
        return pd.Series(dtype=bool)
    if "stake_reason" in p.columns:
        m = p["stake_reason"].astype(str).str.strip() == "STAKED"
        if "track_is_real" in p.columns:
            m &= _truthy(p["track_is_real"])
        return m
    return pd.to_numeric(p.get("stake"), errors="coerce").fillna(0) > 0


def due_mask(f: pd.DataFrame, start, day, week_freeze: dict) -> pd.Series:
    """Fixtures TOP-3 should have seen: kickoff in [start, day) and their league-week not frozen before `start`."""
    from src.gameweek import league_week_id
    m = (f["kickoff"] >= start) & (f["kickoff"] < day)
    if not m.any() or not week_freeze:
        return m
    cache: dict = {}
    early = pd.Series(False, index=f.index)
    for i, lg, ko in zip(f.index[m], f.loc[m, "league"], f.loc[m, "kickoff"]):
        k = (lg, ko.floor("h"))
        if k not in cache:
            w = league_week_id(lg, ko)
            cache[k] = bool(w) and w in week_freeze and week_freeze[w] < start
        early.at[i] = cache[k]
    return m & ~early


def calendar_counts(frames: list, start, end, day=None, week_freeze: dict | None = None) -> dict:
    """{league: distinct fixtures} = the MAX over sources: kickoff in [start, end), or - with `day` - the due
    fixtures only (due_mask)."""
    out: dict = {}
    for f in frames:
        if f is None or f.empty:
            continue
        g = f[due_mask(f, start, day, week_freeze or {})] if day is not None else             f[(f["kickoff"] >= start) & (f["kickoff"] < end)]
        for lg, n in g.groupby("league")["fixture"].nunique().items():
            out[lg] = max(out.get(lg, 0), int(n))
    return out


def team_universe(registry: list, cands: pd.DataFrame, paper) -> list[tuple]:
    from src.top3_league import MARKETS
    paper = set(paper or ())
    reg = {(e.get("league"), e.get("market"), e.get("side")) for e in registry
           if e.get("enabled") is not False and not e.get("paper_league") and e.get("league") not in paper}
    pairs = set()
    if cands is not None and len(cands):
        pairs = {(str(m), str(s)) for m, s in zip(cands["market"], cands["side"]) if str(m) in MARKETS}
    return sorted((lg, m, s, t) for (lg, m, s) in reg if (m, s) in pairs for t in TIERS)


def props_universe(prop_leagues, cands: pd.DataFrame, paper) -> list[tuple]:
    paper = set(paper or ())
    mks = sorted({str(m) for m in cands["market"]}) if cands is not None and len(cands) else []
    return sorted((lg, m, "", t) for lg in prop_leagues if lg not in paper for m in mks for t in TIERS)


def _decided_weeks(weeks: pd.DataFrame, start, end) -> pd.DataFrame:
    if weeks is None or weeks.empty:
        return pd.DataFrame(columns=["league_week_id", "league", "excluded"])
    f = _ts(weeks["freeze_ts"])
    return weeks[(f >= start) & (f < end)]


def _tier_picks(led: pd.DataFrame, tier_col: str, mode: str | None) -> pd.DataFrame:
    if led is None or led.empty:
        return pd.DataFrame()
    d = led.copy()
    if mode and "selection_mode" in d.columns:
        d = d[d["selection_mode"].astype(str) == mode]
    d = d[d[tier_col].astype(str).isin(TIERS)].copy()
    d["_tier"] = d[tier_col].astype(str)
    d["_at"] = _ts(d["issued_at"])
    d["_staked"] = staked_mask(d).values if len(d) else []
    return d


def _qualify_reasons(cands: pd.DataFrame, ctx) -> pd.Series:
    """qualify()'s reason per candidate under the serving bars (read-only); '' when it cannot run."""
    if cands.empty or ctx is None:
        return pd.Series("", index=cands.index)
    try:
        from src.top3_league import qualify
        q = qualify(cands.reset_index(drop=True), ctx)
        return pd.Series(list(q["reason"].astype(str)), index=cands.index)
    except Exception:                                                  # noqa: BLE001
        return pd.Series("", index=cands.index)


def _pick_cause(sr: str, props: bool, cell_status: str = "") -> str:
    sr = str(sr or "").strip()
    if _is_gate(sr):
        return C_GATE
    if sr in (C_NOBAR, C_BELOW, C_NOVAL):
        return sr
    if props and sr in ("", "nan") and str(cell_status).upper() == "UNPROVEN":
        return C_NOBAR
    return C_NS_OTHER


def _dominant(counts: dict) -> str:
    best = max(counts.values()) if counts else 0
    if best <= 0:
        return C_FIX
    return next(c for c in CAUSES if counts.get(c, 0) == best)


def _gate_week_count(weeks: pd.DataFrame, league: str) -> int:
    n = 0
    for raw in weeks.loc[weeks["league"].astype(str) == league, "excluded"] if len(weeks) else []:
        try:
            ex = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
        except ValueError:
            ex = {}
        n += sum(int(v) for k, v in ex.items() if _is_gate(k))
    return n


def evaluate_product(product: str, universe: list, ledger: pd.DataFrame, weeks: pd.DataFrame,
                     cands: pd.DataFrame, calendar: list, now, product_start, ctx=None,
                     predictions: pd.DataFrame | None = None) -> list[dict]:
    """One row per family. Pure (apart from the read-only qualify() call on the serving bars)."""
    props = product == PROPS
    w = window(now, product_start)
    start, end, day = w["start"], w["end"], as_of_day(now)
    tier_col = "role" if props else "slot"
    mode = "PROPS_TOP3_LEAGUE_GW" if props else "TOP3_LEAGUE_GW"
    picks = _tier_picks(ledger, tier_col, mode)
    if len(picks):
        picks = picks[picks["_at"] >= w["observed_start"]]
    pw = picks[(picks["_at"] >= start) & (picks["_at"] < end)] if len(picks) else picks
    dw = _decided_weeks(weeks, start, end)
    wk_ids = set(dw["league_week_id"].astype(str)) if len(dw) else set()
    c = cands[cands["league_week_id"].astype(str).isin(wk_ids)].copy() if cands is not None and len(cands) \
        else pd.DataFrame(columns=["league", "market", "side", "fixture_id", "league_week_id"])
    if not props:
        if "side" not in c.columns:
            c["side"] = ""
        c["_qual"] = _qualify_reasons(c, ctx) if len(c) else pd.Series(dtype=str)
    wf = {}
    if weeks is not None and len(weeks):
        wf = dict(zip(weeks["league_week_id"].astype(str), _ts(weeks["freeze_ts"])))
    cal = calendar_counts(calendar, start, end)
    due = calendar_counts(calendar, start, end, day, wf)
    pred = None
    if props and predictions is not None:
        pr = predictions[due_mask(predictions, start, day, wf)] if len(predictions) else predictions
        pred = pr.groupby(["league", "market"])["fixture"].nunique().to_dict() if len(pr) else {}
    pk = ["league_week_id", "fixture_id", "market"] + (["player_id"] if props else ["side"])
    pick_tier = {}
    if len(pw):
        for r in pw.to_dict("records"):
            pick_tier[tuple(str(r.get(k)) for k in pk)] = r
    rows = []
    for lg, mk, sd, tier in universe:
        fam_c = c[(c["league"].astype(str) == lg) & (c["market"].astype(str) == mk)] if len(c) else c
        mkt_fix = int(fam_c["fixture_id"].astype(str).nunique()) if len(fam_c) else 0
        n_other_side = 0
        if not props and len(fam_c):
            n_other_side = int((fam_c["side"].astype(str) != sd).sum())
            fam_c = fam_c[fam_c["side"].astype(str) == sd]
        fp = picks[(picks["league"].astype(str) == lg) & (picks["market"].astype(str) == mk)
                   & (picks["_tier"] == tier)] if len(picks) else picks
        if not props and len(fp):
            fp = fp[fp["side"].astype(str) == sd]
        fpw = fp[(fp["_at"] >= start) & (fp["_at"] < end)] if len(fp) else fp
        n_staked = int(fpw["_staked"].sum()) if len(fpw) else 0
        last = fp.loc[fp["_staked"], "_at"].max() if len(fp) and fp["_staked"].any() else None
        n_cal, n_due = int(cal.get(lg, 0)), int(due.get(lg, 0))
        counts = {k: 0 for k in CAUSES}
        counts[C_SIDE] = n_other_side
        for r in fam_c.to_dict("records"):
            key = tuple(str(r.get(k)) for k in pk)
            p = pick_tier.get(key)
            if p is not None:
                if p["_tier"] == tier:
                    if not p["_staked"]:
                        counts[_pick_cause(p.get("stake_reason"), props, p.get("cell_status"))] += 1
                else:
                    counts[C_OTHER_TIER] += 1
                continue
            if props:
                counts[C_NOVAL if not (pd.to_numeric(r.get("value"), errors="coerce") > 0) else C_NOT_PICKED] += 1
                continue
            pm = pd.to_numeric(r.get("p_model_raw"), errors="coerce")
            nb = pd.to_numeric(r.get("n_books"), errors="coerce")
            od = pd.to_numeric(r.get("odds"), errors="coerce")
            qr = str(r.get("_qual") or "")
            if pd.isna(pm):
                counts[C_MODEL] += 1
            elif (str(r.get("price_quality_state")) not in ("TWO_SIDED_MULTI_BOOK", "TWO_SIDED_SINGLE_BOOK")
                  or not (od > 1) or not (nb >= MIN_BOOKS) or qr == "NO_TWO_SIDED_PRICE"):
                counts[C_PRICE] += 1
            elif qr in _QUAL_NOBAR:
                counts[C_NOBAR] += 1
            elif qr in _QUAL_BELOW:
                counts[C_BELOW] += 1
            elif _is_gate(qr):
                counts[C_GATE] += 1
            else:
                counts[C_NOT_PICKED] += 1
        if props:
            n_pred = None if pred is None else int(pred.get((lg, mk), 0))
            if n_pred is not None:
                counts[C_MODEL] += max(0, n_due - n_pred)
                counts[C_PRICE] += max(0, n_pred - mkt_fix)
            else:
                counts[C_FIX] += max(0, n_due - mkt_fix)
        else:
            counts[C_FIX] += max(0, n_due - mkt_fix)
        counts[C_GATE] += _gate_week_count(dw, lg)
        if n_staked > 0:
            state = OK
        elif w["pending"]:
            state = PENDING
        elif n_cal == 0 and len(fam_c) == 0 and mkt_fix == 0:
            state = OFF_SEASON
        else:
            state = FLAG
        rows.append({
            "product": product, "league": lg, "market": mk, "side": sd, "tier": tier, "state": state,
            "dominant_cause": _dominant(counts) if state == FLAG else "",
            "days_since_last_stake": int((day - last.floor("D")).days) if last is not None else None,
            "last_stake_at": last.strftime("%Y-%m-%dT%H:%M:%SZ") if last is not None else "",
            "n_staked_window": n_staked, "n_picks_window": int(len(fpw)),
            "n_candidates_window": int(len(fam_c)), "calendar_fixtures_window": n_cal,
            "window_start": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "window_end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            **{f"n_{k.lower()}": int(v) for k, v in counts.items()},
        })
    return rows


def evaluate(inputs: dict, now=None) -> dict:
    """The whole document (families + summary). Pure over `inputs`."""
    team_u = team_universe(inputs.get("registry") or [], inputs.get("team_cands"), inputs.get("paper"))
    props_u = props_universe(inputs.get("prop_leagues") or [], inputs.get("props_cands"), inputs.get("paper"))
    rows = evaluate_product(TEAM, team_u, inputs.get("team_ledger"), inputs.get("team_weeks"),
                            inputs.get("team_cands"), inputs.get("team_calendar") or [], now,
                            inputs.get("team_start"), inputs.get("bars_ctx"))
    rows += evaluate_product(PROPS, props_u, inputs.get("props_ledger"), inputs.get("props_weeks"),
                             inputs.get("props_cands"), inputs.get("props_calendar") or [], now,
                             inputs.get("props_start"), predictions=inputs.get("props_predictions"))
    order = {s: i for i, s in enumerate(STATES)}
    rows.sort(key=lambda r: (order[r["state"]], r["product"], r["league"], r["market"], r["side"],
                             TIERS.index(r["tier"])))
    counts = {p: {s: sum(1 for r in rows if r["product"] == p and r["state"] == s) for s in STATES}
              for p in (TEAM, PROPS)}
    tw, pw = window(now, inputs.get("team_start")), window(now, inputs.get("props_start"))
    return {
        "about": "Stake drought monitor (src/stake_drought.py). FLAGS ONLY: never changes bars, stakes, tiers or "
                 "models. One row per family = product x league x market (side) x tier; see the module docstring "
                 "for the universe, window, states and causes.",
        "rule": RULE, "version": VERSION, "as_of": as_of_day(now).strftime("%Y-%m-%d"),
        "rule_start_utc": RULE_START.strftime("%Y-%m-%dT%H:%M:%SZ"), "window_days": WINDOW_DAYS,
        "windows": {TEAM: {k: (v.strftime("%Y-%m-%dT%H:%M:%SZ") if isinstance(v, pd.Timestamp) else v)
                           for k, v in tw.items()},
                    PROPS: {k: (v.strftime("%Y-%m-%dT%H:%M:%SZ") if isinstance(v, pd.Timestamp) else v)
                            for k, v in pw.items()}},
        "first_possible_flag": {TEAM: (tw["observed_start"].floor("D") + pd.Timedelta(days=WINDOW_DAYS))
                                .strftime("%Y-%m-%d"),
                                PROPS: (pw["observed_start"].floor("D") + pd.Timedelta(days=WINDOW_DAYS))
                                .strftime("%Y-%m-%d")},
        "universe": {TEAM: "track_registry enabled non-paper leagues x TOP-3 market/sides seen in "
                           "top3_league_candidates.csv x 3 tiers",
                     PROPS: "registry prop leagues (non-paper) x prop markets seen in props_top3_candidates.csv "
                            "x 3 roles",
                     "excluded_paper_leagues": sorted(inputs.get("paper") or [])},
        "n_families": len(rows), "counts": counts,
        "total": {s: sum(c[s] for c in counts.values()) for s in STATES},
        "families": rows,
    }


def _clean(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def render(doc: dict) -> tuple[str, str]:
    d = _clean(doc)
    head = json.dumps({k: v for k, v in d.items() if k != "families"}, indent=1, ensure_ascii=False)
    fams = ",\n".join("  " + json.dumps(r, ensure_ascii=False) for r in d.get("families") or [])
    # one line per family: a small file and readable diffs
    js = head[:-2] + ',\n "families": [\n' + fams + ("\n" if fams else "") + " ]\n}\n"
    df = pd.DataFrame(doc["families"]).reindex(columns=CSV_COLS)
    df["days_since_last_stake"] = df["days_since_last_stake"].astype("Int64")
    return js, df.to_csv(index=False, lineterminator="\n")


def _write_if_changed(path: Path, text: str) -> bool:
    try:
        if path.read_text(encoding="utf-8") == text:
            return False
    except FileNotFoundError:
        pass
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return True


def write(now=None, out_dir: Path | None = None, root: Path | None = None) -> dict:
    """Write output/stake_drought.json + .csv only when their bytes change (idempotent). Writes nothing else."""
    od = Path(out_dir) if out_dir else OUT
    doc = evaluate(load_inputs(od, root), now)
    js, cs = render(doc)
    changed = _write_if_changed(od / JSON_NAME, js)
    changed = _write_if_changed(od / CSV_NAME, cs) or changed
    return {"changed": changed, "counts": doc["counts"], "total": doc["total"]}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--print" in argv:
        doc = evaluate(load_inputs())
        print(json.dumps({"as_of": doc["as_of"], "counts": doc["counts"], "total": doc["total"]}, indent=1))
        return 0
    res = write()
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
