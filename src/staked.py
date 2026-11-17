"""THE definition of "Staked" — one function every page, view and report reads (owner, 2026-11-04).

WHY THIS EXISTS. Until 2026-11-04 "Staked" meant two different things and neither was money:
Home / Overview / Tips / Success Rates (dashboard_data.today_view.is_staked, v9.performance
`staked_only`) meant the TIER (SNIPER + MARKSMAN on any track), and the Bankroll page meant Pro's
review tier rule (O/U 2.5 incl. VALUABLE). On 2026-11-03 the tier basis held 805 bets, of which 495
(-61.32u) sat on tracks the policy has as PAPER (reports/P1_CROSS_REPOSITORY_CONTRACTS.md, finding 1).
Owner 2026-11-04: "1 similar staked" — ONE definition everywhere, and it is the policy's:

    A tip is STAKED when the bankroll policy recorded, at the time, a stake > 0 for it on a
    real-money track: deployment_state in src.bankroll_policy.REAL_STATES, base_fraction > 0 and
    stake_amount > 0, as written by src/stake_plan.py into output/stake_audit.csv (append-only,
    first sight per policy_version x fixture x track) or output/stake_plan.csv (the latest run).

WHAT COUNTS AS PROOF. Only a recorded row. A position (match date, home, away, policy market, side,
model_type) is STAKED when its LAST recorded decision BEFORE KICKOFF qualifies, and its stake is that
decision's stake (owner 2026-11-04, fix round 8). Until then the EARLIEST qualifying record decided,
so a position re-sized to 0 by a later run before kickoff still counted as staked. A restatement
under a renamed track or a new policy_version is the same position, never a second stake
(P1_STAKE_CONTRACT_RECONCILIATION section 4). A stake withdrawn before kickoff is NOT staked; it is
counted in `withdrawn` so it stays visible. A decision recorded AFTER kickoff decides nothing (counted
in `n_after_kickoff`): the bet could no longer be placed. A record without a generation time cannot
be placed before or after kickoff and decides nothing either.

KICKOFF. The recorded kickoff_utc of the same fixture key (date|home|away, the predict frame's own
strings) in KICKOFF_FILES, the files the predict run writes beside the stake plan. Two different
recorded kickoffs for one key are refused (ambiguity). Without a single recorded kickoff the deadline
is the END of the match date, UTC (owner 2026-11-04), and those positions are counted
(`kickoff_basis` = 'match_date_eod', never hidden).

EVERY OTHER ROW IS LABELLED, NEVER GUESSED (`stake_status`):
  STAKED                  the last decision before kickoff qualifies
  NOT_STAKED              records exist, the last one before kickoff does not qualify (PAPER track,
                          tier, odds cap, withdrawn, no decision before kickoff ...) — reason kept
  PRE_POLICY_NO_RECORD    no record and the tip was generated before the first recorded policy decision
                          (the policy was wired 2026-10-14; earlier tips have no stake to prove)
  NO_STAKE_RECORD         no record although the tip is from after the policy start — UNIDENTIFIABLE,
                          counted, never assumed staked
  NOT_A_POLICY_MARKET     the market has no bankroll-policy track (HT O/U, player props): never staked
No stake is invented: a row without a record has stake_units = NaN, not 0.

P/L. Ledgers record flat-1u P/L (pnl per 1 unit). Pages keep that as a separate, labelled number
("flat 1u") and may add `stake_pnl` = stake_units x flat-1u pnl, the P/L at the recorded stake on the
policy's NOTIONAL 100-unit bankroll (stake_plan.BASIS).

MATCHING. Ledgers carry names, not fixture ids, so the key is the exact recorded strings
(date|home|away is the stake plan's own fixture key, built from the same predict frame). Single-
direction markets (BTTS YES, Over 1.5/3.5 OVER) match on the market alone, because the 110 R0 audit
rows wrote the invalid side `YES/OVER` (all zero stake). A ledger row whose model_type is missing, or
an audit row whose model_type is `unknown`, falls back to the key without model_type ONLY when that
key holds exactly one recorded position; otherwise it stays unmatched (ambiguity refused).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.bankroll_policy import REAL_STATES

#: the recorded stake decisions, newest file last
RECORD_FILES = ("stake_audit.csv", "stake_plan.csv")
#: recorded kickoffs (file, match-date column), written by the predict run from the same frame as the
#: stake plan (home_team / away_team / kickoff_utc). Union; a key with two kickoffs is refused.
KICKOFF_FILES = (("predictions.csv", "date"), ("edge_engine_shadow.csv", "match_date"),
                 ("drift_shadow.csv", "match_date"), ("quality_tier_shadow.csv", "match_day"),
                 ("league_weekly_candidates.csv", "match_date"))
KICKOFF, EOD = "kickoff", "match_date_eod"
NO_PRE_KICKOFF = "NO_DECISION_BEFORE_KICKOFF"
AMBIGUOUS = "AMBIGUOUS_LAST_DECISION"
#: ledger / display market -> policy market. Anything else has no policy track.
POLICY_MARKET = {"ou25": "OU25", "o/u 2.5": "OU25", "OU25": "OU25", "btts": "BTTS", "BTTS": "BTTS",
                 "over15": "OVER15", "OVER15": "OVER15", "over 1.5": "OVER15",
                 "over35": "OVER35", "OVER35": "OVER35", "over 3.5": "OVER35"}
#: markets with one bettable direction: matched without the side (see MATCHING above)
SINGLE_DIRECTION = ("BTTS", "OVER15", "OVER35")
MODEL_TYPES = ("new_format", "standard")

STAKED, NOT_STAKED = "STAKED", "NOT_STAKED"
PRE_POLICY, NO_RECORD, NOT_POLICY = "PRE_POLICY_NO_RECORD", "NO_STAKE_RECORD", "NOT_A_POLICY_MARKET"
STATUSES = (STAKED, NOT_STAKED, PRE_POLICY, NO_RECORD, NOT_POLICY)

#: the words pages print next to every "Staked" figure
DEFINITION = ("Staked = the bankroll policy recorded a stake > 0 on a real-money track "
              "(stake plan / stake audit)")
SHORT = "policy stake > 0"
FLAT_LABEL = "flat 1u"

LABEL_COLS = ["stake_status", "stake_units", "stake_fraction", "stake_reason", "stake_policy_version"]


# ── recorded decisions ─────────────────────────────────────────────────────────────────────────
def qualifies(state, base_fraction, stake_amount) -> pd.Series:
    """Row mask of THE rule on recorded stake rows: real-money state, base > 0, stake > 0.

    A recorded base_fraction of 0 (or below) always disqualifies. A base_fraction that is NOT
    recorded (a frame that never carried the column) does not, because bankroll_policy.size_day
    computes requested = base x multipliers, so a recorded stake > 0 can only come from base > 0:
    the stake itself is the proof. Nothing is filled in; the state and the stake must be recorded."""
    st = pd.Series(state, dtype=object).map(lambda v: str(v).strip().upper() if pd.notna(v) else None)
    idx = st.index
    base = pd.to_numeric(pd.Series(base_fraction, index=idx), errors="coerce")
    amt = pd.to_numeric(pd.Series(stake_amount, index=idx), errors="coerce")
    base_ok = base.isna() | (base > 0)
    return (st.isin(REAL_STATES) & base_ok & (amt > 0)).fillna(False).astype(bool)


def _canon_market(m) -> pd.Series:
    s = pd.Series(m, dtype=object)
    return s.map(lambda v: (POLICY_MARKET.get(str(v).strip(), POLICY_MARKET.get(str(v).strip().lower()))
                            if pd.notna(v) else None)).astype(object)


def _side_key(market: pd.Series, side) -> pd.Series:
    sd = pd.Series(side, index=market.index, dtype=object).map(
        lambda v: str(v).strip().upper() if pd.notna(v) else None).astype(object)
    return sd.where(~market.astype(object).isin(SINGLE_DIRECTION), "*")


def _model(m, index) -> pd.Series:
    s = pd.Series(m, index=index, dtype=object).map(lambda v: str(v).strip() if pd.notna(v) else None)
    return s.where(s.isin(MODEL_TYPES), None).astype(object)               # 'unknown' / blank -> missing


def records_from(*frames: pd.DataFrame) -> pd.DataFrame:
    """Stake rows (stake_audit / stake_plan schema) -> one normalised row per recorded decision."""
    cols = ["date", "home", "away", "market", "side_key", "model_type", "generated_at", "qualifies",
            "stake_amount", "final_fraction", "reason", "policy_version", "deployment_state", "kickoff"]
    parts = []
    for f in frames:
        if f is None or f.empty or "fixture" not in f.columns:
            continue
        fx = f["fixture"].astype(object).map(lambda v: str(v) if pd.notna(v) else "").str.split(
            "|", n=2, expand=True)
        if fx.shape[1] < 3:
            continue
        mk = _canon_market(f.get("market"))
        mk.index = f.index
        parts.append(pd.DataFrame({
            "date": fx[0].str.strip().str[:10], "home": fx[1], "away": fx[2], "market": mk,
            "side_key": _side_key(mk, f.get("side")), "model_type": _model(f.get("model_type"), f.index),
            "generated_at": pd.to_datetime(f.get("generated_at"), errors="coerce", utc=True, format="mixed"),
            "qualifies": qualifies(f.get("deployment_state"), f.get("base_fraction"),
                                   f.get("stake_amount")).to_numpy(),
            "stake_amount": pd.to_numeric(f.get("stake_amount"), errors="coerce"),
            "final_fraction": pd.to_numeric(f.get("final_fraction"), errors="coerce"),
            "reason": f.get("reason"), "policy_version": f.get("policy_version"),
            "deployment_state": f.get("deployment_state"),
            # a record frame that carries its own recorded kickoff (none of the stake files does yet)
            "kickoff": (pd.to_datetime(f["kickoff_utc"], errors="coerce", utc=True, format="mixed")
                        if "kickoff_utc" in f.columns
                        else pd.Series(pd.NaT, index=f.index, dtype="datetime64[ns, UTC]"))},
            index=f.index))
    if not parts:
        return pd.DataFrame(columns=cols)
    r = pd.concat(parts, ignore_index=True)
    r = r[r["market"].notna()]
    return r.drop_duplicates(subset=[c for c in cols if c not in ("qualifies", "kickoff")]).reset_index(drop=True)


def _read(out_dir, name, reader):
    p = Path(out_dir) / name
    try:
        return reader(p) if reader is not None else (pd.read_csv(p, low_memory=False) if p.exists() else None)
    except Exception:                                                      # noqa: BLE001 - absent = no record
        return None


def kickoffs_from(frames) -> pd.DataFrame:
    """[(frame, date column)] -> one row per fixture key (date, home, away) with its recorded kickoff
    (UTC). A key recorded with two different kickoffs gets NaT (ambiguity refused, never picked)."""
    parts = []
    for f, dc in frames:
        if f is None or f.empty or not {dc, "home_team", "away_team", "kickoff_utc"} <= set(f.columns):
            continue
        parts.append(pd.DataFrame({
            "date": f[dc].astype(object).map(lambda v: str(v).strip()[:10] if pd.notna(v) else None),
            "home": f["home_team"].astype(object), "away": f["away_team"].astype(object),
            "kickoff": pd.to_datetime(f["kickoff_utc"], errors="coerce", utc=True, format="mixed")}))
    if not parts:
        return pd.DataFrame(columns=["date", "home", "away", "kickoff"])
    k = pd.concat(parts, ignore_index=True).dropna()
    g = k.groupby(["date", "home", "away"])["kickoff"]
    one = g.nunique() == 1
    out = g.first().where(one).rename("kickoff").reset_index()
    return out


def load_kickoffs(out_dir=None, reader=None) -> pd.DataFrame:
    """kickoffs_from(KICKOFF_FILES in `out_dir`); a missing file contributes nothing."""
    if out_dir is None:
        import config
        out_dir = config.OUTPUT_DIR
    return kickoffs_from([(_read(out_dir, name, reader), dc) for name, dc in KICKOFF_FILES])


def with_kickoffs(records: pd.DataFrame, kickoffs: pd.DataFrame) -> pd.DataFrame:
    """records with `kickoff` taken from `kickoffs` where the record carries none."""
    if records is None or records.empty or kickoffs is None or kickoffs.empty:
        return records
    m = records.merge(kickoffs.rename(columns={"kickoff": "_k"}), on=["date", "home", "away"], how="left")
    m["kickoff"] = m["kickoff"].where(m["kickoff"].notna(), m["_k"])
    return m.drop(columns="_k")


def load_records(out_dir=None, reader=None) -> pd.DataFrame:
    """Read RECORD_FILES from `out_dir` (default config.OUTPUT_DIR), with each fixture's recorded
    kickoff (KICKOFF_FILES). `reader(path)` lets the dashboard pass its mirror-aware reader; a missing
    file contributes nothing (no record, never a default)."""
    if out_dir is None:
        import config
        out_dir = config.OUTPUT_DIR
    frames = [f for f in (_read(out_dir, name, reader) for name in RECORD_FILES) if f is not None and len(f)]
    return with_kickoffs(records_from(*frames), load_kickoffs(out_dir, reader))


def policy_start(records: pd.DataFrame):
    """First recorded policy decision (UTC) — read from the records, never typed."""
    if records is None or records.empty:
        return pd.NaT
    return records["generated_at"].min()


def deadline(records: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(deadline UTC, basis) per record: the recorded kickoff, else the end of the match date (UTC).
    A record whose match date does not parse has no deadline (NaT) and decides nothing."""
    ko = (pd.to_datetime(records["kickoff"], errors="coerce", utc=True) if "kickoff" in records.columns
          else pd.Series(pd.NaT, index=records.index, dtype="datetime64[ns, UTC]"))
    day = pd.to_datetime(records["date"], errors="coerce", format="%Y-%m-%d")
    eod = day.dt.tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    basis = pd.Series(KICKOFF, index=records.index, dtype=object).where(ko.notna(), EOD)
    return ko.where(ko.notna(), eod), basis


def decisions(records: pd.DataFrame) -> pd.DataFrame:
    """One row per position: staked? = its LAST decision before kickoff qualifies (owner 2026-11-04),
    the stake of that decision, its reason, withdrawn (a qualifying decision before it, then not),
    the kickoff basis and how many records came after kickoff."""
    key = ["date", "home", "away", "market", "side_key", "model_type"]
    cols = key + ["staked", "stake_units", "stake_fraction", "stake_reason", "stake_policy_version",
                  "withdrawn", "n_records", "kickoff_basis", "n_after_kickoff"]
    if records is None or records.empty:
        return pd.DataFrame(columns=cols)
    r = records.sort_values("generated_at", kind="mergesort").copy()
    r["model_type"] = r["model_type"].astype(object).where(r["model_type"].notna(), "")
    r["_deadline"], r["_basis"] = deadline(r)
    r["_pre"] = (r["generated_at"].notna() & r["_deadline"].notna()
                 & (r["generated_at"] <= r["_deadline"])).astype(bool)
    out = []
    for k, g in r.groupby(key, sort=False, dropna=False):
        pre = g[g["_pre"]]
        basis = KICKOFF if (g["_basis"] == KICKOFF).all() else EOD
        n_after = int((~g["_pre"]).sum())
        if pre.empty:
            last = g.iloc[-1]
            out.append(dict(zip(key, k), staked=False, stake_units=0.0, stake_fraction=0.0,
                            stake_reason=NO_PRE_KICKOFF, stake_policy_version=str(last["policy_version"]),
                            withdrawn=False, n_records=int(len(g)), kickoff_basis=basis,
                            n_after_kickoff=n_after))
            continue
        last = pre.iloc[-1]
        # two records at the same last moment that disagree prove no single decision: refused
        tie = pre[pre["generated_at"] == last["generated_at"]]
        ambiguous = len(tie) > 1 and (tie["qualifies"].nunique() > 1
                                      or tie["stake_amount"].nunique(dropna=False) > 1)
        ok = bool(last["qualifies"]) and not ambiguous
        out.append(dict(zip(key, k), staked=ok,
                        stake_units=float(last["stake_amount"]) if ok else 0.0,
                        stake_fraction=float(last["final_fraction"]) if ok else 0.0,
                        stake_reason=("OK" if ok else AMBIGUOUS if ambiguous else str(last["reason"])),
                        stake_policy_version=str(last["policy_version"]),
                        withdrawn=bool(not ok and pre["qualifies"].any()), n_records=int(len(g)),
                        kickoff_basis=basis, n_after_kickoff=n_after))
    return pd.DataFrame(out, columns=cols)


# ── labelling ledger rows ──────────────────────────────────────────────────────────────────────
def label(frame: pd.DataFrame, records: pd.DataFrame, *, date="match_date", home="home_team",
          away="away_team", market="market", side="side", model_type="model_type",
          generated_at="generated_at", market_const=None) -> pd.DataFrame:
    """LABEL_COLS for every row of `frame` (same index). `market_const` names the market of a ledger
    that has no market column (bets_ledger = O/U 2.5)."""
    idx = frame.index
    out = pd.DataFrame({"stake_status": pd.Series(NO_RECORD, index=idx, dtype=object),
                        "stake_units": float("nan"), "stake_fraction": float("nan"),
                        "stake_reason": pd.Series(None, index=idx, dtype=object),
                        "stake_policy_version": pd.Series(None, index=idx, dtype=object)}, index=idx)
    if frame.empty:
        return out
    mk = (_canon_market(pd.Series(market_const, index=idx)) if market_const is not None
          else _canon_market(frame[market] if market in frame.columns else pd.Series(None, index=idx)))
    mk.index = idx
    k = pd.DataFrame({
        "date": (pd.to_datetime(frame[date], errors="coerce").dt.strftime("%Y-%m-%d").astype(object)
                 if date in frame.columns else None),
        "home": frame[home].astype(object) if home in frame.columns else None,
        "away": frame[away].astype(object) if away in frame.columns else None,
        "market": mk,
        "side_key": _side_key(mk, frame[side] if side in frame.columns else None),
        "model_type": _model(frame[model_type] if model_type in frame.columns else None, idx)}, index=idx)
    out.loc[mk.isna(), "stake_status"] = NOT_POLICY
    dec = decisions(records)
    key5 = ["date", "home", "away", "market", "side_key"]
    if len(dec):
        full = dec.set_index(key5 + ["model_type"])
        # fallback without model_type only where that key holds exactly one recorded position
        one = dec.groupby(key5).filter(lambda g: len(g) == 1).set_index(key5)
        for i, row in k[mk.notna()].iterrows():
            kk = tuple(row[c] for c in key5)
            mt = row["model_type"] if pd.notna(row["model_type"]) else ""
            hit = None
            try:
                hit = full.loc[kk + (mt,)]
            except KeyError:
                if kk in one.index:
                    cand = one.loc[[kk]].iloc[0]
                    if not mt or cand["model_type"] in ("", mt):     # never cross a known model_type
                        hit = cand
            if hit is None:
                continue
            if isinstance(hit, pd.DataFrame):
                hit = hit.iloc[0]
            out.at[i, "stake_status"] = STAKED if bool(hit["staked"]) else NOT_STAKED
            out.at[i, "stake_units"] = float(hit["stake_units"])
            out.at[i, "stake_fraction"] = float(hit["stake_fraction"])
            out.at[i, "stake_reason"] = hit["stake_reason"]
            out.at[i, "stake_policy_version"] = hit["stake_policy_version"]
    start = policy_start(records)
    # format="mixed": ledgers hold "2026-04-26 12:56" and, on one 2026-08-08 row, "...T12:59:30Z"; an
    # inferred single format turned that row into NaT, i.e. an unprovable tip time
    gen = (pd.to_datetime(frame[generated_at], errors="coerce", utc=True, format="mixed")
           if generated_at in frame.columns
           else pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]"))
    # no records at all (file missing) proves nothing about the policy start: rows stay NO_STAKE_RECORD
    pre = (out["stake_status"] == NO_RECORD) & (gen.notna() & (gen < start) if pd.notna(start) else False)
    out.loc[pre, "stake_status"] = PRE_POLICY
    return out


#: ledger file -> (market column, constant market). None/None = no policy market (HT O/U, props).
LEDGER_MARKET = {"bets_ledger.csv": (None, "ou25"), "side_bets_ledger.csv": ("market", None),
                 "ht_ledger.csv": (None, None), "player_ledger.csv": (None, None)}


def label_ledger(frame: pd.DataFrame, ledger: str, records: pd.DataFrame) -> pd.DataFrame:
    """label() with the ledger's own market convention (LEDGER_MARKET). An unlisted ledger, or one
    without a policy market, is NOT_A_POLICY_MARKET on every row — it can never hold a stake."""
    mc, const = LEDGER_MARKET.get(Path(str(ledger)).name, (None, None))
    if mc is None and const is None:
        out = label(frame.iloc[0:0], records)
        out = out.reindex(frame.index)
        out["stake_status"] = NOT_POLICY
        return out
    return label(frame, records, market=mc or "market", market_const=const)


def is_staked(labels: pd.DataFrame | pd.Series) -> pd.Series:
    """THE mask: True only where a qualifying record proves the stake."""
    s = labels["stake_status"] if isinstance(labels, pd.DataFrame) else labels
    return s.astype(object).eq(STAKED)


def plan_staked(plan: pd.DataFrame) -> pd.Series:
    """THE mask on stake-plan / stake-audit rows themselves (Home 'Active tips', Production, Bankroll)."""
    if plan is None or plan.empty:
        return pd.Series(False, index=getattr(plan, "index", None), dtype=bool)

    def _c(name):
        return (pd.to_numeric(plan[name], errors="coerce") if name in plan.columns
                else pd.Series(float("nan"), index=plan.index))
    # the stake as recorded: stake_amount (units), or where a reader did not carry it, the recorded
    # final_fraction of the same decision. Either is > 0 exactly when the policy staked the row.
    amt = _c("stake_amount").where(_c("stake_amount").notna(), _c("final_fraction"))
    state = plan["deployment_state"] if "deployment_state" in plan.columns else pd.Series(None, index=plan.index)
    return pd.Series(qualifies(state.to_numpy(), _c("base_fraction").to_numpy(), amt.to_numpy()).to_numpy(),
                     index=plan.index)


def summary(labels: pd.DataFrame, pnl=None) -> dict:
    """Counts per status, staked n / units, flat-1u P/L and P/L at the recorded stake."""
    st = labels["stake_status"]
    m = is_staked(labels)
    out = {"by_status": {k: int((st == k).sum()) for k in STATUSES}, "staked_n": int(m.sum()),
           "staked_units": round(float(labels.loc[m, "stake_units"].sum()), 4)}
    if pnl is not None:
        p = pd.to_numeric(pd.Series(pnl, index=labels.index), errors="coerce")
        out["flat_pnl"] = round(float(p[m].sum()), 4)
        out["stake_pnl"] = round(float((p[m] * labels.loc[m, "stake_units"]).sum()), 4)
    return out
