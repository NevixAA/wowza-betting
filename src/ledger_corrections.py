"""Live-ledger hygiene corrections BESIDE the ledgers (cross-repo certification findings, 2026-11-11).

    python -m src.ledger_corrections build --ledger-dir <dir holding the live ledgers>   # write sidecars
    python -m src.ledger_corrections check  --ledger-dir <dir>                          # counts only

The committed ledgers are never rewritten. Each finding is a sidecar in output/corrections/ with the
original key and the evidence; the readers apply it (src/trustworthy_performance.correct_live, the CLV
consumers through src/clv_corrections, the v91 shadow readers through read_v91_shadow), and data truth
rule 6 (src/integrity/data_truth.ledgers) goes RED if any of them recurs from the Clean Record start.

  M  MULTI-TIP FIXTURES  bets_ledger_multi_tip_fixtures.csv: every row of a fixture holding more than one
     live O/U 2.5 row (a later predict run logged another side). The first-sight row (earliest
     generated_at) is marked; OVER + UNDER on one fixture is flagged `opposite_tip_pair`. FLAGGED, NOT
     DELETED and NOT REGRADED: how such a pair counts in a record is an owner decision (open).
  K  POST-KICKOFF        bets_ledger_post_kickoff.csv: live rows generated after the provider's kick-off
     (API-Football fixture claimed by src.integrity.fixture_claims, kick-off from the HT store). A tip
     issued in play is not a pre-match tip: EXCLUDED from every figure (the top-picks precedent).
  C  INVALID CLOSE       clv_invalid_close.csv: closing_odds <= 1.0 (no decimal price can be 1.0) - CLV
     blanked like the first-half closes (src.clv_corrections.contaminated_mask applies it; P/L kept).
  P  PLAYER DUPLICATES   player_ledger_duplicates.csv: a later copy of the same player, market and match
     (same provider fixture id + player id + match date, or - without ids - the same claimed provider
     fixture, player name, team and market under another club spelling). FIRST SIGHT wins: the earliest
     signal_date row stays graded, the later copy is EXCLUDED. A rescheduled match (match_date moved)
     is NOT a duplicate (the 48 h rule settles it).
  V  V91 DUPLICATES      v91_shadow_duplicates.csv: one provider fixture logged on two match dates; the
     copy whose match_date is not the provider's kick-off date (else the later logged one) is dropped by
     read_v91_shadow().
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORR = ROOT / "output" / "corrections"
MULTI = "bets_ledger_multi_tip_fixtures.csv"
POST_KO = "bets_ledger_post_kickoff.csv"
BAD_CLOSE = "clv_invalid_close.csv"
PLAYER_DUP = "player_ledger_duplicates.csv"
V91_DUP = "v91_shadow_duplicates.csv"

BETS_KEY = ["generated_at", "match_date", "league", "home_team", "away_team", "side"]
FIXTURE = ["match_date", "league", "home_team", "away_team"]
PLAYER_KEY = ["signal_date", "match_date", "league", "home_team", "away_team", "player_name", "market"]
V91_KEY = ["fixture_id", "market", "match_date", "logged_at"]

log = logging.getLogger(__name__)


def _dir(out_dir=None) -> Path:
    return Path(out_dir) / "corrections" if out_dir is not None else CORR


def _read(name: str, out_dir=None) -> pd.DataFrame | None:
    p = _dir(out_dir) / name
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, dtype=str, keep_default_na=False)
    except Exception as e:                                             # noqa: BLE001
        log.warning("ledger sidecar %s unreadable (%s)", name, e)
        return None


def _keys(d: pd.DataFrame, cols: list[str]) -> pd.Series:
    k = pd.DataFrame(index=d.index)
    for c in cols:
        v = d[c] if c in d.columns else pd.Series("", index=d.index)
        v = v.astype(str).str.strip().replace({"nan": "", "None": "", "<NA>": "", "NaT": ""})
        if c in ("match_date", "signal_date"):
            v = v.str[:10]
        if c == "side":
            v = v.str.upper()
        if c == "fixture_id":
            v = v.str.replace(r"\.0$", "", regex=True)
        k[c] = v
    return k.apply(tuple, axis=1) if len(k) else pd.Series([], dtype=object)


def member(d: pd.DataFrame, listed: pd.DataFrame | None, cols: list[str]) -> pd.Series:
    if listed is None or listed.empty or d is None or d.empty:
        return pd.Series(False, index=getattr(d, "index", None), dtype=bool)
    return _keys(d, cols).isin(set(_keys(listed, cols)))


# ── finders (pure: a ledger frame in, the rows to list out) ──────────────────────────────────────
def _ou_dir(side: pd.Series) -> pd.Series:
    s = side.astype(str).str.upper()
    return s.where(s.isin(["OVER", "UNDER"]))


def multi_tip_rows(bets: pd.DataFrame) -> pd.DataFrame:
    """Every live row of a fixture with more than one live row, with first_sight / opposite flags."""
    b = bets[bets.get("source", pd.Series("live", index=bets.index)).astype(str) == "live"].copy()
    if b.empty:
        return pd.DataFrame(columns=BETS_KEY + ["n_rows", "first_sight", "opposite_tip_pair", "sides"])
    b["_gen"] = pd.to_datetime(b["generated_at"], errors="coerce", utc=True, format="mixed")
    b["_fx"] = _keys(b, FIXTURE)
    n = b.groupby("_fx")["_fx"].transform("size")
    m = b[n > 1].sort_values(["_fx", "_gen"], kind="mergesort").copy()
    if m.empty:
        return pd.DataFrame(columns=BETS_KEY + ["n_rows", "first_sight", "opposite_tip_pair", "sides"])
    m["n_rows"] = m.groupby("_fx")["_fx"].transform("size")
    m["first_sight"] = ~m["_fx"].duplicated()
    dirs = m.assign(_d=_ou_dir(m["side"])).groupby("_fx")["_d"].agg(lambda s: set(s.dropna()))
    m["opposite_tip_pair"] = m["_fx"].map(lambda f: dirs[f] >= {"OVER", "UNDER"})
    m["sides"] = m["_fx"].map(m.groupby("_fx")["side"].agg(lambda s: "|".join(map(str, s))))
    cols = BETS_KEY + ["odds", "result", "pnl", "n_rows", "first_sight", "opposite_tip_pair", "sides"]
    return m[[c for c in cols if c in m.columns]].reset_index(drop=True)


def post_kickoff_rows(bets: pd.DataFrame, pool: pd.DataFrame | None = None,
                      kickoffs: pd.Series | None = None) -> pd.DataFrame:
    """Live rows generated after the provider kick-off of the fixture they claim."""
    from src import league_registry as LR
    from src.integrity import fixture_claims as FC
    cols = BETS_KEY + ["odds", "result", "provider_fixture_id", "provider_kickoff_utc"]
    if bets is None or bets.empty:
        return pd.DataFrame(columns=cols)
    pool = FC.provider_pool() if pool is None else pool
    if kickoffs is None:
        ht = pd.read_parquet(FC.HT_SCORES, columns=["fixture_id", "kickoff"]).dropna()
        kickoffs = ht.drop_duplicates("fixture_id").set_index("fixture_id")["kickoff"]
    x = bets.rename(columns={"match_date": "date"}).copy()
    x["league"] = x["league"].map(lambda s: LR.canonical_name(s) if isinstance(s, str) else None)
    x["_hg"], x["_ag"] = np.nan, np.nan
    c = FC.claims(x, pool, home_goals="_hg", away_goals="_ag")
    if c.empty:
        return pd.DataFrame(columns=cols)
    ko = pd.to_datetime(c["fixture_id"].map(kickoffs), errors="coerce", utc=True)
    gen = pd.to_datetime(bets.loc[c["idx"], "generated_at"].to_numpy(), errors="coerce", utc=True, format="mixed")
    hit = c[(gen > ko.to_numpy()) & ko.notna().to_numpy()]
    out = bets.loc[hit["idx"]].copy()
    out["provider_fixture_id"] = hit["fixture_id"].to_numpy()
    out["provider_kickoff_utc"] = c.set_index("idx").loc[hit["idx"], "fixture_id"].map(kickoffs).to_numpy()
    return out[[c for c in cols if c in out.columns]].reset_index(drop=True)


def invalid_close_rows(bets: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(bets.get("closing_odds"), errors="coerce")
    out = bets[close <= 1.0]
    cols = BETS_KEY + ["odds", "closing_odds", "clv_pct", "result"]
    return out[[c for c in cols if c in out.columns]].reset_index(drop=True)


def player_duplicate_rows(pl: pd.DataFrame, pool: pd.DataFrame | None = None) -> pd.DataFrame:
    """Later copies (first sight wins) of one player x market x match, by id or by claimed fixture."""
    from src import league_registry as LR
    from src.integrity import fixture_claims as FC
    cols = PLAYER_KEY + ["team", "tier", "result", "pnl", "player_id", "fixture_id", "duplicate_of_signal_date",
                         "duplicate_rule"]
    if pl is None or pl.empty:
        return pd.DataFrame(columns=cols)
    p = pl.copy()
    p["_sig"] = pd.to_datetime(p["signal_date"], errors="coerce")
    p["_md"] = p["match_date"].astype(str).str[:10]
    fid = pd.to_numeric(p.get("fixture_id"), errors="coerce")
    if pool is None:
        pool = FC.provider_pool()
    x = p.rename(columns={"match_date": "date"}).copy()
    x["league"] = x["league"].map(lambda s: LR.canonical_name(s) if isinstance(s, str) else None)
    x["_hg"], x["_ag"] = np.nan, np.nan
    c = FC.claims(x, pool, home_goals="_hg", away_goals="_ag").set_index("idx")["fixture_id"]
    p["_fx"] = fid.where(fid.notna(), c.reindex(p.index))
    pid = pd.to_numeric(p.get("player_id"), errors="coerce")
    name = p["player_name"].astype(str).str.strip().str.lower()
    team = p["team"].astype(str).str.strip().str.lower()
    p["_who"] = np.where(pid.notna(), "id:" + pid.astype("Int64").astype(str), "nm:" + name + "|" + team)
    ok = p["_fx"].notna()
    q = p[ok].sort_values(["_sig"], kind="mergesort")
    grp = ["_fx", "_who", "market", "_md"]
    first = q.groupby(grp)["signal_date"].transform("first")
    dup = q[q.duplicated(grp)].copy()
    dup["duplicate_of_signal_date"] = first[dup.index]
    dup["duplicate_rule"] = np.where(dup["_who"].str.startswith("id:"), "same_fixture_player_id_market_date",
                                     "same_claimed_fixture_player_name_team_market_date")
    return dup[[c for c in cols if c in dup.columns]].reset_index(drop=True)


def v91_duplicate_rows(v: pd.DataFrame) -> pd.DataFrame:
    """For a fixture id x market logged on two match dates: the copy off the provider kick-off date
    (else the later logged copy)."""
    cols = V91_KEY + ["kickoff_utc", "league", "home_team", "away_team", "kept_match_date", "reason"]
    if v is None or v.empty or "fixture_id" not in v.columns:
        return pd.DataFrame(columns=cols)
    d = v[v["fixture_id"].astype(str).str.strip().ne("")].copy()
    d["_k"] = _keys(d, ["fixture_id", "market"])
    d = d[d.duplicated("_k", keep=False)].copy()
    if d.empty:
        return pd.DataFrame(columns=cols)
    d["_md"] = d["match_date"].astype(str).str[:10]
    d["_ko"] = pd.to_datetime(d["kickoff_utc"], errors="coerce", utc=True, format="mixed").dt.strftime("%Y-%m-%d")
    d["_lg"] = pd.to_datetime(d["logged_at"], errors="coerce", utc=True, format="mixed")
    out = []
    for _, g in d.groupby("_k"):
        on = g[g["_md"] == g["_ko"]]
        keep = on.sort_values("_lg").index[0] if len(on) else g.sort_values("_lg").index[0]
        for i, r in g.drop(index=keep).iterrows():
            out.append({**{c: r.get(c, "") for c in V91_KEY + ["kickoff_utc", "league", "home_team", "away_team"]},
                        "kept_match_date": g.loc[keep, "_md"],
                        "reason": "match_date off the provider kick-off date" if len(on)
                        else "later logged copy (no copy on the kick-off date)"})
    return pd.DataFrame(out, columns=cols)


# ── readers ───────────────────────────────────────────────────────────────────────────────────────
def apply_bets(d: pd.DataFrame, out_dir=None) -> tuple[pd.DataFrame, list[dict]]:
    """bets_ledger: post-kickoff rows excluded (`_excluded`), multi-tip rows flagged (`multi_tip`,
    `multi_tip_first_sight`, `opposite_tip_pair`) - never regraded."""
    d = d.copy()
    for c, v in (("_excluded", False), ("_excluded_reason", "")):
        if c not in d.columns:
            d[c] = v
    post = member(d, _read(POST_KO, out_dir), BETS_KEY)
    d.loc[post, "_excluded"] = True
    d.loc[post, "_excluded_reason"] = "tip generated after the provider kick-off (not a pre-match tip)"
    multi = _read(MULTI, out_dir)
    m = member(d, multi, BETS_KEY)
    d["multi_tip"] = m
    if multi is not None and len(multi):
        fs = member(d, multi[multi["first_sight"].astype(str) == "True"], BETS_KEY)
        op = member(d, multi[multi["opposite_tip_pair"].astype(str) == "True"], BETS_KEY)
    else:
        fs = op = pd.Series(False, index=d.index)
    d["multi_tip_first_sight"] = m & fs
    d["opposite_tip_pair"] = op
    return d, [{"sidecar": POST_KO, "effect": "post-kickoff tips excluded", "rows_changed": int(post.sum())},
               {"sidecar": MULTI, "effect": "multi-tip fixtures FLAGGED (owner decision open; not regraded)",
                "rows_changed": 0, "rows_flagged": int(m.sum()), "opposite_tip_rows": int(op.sum())}]


def apply_players(d: pd.DataFrame, out_dir=None) -> tuple[pd.DataFrame, list[dict]]:
    d = d.copy()
    for c, v in (("_excluded", False), ("_excluded_reason", "")):
        if c not in d.columns:
            d[c] = v
    dup = member(d, _read(PLAYER_DUP, out_dir), PLAYER_KEY + ["team"])
    d.loc[dup, "_excluded"] = True
    d.loc[dup, "_excluded_reason"] = "later copy of a logged player tip (first sight wins)"
    return d, [{"sidecar": PLAYER_DUP, "effect": "duplicate player tips excluded", "rows_changed": int(dup.sum())}]


def invalid_close_mask(d: pd.DataFrame, out_dir=None) -> pd.Series:
    """closing_odds <= 1.0, by rule (no decimal price can be 1.0) or listed in the sidecar."""
    close = pd.to_numeric(d.get("closing_odds"), errors="coerce") if d is not None and "closing_odds" in d.columns \
        else pd.Series(np.nan, index=getattr(d, "index", None))
    return (close <= 1.0).fillna(False) | member(d, _read(BAD_CLOSE, out_dir), BETS_KEY)


def read_v91_shadow(path: Path | None = None, out_dir=None, **kw) -> pd.DataFrame:
    """output/v91_shadow.csv without the listed second copies (v91_shadow_duplicates.csv)."""
    p = Path(path) if path is not None else ROOT / "output" / "v91_shadow.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p, **kw)
    return drop_v91_duplicates(d, out_dir)


def drop_v91_duplicates(d: pd.DataFrame, out_dir=None) -> pd.DataFrame:
    if d is None or d.empty:
        return d
    return d[~member(d, _read(V91_DUP, out_dir), V91_KEY).to_numpy()]


# ── build ─────────────────────────────────────────────────────────────────────────────────────────
def build(ledger_dir: Path, out_dir: Path | None = None, write: bool = True) -> dict:
    ld = Path(ledger_dir)
    rd = lambda n: pd.read_csv(ld / n, low_memory=False) if (ld / n).exists() else pd.DataFrame()  # noqa: E731
    bets, pl, v = rd("bets_ledger.csv"), rd("player_ledger.csv"), rd("v91_shadow.csv")
    from src.integrity import fixture_claims as FC
    pool = FC.provider_pool()
    out = {MULTI: multi_tip_rows(bets), POST_KO: post_kickoff_rows(bets, pool), BAD_CLOSE: invalid_close_rows(bets),
           PLAYER_DUP: player_duplicate_rows(pl, pool), V91_DUP: v91_duplicate_rows(v)}
    stats = {n: int(len(f)) for n, f in out.items()}
    m = out[MULTI]
    stats["multi_tip_fixtures"] = int(m[["match_date", "league", "home_team", "away_team"]].drop_duplicates().shape[0]) if len(m) else 0
    stats["opposite_tip_fixtures"] = int(m[m["opposite_tip_pair"]][["match_date", "league", "home_team", "away_team"]]
                                         .drop_duplicates().shape[0]) if len(m) else 0
    stats["ledger_rows"] = {"bets_ledger.csv": len(bets), "player_ledger.csv": len(pl), "v91_shadow.csv": len(v)}
    if write:
        d = _dir(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        for n, f in out.items():
            f.to_csv(d / n, index=False, lineterminator="\n")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check"])
    ap.add_argument("--ledger-dir", default=str(ROOT / "output"))
    a = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    print(json.dumps(build(Path(a.ledger_dir), write=a.cmd == "build"), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
