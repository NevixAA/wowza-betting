"""ONE-OFF (owner-approved 2026-10-26): make the injury columns of player_history.parquet
point-in-time, or NaN where that cannot be proven.

THE CONTAMINATION. Until 2026-10-26 `--mode enrich-sidelined-live` computed
chronic_injury_risk / days_since_last_injury / return_from_injury_flag as of the day it RAN (D)
and wrote that one triple onto every row of every active player. A row for a match on date
M < D therefore says "days since last injury = D - L" (L = last spell end before D), which is
both wrong for M and built from events after M. Every row whose stored triple is not the
neutral 0 / 365 / 0 is such a stamp.

WHY NOT JUST RE-FETCH. The raw /sidelined history lives only in the CI cache
(player_match_cache/, never committed) and re-fetching ~21k players is an API job; this repair
uses no API. It uses what IS stored: the value every committed weekly refresh wrote, read back
from git, each with its own known stamp date D.

RECONSTRUCTION (history_integrity.reconstruct_from_stamp, proven in tests against
injury_features_asof). For a stamp (D, chronic c, days v) and a row at M <= D:
    v == 365 (no spell ended before D)      -> days(M) = 365
    L = D - v < M (no end in (L, D))        -> days(M) = M - L
    L >= M                                  -> unknown
    c == 0 and (no L or L < M - 365)        -> chronic(M) = 0, else unknown
Of all stamps with D >= M the LATEST one that determines a value is used (freshest data); where
two stamps both determine days for the same row they are compared, and that agreement rate is
the empirical proof printed below.

STAMP TIMELINE. For each refresh snapshot, a player's state is the single non-default triple
on his rows (rows re-fetched by a collect carry the 0/365/0 default and are ignored), DEFAULT
when every row is 0/365/0, or MIXED. A non-default triple is a NEW stamp dated D the first time
it is seen (a later reappearance is an extend restoring an older copy, not a new stamp). A
value already present in a BOUNDARY snapshot (06-10, 08-15: daily refreshes before 08-16 have
no surviving snapshot) has an unknown D and is never used. A DEFAULT state is a 0/365 stamp only
where the refresh provably fetched the player (active in a prop league in the 60 days before D,
on a date when the crawl cap did not bite) and only if no earlier stamp shows an injury end
(that combination is a failed fetch, which the old code also wrote as 0/365/0).

RESULT PER ROW.
    stored triple non-default (contaminated): recovered values where determined, NaN otherwise
    stored 0/365/0: replaced by the reconstruction where one exists, otherwise kept (it is
        either an older 0/365 stamp, which equals the point-in-time value, or the legacy
        neutral default a collect wrote -- neither carries information from after the match)
return_from_injury_flag is derived from the recovered days (NaN when days is NaN).

Run from the v9 root:
    python scripts/oneoff_injury_pit_repair_2026_10.py            # dry run, prints the report
    python scripts/oneoff_injury_pit_repair_2026_10.py --write    # rewrites the month parts

The pre-cleanup snapshot SHAs below exist in the local object store (and in the bundle
_git_backups/wowza-betting-before-history-cleanup-20261026.bundle); see
docs/history_cleanup_2026_10/commit-map.txt for their new IDs.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from player_model import history_integrity as HI  # noqa: E402

INJ = list(HI.INJURY_COLUMNS)
COLS = HI.KEY + ["date", "league"] + INJ

# (sha, stamp date or None for a boundary snapshot). Every committed injury refresh from
# 2026-08-16 on, plus HEAD, whose non-default values are the 2026-10-24 refresh (the only
# refresh after 10-17; the collects since only reset re-fetched rows to the default).
SNAPSHOTS = [
    ("708dd601", None),            # 2026-06-10 data commit
    ("50ad79bd", None),            # 2026-08-15 10:56, after that day's refresh
    ("45e4b744", "2026-08-16"), ("24a39b6e", "2026-08-17"), ("48cceaf2", "2026-08-18"),
    ("ec2aa594", "2026-08-22"), ("63fb324a", "2026-08-29"), ("c2195f98", "2026-09-05"),
    ("416d2694", "2026-09-12"), ("9404d06c", "2026-09-19"), ("937ec2d1", "2026-09-26"),
    ("d9b2902b", "2026-10-03"), ("b881e58d", "2026-10-10"), ("b2da13a4", "2026-10-17"),
    ("HEAD", "2026-10-24"),
]
# 0/365 stamps are trusted only from snapshots on/after this date: PROP_LEAGUES reached its
# current shape on 2026-09-24, so "active in a prop league" can be recomputed from the config.
DEFAULT_STAMPS_FROM = pd.Timestamp("2026-09-26")
CRAWL_CAP = 8000


def _git_bytes(*args) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout


def load_snapshot(sha: str) -> pd.DataFrame:
    if sha == "HEAD":
        return pd.read_parquet(ROOT / "player_history.parquet", columns=COLS)
    names = _git_bytes("ls-tree", "-r", "--name-only", sha, "player_history.parquet").decode().split()
    frames = []
    for n in names:
        raw = _git_bytes("show", f"{sha}:{n}")
        t = pd.read_parquet(io.BytesIO(raw))
        frames.append(t[[c for c in COLS if c in t.columns]])
    return pd.concat(frames, ignore_index=True)


def player_states(snap: pd.DataFrame) -> pd.DataFrame:
    """One row per player: state (c, v) of his single non-default triple, or DEFAULT / MIXED."""
    s = snap[["player_id", "chronic_injury_risk", "days_since_last_injury"]].copy()
    s["c"] = s["chronic_injury_risk"].astype(float)
    s["v"] = s["days_since_last_injury"].astype(float)
    nondef = ~((s.c == 0) & (s.v == HI.NO_INJURY_DAYS)) & s.v.notna()
    nd = s[nondef].drop_duplicates(["player_id", "c", "v"])
    cnt = nd.groupby("player_id").size()
    one = nd[nd.player_id.isin(cnt[cnt == 1].index)].set_index("player_id")[["c", "v"]]
    allp = pd.Index(s.player_id.unique())
    out = pd.DataFrame(index=allp)
    out["state"] = "DEFAULT"
    out.loc[cnt[cnt > 1].index, "state"] = "MIXED"
    out.loc[one.index, "state"] = "STAMP"
    out["c"] = one["c"]
    out["v"] = one["v"]
    return out


def active_players(snap: pd.DataFrame, d: pd.Timestamp, prop_leagues: set) -> set:
    dates = pd.to_datetime(snap["date"], errors="coerce")
    m = (dates >= d - pd.Timedelta(days=60)) & (dates <= d)
    act = set(snap.loc[m, "player_id"].dropna().unique())
    prop = set(snap.loc[snap["league"].isin(prop_leagues), "player_id"].dropna().unique())
    return act & prop


def build_events(prop_leagues: set) -> tuple[pd.DataFrame, dict]:
    seen: dict = {}            # (pid, c, v) -> stamp date or None (boundary)
    events = []                # (pid, D, c, v, kind)
    diag = {}
    for sha, d in SNAPSHOTS:
        snap = load_snapshot(sha)
        if not {"chronic_injury_risk", "days_since_last_injury"}.issubset(snap.columns):
            diag[f"{sha} {d or 'boundary'}"] = {"rows": len(snap), "note": "no injury columns yet"}
            print(f"[snap] {sha}: no injury columns yet", flush=True)
            continue
        st = player_states(snap)
        info = {"rows": len(snap), "players": len(st),
                "stamp": int((st.state == "STAMP").sum()),
                "default": int((st.state == "DEFAULT").sum()),
                "mixed": int((st.state == "MIXED").sum()), "new_stamps": 0, "default_stamps": 0}
        stamped = st[st.state == "STAMP"]
        for pid, c, v in zip(stamped.index, stamped.c, stamped.v):
            k = (pid, c, v)
            if k in seen:
                continue
            seen[k] = None if d is None else pd.Timestamp(d)
            if d is not None:
                events.append((pid, pd.Timestamp(d), c, v, "stamp"))
                info["new_stamps"] += 1
        if d is not None and pd.Timestamp(d) >= DEFAULT_STAMPS_FROM:
            act = active_players(snap, pd.Timestamp(d), prop_leagues)
            info["active"] = len(act)
            if len(act) <= CRAWL_CAP:
                dp = st[(st.state == "DEFAULT") & st.index.isin(list(act))].index
                for pid in dp:
                    events.append((pid, pd.Timestamp(d), 0.0, HI.NO_INJURY_DAYS, "default"))
                info["default_stamps"] = int(len(dp))
            else:
                info["default_stamps"] = "skipped: crawl cap bit, fetched set unknown"
        diag[f"{sha} {d or 'boundary'}"] = info
        print(f"[snap] {sha} {d or 'boundary':>10}: {info}", flush=True)
        del snap
    ev = pd.DataFrame(events, columns=["player_id", "D", "c", "v", "kind"])
    # A 0/365 stamp AFTER a stamp that already showed an injury end before it is impossible
    # (ends do not disappear): it is a failed fetch written as "no injury". Drop it.
    first_end = (ev[ev.v != HI.NO_INJURY_DAYS]
                 .assign(L=lambda x: x.D - pd.to_timedelta(x.v, unit="D"))
                 .groupby("player_id")["L"].min())
    lv = ev["player_id"].map(first_end)
    bad = (ev.kind == "default") & lv.notna() & (ev.D > lv)
    diag["default_stamps_dropped_as_failed_fetch"] = int(bad.sum())
    ev = ev[~bad].reset_index(drop=True)
    return ev, diag


def reconstruct(rows: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Vectorised reconstruct_from_stamp over every (row, stamp with D >= M)."""
    r = rows[["player_id", "M"]].reset_index().rename(columns={"index": "ri"})
    j = r.merge(ev, on="player_id", how="inner")
    j = j[j.D >= j.M]
    sentinel = j.v == HI.NO_INJURY_DAYS
    L = j.D - pd.to_timedelta(j.v.where(~sentinel, 0), unit="D")
    days = np.where(sentinel, HI.NO_INJURY_DAYS,
                    np.where(L < j.M, (j.M - L).dt.days.astype(float), np.nan))
    chronic = np.where((j.c == 0) & (sentinel | (L < j.M - pd.Timedelta(days=HI.CHRONIC_LOOKBACK_DAYS))),
                       0.0, np.nan)
    j = j.assign(days=days, chronic=chronic)
    # proof: rows where >= 2 stamps determine days, do they agree?
    dd = j[j.days.notna()].groupby("ri")["days"].agg(["nunique", "size"])
    multi = dd[dd["size"] >= 2]
    agree = {"rows_with_2plus_determining_stamps": int(len(multi)),
             "rows_where_they_disagree": int((multi["nunique"] > 1).sum())}
    jd = j[j.days.notna()].sort_values("D").groupby("ri").tail(1).set_index("ri")
    jc = j[j.chronic.notna()].sort_values("D").groupby("ri").tail(1).set_index("ri")
    out = pd.DataFrame(index=rows.index)
    out["days"] = jd["days"]
    out["days_D"] = jd["D"]
    out["chronic"] = jc["chronic"]
    # Two stamps that both claim to determine the value and DISAGREE mean the sidelined data
    # was revised between them; neither is provable, so the row gets NaN, not the newer guess.
    out["disagree"] = out.index.isin(multi.index[multi["nunique"] > 1])
    out.loc[out["disagree"], "days"] = np.nan
    return out, agree, j


def scalar_spotcheck(rows, ev, out, n=3000, seed=7):
    """The vectorised result must equal history_integrity.reconstruct_from_stamp, row by row."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(rows.index.to_numpy(), size=min(n, len(rows)), replace=False)
    evg = {p: g for p, g in ev.groupby("player_id")}
    bad = 0
    for i in pick:
        p, m = rows.at[i, "player_id"], rows.at[i, "M"]
        g = evg.get(p)
        best_d, best_c = (None, None), (None, None)
        if g is not None:
            for D, c, v in zip(g.D, g.c, g.v):
                if D < m:
                    continue
                d_, c_ = HI.reconstruct_from_stamp(m, D, c, v)
                if d_ is not None and (best_d[0] is None or D >= best_d[0]):
                    best_d = (D, d_)
                if c_ is not None and (best_c[0] is None or D >= best_c[0]):
                    best_c = (D, c_)
        vd, vc = out.at[i, "days"], out.at[i, "chronic"]
        ok_d = (best_d[1] is None and pd.isna(vd)) or (best_d[1] is not None and vd == best_d[1])
        ok_c = (best_c[1] is None and pd.isna(vc)) or (best_c[1] is not None and vc == best_c[1])
        bad += int(not (ok_d and ok_c))
    return {"checked": int(len(pick)), "mismatches": int(bad)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--report", default=str(ROOT / "output" / "injury_pit_repair_2026_10.json"))
    args = ap.parse_args()

    from player_model import config
    from player_model.pipeline import _save_history, HISTORY_CACHE

    ev, diag = build_events(set(config.PROP_LEAGUES.keys()))
    full = pd.read_parquet(HISTORY_CACHE)
    n0 = len(full)
    rows = full[["player_id"] + INJ].copy()
    rows["M"] = pd.to_datetime(full["date"], errors="coerce").dt.normalize()
    out, agree, _ = reconstruct(rows, ev)
    spot = scalar_spotcheck(rows[~out["disagree"]], ev, out)

    c0 = full["chronic_injury_risk"].astype(float)
    v0 = full["days_since_last_injury"].astype(float)
    is_default = (c0 == 0) & (v0 == HI.NO_INJURY_DAYS)
    contaminated = ~is_default & v0.notna()
    has_rec = out["days"].notna() | out["chronic"].notna()

    new_days = v0.copy()
    new_chr = c0.copy()
    # contaminated: recovered or NaN
    new_days[contaminated] = out.loc[contaminated, "days"]
    new_chr[contaminated] = out.loc[contaminated, "chronic"]
    # default rows with a reconstruction: take it (NaN where only part is determined)
    upd = is_default & has_rec
    new_days[upd] = out.loc[upd, "days"]
    new_chr[upd] = out.loc[upd, "chronic"]
    new_flag = pd.Series(np.where(new_days.notna(),
                                  ((new_days > 0) & (new_days <= HI.RETURN_WINDOW_DAYS)).astype(float),
                                  np.nan), index=full.index)

    def _changed(a, b):
        return int((~((a == b) | (a.isna() & b.isna()))).sum())

    rep = {
        "rows": n0, "players": int(full["player_id"].nunique()),
        "stamp_events": int(len(ev)), "stamp_events_by_kind": ev["kind"].value_counts().to_dict(),
        "contaminated_rows": int(contaminated.sum()),
        "contaminated_days_recovered": int((contaminated & out["days"].notna()).sum()),
        "contaminated_days_nulled": int((contaminated & out["days"].isna()).sum()),
        "contaminated_chronic_recovered": int((contaminated & out["chronic"].notna()).sum()),
        "contaminated_chronic_nulled": int((contaminated & out["chronic"].isna()).sum()),
        "default_rows": int(is_default.sum()),
        "default_rows_reconstructed": int(upd.sum()),
        "default_rows_value_changed": _changed(v0[upd], new_days[upd]) ,
        "default_rows_chronic_set_nan": int((upd & out["chronic"].isna()).sum()),
        "default_rows_kept_no_stamp": int((is_default & ~has_rec).sum()),
        "changed_cells": {"days_since_last_injury": _changed(v0, new_days),
                          "chronic_injury_risk": _changed(c0, new_chr),
                          "return_from_injury_flag": _changed(full["return_from_injury_flag"].astype(float), new_flag)},
        "nan_after": {"days_since_last_injury": int(new_days.isna().sum()),
                      "chronic_injury_risk": int(new_chr.isna().sum())},
        "proof_agreement_between_independent_stamps": agree,
        "rows_nulled_because_stamps_disagree": int(out["disagree"].sum()),
        "proof_vectorised_vs_scalar": spot,
        "contaminated_by_year": {str(y): int(n) for y, n in
                                 contaminated.groupby(rows["M"].dt.year).sum().items()},
        "snapshots": diag,
    }
    print(json.dumps({k: v for k, v in rep.items() if k != "snapshots"}, indent=2, default=str))

    if not args.write:
        print("[dry-run] nothing written")
        return 0

    # Only the three injury columns are assigned, so `full`'s other columns ARE the originals
    # and comparing the re-read history against them checks what reached the disk.
    before = full
    out_frame = full.assign(days_since_last_injury=new_days.astype("float32"),
                            chronic_injury_risk=new_chr.astype("float32"),
                            return_from_injury_flag=new_flag.astype("float32"))
    st = _save_history(out_frame)
    del out_frame
    back = pd.read_parquet(HISTORY_CACHE)
    # VERIFY: same row count, same keys in the same order, every non-injury column identical.
    k = HI.KEY
    ok_rows = len(back) == n0
    ok_keys = back[k].reset_index(drop=True).equals(before[k].reset_index(drop=True))
    other = [c for c in before.columns if c not in INJ]
    diff_cols = [c for c in other if not back[c].reset_index(drop=True).equals(before[c].reset_index(drop=True))]
    rep["write"] = {"parts_written": len(st["written"]), "bytes_written": st["bytes_written"],
                    "rows_unchanged": ok_rows, "keys_unchanged_in_order": ok_keys,
                    "non_injury_columns_checked": len(other), "non_injury_columns_differing": diff_cols}
    print(json.dumps(rep["write"], indent=2))
    Path(args.report).write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(f"report -> {args.report}")
    return 0 if (ok_rows and ok_keys and not diff_cols) else 1


if __name__ == "__main__":
    sys.exit(main())
