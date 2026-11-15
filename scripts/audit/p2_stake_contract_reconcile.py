"""Phase 2 / P1 stake-contract reconciliation (read-only on every committed artifact).

Reads output/stake_audit.csv, output/stake_plan.csv and the two tip ledgers AS COMMITTED at a git ref
(default origin/main = production, where predict.yml commits them), validates them with the repo's own
contract (src/contracts STAKE_AUDIT_ROW / STAKE_PLAN_ROW) AND with independent row-level checks written
here, and writes a DERIVED, sidecar view - the originals are never touched:

    output/corrections/stake_audit_reconciled.csv   one row per stake_audit row that violates anything
    reports/p2_data/stake_contract_numbers.json     every number quoted in
                                                    reports/P1_STAKE_CONTRACT_RECONCILIATION.md

    python scripts/audit/p2_stake_contract_reconcile.py [--ref origin/main]

Light: four CSVs (< 6 MB together), column-projected ledgers. No network beyond `git show`.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
GIT = r"C:\Program Files\Git\cmd\git.exe"

# Regime boundaries: commit times (UTC) on origin/main of the change that defines each regime.
REGIMES = [
    ("R0_pre_side_fix", None, "before 219c1848 (2026-10-22 10:45Z): side tips lost side/odds/model_type"),
    ("R1_side_fixed", "2026-10-22 10:45:28", "219c1848: side tips reach the policy with real model_type/odds"),
    ("R2_direction_guard", "2026-10-25 12:17:16", "61462e71: one goal direction per fixture"),
    ("R3_ou25_side_split", "2026-10-26 11:48:13", "ab23726d: O/U 2.5 tracks split by side"),
]
R4_POLICY = "balanced-2026-10-13+nf-under-paper-2026-11-02+league-rating-off-2026-11-02"
SIDES_FOR = {"BTTS": ("YES", "NO"), "OU25": ("OVER", "UNDER"), "OVER15": ("OVER", "UNDER"),
             "OVER35": ("OVER", "UNDER")}


def git(*a) -> bytes:
    return subprocess.run([GIT, "-C", str(REPO), *a], capture_output=True, check=True).stdout


def show(ref, path, **kw) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(git("show", f"{ref}:{path}")), **kw)


def regime(gen: str, pv: str) -> str:
    if pv == R4_POLICY:
        return "R4_policy_2026-11-02"
    r = REGIMES[0][0]
    for name, start, _ in REGIMES[1:]:
        if gen >= start:
            r = name
    return r


def main(ref: str) -> dict:
    sha = git("rev-parse", ref).decode().strip()
    code_sha = git("rev-parse", "HEAD").decode().strip()
    sa = show(ref, "output/stake_audit.csv", dtype=str, keep_default_na=False)
    sp = show(ref, "output/stake_plan.csv", dtype=str, keep_default_na=False)
    bl = show(ref, "output/bets_ledger.csv", dtype=str, keep_default_na=False,
              usecols=["generated_at", "match_date", "home_team", "away_team", "side", "odds", "signal_tier",
                       "model_type", "result", "pnl"])
    sl = show(ref, "output/side_bets_ledger.csv", dtype=str, keep_default_na=False,
              usecols=["generated_at", "match_date", "home_team", "away_team", "market", "side", "odds",
                       "signal_tier", "model_type", "result", "pnl"])

    # 1. The repo's own contract, unchanged.
    from src.contracts import SCHEMAS, validate
    own = {"stake_audit": validate(sa, SCHEMAS["STAKE_AUDIT_ROW"]),
           "stake_plan": validate(sp, SCHEMAS["STAKE_PLAN_ROW"])}

    # 2. Independent row-level checks.
    a = sa.copy()
    a.insert(0, "audit_row", range(len(a)))                    # 0-based data row; csv line = audit_row + 2
    ff = pd.to_numeric(a["final_fraction"], errors="coerce")
    a["staked"] = ff > 0
    a["regime"] = [regime(g, p) for g, p in zip(a["generated_at"], a["policy_version"])]
    parts = a["fixture"].str.split("|", n=2, expand=True)
    a["fx_date"], a["home"], a["away"] = parts[0], parts[1], parts[2]
    allowed = a.apply(lambda r: r["side"] in SIDES_FOR.get(r["market"], ()), axis=1)
    a["v_side_invalid"] = ~allowed
    a["v_odds_blank"] = pd.to_numeric(a["odds"], errors="coerce").isna()
    a["v_stake_not_ok"] = a["staked"] & (a["reason"] != "OK")
    from src.bankroll_policy import REAL_STATES, track_key
    a["v_stake_not_real_state"] = a["staked"] & ~a["deployment_state"].isin(REAL_STATES)
    want = [track_key(m, k, s) for m, k, s in zip(a["model_type"], a["market"], a["side"])]
    a["v_track_mismatch"] = ~((a["track"] == pd.Series(want, index=a.index))
                              | (a["track"] == a["model_type"] + "|" + a["market"]))
    a["v_dup_key"] = a.duplicated(["policy_version", "fixture", "track"], keep="first")
    a["v_odds_at_or_above_hard_cap_staked"] = a["staked"] & (pd.to_numeric(a["odds"], errors="coerce") >= 3.0)

    # one position per fixture: every staked row on a fixture with > 1 staked row (the contract's rule)
    a["n_staked_on_fixture"] = a.groupby("fixture")["staked"].transform("sum").astype(int)
    a["v_one_position"] = a["staked"] & (a["n_staked_on_fixture"] > 1)
    # what each extra staked row IS: the position = (fixture, market, side)
    a["position"] = a["fixture"] + "#" + a["market"] + "#" + a["side"]
    st = a[a["staked"]].sort_values(["generated_at", "audit_row"], kind="mergesort")
    first_pos = st.drop_duplicates("fixture", keep="first").set_index("fixture")["position"]
    first_row = st.drop_duplicates("fixture", keep="first").set_index("fixture")["audit_row"]
    first_pos_row = st.drop_duplicates("position", keep="first").set_index("position")["audit_row"]
    from src.bankroll_policy import goal_direction
    a["direction"] = [goal_direction(m, s) for m, s in zip(a["market"], a["side"])]
    cls, ref_row = [], []
    for r in a.itertuples(index=False):
        if not r.v_one_position:
            cls.append(""); ref_row.append(""); continue
        fp, fr = first_pos.get(r.fixture), first_row.get(r.fixture)
        if r.audit_row == fr:
            cls.append("FIRST_STAKED_POSITION"); ref_row.append(""); continue
        pr = first_pos_row.get(r.position)
        if pr != r.audit_row:                                   # same position staked earlier
            prev = a.loc[pr]
            why = ("TRACK_RENAME_RESTATEMENT" if prev["track"] != r.track and prev["policy_version"] == r.policy_version
                   else "POLICY_VERSION_RESTATEMENT" if prev["policy_version"] != r.policy_version
                   else "SAME_KEY_RESTATEMENT")
            cls.append(why); ref_row.append(str(int(pr))); continue
        same_run = a.loc[fr, "generated_at"] == r.generated_at
        d0 = a.loc[fr, "direction"]
        d0 = d0 if isinstance(d0, str) else None
        d1 = r.direction if isinstance(r.direction, str) else None
        dirn = ("OPPOSITE_DIRECTION" if d0 and d1 and d0 != d1 else "SAME_DIRECTION" if d0 and d1
                else "NO_DIRECTION_BTTS")
        # was the first position still standing when this one was staked? Its latest decision (any track
        # / policy version of the same fixture+market+side) at or before this row's generated_at.
        hist = a[(a["position"] == a.loc[fr, "position"]) & (a["generated_at"] <= r.generated_at)]
        last = hist.sort_values(["generated_at", "audit_row"], kind="mergesort").iloc[-1]
        stands = bool(last["staked"])
        kind = ("SECOND_POSITION_SAME_RUN" if same_run else "SECOND_POSITION_LATER_RUN"
                if stands else "SECOND_POSITION_AFTER_FIRST_WITHDRAWN")
        cls.append(f"{kind}_{dirn}")
        ref_row.append(str(int(fr)))
    a["one_position_class"], a["one_position_ref_row"] = cls, ref_row

    # 3. Where each row went: current plan, ledgers (settlement), execution (none), ROI consumers.
    pk = set(zip(sp["policy_version"], sp["fixture"], sp["track"]))
    a["in_current_stake_plan"] = [(p, f, t) in pk for p, f, t in zip(a["policy_version"], a["fixture"], a["track"])]
    blk = bl.drop_duplicates(["match_date", "home_team", "away_team", "side"], keep="first").set_index(
        ["match_date", "home_team", "away_team", "side"])
    sl["MK"] = sl["market"].str.upper()
    slk_side = sl.drop_duplicates(["match_date", "home_team", "away_team", "MK", "side"], keep="first").set_index(
        ["match_date", "home_team", "away_team", "MK", "side"])
    slk_any = sl.drop_duplicates(["match_date", "home_team", "away_team", "MK"], keep="first").set_index(
        ["match_date", "home_team", "away_team", "MK"])
    sl_sides = sl.groupby(["match_date", "home_team", "away_team", "MK"])["side"].nunique().to_dict()
    led = []
    for r in a.itertuples(index=False):
        hit, how = None, ""
        if r.market == "OU25":
            k = (r.fx_date, r.home, r.away, r.side)
            if k in blk.index:
                hit, how = blk.loc[k], "bets_ledger (date, home, away, side)"
        else:
            k = (r.fx_date, r.home, r.away, r.market, r.side)
            if k in slk_side.index:
                hit, how = slk_side.loc[k], "side_bets_ledger (date, home, away, market, side)"
            elif (r.fx_date, r.home, r.away, r.market) in slk_any.index:
                k4 = (r.fx_date, r.home, r.away, r.market)
                n_sides = sl_sides.get(k4, 0)
                hit, how = slk_any.loc[k4], ("side_bets_ledger (date, home, away, market); the audit side "
                                             "is not a valid side so only the market keys the join"
                                             + ("" if n_sides == 1 else f" - AMBIGUOUS: {n_sides} ledger sides"))
        led.append({"ledger_match": how, "ledger_side": "" if hit is None else hit.get("side", ""),
                    "ledger_odds": "" if hit is None else hit["odds"],
                    "ledger_generated_at": "" if hit is None else hit["generated_at"],
                    "ledger_result": "" if hit is None else hit["result"]})
    a = pd.concat([a, pd.DataFrame(led, index=a.index)], axis=1)

    vcols = [c for c in a.columns if c.startswith("v_")]
    a["any_violation"] = a[vcols].any(axis=1)
    viol = a[a["any_violation"]].copy()

    # 4. The reconciled (DERIVED) view: original value, canonical value, reason, provenance, confidence.
    rec = []
    for r in viol.itertuples(index=False):
        base = {"audit_row": r.audit_row, "csv_line": r.audit_row + 2, "fixture": r.fixture, "league": r.league,
                "market": r.market, "track": r.track, "policy_version": r.policy_version,
                "generated_at": r.generated_at, "regime": r.regime, "deployment_state": r.deployment_state,
                "reason": r.reason, "final_fraction": r.final_fraction, "stake_amount": r.stake_amount,
                "in_current_stake_plan": r.in_current_stake_plan, "ledger_match": r.ledger_match,
                "ledger_result": r.ledger_result,
                "reached_execution": "NO - no execution consumer of stake_plan/stake_audit "
                                     "(wowza-exec origin/main has none; stakes are recommendations)"}
        if r.v_side_invalid:
            ls = r.ledger_side
            ok = ls in SIDES_FOR.get(r.market, ()) and "AMBIGUOUS" not in r.ledger_match
            only = SIDES_FOR.get(r.market, ("",))[0]
            rec.append({**base, "violation": "side_invalid", "field": "side", "original_value": r.side,
                        "canonical_value": ls if ok else "",
                        "correction_reason": "'YES/OVER' is the pre-219c1848 union label; the tip's own ledger row "
                                             "states its side" if ok else "no ledger row states the side",
                        "provenance": f"{r.ledger_match} @ {sha[:8]}" if ok else "",
                        "confidence": "HIGH (ledger row of the same tip)" if ok else "UNRESOLVED",
                        "market_only_side_would_be": only})
        if r.v_odds_blank:
            lo = pd.to_numeric(pd.Series([r.ledger_odds]), errors="coerce").iloc[0]
            ok = pd.notna(lo) and "AMBIGUOUS" not in r.ledger_match
            rec.append({**base, "violation": "odds_blank", "field": "odds", "original_value": r.odds,
                        "canonical_value": "" if not ok else f"{lo:g}",
                        "correction_reason": ("ledger price of the same tip at the ledger's first sight "
                                              f"({r.ledger_generated_at}); NOT necessarily the price the planner "
                                              "saw at " + r.generated_at) if ok else "no ledger price",
                        "provenance": f"{r.ledger_match} @ {sha[:8]}" if ok else "",
                        "confidence": "MEDIUM (same tip, ledger timestamp)" if ok else "UNRESOLVED"})
        if r.v_one_position:
            c = r.one_position_class
            canon = ("stake stands (first staked position on the fixture)" if c == "FIRST_STAKED_POSITION"
                     else f"same position as audit_row {r.one_position_ref_row}: a restatement, NOT a second position"
                     if c.endswith("RESTATEMENT")
                     else f"second position (first: audit_row {r.one_position_ref_row}); under the policy flag "
                          "one_position_per_fixture=true it would be ONE_POSITION_PER_FIXTURE (0)")
            rec.append({**base, "violation": "one_position_per_fixture", "field": "final_fraction",
                        "original_value": r.final_fraction,
                        "canonical_value": r.final_fraction if (c == "FIRST_STAKED_POSITION" or c.endswith("RESTATEMENT"))
                        else "0", "correction_reason": f"{c}: {canon}",
                        "provenance": f"stake_audit.csv @ {sha[:8]} (generated_at order); config/bankroll_policy.json",
                        "confidence": ("HIGH" if c == "FIRST_STAKED_POSITION" or c.endswith("RESTATEMENT")
                                       else "OWNER_DECISION (baseline note says same-direction positions are fine; "
                                            "policy flag says one per fixture)")})
        for v in ("v_stake_not_ok", "v_stake_not_real_state", "v_track_mismatch", "v_dup_key",
                  "v_odds_at_or_above_hard_cap_staked"):
            if getattr(r, v):
                rec.append({**base, "violation": v[2:], "field": "", "original_value": "", "canonical_value": "",
                            "correction_reason": "", "provenance": "", "confidence": "UNRESOLVED"})
    recon = pd.DataFrame(rec)

    # 5. Numbers.
    yo = a[a["v_side_invalid"]]
    op = a[a["v_one_position"]]
    nums = {
        "artifact_ref": ref, "artifact_sha": sha, "code_sha": code_sha,
        "stake_audit_rows": int(len(a)), "stake_audit_generated": [a["generated_at"].min(), a["generated_at"].max()],
        "stake_plan_rows": int(len(sp)), "stake_plan_generated": sorted(set(sp["generated_at"])),
        "repo_contract_violations": {k: [{x: v[x] for x in ("kind", "column", "n_rows")} for v in vs]
                                     for k, vs in own.items()},
        "independent": {c: int(a[c].sum()) for c in vcols},
        "rows_with_any_violation": int(len(viol)),
        "staked_rows": int(a["staked"].sum()),
        "regime_counts": a["regime"].value_counts().to_dict(),
        "side_invalid": {"values": yo["side"].value_counts().to_dict(), "markets": yo["market"].value_counts().to_dict(),
                         "staked": int(yo["staked"].sum()), "reasons": yo["reason"].value_counts().to_dict(),
                         "states": yo["deployment_state"].value_counts().to_dict(),
                         "generated": [yo["generated_at"].min(), yo["generated_at"].max()],
                         "regimes": yo["regime"].value_counts().to_dict(),
                         "odds_blank_too": int((yo["v_odds_blank"]).sum()),
                         "ledger_side": yo["ledger_side"].replace("", "(no ledger row)").value_counts().to_dict(),
                         "ledger_match": yo["ledger_match"].replace("", "(none)").value_counts().to_dict(),
                         "ledger_settled_WIN_LOSS": int(yo["ledger_result"].isin(["WIN", "LOSS"]).sum()),
                         "in_current_plan": int(yo["in_current_stake_plan"].sum())},
        "odds_blank_total": int(a["v_odds_blank"].sum()),
        "odds_blank_not_side_invalid": int((a["v_odds_blank"] & ~a["v_side_invalid"]).sum()),
        "one_position": {"rows": int(len(op)), "fixtures": int(op["fixture"].nunique()),
                         "classes": op["one_position_class"].value_counts().to_dict(),
                         "by_regime": op["regime"].value_counts().to_dict(),
                         "class_by_regime": {f"{c} / {g}": int(n) for (c, g), n in
                                             op.groupby(["one_position_class", "regime"]).size().items()},
                         "in_current_plan": int(op["in_current_stake_plan"].sum()),
                         "ledger_settled_WIN_LOSS": int(op["ledger_result"].isin(["WIN", "LOSS"]).sum()),
                         "stake_units_all_rows": round(float(pd.to_numeric(op["stake_amount"]).sum()), 2),
                         "stake_units_restatements": round(float(pd.to_numeric(
                             op[op["one_position_class"].str.endswith("RESTATEMENT")]["stake_amount"]).sum()), 2),
                         "stake_units_second_positions": round(float(pd.to_numeric(
                             op[op["one_position_class"].str.startswith("SECOND_POSITION")]["stake_amount"]).sum()), 2)},
        "current_plan": {"rows": int(len(sp)),
                         "staked": int((pd.to_numeric(sp["final_fraction"], errors="coerce") > 0).sum()),
                         "fixtures_with_2plus_staked": int(sp.assign(_s=pd.to_numeric(sp["final_fraction"],
                                                                                      errors="coerce") > 0)
                                                           .groupby("fixture")["_s"].sum().gt(1).sum()),
                         "side_invalid": int((~sp.apply(lambda r: r["side"] in SIDES_FOR.get(r["market"], ()),
                                                        axis=1)).sum()),
                         "odds_blank": int(pd.to_numeric(sp["odds"], errors="coerce").isna().sum()),
                         "staked_fixtures_already_staked_earlier_other_position": None},
    }
    # current-plan stakes on a fixture that already holds a DIFFERENT staked position in the audit
    sps = sp[pd.to_numeric(sp["final_fraction"], errors="coerce") > 0]
    earlier = a[a["staked"]]
    n = 0
    ex = []
    for r in sps.itertuples(index=False):
        e = earlier[(earlier["fixture"] == r.fixture) & (earlier["generated_at"] < r.generated_at)
                    & ~((earlier["market"] == r.market) & (earlier["side"] == r.side))]
        if len(e):
            n += 1
            ex.append(f"{r.fixture} {r.market} {r.side} vs earlier {sorted(set(e['market'] + ' ' + e['side']))}")
    nums["current_plan"]["staked_fixtures_already_staked_earlier_other_position"] = n
    nums["current_plan"]["examples"] = ex[:10]
    nums["one_position_rows_detail"] = op[["audit_row", "fixture", "market", "side", "track", "odds", "policy_version",
                                           "generated_at", "regime", "stake_amount", "one_position_class",
                                           "one_position_ref_row", "in_current_stake_plan", "ledger_result"]] \
        .to_dict("records")
    nums["side_invalid_rows_detail"] = yo[["audit_row", "fixture", "market", "side", "odds", "track", "signal_tier",
                                           "generated_at", "regime", "deployment_state", "reason", "ledger_side",
                                           "ledger_odds", "ledger_result"]].to_dict("records")

    tag = "" if ref == "origin/main" else "_" + ref.replace("/", "_")
    if not tag:                                   # the derived view is built from production artifacts only
        out_csv = REPO / "output" / "corrections" / "stake_audit_reconciled.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        recon.to_csv(out_csv, index=False)
    out_json = REPO / "reports" / "p2_data" / f"stake_contract_numbers{tag}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(nums, indent=1, default=str) + "\n", encoding="utf-8")
    return nums


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="origin/main")
    n = main(ap.parse_args().ref)
    print(json.dumps({k: v for k, v in n.items() if not k.endswith("_detail")}, indent=1, default=str))
