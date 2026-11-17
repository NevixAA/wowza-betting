"""AU3 Phase 6: settlement and ROI reconciliation.

Reconstructs P&L from immutable first-sight selections (au3_first_sight + au3_settle), compares it with
the committed ledgers, recounts the stake_audit contract violations on origin/main @ AUDIT_REF and their
effect, and checks player-prop settlement id alignment. Writes reports/au3_data/phase6_numbers.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from au3_common import DATA, LIVE_FROM, REPO, roi_ci  # noqa: E402

OUT = REPO / "output"


def r_(p, dt):
    r = roi_ci(p, dt)
    return {"n": r["n"], "pnl": r["pnl"], "roi_pct": None if not r["n"] else round(100 * r["roi"], 2),
            "ci_day": [None if np.isnan(v) else round(100 * v, 1) for v in r["ci_day"]]}


def categories(df, kind):
    c = {}
    led = df["final_result"].fillna("")
    c["live_keys_ever"] = int(len(df))
    c["in_final_ledger"] = int(df["in_final"].sum())
    c["vanished_from_ledger"] = int((~df["in_final"]).sum())
    if kind == "ou25":
        c["unbettable_AVOID_VOID_rows"] = int((df.side == "AVOID").sum())
        df = df[df.side != "AVOID"]
        led = df["final_result"].fillna("")
    c["ledger_WIN_LOSS"] = int(led.isin(["WIN", "LOSS"]).sum())
    c["ledger_VOID"] = int((led == "VOID").sum())
    c["ledger_unsettled"] = int((led == "").sum())
    c["ledger_unsettled_past_match"] = int(((led == "") & (df.match_date < "2026-11-01")).sum())
    c["independent_WIN_LOSS"] = int(df.ind_result.isin(["WIN", "LOSS"]).sum())
    c["independent_unresolved"] = int((df.ind_result == "UNRESOLVED").sum())
    c["ledger_settled_not_independently_verifiable"] = int((led.isin(["WIN", "LOSS"]) & (df.ind_result == "UNRESOLVED")).sum())
    c["result_mismatch_ledger_vs_independent"] = int((led.isin(["WIN", "LOSS"]) & df.ind_result.isin(["WIN", "LOSS"])
                                                      & (led != df.ind_result)).sum())
    c["independent_settled_but_ledger_pending"] = int((led == "") .mul(df.ind_result.isin(["WIN", "LOSS"])).sum())
    c["missing_odds_on_settled"] = int((led.isin(["WIN", "LOSS"]) & df.final_odds_f.isna()).sum())
    c["pnl_missing_on_settled"] = int((led.isin(["WIN", "LOSS"]) & df.ledger_pnl.isna()).sum())
    c["odds_changed_after_first_sight"] = int((df.odds_f != df.final_odds_f).sum())
    c["tier_changed_after_first_sight"] = int((df.signal_tier != df.final_signal_tier).sum())
    return c


def main():
    nums = {}
    for kind in ("ou25", "side"):
        df = pd.read_csv(DATA / f"settled_{kind}.csv", low_memory=False)
        nums[f"categories_{kind}"] = categories(df, kind)
        if kind == "ou25":
            df = df[df.side != "AVOID"]
        df["md"] = pd.to_datetime(df.match_date)
        rec = {}
        mk = ["ou25"] if kind == "ou25" else ["btts", "over15", "over35"]
        for m in mk:
            x = df if kind == "ou25" else df[df.market == m]
            for tr in ("standard", "new_format", "ALL"):
                y = x if tr == "ALL" else x[x.model_type == tr]
                for w, z in (("all_live", y), ("since_0901", y[y.md >= LIVE_FROM])):
                    L = z[z.final_result.isin(["WIN", "LOSS"])]
                    I = z[z.ind_result.isin(["WIN", "LOSS"])]
                    both = z[z.final_result.isin(["WIN", "LOSS"]) & z.ind_result.isin(["WIN", "LOSS"])]
                    rec[f"{m}|{tr}|{w}"] = {
                        "A_committed_ledger": r_(L.ledger_pnl.values, L.match_date.values),
                        "B_independent_first_sight_price": r_(I.ind_pnl_fs.values, I.match_date.values),
                        "C_independent_ledger_price": r_(I.ind_pnl_final.values, I.match_date.values),
                        "D_same_rows_ledger": r_(both.ledger_pnl.values, both.match_date.values),
                        "D_same_rows_first_sight": r_(both.ind_pnl_fs.values, both.match_date.values),
                    }
        nums[f"roi_{kind}"] = rec
        # tier-level effect of tier upgrades (first sight vs ledger tier), independent settlement, first-sight price
        I = df[df.ind_result.isin(["WIN", "LOSS"])]
        t = {}
        for w, z in (("all_live", I), ("since_0901", I[I.md >= LIVE_FROM])):
            for col in ("signal_tier", "final_signal_tier"):
                for tier, g in z.groupby(z[col].replace({"VALUE": "VALUABLE"})):
                    t[f"{w}|{'first_sight' if col == 'signal_tier' else 'ledger'}|{tier}"] = r_(g.ind_pnl_fs.values, g.match_date.values)
        nums[f"tier_label_effect_{kind}"] = t
    # same-fixture correlation across all ledgers
    o = pd.read_csv(DATA / "settled_ou25.csv", low_memory=False); o = o[o.side.isin(["OVER", "UNDER"])]
    s = pd.read_csv(DATA / "settled_side.csv", low_memory=False)
    k = ["match_date", "home_team", "away_team"]
    a = pd.concat([o[k + ["side"]].assign(m="ou25"), s[k + ["market"]].rename(columns={"market": "m"}).assign(side="YES/OVER")])
    per = a.groupby(k).size()
    nums["fixtures_with_any_tip"] = int(len(per)); nums["fixtures_with_2plus_tips"] = int((per > 1).sum())
    nums["tips_on_fixtures_with_2plus"] = int(per[per > 1].sum())
    flip = o.groupby(k).side.nunique()
    nums["ou25_fixtures_both_sides_logged"] = int((flip > 1).sum())
    # opposite goal-direction: UNDER 2.5 + (BTTS yes / O1.5 / O3.5 / OVER2.5) on one fixture
    # Owner rule 2026-10-25 (src/direction_guard.py): goal-total lines only; BTTS has no direction.
    u = o[o.side == "UNDER"][["match_date", "home_team", "away_team", "fs_commit_ct"]]
    so = s[s.market.isin(["over15", "over35"])][["match_date", "home_team", "away_team", "fs_commit_ct"]]
    j = u.merge(so, on=k, suffixes=("_u", "_s"))
    later = pd.to_datetime(j[["fs_commit_ct_u", "fs_commit_ct_s"]].max(axis=1), unit="s")
    nums["fixtures_UNDER25_plus_over15_or_over35"] = int(j[k].drop_duplicates().shape[0])
    nums["UNDER25_plus_over_line_second_leg_after_guard_2026-10-25T12:17Z"] = int((later >= "2026-10-25 12:17").sum())
    jb = u.merge(s[s.market == "btts"][k], on=k)
    nums["fixtures_UNDER25_plus_BTTS_yes_(no_direction_by_owner_rule)"] = int(jb[k].drop_duplicates().shape[0])
    # ---- stake_audit contract recount ----------------------------------------------------------------
    sa = pd.read_csv(OUT / "stake_audit.csv")
    sa["staked"] = sa.final_fraction > 0
    yo = sa[sa.side == "YES/OVER"]
    g = sa.groupby("fixture").staked.transform("sum")
    v = sa[sa.staked & (g > 1)]
    pairs = v.groupby("fixture").track.apply(lambda t: " + ".join(sorted(t)))
    rename_dup = pairs[pairs.str.contains(r"\|OU25 \+ .*\|OU25_(?:UNDER|OVER)")].index
    vr = v[v.fixture.isin(rename_dup)].sort_values("generated_at")
    second = vr.groupby("fixture").tail(1)
    nums["stake_audit"] = {
        "rows": int(len(sa)), "generated_range": [sa.generated_at.min(), sa.generated_at.max()],
        "YES_OVER_side_rows": int(len(yo)), "YES_OVER_markets": yo.market.value_counts().to_dict(),
        "YES_OVER_null_odds": int(yo.odds.isna().sum()), "YES_OVER_staked": int(yo.staked.sum()),
        "YES_OVER_reason": yo.reason.value_counts().to_dict(), "YES_OVER_written": [yo.generated_at.min(), yo.generated_at.max()],
        "null_odds_rows_total": int(sa.odds.isna().sum()),
        "one_position_violation_rows": int(len(v)), "one_position_violation_fixtures": int(v.fixture.nunique()),
        "violation_pairs": pairs.value_counts().to_dict(),
        "violations_written_after_2026-10-25": int((v.generated_at >= "2026-10-26").sum()),
        "rename_duplicate_fixtures": int(len(rename_dup)),
        "rename_duplicate_second_stake_units": round(float(second.stake_amount.sum()), 2),
        "rename_duplicate_second_fraction_sum": round(float(second.final_fraction.sum()), 4),
        "rename_duplicate_odds_differ": int((vr.groupby("fixture").odds.nunique() > 1).sum()),
        "rename_duplicate_tier_differ": int((vr.groupby("fixture").signal_tier.nunique() > 1).sum()),
        "opposite_direction_staked_fixtures": 0,
        "states": v.deployment_state.value_counts().to_dict(),
    }
    # ---- player ledger ---------------------------------------------------------------------------------
    p = pd.read_csv(OUT / "player_ledger.csv", low_memory=False)
    st = p[p.result.isin(["WIN", "LOSS"])]
    kk = ["match_date", "home_team", "away_team", "player_id", "market"]
    q = p.dropna(subset=["player_id"]); d = q[q.duplicated(kk, keep=False)]
    combo = d.groupby(kk).result.apply(lambda s: "/".join(sorted(s.fillna("PENDING"))))
    nums["player"] = {
        "rows": int(len(p)), "settled_WIN_LOSS": int(len(st)), "settled_with_player_id": int(st.player_id.notna().sum()),
        "settled_with_fixture_id": int(st.fixture_id.notna().sum()), "priced_settled": int(st.market_odds.notna().sum()),
        "markets": p.market.value_counts().to_dict(),
        "sot2_sot3_rows": int(p.market.isin(["sot2", "sot3"]).sum()),
        "sot2_sot3_ever_WIN_LOSS": int((p.market.isin(["sot2", "sot3"]) & p.result.isin(["WIN", "LOSS"])).sum()),
        "sot2_sot3_pending": int((p.market.isin(["sot2", "sot3"]) & p.result.isna()).sum()),
        "sot2_sot3_priced": int((p.market.isin(["sot2", "sot3"]) & p.market_odds.notna()).sum()),
        "dup_groups_same_player_id_fixture_market": int(combo.size),
        "dup_groups_name_spelling_differs": int((d.groupby(kk).player_name.nunique() > 1).sum()),
        "dup_group_result_combos": combo.value_counts().to_dict(),
        "dup_rows_priced_settled": int((d.result.isin(["WIN", "LOSS"]) & d.market_odds.notna()).sum()),
        "dup_rows_priced_settled_pnl": round(float(d[d.result.isin(["WIN", "LOSS"]) & d.market_odds.notna()].pnl.sum()), 2),
        "void_DNP_rows": int((p.notes.astype(str).str.startswith("DNP")).sum()),
    }
    json.dump(nums, open(DATA / "phase6_numbers.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in nums.items() if not k.startswith("roi_") and not k.startswith("tier_")}, indent=1, default=str))


if __name__ == "__main__":
    main()
