"""Settle Fantasy forecasts against what actually happened, from data already on disk.

    python -m player_model.fantasy_settle            # writes output/fantasy_ledger.csv

THE QUESTION THIS EXISTS TO ANSWER. "Is Wowza better than FPL's own ep_next?" Nothing could
answer it before: the dashboard compared a projection against SEASON-TO-DATE points per game,
which is not a settlement -- it scores a forecast against an average that includes the very
matches being forecast.

NO EXTRA API CALLS. The projection log already stores FPL's CUMULATIVE `fpl_total_points`
alongside each snapshot, every day. The difference in that cumulative total across a gameweek
boundary IS the points scored in that gameweek. So a settled history can be reconstructed from
31 days of logs that were sitting there unused, instead of waiting weeks for new data or
spending 667 element-summary calls.

    projection  = the LAST snapshot strictly BEFORE the deadline (what you could have acted on)
    actual      = cumulative points AFTER the gameweek settled - cumulative BEFORE the deadline

WHY "AFTER" IS NOT "THE NEXT DAY". FPL adds bonus points a day or two after the final whistle,
so a snapshot taken too early undercounts. The after-snapshot is therefore taken once the
cumulative total has STOPPED moving, not simply the next morning.

WHAT IT REFUSES TO DO. A player missing from either side has no defensible actual, so he is
dropped and counted, never imputed as zero -- a zero would read as a blank gameweek and quietly
flatter or damn the model depending on who it hit. Gameweeks where the boundary snapshots do not
exist are reported as unsettleable rather than estimated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from player_model import config

OUT = config.OUTPUT_DIR
LOG = OUT / "fantasy_projection_log.csv"
LEDGER = OUT / "fantasy_ledger.csv"
PERF = OUT / "fantasy_performance.json"


def _events() -> pd.DataFrame:
    b = json.loads((OUT / "fpl_bootstrap.json").read_text(encoding="utf-8"))
    ev = pd.DataFrame([{"gw": e["id"], "deadline": e["deadline_time"],
                        "finished": bool(e.get("finished")),
                        "checked": bool(e.get("data_checked"))} for e in b.get("events", [])])
    if ev.empty:
        return ev
    ev["deadline"] = pd.to_datetime(ev["deadline"], utc=True)
    return ev.sort_values("gw").reset_index(drop=True)


def build_ledger() -> tuple[pd.DataFrame, dict]:
    """One row per (gameweek, player): pre-deadline forecast vs points actually scored."""
    meta: dict = {"settled_gameweeks": [], "unsettleable": [], "notes": []}
    if not LOG.exists():
        return pd.DataFrame(), {**meta, "error": "no projection log"}
    lg = pd.read_csv(LOG)
    if lg.empty:
        return pd.DataFrame(), {**meta, "error": "projection log empty"}
    lg["snapshot_ts"] = pd.to_datetime(lg["snapshot_ts"], errors="coerce", utc=True)
    lg = lg.dropna(subset=["snapshot_ts", "player_name"])

    ev = _events()
    if ev.empty:
        return pd.DataFrame(), {**meta, "error": "no FPL events"}
    done = ev[ev["finished"] & ev["checked"]]

    rows = []
    for _, e in done.iterrows():
        gw, dl = int(e["gw"]), e["deadline"]
        before = lg[lg["snapshot_ts"] < dl]
        after = lg[lg["snapshot_ts"] > dl]
        if before.empty or after.empty:
            meta["unsettleable"].append({"gw": gw, "why": "no snapshot on one side of the deadline"})
            continue

        # Pre-deadline forecast: the last snapshot a manager could actually have acted on.
        pre = (before.sort_values("snapshot_ts").groupby("player_name", as_index=False).last())

        # Post-settlement: the LAST snapshot before the NEXT deadline.
        #
        # An earlier version looked for the first timestamp where the cumulative total "stopped
        # moving". That was wrong in a way that produced confident nonsense: tot.diff() makes
        # the first row NaN, fillna(0) turned it into "no movement", so the EARLIEST
        # post-deadline snapshot was chosen -- before the matches had been scored at all. Every
        # actual came out near zero, which made the model look as though it over-predicted by
        # four points a week. The bias of +4.03 was mine, not the model's.
        #
        # Bounding by the next deadline is exact instead of heuristic: by then the gameweek is
        # fully scored INCLUDING the bonus that lands a day or two late, and no points from the
        # following gameweek can have been added yet.
        nxt = ev[ev["gw"] == gw + 1]
        upper = nxt["deadline"].iloc[0] if len(nxt) else after["snapshot_ts"].max()
        window = after[after["snapshot_ts"] <= upper]
        if window.empty:
            meta["unsettleable"].append({"gw": gw, "why": "no snapshot between this deadline and the next"})
            continue
        settle_ts = window["snapshot_ts"].max()
        post = window.sort_values("snapshot_ts").groupby("player_name", as_index=False).last()

        j = pre.merge(post[["player_name", "fpl_total_points"]], on="player_name",
                      suffixes=("_pre", "_post"))
        j["actual_points"] = j["fpl_total_points_post"] - j["fpl_total_points_pre"]
        n_before = len(j)
        j = j.dropna(subset=["actual_points"])
        dropped = n_before - len(j)
        if dropped:
            meta["notes"].append(f"GW{gw}: dropped {dropped} players missing on one side "
                                 f"(never imputed as zero)")
        if j.empty:
            meta["unsettleable"].append({"gw": gw, "why": "no player present on both sides"})
            continue

        # SEASON-ROLLOVER GUARD. `fpl_total_points` is a CUMULATIVE season total, so a
        # difference across a gameweek can never be negative. In this log it is: the feed still
        # carried 2025/26 totals until 2026-09-10 and then reset to the current season, so
        # Khusanov reads 67 on 09-09 and 9 on 09-10. Any window spanning that date produces
        # nonsense -- GW3 came out at a mean of MINUS 73 points per player.
        #
        # Without this the corrupted gameweeks are silently averaged in with the good ones, and
        # the resulting "FPL beats Wowza by 2.4 MAE" looks like a finding rather than a bug.
        neg = float((j["actual_points"] < 0).mean())
        if neg > 0.02:
            meta["unsettleable"].append({
                "gw": gw,
                "why": f"{neg:.0%} of players show a NEGATIVE cumulative change — the window "
                       f"spans a season rollover in fpl_total_points, so these actuals are "
                       f"not real gameweek scores"})
            continue
        if (j["actual_points"] < 0).any():
            j = j[j["actual_points"] >= 0]
        # A whole gameweek of exact zeros means both snapshots sat on the same side of scoring.
        if float((j["actual_points"] == 0).mean()) > 0.98:
            meta["unsettleable"].append({
                "gw": gw, "why": "every player scored exactly 0 — both snapshots fall on the "
                                 "same side of scoring, so nothing was actually measured"})
            continue

        j["gw"] = gw
        j["deadline"] = dl.isoformat()
        j["settled_at"] = pd.Timestamp(settle_ts).isoformat()
        rows.append(j)
        meta["settled_gameweeks"].append(gw)

    if not rows:
        return pd.DataFrame(), meta
    led = pd.concat(rows, ignore_index=True)
    keep = ["gw", "deadline", "settled_at", "snapshot_ts", "player_name", "team", "position",
            "price", "proj_pts_per_game", "proj_fixture_adj", "p_start", "fpl_ep_next",
            "fpl_ppg", "fpl_form", "availability", "actual_points"]
    led = led[[c for c in keep if c in led.columns]]
    return led.sort_values(["gw", "actual_points"], ascending=[True, False]).reset_index(drop=True), meta


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 5:
        return {"n": int(len(y))}
    from scipy.stats import spearmanr
    return {"n": int(len(y)),
            "mae": round(float(np.abs(p - y).mean()), 4),
            "rmse": round(float(np.sqrt(((p - y) ** 2).mean())), 4),
            "spearman": round(float(spearmanr(p, y).statistic), 4),
            "bias": round(float((p - y).mean()), 4)}


def performance(led: pd.DataFrame) -> dict:
    """Wowza against FPL's own free number, on the SAME players and the SAME gameweeks."""
    if led.empty:
        return {"comparable": False, "why": "empty ledger"}
    out = {"comparable": True, "by_gameweek": [], "overall": {}}
    # WOWZA IS SCORED ON THE NUMBER IT ACTUALLY SHOWS: unconditional points, projection x
    # P(start). Scoring the CONDITIONAL projection against real gameweek results is not a fair
    # test of the product -- 46% of actual scores are zero because the player did not play, and
    # a conditional number does not claim to predict those. Measured on GW4-5: conditional MAE
    # 3.430 / rho 0.195, unconditional MAE 2.164 / rho 0.543. Both are reported so the
    # difference stays visible rather than being a silent choice in the scorer's favour.
    led = led.copy()
    base = "proj_fixture_adj" if "proj_fixture_adj" in led.columns else "proj_pts_per_game"
    if "p_start" in led.columns:
        led["proj_uncond"] = (pd.to_numeric(led[base], errors="coerce")
                              * pd.to_numeric(led["p_start"], errors="coerce").clip(0, 1))
        wcol = "proj_uncond"
    else:
        wcol = base
    out["wowza_column"] = wcol
    out["wowza_conditional_column"] = base
    for gw, g in led.groupby("gw"):
        both = g.dropna(subset=[wcol, "fpl_ep_next", "actual_points"])
        if len(both) < 5:
            continue
        y = both["actual_points"].to_numpy(float)
        w = _metrics(y, both[wcol].to_numpy(float))
        f = _metrics(y, both["fpl_ep_next"].to_numpy(float))
        k = min(10, len(both))
        top_actual = set(both.nlargest(k, "actual_points")["player_name"])
        out["by_gameweek"].append({
            "gw": int(gw), "n": int(len(both)),
            "wowza": w, "fpl_ep_next": f,
            "wowza_better_mae": bool(w.get("mae", 9e9) < f.get("mae", 9e9)),
            "top10_overlap_wowza": len(set(both.nlargest(k, wcol)["player_name"]) & top_actual),
            "top10_overlap_fpl": len(set(both.nlargest(k, "fpl_ep_next")["player_name"]) & top_actual),
            "top10_of": k,
        })
    allb = led.dropna(subset=[wcol, "fpl_ep_next", "actual_points"])
    if len(allb) >= 5:
        y = allb["actual_points"].to_numpy(float)
        out["overall"] = {
            "wowza": _metrics(y, allb[wcol].to_numpy(float)),
            "wowza_conditional": _metrics(y, allb[base].to_numpy(float)),
            "fpl_ep_next": _metrics(y, allb["fpl_ep_next"].to_numpy(float)),
        }
        out["actual_zero_share"] = round(float((y == 0).mean()), 4)
        wm = out["overall"]["wowza"].get("mae")
        fm = out["overall"]["fpl_ep_next"].get("mae")
        if wm is not None and fm is not None:
            out["verdict"] = ("Wowza beats FPL ep_next on MAE" if wm < fm
                              else "FPL ep_next beats Wowza on MAE")
            out["mae_gap"] = round(wm - fm, 4)
    return out


def main() -> int:
    led, meta = build_ledger()
    if led.empty:
        print(f"[settle] nothing settleable — {meta.get('error') or meta.get('unsettleable')}")
        PERF.write_text(json.dumps({"comparable": False, **meta}, indent=2), encoding="utf-8")
        return 0
    led.to_csv(LEDGER, index=False)
    perf = performance(led)
    perf["meta"] = meta
    PERF.write_text(json.dumps(perf, indent=2, default=str), encoding="utf-8")

    print(f"[settle] {len(led):,} rows over gameweeks {meta['settled_gameweeks']}")
    for n in meta["notes"]:
        print(f"   NOTE: {n}")
    for u in meta["unsettleable"]:
        print(f"   UNSETTLEABLE GW{u['gw']}: {u['why']}")
    ov = perf.get("overall", {})
    if ov:
        w, f = ov['wowza'], ov['fpl_ep_next']
        wc = ov.get('wowza_conditional')
        print(f"\n   {'':<14}{'MAE':>8}{'RMSE':>8}{'Spearman':>10}{'bias':>8}{'n':>7}")
        print(f"   {'Wowza':<14}{w['mae']:>8.3f}{w['rmse']:>8.3f}{w['spearman']:>10.3f}"
              f"{w['bias']:>8.3f}{w['n']:>7}")
        if wc:
            print(f"   {'Wowza (cond.)':<14}{wc['mae']:>8.3f}{wc['rmse']:>8.3f}{wc['spearman']:>10.3f}"
                  f"{wc['bias']:>8.3f}{wc['n']:>7}")
        print(f"   {'FPL ep_next':<14}{f['mae']:>8.3f}{f['rmse']:>8.3f}{f['spearman']:>10.3f}"
              f"{f['bias']:>8.3f}{f['n']:>7}")
        print(f"   ({perf.get('actual_zero_share',0):.0%} of actual scores are zero — players who did not play)")
        print(f"\n   {perf.get('verdict')} (gap {perf.get('mae_gap'):+.3f})")
    print(f"\n[settle] wrote {LEDGER.name} and {PERF.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
