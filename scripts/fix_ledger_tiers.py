"""One-time backfill: reconcile bets_ledger.csv tiers with the ACTUAL send record.

Why: append_tips used to freeze each fixture's signal_tier at first sighting (usually
VALUABLE, the lowest tier). Tips that later climbed and were SENT as SNIPER/MARKSMAN kept
the stale VALUABLE label, so the weekly summary under-counted sharp tips (the "MARKSMAN=4"
symptom). telegram_bot/notified.json is the ground truth of what was actually pinged.

Fix (conservative, evidence-based — only touches provably-wrong rows):
  * Only LIVE O/U 2.5 ledger rows (source == "live").
  * Only rows whose key IS in notified.json (i.e. it WAS sent) but whose tier is NOT already
    SNIPER/MARKSMAN. A sent O/U tip is by definition SNIPER or MARKSMAN, never VALUABLE.
  * SNIPER vs MARKSMAN decided by re-deriving from the recorded edge + league; if that does
    not clear SNIPER we fall to MARKSMAN (the honest floor — the peak edge was never stored,
    so we cannot prove SNIPER; MARKSMAN is the minimum a sent tip can be).
  * Rows that were NOT sent are left untouched (no rewriting of history we cannot verify).

Usage:  python scripts/fix_ledger_tiers.py            # dry run (report only)
        python scripts/fix_ledger_tiers.py --apply    # write corrected ledger
"""
import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
import config  # noqa: E402
from src.betting import _base_tier  # noqa: E402

LEDGER   = config.OUTPUT_DIR / "bets_ledger.csv"
NOTIFIED = Path(__file__).resolve().parents[1] / "telegram_bot" / "notified.json"
SHARP    = {"SNIPER", "MARKSMAN"}
PREFIXES = {"SIDE", "HT", "AGENT", "WC", "SHARP", "PLAYER"}


def load_sent_ou_keys() -> set:
    """Plain O/U keys are 'date|home|away|side' with no type prefix."""
    if not NOTIFIED.exists():
        return set()
    data = json.loads(NOTIFIED.read_text(encoding="utf-8-sig"))
    keys = set(data.get("keys", []) if isinstance(data, dict) else data)
    return {k for k in keys
            if len(k.split("|")) == 4 and k.split("|")[0] not in PREFIXES}


def main(apply: bool = False) -> None:
    if not LEDGER.exists():
        print(f"No ledger at {LEDGER}")
        return
    df   = pd.read_csv(LEDGER)
    sent = load_sent_ou_keys()

    changes = []
    for i, r in df.iterrows():
        if str(r.get("source", "")) != "live":
            continue
        key = f"{str(r['match_date'])[:10]}|{r['home_team']}|{r['away_team']}|{str(r['side']).strip()}"
        if key not in sent:
            continue
        cur = str(r.get("signal_tier", ""))
        if cur in SHARP:
            continue  # already correct
        try:
            edge = float(r.get("edge_pct", 0) or 0) / 100.0
        except (TypeError, ValueError):
            edge = 0.0
        rederived = _base_tier(edge, str(r.get("side", "")), str(r.get("league", "")))
        new_tier  = "SNIPER" if rederived == "SNIPER" else "MARKSMAN"
        changes.append((i, cur, new_tier))
        if apply:
            df.at[i, "signal_tier"] = new_tier

    live_n = int((df["source"] == "live").sum()) if "source" in df.columns else 0
    print(f"sent O/U keys: {len(sent)} | live ledger rows: {live_n} | corrections: {len(changes)}")
    print("  from -> to:", dict(Counter((c[1] or "(blank)", c[2]) for c in changes)))
    if apply and changes:
        df.to_csv(LEDGER, index=False)
        print(f"  WROTE {LEDGER}")
    elif not apply:
        print("  DRY RUN — pass --apply to write.")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
