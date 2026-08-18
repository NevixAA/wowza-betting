"""The collect merge must NEVER lose a row or an enrichment value.

Replays the exact failure modes the old league-replacement merge was vulnerable to.
"""
import sys
from pathlib import Path

import pandas as pd

V9 = Path(r"c:\Users\nevo\OneDrive - Apollo\archive\נבו אישי\אישי\Wowza\mixed\v9")
sys.path.insert(0, str(V9))

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'' if cond else '  <- ' + detail}")
    if not cond:
        FAILS.append(name)


def union(existing, fresh):
    """Mirror of the merge in mode_collect."""
    key = ["fixture_id", "player_id"]
    a = existing.drop_duplicates(subset=key, keep="last").set_index(key)
    b = fresh.drop_duplicates(subset=key, keep="last").set_index(key)
    return b.combine_first(a).reset_index()


# Existing history: two leagues, plus an enrichment column a collect never produces.
existing = pd.DataFrame([
    {"fixture_id": 1, "player_id": 10, "league": "Championship", "goals": 0,
     "chronic_injury_risk": 0.4, "days_since_last_injury": 30},
    {"fixture_id": 2, "player_id": 11, "league": "Championship", "goals": 1,
     "chronic_injury_risk": 0.1, "days_since_last_injury": 90},
    {"fixture_id": 3, "player_id": 12, "league": "La Liga", "goals": 2,
     "chronic_injury_risk": 0.2, "days_since_last_injury": 15},
])

print("\n== a PARTIAL fetch of a league must not shrink it ==")
# The old merge deleted every Championship row and kept only this one.
partial = pd.DataFrame([{"fixture_id": 1, "player_id": 10, "league": "Championship",
                         "goals": 0}])
out = union(existing, partial)
check("no row is lost on a partial league fetch", len(out) == 3, f"{len(out)} rows")
check("the untouched league survives", (out.league == "La Liga").sum() == 1)
check("the other Championship row survives",
      set(out.fixture_id) == {1, 2, 3}, str(sorted(out.fixture_id)))

print("\n== enrichment columns must not be overwritten with NaN ==")
# A collect does not produce chronic_injury_risk. keep='last' would have nulled it.
check("chronic_injury_risk preserved on the re-fetched row",
      out.loc[out.fixture_id == 1, "chronic_injury_risk"].iloc[0] == 0.4,
      str(out.loc[out.fixture_id == 1, "chronic_injury_risk"].iloc[0]))
check("days_since_last_injury preserved",
      out.loc[out.fixture_id == 1, "days_since_last_injury"].iloc[0] == 30)

print("\n== new rows are added, re-fetched values refreshed ==")
fresh = pd.DataFrame([
    {"fixture_id": 1, "player_id": 10, "league": "Championship", "goals": 3},   # corrected
    {"fixture_id": 4, "player_id": 13, "league": "Championship", "goals": 1},   # new
])
out2 = union(existing, fresh)
check("new row added", len(out2) == 4, f"{len(out2)} rows")
check("re-fetched value wins", out2.loc[out2.fixture_id == 1, "goals"].iloc[0] == 3)
check("its enrichment still preserved",
      out2.loc[out2.fixture_id == 1, "chronic_injury_risk"].iloc[0] == 0.4)

print("\n== an EMPTY fetch must be a no-op, not a wipe ==")
out3 = union(existing, existing.iloc[0:0])
check("empty fetch keeps everything", len(out3) == 3, f"{len(out3)} rows")

print("\n== duplicate keys on either side are collapsed, not multiplied ==")
dup = pd.concat([existing, existing], ignore_index=True)
out4 = union(dup, fresh)
check("no row multiplication", len(out4) == 4, f"{len(out4)} rows")

print("\n== the guard: result can never be smaller than the input ==")
for label, f in [("partial", partial), ("empty", existing.iloc[0:0]), ("fresh", fresh)]:
    o = union(existing, f)
    check(f"union({label}) >= existing", len(o) >= len(existing),
          f"{len(o)} < {len(existing)}")

print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all checks passed'}")
sys.exit(1 if FAILS else 0)
