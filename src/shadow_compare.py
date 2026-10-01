"""§18 — per-fixture shadow log: v9 against every v9.1 challenger. §14 — FDR control.

    from src.shadow_compare import append_shadow, fdr
    append_shadow(rows)      # one row per fixture per predict run

WHY A PER-FIXTURE LOG. Every comparison so far has been retrospective — fitted and scored on
history that already existed when the model was built. §19 requires PROSPECTIVE predictions:
fixtures that had not been played when the prediction was written. Nothing in the estate records
that today, so the second-period check in the promotion gate can never pass, and every
challenger is permanently stuck at CHALLENGER.

This is the file that unsticks it. One row per fixture per run, written before kickoff, carrying
what each model said. In November it becomes the independent forward period.

NOTHING HERE BETS. The v9.1 columns are recorded and never acted on.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

SHADOW_FILE = config.OUTPUT_DIR / "v91_shadow.csv"

#: The schema §18 asks for. Fixed so the file stays readable as challengers come and go — a new
#: model adds a column, it does not restructure the file.
COLUMNS = [
    "fixture_id", "logged_at", "match_date", "kickoff_utc", "league", "model_type",
    # what each model believed, before kickoff
    "p_v9", "p_v91", "p_pro_hgb", "p_goal_distribution", "p_market",
    # what each would have done
    "v9_side", "v91_side", "v9_tier", "v91_tier", "v9_edge", "v91_edge",
    # execution and outcome, filled at settlement
    "entry_odds", "closing_odds", "clv", "result",
    # provenance — without these a row cannot be tied to the model that wrote it
    "v9_model_sha", "v91_model_sha", "dataset_id",
]


def append_shadow(rows: list[dict] | pd.DataFrame) -> int:
    """Append fixtures to the shadow log, deduped on (fixture_id, match_date).

    APPEND-ONLY AND DEDUPED ON FIRST SIGHT. A logged opinion is evidence precisely because it
    cannot be revised once the result is known. predict runs every 15 minutes, so without the
    dedup the same fixture would be written ~100 times and the later rows would carry a model
    that had seen more of the price curve — which would quietly turn a prospective log into a
    retrospective one.
    """
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()
    if df.empty:
        return 0
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLUMNS]

    if SHADOW_FILE.exists():
        old = pd.read_csv(SHADOW_FILE)
        seen = set(zip(old["fixture_id"].astype(str),
                       old["match_date"].astype(str).str[:10]))
        keep = ~df.apply(lambda r: (str(r["fixture_id"]),
                                    str(r["match_date"])[:10]) in seen, axis=1)
        df = df[keep]
        if df.empty:
            return 0
        out = pd.concat([old, df], ignore_index=True)
    else:
        out = df
    SHADOW_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SHADOW_FILE, index=False)
    log.info(f"shadow: +{len(df)} fixtures -> {SHADOW_FILE.name} ({len(out)} total)")
    return int(len(df))


def prospective_window(since: str) -> pd.DataFrame:
    """Settled shadow rows LOGGED AFTER `since` — the only rows a forward test may use.

    Filters on `logged_at`, not `match_date`: a fixture played after registration but predicted
    before it is still retrospective for any model that existed at registration time.
    """
    if not SHADOW_FILE.exists():
        return pd.DataFrame()
    d = pd.read_csv(SHADOW_FILE)
    if d.empty:
        return d
    d["logged_at"] = pd.to_datetime(d["logged_at"], errors="coerce")
    out = d[(d["logged_at"] >= pd.Timestamp(since))
            & (d["result"].astype(str).str.upper().isin(["WIN", "LOSS"]))]
    return out.reset_index(drop=True)


# ── §14 multiple-comparison control ───────────────────────────────────────────────────────────
def fdr(pvalues, q: float = 0.10) -> pd.DataFrame:
    """Benjamini-Hochberg. Returns each hypothesis with its threshold and whether it survives.

    WHY THIS IS NOT OPTIONAL HERE. The cells available are league x market x side x odds band x
    tier x movement x model type — thousands of them. Searching that many guarantees apparent
    winners: the movement study produced 19 cells of which exactly 1 cleared at 5%, and 0.05 x 19
    = 1.0 is what chance alone produces. Without FDR that single cell reads as a discovery.

    q=0.10 means we accept that up to 10% of the cells we call real are not.
    """
    p = pd.Series(pvalues, dtype=float).dropna().sort_values()
    m = len(p)
    if m == 0:
        return pd.DataFrame(columns=["p", "rank", "bh_threshold", "survives"])
    rank = np.arange(1, m + 1)
    thresh = rank / m * q
    below = p.to_numpy() <= thresh
    # BH: find the largest rank that passes, then everything up to it survives.
    k = int(np.max(np.where(below)[0]) + 1) if below.any() else 0
    out = pd.DataFrame({"p": p.to_numpy(), "rank": rank, "bh_threshold": thresh,
                        "survives": rank <= k}, index=p.index)
    return out


def whites_reality_check(best_stat: float, boot_stats: np.ndarray) -> float:
    """p-value for "the BEST of N searched strategies beats zero", by White's Reality Check.

    The naive p-value of a single cell ignores that it was the winner of a search. This compares
    the observed best against the distribution of the best under the null, which is the correct
    null when the cell was chosen for being good.
    """
    b = np.asarray(boot_stats, dtype=float)
    if b.size == 0:
        return float("nan")
    return float((b >= best_stat).mean())
