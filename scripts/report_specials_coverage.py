"""Coverage of the special-market sidecar (output/specials_quotes, src/specials_quotes.py), per league.

    python scripts/report_specials_coverage.py [--since YYYY-MM-DD] [--fixtures PARQUET] [--out MD]

Per covered league (the std + new-format capture leagues, config.STANDARD_FORMAT_LEAGUES |
NEW_FORMAT_LEAGUES): fixtures the captures priced (book_quotes or the sidecar, since --since), fixtures
with any special quote, fixtures with Exact Score quoted, and the bookmakers quoting Exact Score.

Registry leagues that NO existing capture reaches are listed with the calls a day covering them would
take. Nothing here makes an API call, and no call is added: the owner decides.
  * fixtures a day = that league's fixtures in the last 365 days of the local fixture list (--fixtures,
    default output/strength/competition_fixtures.parquet), divided by 365. Missing file -> "unknown".
  * one pre-kickoff snapshot per fixture = fixtures a day /odds calls (+ one shared /fixtures?date list
    call a day for all of them).
  * at the existing captures' cadence = fixtures a day x the MEASURED /odds calls per fixture of the
    existing captures (distinct capture times per fixture in book_quotes since --since), plus their
    per-league fixture-list calls.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
import config  # noqa: E402
from src import book_quotes as BQ  # noqa: E402
from src import specials_quotes as SQ  # noqa: E402

FIXTURES = config.OUTPUT_DIR / "strength" / "competition_fixtures.parquet"


def _book_quotes_since(since: str) -> pd.DataFrame:
    cols = ["snapshot_ts", "fixture_id", "league", "bookmaker_id"]
    files = [f for f in sorted(BQ.QUOTES_DIR.glob("*.csv")) if f.stem >= since] if BQ.QUOTES_DIR.exists() else []
    frames = [pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False) for f in files]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


def covered_table(since: str) -> pd.DataFrame:
    covered = sorted((set(config.STANDARD_FORMAT_LEAGUES) | set(config.NEW_FORMAT_LEAGUES))
                     & set(config.API_FOOTBALL_IDS))
    bq = _book_quotes_since(since)
    sq = SQ.load()
    if len(sq):
        sq = sq[sq["captured_at"].astype(str).str[:10] >= since]
    out = []
    for lg in covered:
        b = bq[bq["league"] == lg] if len(bq) else bq
        s = sq[sq["league"] == lg] if len(sq) else sq
        es = s[s["bet"] == SQ.BET_EXACT] if len(s) else s
        fx = set(b["fixture_id"].dropna().astype("int64")) | set(s["af_fixture_id"].dropna().astype("int64"))
        out.append({
            "league": lg,
            "track": "standard" if lg in config.STANDARD_FORMAT_LEAGUES else "new_format",
            "fixtures_captured": len(fx),
            "fixtures_with_specials": int(s["af_fixture_id"].nunique()) if len(s) else 0,
            "fixtures_with_exact_score": int(es["af_fixture_id"].nunique()) if len(es) else 0,
            "exact_score_bookmakers": int(es["bookmaker_id"].nunique()) if len(es) else 0,
            "exact_score_bookmaker_names": ", ".join(sorted(es["bookmaker"].astype(str).unique())) if len(es) else "",
            "specials_rows": int(len(s)),
        })
    return pd.DataFrame(out)


def odds_calls_per_fixture(since: str) -> float | None:
    """Measured: distinct capture times per fixture in book_quotes (one /odds call each)."""
    bq = _book_quotes_since(since)
    if bq.empty:
        return None
    return float(bq.groupby("fixture_id")["snapshot_ts"].nunique().mean())


def unreached_table(fixtures_path: Path, calls_per_fx: float | None) -> pd.DataFrame:
    from src import league_registry as LR
    reached = set(config.STANDARD_FORMAT_LEAGUES) | set(config.NEW_FORMAT_LEAGUES)
    rows = [e for e in LR.leagues() if e["name"] not in reached and e.get("api_football_id") is not None]
    fx = pd.read_parquet(fixtures_path) if fixtures_path.exists() else None
    out = []
    for e in rows:
        per_day = None
        if fx is not None:
            d = pd.to_datetime(fx["date"], errors="coerce", utc=True)
            end = d.max()
            m = (fx["league_id"] == int(e["api_football_id"])) & (d > end - timedelta(days=365)) & (d <= end)
            per_day = round(int(m.sum()) / 365.0, 2) if m.any() else None
        out.append({
            "league": e["name"], "api_football_id": int(e["api_football_id"]),
            "production_state": e.get("production_state"),
            "fixtures_per_day": per_day,
            "calls_per_day_one_snapshot": None if per_day is None else round(per_day, 1),
            "calls_per_day_capture_cadence":
                None if per_day is None or calls_per_fx is None else round(per_day * calls_per_fx, 1),
        })
    return pd.DataFrame(out)


def _md(df: pd.DataFrame) -> str:
    """A markdown table without the optional `tabulate` dependency. Missing values print as blank."""
    cols = list(df.columns)
    fmt = lambda v: "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)  # noqa: E731
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", default=None, help="first capture day counted (default: first sidecar day)")
    ap.add_argument("--fixtures", default=str(FIXTURES))
    ap.add_argument("--out", default=str(PROJ / "reports" / "SPECIALS_QUOTES_COVERAGE.md"))
    a = ap.parse_args(argv)
    sq = SQ.load()
    since = a.since or (str(sq["captured_at"].astype(str).min())[:10] if len(sq)
                        else datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    cov = covered_table(since)
    cpf = odds_calls_per_fixture(since)
    un = unreached_table(Path(a.fixtures), cpf)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Special-market quotes: coverage per league",
        "",
        f"Generated {now} by `scripts/report_specials_coverage.py` from `output/specials_quotes/` and",
        f"`output/book_quotes/`, captures since {since}. Source: the std / nf odds captures' existing",
        "/odds answers (zero extra API calls). See `src/specials_quotes.py`.",
        "",
        f"Total: {int(cov['fixtures_captured'].sum())} fixtures captured, "
        f"{int(cov['fixtures_with_exact_score'].sum())} with Exact Score quoted, "
        f"{int(cov['specials_rows'].sum())} special rows, "
        f"{int((cov['fixtures_with_exact_score'] > 0).sum())} of {len(cov)} covered leagues with Exact Score so far.",
        "",
        "## Covered leagues",
        "",
        _md(cov.drop(columns=["exact_score_bookmaker_names"])),
        "",
        "## Registry leagues no existing capture reaches",
        "",
        "These are COLLECTION_ONLY registry leagues: neither the std nor the new-format capture lists their",
        "fixtures. No call was added for them; the owner decides. Fixtures a day come from the last 365 days",
        f"of `{Path(a.fixtures).as_posix().split('/output/')[-1] if '/output/' in Path(a.fixtures).as_posix() else a.fixtures}`;",
        "calls a day = fixtures a day x 1 (one pre-kickoff snapshot each) or x "
        + (f"{cpf:.1f}" if cpf is not None else "unknown")
        + " (the measured /odds calls per fixture of the existing captures since " + since + ").",
        "Either option also needs fixture-list calls (one shared /fixtures?date call a day, or the captures'",
        "per-league list calls).",
        "",
        _md(un) if len(un) else "(none)",
        "",
    ]
    tot1 = un["calls_per_day_one_snapshot"].dropna().sum() if len(un) else 0
    totc = un["calls_per_day_capture_cadence"].dropna().sum() if len(un) else 0
    lines += [f"Totals (leagues with a known fixture rate): {tot1:.0f} calls a day for one snapshot per fixture; "
              f"{totc:.0f} calls a day at the existing capture cadence.", ""]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
