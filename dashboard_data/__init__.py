"""Read-only adapters over the three Wowza repos, for the dashboard.

    from dashboard_data import v9, pro, v11, fantasy, freshness

WHY THIS LAYER EXISTS. The dashboard grew nine pages that each parse their own CSVs inline, and
the result is a v9-only view: of nine pages, exactly one reads Pro (Bet Builder, via a sibling
checkout) and NOTHING reads v11. A human opening Wowza cannot see what Pro is validating or what
v11 is observing, because the dashboard has no way to reach them.

Every adapter here READS. Nothing in this package trains, promotes, stakes, writes a model
input, or touches a collector. Pro and v11 are read from sibling checkouts if present and
reported as absent if not -- the dashboard degrades to what it can see rather than failing.

FRESHNESS COMES FROM CONTENT, NEVER FROM FILE MTIME. `git checkout` rewrites mtime on every CI
run, so an mtime-based age reads as ~0 hours and never goes stale -- a trap this estate has
fallen into three separate times, in provenance._model_sha, fpl_api._cached_fetch and again in
the first version of the Fantasy health banner. Age here is taken from the newest RECORD inside
a file, or from a recorded timestamp field, and an age that cannot be established is reported
UNKNOWN and treated as old rather than fresh.
"""
from __future__ import annotations

from dashboard_data.core import (REPOS, Freshness, freshness_from_timestamp,
                                 newest_date, read_csv, read_json, repo_path)

__all__ = ["REPOS", "Freshness", "freshness_from_timestamp", "newest_date",
           "read_csv", "read_json", "repo_path"]
