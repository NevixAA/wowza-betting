# All-data test (owner 2026-11-15): team info + league level + logit calibration + bet-time market

Evidence only. Nothing was promoted, installed or refitted. Branch `feat/team-info-features`.
Protocol: config/phase2_info.json = the Nov 9/10 holdout (2026-04-01..2026-11-08), folds, PG-1.0
thresholds and Holm, on O/U 1.5 / 2.5 / 3.5 and BTTS. One Holm family per run.

| Run | Where | Code | Frame hash | Candidates |
|---|---|---|---|---|
| info_standard_gh_run_20261115T111318Z | GitHub 43706983618 | 378c59148 | f75ea2d98e98ad22 | champion family (6) |
| info_new_format_run_20261115T154301Z | local, 1 thread | 411dda3c0 | f75ea2d98e98ad22 | all 11 |
| info_standard_run_20261115T190707Z | local, 1 thread | d7ded2ef5 | 974e04bc77c4d438 | all 11 |

The bet-time prices (OddsAPI tier BT, role BET, 125 h Standard / 153 h New-Format before kickoff) are
PRIVATE (wowza-exec odds-history; no redistribution): only counts and scores are here, no price.

Files: team_info_coverage.csv / _summary.json (scripts/team_info_coverage.py), parity metas
(scripts/team_info_parity.py: 65 dates, 1,311 rows, 600 tip times, 0 mismatched cells;
scripts/team_train_serve_parity.py: 6 dates, 125 fixtures, 72 raw columns incl. 22 info and 7
league-level, 0 mismatched cells, 0.0 pp probability difference). Each run folder: comparison.csv
(verdicts), metrics.json, per_league.csv, league_levels.csv (league mean predicted vs observed:
level_slope, mean_abs_gap, spreads), coverage.csv, manifest.json.
