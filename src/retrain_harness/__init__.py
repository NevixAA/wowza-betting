"""Phase 2 part 4 - the CLEAN RETRAIN HARNESS for the team models (BUILD ONLY).

    python scripts/phase2_retrain.py --dry-run          # plan, row counts, feature list; no fitting
    python scripts/phase2_retrain.py --out DIR          # GitHub Actions only (phase2_retrain.yml)

Pre-registered protocol: config/phase2_retrain.json. Plan and owner decisions:
reports/P0_NEW_FORMAT_CLEAN_RETRAIN_PLAN.md.

    protocol    config, guards, eligible rows, feature list, walk-forward folds, final holdout
    candidates  champion recipe, logistic, NaN-native HGB, Dixon-Coles, market-only, model+market
    evaluate    log loss, Brier, calibration, CLV, the pre-registered paired comparison (Holm)
    runner      walk-forward + holdout stages, everything written to one run folder

Owner rules this package enforces in code: no fitting unless the holdout is approved in the config
AND PHASE2_RETRAIN_ENABLED is 'true'; nothing is written outside the run folder; no serving model,
registry or promotion file is touched; no value is invented for a missing input.
"""
