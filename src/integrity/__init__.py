"""Model-integrity helpers (R2-C, 2026-11-01).

    joins            safe_left_enrich: fixture-level left enrichment that can never expand, shift or
                     guess; every call leaves a JoinReport in a process-wide registry
    parity           train/serve feature parity (mean/std, missing, constant-live, PSI, rule)
    price_provenance synthetic-price / market-mapping labels and the price contract checks
    side_edge_devig  SHADOW side-market edge on two-sided de-vigged prices beside production's 1.08
    oddsapi_quota    OddsAPI credit accounting from run logs
    model_health     output/model_integrity_health.json aggregator

Nothing in this package changes a tip, a tier, a threshold or a stake.
"""
