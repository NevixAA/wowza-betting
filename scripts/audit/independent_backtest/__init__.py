"""AU5 independent backtest (forensic audit phase 10).

READ-ONLY, SEPARATE implementation. Nothing in this package imports v9 selection
(src/betting.py, pipeline side bets) or settlement (update_results.py, src/ledger.py) code,
nor src/team_names. Inputs are frozen committed files read from git history.
"""
