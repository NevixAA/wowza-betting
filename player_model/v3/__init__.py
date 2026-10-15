"""Player V3 — the parts of wowzaV9-Pro's Player V3 that Fantasy uses, ported into v9 (2026-10-15).

Validated in Pro (src/studies/fantasy_v3.py, output/studies/FANTASY_V3.md): on Premier League
players who played their club's previous match, V3 projections picked a top-11 worth 42.0 real FPL
points per week against v9's 39.0 (90% CI +1.2 to +4.8), over 42 weeks. Owner-approved port.

Only what that study used is ported:
  data.py      history loader, team totals, as-of rolling helpers (unchanged from Pro)
  features.py  FORM (intermediate), FORM_SHRUNK, TEAM_ENV  (+ v9's own BASE features)
  minutes.py   the squad panel (every club fixture, absences = 0 minutes) and the expected-minutes
               HGB models: P(start), E[minutes], P(60+/75+/90)

Used ONLY by Fantasy (player_model/fantasy_v3.py -> output/fantasy_v3_inputs.csv). Prop tips,
tiers and betting are untouched.
"""
