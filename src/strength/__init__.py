"""
src/strength - club and league strength MEASURED from the results we hold (owner decision A,
2026-11-02: "measure our own strength").

  results.py  every real result we hold, one row per match, each club keyed by ONE identity
              (API-Football team id when a collected fixture names it, else '<scope>|<name>')
  elo.py      a time-ordered Elo rating on the log-odds scale; K and home advantage are FITTED on
              held-out prediction log loss, never typed
  build.py    CLI: fit + run + write output/strength/{club_ratings,league_strength}.parquet and
              elo_fit.json
  lookup.py   as-of-date reads for feature code (a match only sees ratings from EARLIER days)

No hand tiers and no prior strengths: a club with fewer than MIN_CLUB_MATCHES rated matches has a
NaN rating, and a league is NaN until enough of its clubs are rated AND its country is linked to the
reference component by real cross-league matches (UEFA / continental club cups).
"""
