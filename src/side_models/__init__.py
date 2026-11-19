"""Side-market count models (corners, cards) for the prediction ledger - LEARNING ONLY, never bets.

    src/side_models/data.py      verified match table (two sources agree per fixture)
    src/side_models/features.py  as-of team features (the SAME code for training and serving)
    src/side_models/model.py     per-league negative-binomial count models -> P(OVER line)
    src/side_models/ledger.py    every upcoming fixture of a covered league -> prediction ledger rows

Config: config/side_models.json (coverage, lines, count convention). Artifacts: models/side_models/.
"""
