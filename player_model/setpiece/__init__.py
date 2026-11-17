"""
player_model/setpiece - set-piece features from REAL data only (owner decision B, 2026-11-02).

  understat.py  parse Understat league / match payloads (shots with their situation, rosters)
  mapping.py    Understat match -> our fixture_id (date +-1 day, both clubs through
                src.team_names.resolve); Understat player -> our player_id (name + club + date,
                ambiguity refused, conflicting evidence refused)
  features.py   as-of-date set-piece taker / receiver / threat features from the mapped shots

Leagues Understat does not cover (everything but EPL, La Liga, Bundesliga, Serie A, Ligue 1) and
players it cannot map get NaN. The corner matchup (team corners for / opponent corners against)
comes from the API-Football team statistics already in player_history (team_corners) and lives in
feature_engineering.team_match_table.
"""
