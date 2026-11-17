"""Owner serving rules for props tips (2026-11-08). Light module: read by player_model.predict (tip
generation) and dashboard_data.props_view (the board), so both apply the same rule.

  * "No GK attacking tips": a goalkeeper is never tipped on shots on target, goals or assists
    (config.GK_BLOCKED_MARKETS); cards stay allowed.
  * "Retrain them too": goals2 / goals3 / sot3 / sot4 still serve placeholder-trained models and are
    HIDDEN until the props clean retrain covers them (config.HIDDEN_PENDING_CLEAN_RETRAIN).
  * An unknown position is never guessed: the tip is kept and labelled UNKNOWN.
"""
from __future__ import annotations

from . import config


def position_status(pos) -> str:
    """'G' / 'D' / 'M' / 'F' from the player's recorded position (its first letter), or
    config.POSITION_UNKNOWN when none is recorded. Never guessed."""
    if pos is None or (isinstance(pos, float) and pos != pos):
        return config.POSITION_UNKNOWN
    s = str(pos).strip().upper()
    if not s or s in ("NAN", "NONE", "UNKNOWN"):
        return config.POSITION_UNKNOWN
    return s[0] if s[0] in "GDMF" else config.POSITION_UNKNOWN


def serving_skip_reason(market: str, pos_status: str) -> str | None:
    """The owner rule that blocks this player-market, or None. The hidden market comes first (no
    tip from a placeholder-trained model, whoever the player is), then the goalkeeper rule."""
    if market in config.HIDDEN_PENDING_CLEAN_RETRAIN:
        return config.SKIP_HIDDEN_MARKET
    if pos_status == "G" and market in config.GK_BLOCKED_MARKETS:
        return config.SKIP_GK_ATTACKING
    return None
