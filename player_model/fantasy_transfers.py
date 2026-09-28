"""Transfers as a decision: what you gain, what it costs, and whether you can afford it.

    from player_model.fantasy_transfers import transfer_options
    opts = transfer_options(squad_df, proj_df, bank=1.4, free_transfers=1)

WHAT WAS THERE BEFORE. transfer_suggestions() found the weakest owned players by CONDITIONAL
points and offered the highest-CONDITIONAL-points replacements in the same position. It checked
no budget, so it would cheerfully suggest a £14m striker to a manager with £0.2m in the bank; it
had no horizon, so a one-week fixture blip looked the same as a lasting upgrade; and it never
mentioned the 4-point hit, which is the entire question when you are past your free transfer.

WHAT A TRANSFER ACTUALLY COSTS. A transfer beyond the free one costs 4 points. So the only
number that decides anything is:

    net = (points the incoming player adds over the horizon) - 4 x (hits taken)

A move worth +1.2 next week and +5.8 over three is a GOOD transfer on a hit and a BAD one if you
were only ever looking at next week. Both numbers are reported, because they answer different
questions.

AFFORDABILITY IS PART OF THE ANSWER, NOT A FOOTNOTE. FPL sells at the player's selling price and
buys at the current price; a suggestion you cannot fund is not a suggestion. Where bank or
selling price cannot be established, this says so rather than assuming unlimited money -- the
brief is explicit on that point and it is the difference between advice and noise.

DOUBLE AND BLANK GAMEWEEKS ARE THE POINT, NOT AN EDGE CASE. The horizon gain uses each player's
OWN upcoming fixture count, so a player with two fixtures shows twice the window, and a player
with none shows zero. Averaging that away would hide exactly the situation transfers exist for,
so the fixture counts are reported alongside the gain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HIT_COST = 4.0
MAX_PER_CLUB = 3


def _num(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default)


def _per_gw(df: pd.DataFrame) -> pd.Series:
    """Unconditional points for the NEXT gameweek."""
    for c in ("xpts_uncond", "xpts_rot"):
        if c in df.columns:
            return _num(df[c])
    return _num(df.get("fantasy_pts", 0)) * _num(df.get("p_start", 1)).clip(0, 1)


def _window(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(points over each player's own upcoming fixtures, how many fixtures that is)."""
    n = _num(df.get("n_fixtures_next", 1), 1.0)
    for c in ("total_xpts_rot", "total_xpts_next"):
        if c in df.columns:
            return _num(df[c]), n
    return _per_gw(df) * n, n


def transfer_options(squad: pd.DataFrame, proj: pd.DataFrame, *,
                     bank: float | None = None,
                     free_transfers: int = 1,
                     n_out: int = 5, n_in: int = 3,
                     respect_club_limit: bool = True) -> dict:
    """Rank sell->buy moves by what they actually gain, net of any hit.

    `squad` needs player_name, position, price (and ideally team, selling_price).
    Returns {moves:[...], bank, free_transfers, assumptions:[...], warnings:[...]}.
    """
    warnings: list[str] = []
    assumptions: list[str] = []
    if squad is None or squad.empty:
        return {"error": "No squad supplied."}
    if proj is None or proj.empty:
        return {"error": "No projections available."}

    sq, pr = squad.copy(), proj.copy()
    for d in (sq, pr):
        d["_key"] = d["player_name"].astype(str).str.lower().str.strip()

    pr["_gw"] = _per_gw(pr)
    pr["_win"], pr["_nfix"] = _window(pr)

    # Owned players, with projections attached where we have them.
    look = pr.set_index("_key")
    sq["_gw"] = [float(look["_gw"].get(k, 0.0)) for k in sq["_key"]]
    sq["_win"] = [float(look["_win"].get(k, 0.0)) for k in sq["_key"]]
    sq["_nfix"] = [float(look["_nfix"].get(k, 0.0)) for k in sq["_key"]]
    unknown = int(sum(1 for k in sq["_key"] if k not in look.index))
    if unknown:
        warnings.append(f"{unknown} of {len(sq)} owned players have no projection — they are "
                        f"treated as 0 points, which will make them look like obvious sells")

    # Selling price: FPL sells at the price you bought at plus half any rise. Without the
    # manager's purchase price we cannot know it, so say so instead of quietly using list price.
    if "selling_price" in sq.columns:
        sq["_sell_at"] = _num(sq["selling_price"])
    else:
        sq["_sell_at"] = _num(sq.get("price", 0))
        assumptions.append("selling price assumed equal to current price (purchase price "
                           "unknown) — a player who has risen is worth slightly less than this")

    if bank is None:
        bank = 0.0
        assumptions.append("bank assumed £0.0m (not supplied) — only moves that free up enough "
                           "money by themselves are marked affordable")

    owned = set(sq["_key"])
    club_counts = sq["team"].value_counts().to_dict() if "team" in sq.columns else {}

    # Weakest first, on UNCONDITIONAL points -- a fit bench player beats an injured star.
    outs = sq.sort_values("_gw").head(n_out)
    moves = []
    for _, o in outs.iterrows():
        pos = o.get("position")
        cands = pr[(pr.get("position") == pos) & (~pr["_key"].isin(owned))].copy()
        if "availability" in cands.columns:
            cands = cands[~cands["availability"].astype(str).str.lower()
                          .isin(("injured", "unavailable", "suspended"))]
        if cands.empty:
            continue
        budget_for = float(o["_sell_at"]) + float(bank)
        cands["_afford"] = _num(cands.get("price", 0)) <= budget_for + 1e-9
        cands["_gain_win"] = cands["_win"] - float(o["_win"])
        cands = cands.sort_values(["_afford", "_gain_win"], ascending=[False, False]).head(n_in)

        for _, c in cands.iterrows():
            club = c.get("team")
            club_ok = True
            if respect_club_limit and club and "team" in sq.columns:
                # Selling this player frees a slot at his own club.
                after = club_counts.get(club, 0) + (0 if club == o.get("team") else 1)
                club_ok = after <= MAX_PER_CLUB
            gain_gw = float(c["_gw"]) - float(o["_gw"])
            gain_win = float(c["_win"]) - float(o["_win"])
            price_delta = float(_num(pd.Series([c.get("price", 0)]))[0]) - float(o["_sell_at"])
            moves.append({
                "out": o.get("player_name"), "out_team": o.get("team"),
                "out_pos": pos, "out_price": round(float(o["_sell_at"]), 1),
                "out_gw": round(float(o["_gw"]), 2), "out_window": round(float(o["_win"]), 2),
                "out_fixtures": int(o["_nfix"]),
                "in": c.get("player_name"), "in_team": club,
                "in_price": round(float(_num(pd.Series([c.get("price", 0)]))[0]), 1),
                "in_gw": round(float(c["_gw"]), 2), "in_window": round(float(c["_win"]), 2),
                "in_fixtures": int(c["_nfix"]),
                "in_start_confidence": c.get("start_confidence"),
                "in_owned_pct": c.get("owned_pct"),
                "gain_next_gw": round(gain_gw, 2),
                "gain_window": round(gain_win, 2),
                "price_delta": round(price_delta, 1),
                "affordable": bool(c["_afford"]),
                "club_ok": bool(club_ok),
                "legal": bool(c["_afford"] and club_ok),
            })

    # HIT MATHS, ON ALTERNATIVES RATHER THAN A SEQUENCE.
    #
    # These moves are mutually exclusive options, not a plan: the same incoming player is often
    # the best answer for two different sells, and the same player cannot be sold twice. An
    # earlier version charged an escalating 4, 8, 12 as if you would take rows 1, 2 and 3 in
    # order, which reads as a coherent strategy and is not one -- row 3 frequently buys a player
    # row 1 already bought.
    #
    # So each move is priced on its own terms: what it gains with a free transfer, and what it
    # gains if it costs you a hit. Choosing a COMBINATION of transfers is a genuinely different
    # optimisation, and pretending this list is one would be worse than not offering it.
    moves.sort(key=lambda m: m["gain_window"], reverse=True)
    for m in moves:
        m["net_if_free"] = m["gain_window"]
        m["net_if_hit"] = round(m["gain_window"] - HIT_COST, 2)
        m["worth_a_hit"] = bool(m["gain_window"] > HIT_COST)
        m["verdict"] = ("use your free transfer" if free_transfers >= 1 and m["legal"]
                        else "worth -4" if m["worth_a_hit"] and m["legal"]
                        else "not worth a hit" if m["legal"]
                        else "cannot afford" if not m["affordable"]
                        else "breaks the 3-per-club limit")

    return {"moves": moves, "bank": bank, "free_transfers": free_transfers,
            "assumptions": assumptions, "warnings": warnings,
            "horizon_note": "Window = each player's OWN upcoming fixtures, so a double "
                            "gameweek counts twice and a blank counts zero.",
            "exclusivity_note": "These are ALTERNATIVES, not a sequence. The same incoming "
                                "player often answers two different sells, and nobody can be "
                                "sold twice — so each row is priced on its own, never as "
                                "'transfer number three'."}
