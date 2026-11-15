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

import logging

import numpy as np
import pandas as pd

from player_model.fantasy import fpl_ids

log = logging.getLogger(__name__)

HIT_COST = 4.0
MAX_PER_CLUB = 3


def _num(s):
    # Unknown stays NaN: a missing price/points value is never read as 0.
    return pd.to_numeric(s, errors="coerce")


def _col(df: pd.DataFrame, c: str) -> pd.Series:
    return _num(df[c]) if c in df.columns else pd.Series(np.nan, index=df.index)


def _per_gw(df: pd.DataFrame) -> pd.Series:
    """Unconditional points for the NEXT gameweek (NaN where unknown - no P(start)=1 assumed)."""
    for c in ("xpts_uncond", "xpts_rot"):
        if c in df.columns:
            return _num(df[c])
    return _col(df, "fantasy_pts") * _col(df, "p_start").clip(0, 1)


def _window(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(points over each player's own upcoming fixtures, how many fixtures that is).
    An unknown fixture count stays NaN - it is not one notional fixture."""
    n = _col(df, "n_fixtures_next")
    for c in ("total_xpts_rot", "total_xpts_next"):
        if c in df.columns:
            return _num(df[c]), n
    return _per_gw(df) * n, n


def _r(v, nd):
    """round() that keeps an unknown value unknown (None) instead of raising on NaN."""
    return round(float(v), nd) if v is not None and v == v else None


def _int(v):
    return int(v) if v is not None and v == v else None


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
    # Joined on the FPL element id, never on a lower-cased name (two players can share a name,
    # and accents/short names broke the old join). A side without ids cannot be joined.
    sq["_key"], pr["_key"] = fpl_ids(sq), fpl_ids(pr)
    if sq["_key"].isna().all() or pr["_key"].isna().all():
        log.warning("transfer_options: squad or projections carry no FPL element ids - no advice")
        return {"error": "Squad and projections must both carry FPL element ids (fpl_id/element)."}

    pr["_gw"] = _per_gw(pr)
    pr["_win"], pr["_nfix"] = _window(pr)

    # Owned players, with projections attached where we have them; the rest stay NaN.
    look = pr.dropna(subset=["_key"]).drop_duplicates("_key").set_index("_key")
    for c in ("_gw", "_win", "_nfix"):
        sq[c] = sq["_key"].map(look[c])
    unknown = sq["_gw"].isna()
    if unknown.any():
        names = (", ".join(sq.loc[unknown, "player_name"].astype(str))
                 if "player_name" in sq.columns else "")
        log.info("transfer_options: %d owned player(s) without a projection: %s",
                 int(unknown.sum()), names)
        warnings.append(f"{int(unknown.sum())} of {len(sq)} owned players have no projection "
                        f"({names}) — they are left out of the sell list, not valued at 0")

    # Selling price: FPL sells at the price you bought at plus half any rise. Without the
    # manager's purchase price we cannot know it, so say so instead of quietly using list price.
    if "selling_price" in sq.columns:
        sq["_sell_at"] = _num(sq["selling_price"])
    else:
        sq["_sell_at"] = _col(sq, "price")
        assumptions.append("selling price assumed equal to current price (purchase price "
                           "unknown) — a player who has risen is worth slightly less than this")

    if bank is None:
        # Not assumed £0.0m: a bank is never negative, so a move the sale alone funds is
        # affordable; any dearer move is "affordability unknown", not "cannot afford".
        assumptions.append("bank not supplied — moves the sale alone funds are affordable; "
                           "dearer moves are marked affordability unknown")

    owned = set(sq["_key"].dropna())
    club_counts = sq["team"].value_counts().to_dict() if "team" in sq.columns else {}

    # Weakest first, on UNCONDITIONAL points -- a fit bench player beats an injured star.
    # Owned players with no projection are not sell candidates (unknown is not "weakest").
    outs = sq[sq["_gw"].notna()].sort_values("_gw").head(n_out)
    moves = []
    for _, o in outs.iterrows():
        pos = o.get("position")
        cands = pr[(pr.get("position") == pos) & (~pr["_key"].isin(owned))
                   & pr["_win"].notna() & pr["_gw"].notna()].copy()
        if "availability" in cands.columns:
            # Ruled out OR availability unknown: never a transfer target (Data Truth 2026-11-02).
            from player_model.fpl_api import availability_unknown, not_recommendable
            n_unknown = int(availability_unknown(cands["availability"]).sum())
            if n_unknown:
                log.info("transfers: %d %s candidate(s) with unknown availability not offered",
                         n_unknown, pos)
            cands = cands[~not_recommendable(cands["availability"])]
        if cands.empty:
            continue
        sell_at = float(o["_sell_at"])
        price_in = _col(cands, "price")
        # A NaN price (or unknown selling price) is NOT affordable-known: NaN comparisons are
        # False, so a player with no FPL price no longer looks free.
        if bank is not None:
            cands["_afford"] = (price_in <= sell_at + float(bank) + 1e-9).astype(object)
        else:
            cands["_afford"] = [True if (pi <= sell_at + 1e-9) else None for pi in price_in]
        cands["_afford_rank"] = cands["_afford"].map({True: 1, False: 0, None: 0})
        cands["_gain_win"] = cands["_win"] - float(o["_win"])
        cands = cands.sort_values(["_afford_rank", "_gain_win"],
                                  ascending=[False, False]).head(n_in)

        for _, c in cands.iterrows():
            club = c.get("team")
            club_ok = True
            if respect_club_limit and club and "team" in sq.columns:
                # Selling this player frees a slot at his own club.
                after = club_counts.get(club, 0) + (0 if club == o.get("team") else 1)
                club_ok = after <= MAX_PER_CLUB
            gain_gw = float(c["_gw"]) - float(o["_gw"])
            gain_win = float(c["_win"]) - float(o["_win"])
            in_price = float(_num(pd.Series([c.get("price")], dtype="object"))[0])
            price_delta = in_price - float(o["_sell_at"])
            moves.append({
                "out": o.get("player_name"), "out_team": o.get("team"),
                "out_pos": pos, "out_price": _r(o["_sell_at"], 1),
                "out_gw": _r(o["_gw"], 2), "out_window": _r(o["_win"], 2),
                "out_fixtures": _int(o["_nfix"]),
                "in": c.get("player_name"), "in_team": club,
                "in_price": _r(in_price, 1),
                "in_gw": _r(c["_gw"], 2), "in_window": _r(c["_win"], 2),
                "in_fixtures": _int(c["_nfix"]),
                "in_start_confidence": c.get("start_confidence"),
                "in_owned_pct": c.get("owned_pct"),
                "gain_next_gw": _r(gain_gw, 2),
                "gain_window": _r(gain_win, 2),
                "price_delta": _r(price_delta, 1),
                "affordable": None if c["_afford"] is None else bool(c["_afford"]),
                "club_ok": bool(club_ok),
                "legal": bool(c["_afford"] is True and club_ok),
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
                        else "affordability unknown (bank not supplied)" if m["affordable"] is None
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
