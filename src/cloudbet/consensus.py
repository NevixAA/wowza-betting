"""Two different questions about a Cloudbet price. Keeping them apart is the whole point.

    from src.cloudbet.consensus import consensus_probability, soft_price_edge

    edge_vs_model      p_model      - 1/cloudbet_odds      "our model disagrees with them"
    edge_vs_consensus  p_consensus  - 1/cloudbet_odds      "they are cheap against the market"

THESE ARE NOT THE SAME EDGE AND MUST NOT BE ADDED TOGETHER.

The first is what v9 does today: a model opinion against one book's price. This estate has
measured that the de-vigged market beats every model it has — log loss 0.67702 vs 0.68752 on
standard, 0.67899 vs 0.68714 on new-format, scored on identical rows. So a disagreement with
the market is, on the evidence here, more often the model being wrong than the market.

The second asks something the model is not involved in at all: is CLOUDBET cheap relative to
the other books? That is a market-microstructure claim, it does not depend on the model being
right, and it is the only one of the two with a mechanism behind it — a smaller book pricing
a thin second division more slowly than the market as a whole.

LEAVE-ONE-OUT IS NOT OPTIONAL. If Cloudbet is inside the consensus it is being compared with
itself, and the edge shrinks toward zero by construction — with 7 books, roughly 1/7 of any
genuine softness disappears. So `consensus_probability` takes `exclude` and the soft-price
edge always excludes the book being priced. Adding Cloudbet to the consensus for OTHER purposes
(a better market anchor for research) is fine and is what `include_cloudbet` is for; the two use
cases simply must not share one number.

WHY THE CONSENSUS IS DE-VIGGED AND THE CLOUDBET SIDE IS NOT. The consensus is an estimate of
the true probability, so its margin must come out. `1/cloudbet_odds` is what we actually pay,
margin included — that is the point. Comparing a de-vigged estimate against a raw price is
deliberate: the difference IS the cost of doing business, and a bet is only worth making when
the estimated edge exceeds it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.market_anchor import MIN_BOOKS_FOR_MEDIAN, power_devig

log = logging.getLogger(__name__)

OPPOSITE = {"over": "under", "under": "over", "yes": "no", "no": "yes"}


@dataclass(frozen=True)
class Consensus:
    probability: float | None
    n_books: int
    books: tuple
    dispersion: float | None      # stdev of the per-book de-vigged probabilities
    status: str

    @property
    def usable(self) -> bool:
        return self.probability is not None and self.n_books >= MIN_BOOKS_FOR_MEDIAN


def consensus_probability(quotes: pd.DataFrame, fixture_id, market: str, side: str,
                          line: float | None = None,
                          exclude: tuple = ()) -> Consensus:
    """Median de-vigged probability across books that priced BOTH sides.

    `exclude` drops books by name before the median — pass the book you are about to price so
    it is not compared with itself.

    A book contributes only when it quotes both sides. Ten books carrying only the Over are
    still impossible to de-vig: with one side you cannot separate the margin from the opinion,
    and `1/odds` with the vig left in is biased high every time, in the same direction.
    """
    side = str(side).lower()
    other = OPPOSITE.get(side)
    if other is None:
        return Consensus(None, 0, (), None, "NO_OPPOSITE_SIDE")

    d = quotes[(quotes["fixture_id"].astype(str) == str(fixture_id))
               & (quotes["market"].astype(str) == market)]
    if line is not None and "line" in d.columns:
        d = d[pd.to_numeric(d["line"], errors="coerce").fillna(-1) == line]
    if exclude:
        low = {str(e).lower() for e in exclude}
        d = d[~d["bookmaker"].astype(str).str.lower().isin(low)]
    if d.empty:
        return Consensus(None, 0, (), None, "NO_QUOTES")

    # The quote CLOSEST TO KICKOFF per book and side — not the first row, not the newest write.
    if "minutes_to_kickoff" in d.columns:
        d = d.assign(_m=pd.to_numeric(d["minutes_to_kickoff"], errors="coerce"))
        d = d[d["_m"] >= 0].sort_values("_m")
    a = d[d["side"].astype(str).str.lower() == side].groupby("bookmaker")["odds"].first()
    b = d[d["side"].astype(str).str.lower() == other].groupby("bookmaker")["odds"].first()

    ps, used = [], []
    for bk in sorted(set(a.index) & set(b.index)):
        fair = power_devig([float(a[bk]), float(b[bk])])
        if fair:
            ps.append(fair[0])
            used.append(bk)
    if len(ps) < MIN_BOOKS_FOR_MEDIAN:
        return Consensus(None, len(ps), tuple(used), None, "INSUFFICIENT_BOOKS")
    return Consensus(round(float(np.median(ps)), 6), len(ps), tuple(used),
                     round(float(np.std(ps, ddof=1)), 6) if len(ps) > 1 else None, "OK")


def soft_price_edge(quotes: pd.DataFrame, fixture_id, market: str, side: str,
                    cloudbet_odds: float, line: float | None = None,
                    book_name: str = "Cloudbet") -> tuple[float | None, Consensus]:
    """How much cheaper Cloudbet is than the rest of the market, in probability points.

    Positive means their implied probability is BELOW the consensus on this side — i.e. they
    are offering a bigger price than the market thinks the outcome deserves. That is value
    whether or not our model is right about the fixture.

    The book being priced is always excluded from the consensus. Comparing a book with itself
    shrinks any real softness toward zero by construction.
    """
    c = consensus_probability(quotes, fixture_id, market, side, line,
                              exclude=(book_name,))
    if not c.usable or not (cloudbet_odds and cloudbet_odds > 1):
        return None, c
    return round(c.probability - 1.0 / float(cloudbet_odds), 6), c


def both_edges(p_model: float | None, cloudbet_odds: float,
               quotes: pd.DataFrame, fixture_id, market: str, side: str,
               line: float | None = None) -> dict:
    """Both numbers side by side, never summed, each with what it rests on.

    `agree` is the interesting column: a selection where the model likes it AND Cloudbet is
    cheap against the market is a different proposition from one where only one holds. Which of
    the three cases actually performs is an empirical question this records the data to answer —
    it is NOT assumed here, and nothing in the bot filters on `agree` yet.
    """
    implied = 1.0 / float(cloudbet_odds) if cloudbet_odds and cloudbet_odds > 1 else None
    soft, c = soft_price_edge(quotes, fixture_id, market, side, cloudbet_odds, line)
    model_edge = (round(float(p_model) - implied, 6)
                  if (p_model is not None and implied is not None) else None)
    return {
        "cloudbet_implied": round(implied, 6) if implied is not None else None,
        "edge_vs_model": model_edge,
        "edge_vs_consensus": soft,
        "consensus_probability": c.probability,
        "consensus_books": c.n_books,
        "consensus_book_names": ",".join(c.books),
        "consensus_dispersion": c.dispersion,
        "consensus_status": c.status,
        "agree": (None if (model_edge is None or soft is None)
                  else bool(model_edge > 0 and soft > 0)),
    }
