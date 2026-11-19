"""ONE fixture-level coherence gate for every tip sender: never send two tips on one match that work against
each other (owner 2026-11-19, "mismatch bets on the same match").

    from src.tip_coherence import CoherenceGate, Candidate, parse_event, fixture_id
    gate = CoherenceGate.load()                  # every tip already SENT, from every notified store
    losers = gate.plan(cands)                    # this run's candidates: the higher-ranked one stands
    why = gate.check(cand)                       # "" = may go out; else "MISMATCH with <tip> (...)"
    gate.commit(cand)                            # after a successful send

THE RULE (principled, no tuned number). Two tips A and B on the same match are a MISMATCH when they are
negatively dependent under the model's own score distribution:

    P(A and B) < P(A) x P(B)

i.e. knowing one of them wins makes the other LESS likely. Backing both is then a partial hedge of yourself:
BTTS Yes + Under 2.5 both win only on 1-1; Over 3.5 + BTTS No only on 4-0 / 5-0 ...; Over 2.5 + Under 3.5 only
on exactly 3 goals. Positively dependent pairs (Over 2.5 + BTTS Yes, Under 2.5 + BTTS No, Over 1.5 + Over 2.5)
pull the same way and are allowed - the owner's own rule ("several bets on one match are fine ... if its not
the opposites", 2026-11-04). The only number in the comparison is a floating-point tolerance (NUM_TOL), not a
betting threshold.

THE SCORE DISTRIBUTION. v9's own probabilities, not a second forecaster:
  * FT pairs: a Dixon-Coles grid (independent Poissons x the low-score tau) whose two rates and rho are FITTED
    to the fixture's own p_over15 / p_over25 / p_over35 / p_btts on the board (output/predictions.csv) - the
    approach of wowzaV9-Pro src/combo/score_model.py, re-implemented here so v9 computes it locally.
  * pairs with an HT leg: each team's first-half goals ~ Poisson(ht_lambda_home / ht_lambda_away, the HT
    model's own rates on the board) and second-half goals ~ Poisson(FT rate - HT rate), independent; FT = HT +
    second half. The joint over (HT score, FT score) is exact on that grid.
  * Only symmetric markets (goal totals, BTTS) are read off a grid: a fit to totals + BTTS cannot tell home
    from away (Pro's identified_markets), so nothing side-specific is ever decided by it.

FALLBACK - THE HARD LIST, when the grid cannot be computed (no board row, too few probabilities, a player
leg): a tip has a goal DIRECTION (Over any line / BTTS Yes / HT Over / a player's goal involvement = MORE
goals; Under any line / BTTS No / HT Under = FEWER goals) and two tips of opposite direction are a mismatch.
On an independent-Poisson grid that list IS the rule above: an Over-type event is increasing in the goal
counts and an Under-type event decreasing, so their covariance is negative (Harris / FKG inequality) and two
same-direction events are positively dependent. It is also the owner's direction rule (2026-10-25 / 11-04,
src/direction_guard.py: "over 2.5 and under 3.5 ... under 2.5 and over 1.5 ... under 1.5 and btts Y"). A tip
with no goal direction (corners, cards, 1X2, a market we cannot parse) is never called a mismatch by the list.

RESOLUTION. The FIRST tip already sent stands (the existing principle: src/direction_guard, the notifier's
both-sides skip). Among this run's candidates the higher-ranked one stands (TOP-3 rank, else calibrated EV);
the other is skipped with the logged reason "MISMATCH with <tip>". Two mismatched tips are never both sent.

WHAT COUNTS AS SENT. Every Telegram dedup store that records a tip: telegram_bot/notified.json (instant O/U 2.5,
side-market, HT and album-marked tips) and the TOP-3 picks of every league-week in output/top3_league_notified.
json. A tip that was only ledgered (never sent) does not block anything here; the pipeline's direction guard
already keeps the ledgers one-directional.

Read-only: no ledger, tip, tier, threshold or stake is written. Prices are never printed.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
NOTIFIED_FILE = ROOT / "telegram_bot" / "notified.json"
TOP3_PICKS = "top3_league_ledger.csv"
TOP3_NOTIFIED = "top3_league_notified.json"

#: floating-point tolerance of P(A and B) - P(A)P(B); a grid sum is exact to ~1e-15, so this is not a bar
NUM_TOL = 1e-9
MAX_FT = 10                      # FT grid 0..10 goals a side (P(>10) renormalised away)
MAX_HALF = 7                     # per-half grid 0..7 goals a side
RHO_GRID = (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10)
GRID_KINDS = ("TOTAL", "BTTS")               # symmetric markets a totals+BTTS fit identifies
GOAL_KINDS = ("TOTAL", "BTTS", "PLAYER")     # kinds with a goal direction (the hard list)


# ── events ──────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Event:
    """One bet outcome. period FT / HT; kind TOTAL (line, OVER/UNDER), BTTS (YES/NO) or PLAYER (a player's
    goal involvement - scorer / assist / shots on target: no line, never on a grid)."""
    period: str
    kind: str
    side: str
    line: float | None = None

    @property
    def direction(self) -> int:
        """+1 = needs more goals, -1 = needs fewer goals, 0 = no known goal relation (kind OTHER)."""
        if self.kind not in GOAL_KINDS:
            return 0
        return 1 if self.side in ("OVER", "YES") else -1

    @property
    def label(self) -> str:
        p = "HT " if self.period == "HT" else ""
        if self.kind == "TOTAL":
            return f"{p}{self.side.title()} {self.line:g}"
        if self.kind == "BTTS":
            return f"{p}BTTS {self.side.title()}"
        return "player goal involvement"


_NUM = re.compile(r"(\d+(?:[.,]\d+)?)")


def _line_from(s: str) -> float | None:
    """'25' -> 2.5, '05' -> 0.5, '1.5' -> 1.5, '35' -> 3.5."""
    m = _NUM.search(s)
    if not m:
        return None
    t = m.group(1).replace(",", ".")
    if "." in t:
        return float(t)
    return float(t) / 10.0 if len(t) == 2 else float(t) + 0.5 if len(t) == 1 and t != "0" else None


def parse_event(market, side=None) -> Event | None:
    """Every spelling the senders use -> an Event, or None (unknown / not a goal market).
    ou25 + OVER, OU25_UNDER, over15 (+ YES/NO), under35, btts + YES/NO, BTTS (panel), ht_over05, HTOVER0.5,
    'HT Over 0.5', 'Over 3.5 goals', 'Under 2.5', 'BTTS Yes', 'Both teams to score - No', '<player> to score'."""
    m = str(market or "").strip()
    s = str(side or "").strip().upper()
    if s in ("NAN", "NONE"):
        s = ""
    low = m.lower()
    if not low and s not in ("OVER", "UNDER"):
        return None
    # free text (Pro's Bet Builder legs, display labels)
    if " " in low or "to score" in low or "to assist" in low:
        period = "HT" if re.search(r"\b(ht|1st half|first half|half[- ]time)\b", low) else "FT"
        if "btts" in low or "both teams" in low:
            if re.search(r"\b(no)\b", low):
                return Event(period, "BTTS", "NO")
            return Event(period, "BTTS", "YES")
        if re.search(r"\bto (score|assist)\b|\bshots? on target\b|\banytime\b|\bscorer\b", low):
            return Event("FT", "PLAYER", "OVER")
        ln = _line_from(low)
        if ln is not None and re.search(r"\bover\b", low):
            return Event(period, "TOTAL", "OVER", ln)
        if ln is not None and re.search(r"\bunder\b", low):
            return Event(period, "TOTAL", "UNDER", ln)
        return None
    period = "FT"
    key = low.replace(" ", "").replace("/", "").replace("_", "").replace(".", "").replace("-", "")
    if key.startswith("ht") and not key.startswith("htft"):
        period, key = "HT", key[2:]
    if key.startswith("btts"):
        sd = s or ("NO" if key.endswith("no") else "YES")
        return Event(period, "BTTS", sd) if sd in ("YES", "NO") else None
    if key in ("ou25over", "ou25under"):
        return Event(period, "TOTAL", key[4:].upper(), 2.5)
    if key.startswith("ou"):
        ln = _line_from(key) if _NUM.search(key) else 2.5
        return Event(period, "TOTAL", s, ln) if s in ("OVER", "UNDER") and ln is not None else None
    if key == "" and s in ("OVER", "UNDER"):
        return Event(period, "TOTAL", s, 2.5)
    for word in ("over", "under"):
        if key.startswith(word):
            ln = _line_from(key[len(word):])
            if ln is None:
                return None
            sd = word.upper()
            if s in ("NO", "UNDER") and word == "over":
                sd = "UNDER"
            elif s in ("NO", "OVER") and word == "under":
                sd = "OVER"
            return Event(period, "TOTAL", sd, ln)
    return None


# ── fixture identity ────────────────────────────────────────────────────────────────────────────
def _norm(name) -> str:
    return " ".join(str(name or "").casefold().split())


def fixture_id(date, home, away) -> str:
    """The canonical fixture key every store can produce: UTC match date | home | away (board names,
    case/space-normalised). Notified keys carry no event id, so the board's oddsapi_event_id cannot be the key;
    two rows with the same date and clubs are the same match."""
    return f"{str(date)[:10]}|{_norm(home)}|{_norm(away)}"


# ── score distributions ─────────────────────────────────────────────────────────────────────────
_LOGFACT = np.cumsum(np.concatenate([[0.0], np.log(np.arange(1, 64))]))


def _pmf(lam: float, n: int) -> np.ndarray:
    lam = max(float(lam), 1e-6)
    k = np.arange(n + 1)
    return np.exp(-lam + k * math.log(lam) - _LOGFACT[k])


def _tau(rho: float, lh: float, la: float, n: int) -> np.ndarray:
    t = np.ones((n + 1, n + 1))
    t[0, 0] = 1.0 - lh * la * rho
    t[0, 1] = 1.0 + lh * rho
    t[1, 0] = 1.0 + la * rho
    t[1, 1] = 1.0 - rho
    return np.clip(t, 1e-9, None)


def ft_matrix(lh: float, la: float, rho: float = 0.0) -> np.ndarray:
    """P(home=i, away=j), Dixon-Coles corrected, renormalised (src/goal_distribution._tau's form)."""
    m = np.outer(_pmf(lh, MAX_FT), _pmf(la, MAX_FT)) * _tau(rho, lh, la, MAX_FT)
    return m / m.sum()


def _ft_probs(m: np.ndarray) -> dict:
    i, j = np.meshgrid(np.arange(MAX_FT + 1), np.arange(MAX_FT + 1), indexing="ij")
    t = i + j
    return {"p_over15": float(m[t >= 2].sum()), "p_over25": float(m[t >= 3].sum()),
            "p_over35": float(m[t >= 4].sum()), "p_btts": float(m[(i >= 1) & (j >= 1)].sum())}


def fit_ft(targets: dict) -> dict | None:
    """(lam_home, lam_away, rho) fitted so the grid reproduces v9's own probabilities. Needs a total target
    (O1.5 / O2.5 / O3.5) AND p_btts - with totals only the split (which BTTS depends on) is unpinned, and a
    BTTS-dependent answer would come from the optimiser's start, not from data. None when not fittable."""
    from scipy.optimize import minimize
    use = {k: float(v) for k, v in (targets or {}).items()
           if k in ("p_over15", "p_over25", "p_over35", "p_btts") and v is not None
           and np.isfinite(float(v)) and 0.0 < float(v) < 1.0}
    if "p_btts" not in use or not ({"p_over15", "p_over25", "p_over35"} & set(use)):
        return None
    best = None
    for rho in RHO_GRID:
        def loss(x, rho=rho):
            lh, la = math.exp(x[0]), math.exp(x[1])
            p = _ft_probs(ft_matrix(lh, la, rho))
            return sum((p[k] - v) ** 2 for k, v in use.items())
        r = minimize(loss, x0=np.log([1.4, 1.1]), method="Nelder-Mead",
                     options={"xatol": 1e-5, "fatol": 1e-10, "maxiter": 400})
        if best is None or r.fun < best[0]:
            best = (float(r.fun), math.exp(r.x[0]), math.exp(r.x[1]), rho)
    if best is None or not all(np.isfinite(best[:3])):
        return None
    return {"lam_home": best[1], "lam_away": best[2], "rho": best[3], "sse": best[0]}


def _ft_mask(e: Event, i, j) -> np.ndarray:
    if e.kind == "TOTAL":
        return (i + j > e.line) if e.side == "OVER" else (i + j < e.line)
    both = (i >= 1) & (j >= 1)
    return both if e.side == "YES" else ~both


@dataclass
class FixtureModel:
    """A fixture's fitted FT grid (+ HT rates when the board has them)."""
    fit: dict | None
    ht: tuple[float, float] | None = None
    _ht_grid: tuple | None = field(default=None, repr=False)

    def _grid_ft(self):
        m = ft_matrix(self.fit["lam_home"], self.fit["lam_away"], self.fit["rho"])
        i, j = np.meshgrid(np.arange(MAX_FT + 1), np.arange(MAX_FT + 1), indexing="ij")
        return m, (lambda e: _ft_mask(e, i, j))

    def _grid_ht(self):
        """Joint (HT home, HT away, 2H home, 2H away). The FT rate is split onto the team whose HT rate is the
        larger (the fit is symmetric); second-half rate = FT - HT, floored at a tiny positive rate."""
        if self._ht_grid is None:
            hh, ha = self.ht
            fh, fa = sorted((self.fit["lam_home"], self.fit["lam_away"]), reverse=bool(hh >= ha))
            sh, sa = max(fh - hh, 1e-3), max(fa - ha, 1e-3)
            n = MAX_HALF
            p = np.einsum("a,b,c,d->abcd", _pmf(hh, n), _pmf(ha, n), _pmf(sh, n), _pmf(sa, n))
            p = p / p.sum()
            a, b, c, d = np.meshgrid(*(np.arange(n + 1),) * 4, indexing="ij")

            def mask(e: Event):
                if e.period == "HT":
                    return _ft_mask(e, a, b)
                return _ft_mask(e, a + c, b + d)
            self._ht_grid = (p, mask)
        return self._ht_grid

    def joint(self, x: Event, y: Event):
        """(P(x), P(y), P(x and y), method) or None when this model cannot price the pair."""
        if self.fit is None or x.kind not in GRID_KINDS or y.kind not in GRID_KINDS:
            return None
        if x.period == "HT" or y.period == "HT":
            if self.ht is None:
                return None
            p, mask = self._grid_ht()
            meth = "grid_ht"
        else:
            p, mask = self._grid_ft()
            meth = "grid_ft"
        mx, my = mask(x), mask(y)
        return float(p[mx].sum()), float(p[my].sum()), float(p[mx & my].sum()), meth


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def models_from_board(board: pd.DataFrame | None) -> dict:
    """fixture_id -> FixtureModel from a predictions board (columns date/home_team/away_team + v9's p_* and
    ht_lambda_*). Rows without enough probabilities give no model (the hard list decides for them)."""
    out = {}
    if board is None or board.empty:
        return out
    cols = set(board.columns)
    dcol = "date" if "date" in cols else "match_date"
    for r in board.to_dict("records"):
        fx = fixture_id(r.get(dcol), r.get("home_team"), r.get("away_team"))
        if fx in out:
            continue
        fit = fit_ft({k: _num(r.get(k)) for k in ("p_over15", "p_over25", "p_over35", "p_btts")})
        hh, ha = _num(r.get("ht_lambda_home")), _num(r.get("ht_lambda_away"))
        out[fx] = FixtureModel(fit, (hh, ha) if hh and ha and hh > 0 and ha > 0 else None)
    return out


class _LazyBoard(dict):
    """fixture_id -> FixtureModel, fitted on first use from output/predictions.csv (a fit is ~10 ms; only the
    fixtures a check actually touches are fitted)."""

    def __init__(self, board: pd.DataFrame | None):
        super().__init__()
        self._rows = {}
        if board is not None and not board.empty:
            dcol = "date" if "date" in board.columns else "match_date"
            for r in board.to_dict("records"):
                self._rows.setdefault(fixture_id(r.get(dcol), r.get("home_team"), r.get("away_team")), r)

    def get(self, fx, default=None):
        if fx in self:
            return self[fx]
        r = self._rows.get(fx)
        if r is None:
            return default
        m = models_from_board(pd.DataFrame([r])).get(fx)
        self[fx] = m
        return m


def load_board(path: Path | None = None) -> pd.DataFrame | None:
    p = Path(path) if path else config.OUTPUT_DIR / "predictions.csv"
    if not p.exists():
        return None
    try:
        hdr = pd.read_csv(p, nrows=0).columns
        want = [c for c in ("date", "match_date", "home_team", "away_team", "p_over15", "p_over25", "p_over35",
                            "p_btts", "ht_lambda_home", "ht_lambda_away") if c in hdr]
        return pd.read_csv(p, usecols=want, low_memory=False)
    except Exception as e:                                             # noqa: BLE001
        log.warning(f"coherence gate: board unreadable ({type(e).__name__}: {e}) - hard list only")
        return None


# ── the decision ────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Verdict:
    mismatch: bool
    method: str              # same_event / grid_ft / grid_ht / hard_list / no_relation
    detail: str = ""


def mismatch(a: Event, b: Event, model: FixtureModel | None = None) -> Verdict:
    """Is the pair (a, b) on ONE match negatively dependent? Grid when the fixture's model prices the pair,
    else the hard list (opposite goal directions)."""
    if a == b:
        return Verdict(False, "same_event")
    j = model.joint(a, b) if model is not None else None
    if j is not None:
        pa, pb, pab, meth = j
        neg = pab < pa * pb - NUM_TOL
        return Verdict(neg, meth, f"P(both)={pab:.3f} {'<' if neg else '>='} P(A)P(B)={pa * pb:.3f}")
    if 0 in (a.direction, b.direction):
        return Verdict(False, "no_relation", "no known goal relation")
    if a.direction != b.direction:
        return Verdict(True, "hard_list", "opposite goal directions")
    return Verdict(False, "hard_list", "same goal direction")


@dataclass
class Candidate:
    """One tip a sender would send. rank: higher stands within a run (TOP-3 rank -> larger is better, so a
    caller passes e.g. -top3_rank; else calibrated EV)."""
    fixture: str
    event: Event
    source: str
    label: str = ""
    rank: float = float("-inf")
    uid: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = self.event.label
        if not self.uid:
            self.uid = f"{self.source}|{self.fixture}|{self.event}"

    @property
    def text(self) -> str:
        return f"{self.label} [{self.source}]"


def calibrated_ev(edge_calibrated, odds, raw_edge=None) -> float:
    """Rank of a non-TOP-3 tip: its calibrated EV per unit, edge_calibrated x odds (p_cal x odds - 1 up to the
    de-vig margin); the raw edge x odds when no calibrated edge exists; -inf when neither."""
    o = _num(odds)
    for e in (edge_calibrated, raw_edge):
        x = _num(e)
        if x is not None and o is not None:
            return x * o
    return float("-inf")


# ── what was already sent ───────────────────────────────────────────────────────────────────────
def parse_notified_key(key: str) -> tuple[str, Event, str] | None:
    """(fixture_id, event, store label) of a telegram_bot/notified.json key that records a tip, else None.
        <date>|<home>|<away>|OVER/UNDER                     instant / album O/U 2.5
        SIDE|<date>|<home>|<away>|<market>[|<side>]          side market (btts / over15 / over35 / ...)
        HT|<date>|<home>|<away>|HT<OVER|UNDER><line>         half-time tip"""
    p = str(key).split("|")
    try:
        if p[0] == "SIDE" and len(p) >= 5:
            ev = parse_event(p[4], p[5] if len(p) > 5 else "YES")
            return (fixture_id(p[1], p[2], p[3]), ev, "notified SIDE") if ev else None
        if p[0] == "HT" and len(p) == 5:
            m = re.match(r"HT(OVER|UNDER)(\d+(?:\.\d+)?)$", p[4])
            if not m:
                return None
            return fixture_id(p[1], p[2], p[3]), Event("HT", "TOTAL", m.group(1), float(m.group(2))), "notified HT"
        if len(p) == 4 and re.match(r"\d{4}-\d{2}-\d{2}$", p[0]) and p[3] in ("OVER", "UNDER"):
            return fixture_id(p[0], p[1], p[2]), Event("FT", "TOTAL", p[3], 2.5), "notified O/U 2.5"
    except Exception:                                                  # noqa: BLE001
        return None
    return None


def sent_from_notified(keys) -> list[Candidate]:
    out = []
    for k in keys or ():
        x = parse_notified_key(k)
        if x:
            out.append(Candidate(x[0], x[1], x[2], uid=f"notified|{k}"))
    return out


def sent_from_top3(out_dir: Path | None = None) -> list[Candidate]:
    """Every pick of every TOP-3 league-week already sent (output/top3_league_notified.json weeks)."""
    od = Path(out_dir) if out_dir else config.OUTPUT_DIR
    try:
        weeks = set(json.loads((od / TOP3_NOTIFIED).read_text(encoding="utf-8")).get("weeks", []))
    except Exception:                                                  # noqa: BLE001
        return []
    p = od / TOP3_PICKS
    if not weeks or not p.exists():
        return []
    try:
        d = pd.read_csv(p, dtype=str)
    except Exception:                                                  # noqa: BLE001
        return []
    d = d[d["league_week_id"].isin(weeks)]
    out = []
    for r in d.to_dict("records"):
        ev = parse_event(r.get("market"), r.get("side"))
        if ev:
            out.append(Candidate(fixture_id(r.get("match_date"), r.get("home_team"), r.get("away_team")), ev,
                                 "sent TOP-3", uid=f"top3|{r.get('league_week_id')}|{r.get('top3_rank')}"))
    return out


# ── the gate ────────────────────────────────────────────────────────────────────────────────────
class CoherenceGate:
    """Tips already sent (per fixture) + this run's planned losers. Never raises from check/plan/commit: on an
    internal error a check falls back to the hard list for that pair."""

    def __init__(self, sent: list[Candidate] | None = None, models: dict | None = None):
        self.sent: dict[str, list[Candidate]] = {}
        for c in sent or ():
            self.sent.setdefault(c.fixture, []).append(c)
        self.models = models if models is not None else {}
        self.losers: dict[str, str] = {}
        self.log: list[str] = []

    @classmethod
    def load(cls, notified_keys=None, out_dir: Path | None = None, board: pd.DataFrame | None = None,
             notified_file: Path | None = None) -> "CoherenceGate":
        if notified_keys is None and notified_file is None:
            try:                                   # the notifier's own reader (its path, its key migration)
                from telegram_bot.notifier import _load_notified
                notified_keys = _load_notified()
            except Exception:                                          # noqa: BLE001
                notified_keys = None
        if notified_keys is None:
            try:
                f = notified_file or NOTIFIED_FILE
                notified_keys = json.loads(f.read_text(encoding="utf-8-sig") or "{}").get("keys", []) \
                    if f.exists() else []
            except Exception as e:                                     # noqa: BLE001
                log.warning(f"coherence gate: notified store unreadable ({type(e).__name__}: {e})")
                notified_keys = []
        sent = sent_from_notified(notified_keys) + sent_from_top3(out_dir)
        return cls(sent, _LazyBoard(board if board is not None else load_board()))

    def _model(self, fx):
        try:
            return self.models.get(fx)
        except Exception:                                              # noqa: BLE001
            return None

    def _verdict(self, a: Candidate, b: Candidate) -> Verdict:
        try:
            return mismatch(a.event, b.event, self._model(a.fixture))
        except Exception as e:                                         # noqa: BLE001
            log.warning(f"coherence gate: grid failed ({type(e).__name__}: {e}) - hard list")
            return mismatch(a.event, b.event, None)

    def conflict(self, cand: Candidate, against: list[Candidate] | None = None) -> str:
        """'' or the reason `cand` may not go out next to `against` (default: the tips already sent)."""
        pool = self.sent.get(cand.fixture, []) if against is None else against
        for other in pool:
            if other.fixture != cand.fixture:
                continue
            v = self._verdict(cand, other)
            if v.mismatch:
                return f"MISMATCH with {other.text} ({v.method}: {v.detail})"
        return ""

    def plan(self, cands: list[Candidate]) -> dict[str, str]:
        """This run's candidates, best rank first: a candidate that mismatches a tip already sent, or a
        higher-ranked candidate kept before it, is a loser {uid: reason}. Ties keep the caller's order."""
        order = sorted(range(len(cands)), key=lambda k: (-(cands[k].rank if cands[k].rank == cands[k].rank
                                                            else float("-inf")), k))
        kept: dict[str, list[Candidate]] = {}
        out = {}
        for k in order:
            c = cands[k]
            why = self.conflict(c) or self.conflict(c, kept.get(c.fixture, []))
            if why:
                out[c.uid] = why
            else:
                kept.setdefault(c.fixture, []).append(c)
        self.losers.update(out)
        return out

    def check(self, cand: Candidate) -> str:
        """'' = may be sent now; else the logged reason it is skipped."""
        return self.losers.get(cand.uid) or self.conflict(cand)

    def commit(self, cand: Candidate) -> None:
        self.sent.setdefault(cand.fixture, []).append(cand)


def filter_frame(rows: pd.DataFrame, gate: CoherenceGate, source: str, market_col: str = "market",
                 side_col: str | None = None, date_col: str = "date", rank=None,
                 sent_mask: pd.Series | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Rows (one tip each) -> (coherent rows, skip log lines). `sent_mask` marks rows that were already sent
    (they are kept and stand); the rest are planned best-`rank` first. A row whose market does not parse is
    kept untouched (no goal relation known)."""
    if rows is None or rows.empty:
        return rows, []
    rk = rank if rank is not None else pd.Series(float("-inf"), index=rows.index)
    sm = sent_mask if sent_mask is not None else pd.Series(False, index=rows.index)
    cands, idx = [], []
    for i, r in rows.iterrows():
        ev = parse_event(r.get(market_col), r.get(side_col) if side_col else None)
        if ev is None:
            continue
        c = Candidate(fixture_id(r.get(date_col), r.get("home_team"), r.get("away_team")), ev, source,
                      rank=float(rk.loc[i]) if pd.notna(rk.loc[i]) else float("-inf"), uid=f"{source}|row{i}")
        if bool(sm.loc[i]):
            gate.commit(c)
            continue
        cands.append(c)
        idx.append(i)
    losers = gate.plan(cands)
    drop = [i for i, c in zip(idx, cands) if c.uid in losers]
    lines = [f"SKIP {rows.loc[i, 'home_team']} v {rows.loc[i, 'away_team']} {c.label} [{source}]: {losers[c.uid]}"
             for i, c in zip(idx, cands) if c.uid in losers]
    for i, c in zip(idx, cands):
        if c.uid not in losers:
            gate.commit(c)
    return rows.drop(index=drop), lines
