"""Markdown tables from reports/sharp_filter/results.json (counts and statistics only; no price)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "reports" / "sharp_filter" / "results.json").read_text(encoding="utf-8"))


def pct(x):
    return "-" if x is None else f"{x * 100:+.1f}%"


def ci(c):
    return "-" if not c or c[0] is None else f"[{c[0] * 100:+.1f}, {c[1] * 100:+.1f}]"


def num(x, d=3):
    return "-" if x is None else f"{x:.{d}g}" if isinstance(x, float) else str(x)


def model_table(window: str, signal: str) -> str:
    cells = [c for c in R["cells"] if c["arm"] == "model" and c["window"] == window]
    keys = sorted({(c["track"], c["side"], c["league"]) for c in cells})
    out = ["| Track | Market-side | League | ALL n / ROI | AGREE n / ROI / CI / Holm p | DISAGREE n / ROI | NONE n / ROI | NO_DATA+MIXED n | AGREE win-implied | AGREE verdict |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        g = {c["cls"]: c for c in cells if (c["track"], c["side"], c["league"]) == k and c["signal"] in (signal, "-")}
        def s(cl):
            x = g.get(cl, {}).get("stats", {})
            return x
        a = s("AGREE")
        nd = (s("NO_DATA").get("bets", 0) or 0) + (s("MIXED").get("bets", 0) or 0)
        out.append(f"| {k[0]} | {k[1]} | {k[2]} | {s('ALL').get('bets', 0)} / {pct(s('ALL').get('roi'))} | "
                   f"{a.get('bets', 0)} / {pct(a.get('roi'))} / {ci(a.get('ci95'))} / {num(g.get('AGREE', {}).get('p_holm'))} | "
                   f"{s('DISAGREE').get('bets', 0)} / {pct(s('DISAGREE').get('roi'))} | {s('NONE').get('bets', 0)} / {pct(s('NONE').get('roi'))} | "
                   f"{nd} | {pct(a.get('win_minus_implied'))} | {g.get('AGREE', {}).get('verdict', '-')} |")
    return "\n".join(out)


def market_table(window: str, signal: str) -> str:
    cells = sorted([c for c in R["cells"] if c["arm"] == "market" and c["window"] == window and c["signal"] == signal],
                   key=lambda c: (c["side"], c["league"]))
    out = ["| Side | League | n | ROI | ROI 95% CI | win-implied | CLV (n) | Holm p | Verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        s = c["stats"]
        out.append(f"| {c['side']} | {c['league']} | {s.get('bets', 0)} | {pct(s.get('roi'))} | {ci(s.get('ci95'))} | "
                   f"{pct(s.get('win_minus_implied'))} | {pct(s.get('clv'))} ({s.get('clv_n', 0)}) | {num(c.get('p_holm'))} | {c.get('verdict', '-')} |")
    return "\n".join(out)


def pooled(arm: str, window: str, signal: str, cls: str, side_prefix: str = "") -> dict:
    """Descriptive sum over cells (not a decision unit): n, ROI weighted by bets."""
    cs = [c for c in R["cells"] if c["arm"] == arm and c["window"] == window and c["signal"] == signal
          and c["cls"] == cls and c["side"].startswith(side_prefix) and c["stats"].get("bets")]
    n = sum(c["stats"]["bets"] for c in cs)
    pl = sum(c["stats"]["bets"] * c["stats"]["roi"] for c in cs)
    return {"n": n, "roi": (pl / n) if n else None}


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "model":
        print(model_table(sys.argv[2], sys.argv[3]))
    elif what == "market":
        print(market_table(sys.argv[2], sys.argv[3]))
