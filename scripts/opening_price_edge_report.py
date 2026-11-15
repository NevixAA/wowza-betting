"""Findings table of the opening-price-edge study (markdown, counts / ROI / CLV / CI / Holm only).

    python scripts/opening_price_edge_report.py > <file>
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "output" / "research" / "opening_price_edge" / "results.json"


def pct(v):
    return "-" if v is None or pd.isna(v) else f"{100 * v:+.1f}%"


def ci(r):
    return "-" if r.get("lo") is None else f"[{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}]"


def hp(v):
    return "-" if v is None else (f"{v:.3g}" if v < 0.001 else f"{v:.3f}")


def main() -> int:
    d = json.loads(RES.read_text(encoding="utf-8"))
    cells = {(c["track"], c["league"], c["market"], c["side"], c["role"]): c for c in d["cells"]}
    keys = sorted({k[:4] for k in cells})
    print("| Track | League | Market | Side | Role | n bets | ROI | ROI 95% CI | Holm p (ROI) | CLV vs close | CLV 95% CI "
          "| RAW share | Verdict |")
    print("|---|---|---|---|---|---:|---:|---|---:|---:|---|---:|---|")
    for k in keys:
        for role in ("OPEN", "BET", "FD_PRE"):
            c = cells.get(k + (role,))
            if c is None or not c["bets"]:
                continue
            side = {"yes": "Over" if k[2] != "btts" else "Yes", "no": "Under" if k[2] != "btts" else "No"}[k[3]]
            print(f"| {'Std' if k[0] == 'standard' else 'NF'} | {k[1]} | {k[2]} | {side} | {role} | {c['bets']} | "
                  f"{pct(c['roi']['mean'])} | {ci(c['roi'])} | {hp(c['roi'].get('p_holm'))} | {pct(c['clv']['mean'])} | "
                  f"{ci(c['clv'])} | {pct(c['raw_share'])} | {c['verdict']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
