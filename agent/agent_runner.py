"""
Agent Runner — calls the Elite Football Analytics agent via Google Gemini (free tier).

Priority:
  1. Google Gemini  — if GOOGLE_API_KEY is set (free tier: gemini-1.5-flash)
  2. Anthropic Claude — if ANTHROPIC_API_KEY is set (paid, very cheap)
  3. Manual fallback — returns filled prompt for paste into Claude.ai

Setup (Gemini free):
  1. Go to https://aistudio.google.com/apikey → create free key
  2. Set env var: GOOGLE_API_KEY=your_key
  3. pip install google-generativeai

Usage:
    from agent.agent_runner import run_agent, build_prompt
    result = run_agent(match_row)
    # result["mode"]     → "gemini" | "claude" | "manual"
    # result["response"] → analysis text or filled prompt
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

# Load .env from v9/ root (one level up from agent/)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

AGENT_DIR   = Path(__file__).resolve().parent
SYSTEM_FILE = AGENT_DIR / ".agent.md"

GEMINI_MODEL  = "gemini-1.5-flash"
CLAUDE_MODEL  = "claude-haiku-4-5-20251001"


def _load_system_prompt() -> str:
    """Extract the body of .agent.md (strips YAML frontmatter)."""
    text = SYSTEM_FILE.read_text(encoding="utf-8")
    text = re.sub(r"^---.*?---\s*", "", text, flags=re.DOTALL)
    return text.strip()


def build_prompt(row: "pd.Series | dict[str, Any]") -> str:
    """Return a filled analysis prompt for one match row from bets.csv."""
    if isinstance(row, dict):
        r = row
    else:
        r = row.to_dict()

    home       = r.get("home_team", "")
    away       = r.get("away_team", "")
    league     = r.get("league", "")
    date       = str(r.get("date", ""))[:10]
    side       = r.get("best_side") or r.get("bet", "")
    edge       = float(r.get("best_edge", 0)) * 100
    odds_over  = r.get("odds_over25",  "N/A")
    odds_under = r.get("odds_under25", "N/A")
    model_type = r.get("model_type", "standard")
    drift      = r.get("drift_signal", "New")
    p_over     = r.get("p_over25",  "N/A")
    p_under    = r.get("p_under25", "N/A")

    ht_lines = ""
    p_ht05 = r.get("p_ht_over05")
    p_ht15 = r.get("p_ht_over15")
    if p_ht05 not in (None, "", "N/A"):
        ht_lines = (
            f"\n- HT Over 0.5: P={float(p_ht05)*100:.0f}%"
            + (f"  |  HT Over 1.5: P={float(p_ht15)*100:.0f}%" if p_ht15 not in (None, "", "N/A") else "")
        )

    return f"""Hunt for sniper value signals in the following match using the Elite Football Analytics & Betting Intelligence framework.
Run the full model pipeline, then surface only the strongest edges.

**Match:**
- Home team: {home}
- Away team: {away}
- Competition: {league}
- Date / kickoff: {date}

**ML Model Output (v9 pipeline):**
- Signal tier: {r.get('signal_tier', '')} — Recommended side: {side} 2.5
- Model edge: {edge:.1f}%  |  Model type: {model_type}
- P(Over 2.5): {p_over}  |  P(Under 2.5): {p_under}
- Drift signal: {drift}

**Market odds:**
- Over 2.5: {odds_over}  |  Under 2.5: {odds_under}{ht_lines}
- Asian Handicap line: [provide if available]
- BTTS Yes / No: [provide if available]
- Correct score top lines: [provide if available]
- 1X2: [provide if available]

**Team news / context:** [injuries, lineup risk, referee, weather, line movement]

---

Required output — lead with the signals:

### 1. STRONGEST SIGNALS
Tier S / A / B only. For each: market, selection, model prob vs implied prob, Edge%, EV, confidence note.

### 2. Expected Goals
λ_H, λ_A, λ_T with derivation.

### 3. Probability Model
Over/Under, BTTS, Correct Score, AH, 1X2 — all markets with fair odds.

### 4. Arbitrage Check
Compute ArbitrageSum for all provided odds. Flag Tier S if < 1.0.

### 5. Cross-Market Consistency
Flag outcomes confirmed by ≥ 3 independent markets or models.

### 6. Player Prop Signals
Flag any prop aligning with team-level model (correlated value cluster).

### 7. Model Assumptions & Risk
Key assumptions, data gaps, confidence intervals on λ.

### 8. Responsible Betting Note

Use rigorous calculations. Show all EV and edge figures. No narrative predictions."""


def _run_gemini(system: str, prompt: str) -> str:
    import google.generativeai as genai  # type: ignore
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system,
    )
    response = model.generate_content(prompt)
    return response.text


def _run_claude(system: str, prompt: str) -> str:
    import anthropic  # type: ignore
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def run_agent(row: "pd.Series | dict[str, Any]") -> dict[str, str]:
    """
    Run the agent for one match row.

    Returns dict with keys:
        mode      → "gemini" | "claude" | "manual"
        response  → analysis text (auto) or filled prompt (manual)
        match     → "Home vs Away"
    """
    r = row.to_dict() if not isinstance(row, dict) else row
    match_label = f"{r.get('home_team', '')} vs {r.get('away_team', '')}"
    prompt = build_prompt(row)
    system = _load_system_prompt()

    if os.getenv("GOOGLE_API_KEY"):
        try:
            text = _run_gemini(system, prompt)
            return {"mode": "gemini", "response": text, "match": match_label}
        except Exception as e:
            print(f"Gemini error: {e} — falling back")

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            text = _run_claude(system, prompt)
            return {"mode": "claude", "response": text, "match": match_label}
        except Exception as e:
            print(f"Claude error: {e} — falling back")

    return {"mode": "manual", "response": prompt, "match": match_label}
