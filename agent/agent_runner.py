"""
Agent Runner v2
===============
Supports the full 9-agent pipeline:
  football-data-collection → minutes-projection + team-goal-expectancy
  → player-prop-modeling + market-intelligence
  → confidence-scoring + simulation
  → portfolio-manager → final signals

Also supports individual agents:
  elite-football-analytics, senior-ml-systems,
  player-prop-analyst, master-player-prop-analytics,
  market-probability-analytics, quant-football-fund

API priority:
  1. Google Gemini (GOOGLE_API_KEY — free tier)
  2. Anthropic Claude (ANTHROPIC_API_KEY)
  3. Manual fallback — returns prompt for paste into Claude.ai
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

AGENT_DIR    = Path(__file__).resolve().parent
GEMINI_MODEL = "gemini-1.5-flash"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# ── All available agents ──────────────────────────────────────────────────────
AGENTS = {f.stem.replace(".agent", ""): f
          for f in AGENT_DIR.glob("*.agent.md")}

# ── Pipeline definition ───────────────────────────────────────────────────────
PIPELINE_STAGES = [
    "football-data-collection",
    "minutes-projection",
    "team-goal-expectancy",
    "player-prop-modeling",
    "market-intelligence",
    "confidence-scoring",
    "simulation",
    "portfolio-manager",
]


def _load_agent(name: str) -> str:
    """Load and strip frontmatter from an agent .md file."""
    path = AGENTS.get(name)
    if not path or not path.exists():
        return f"Agent '{name}' not found."
    text = path.read_text(encoding="utf-8")
    return re.sub(r"^---.*?---\s*", "", text, flags=re.DOTALL).strip()


def _run_gemini(system: str, prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(model_name=GEMINI_MODEL, system_instruction=system)
    return model.generate_content(prompt).text


def _run_claude(system: str, prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _call_agent(agent_name: str, prompt: str) -> dict:
    """Call one agent with a prompt. Returns {mode, response, agent}."""
    system = _load_agent(agent_name)
    match_label = prompt[:80].replace("\n", " ")

    if os.getenv("GOOGLE_API_KEY"):
        try:
            text = _run_gemini(system, prompt)
            return {"mode": "gemini", "response": text, "agent": agent_name}
        except Exception as e:
            print(f"Gemini error ({agent_name}): {e}")

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            text = _run_claude(system, prompt)
            return {"mode": "claude", "response": text, "agent": agent_name}
        except Exception as e:
            print(f"Claude error ({agent_name}): {e}")

    return {"mode": "manual", "response": f"[{agent_name}]\n{prompt}", "agent": agent_name}


# ── Single agent run ──────────────────────────────────────────────────────────

def run_agent(row: "pd.Series | dict[str, Any]",
              agent_name: str = "elite-football-analytics") -> dict:
    """Run a single agent on a match row."""
    r = row.to_dict() if not isinstance(row, dict) else row
    prompt = _build_match_prompt(r)
    result = _call_agent(agent_name, prompt)
    result["match"] = f"{r.get('home_team','')} vs {r.get('away_team','')}"
    return result


def _build_match_prompt(r: dict) -> str:
    """Build a match analysis prompt from a row dict."""
    home  = r.get("home_team", "")
    away  = r.get("away_team", "")
    league= r.get("league", "")
    date  = str(r.get("date", ""))[:10]
    side  = r.get("best_side") or r.get("bet", "")
    edge  = float(r.get("best_edge", 0)) * 100
    odds_o= r.get("odds_over25", "N/A")
    odds_u= r.get("odds_under25", "N/A")
    drift = r.get("drift_signal", "New")
    p_over= r.get("p_over25", "N/A")
    model = r.get("model_type", "standard")

    return f"""Match: {home} vs {away} | {league} | {date}
ML Signal: {r.get('signal_tier','')} — {side} 2.5 | Edge: {edge:.1f}% | Model: {model}
P(Over 2.5): {p_over} | Drift: {drift}
Odds: Over {odds_o} | Under {odds_u}

Run full analysis. Surface only the strongest edges."""


# ── Full pipeline run ─────────────────────────────────────────────────────────

def run_pipeline(match_data: dict) -> dict:
    """
    Run the full 8-agent pipeline on a match.
    Each agent receives the previous agent's output as context.
    Returns final portfolio-manager output.
    """
    context = f"Match context:\n{match_data}\n\n"
    results = {}

    for stage in PIPELINE_STAGES:
        prompt = context + f"Previous outputs:\n{_summarise(results)}\n\nExecute your role."
        result = _call_agent(stage, prompt)
        results[stage] = result["response"]
        context += f"\n\n[{stage} output]:\n{result['response'][:500]}\n"
        print(f"  [{stage}] done ({result['mode']})")

    return {
        "mode": "pipeline",
        "stages": results,
        "final": results.get("portfolio-manager", ""),
        "match": match_data.get("match", ""),
    }


def _summarise(results: dict) -> str:
    if not results:
        return "None yet."
    return "\n".join(f"- {k}: {v[:200]}..." for k, v in results.items())


def _load_system_prompt(agent_name: str = "elite-football-analytics") -> str:
    """Load system prompt for an agent (falls back to first available agent)."""
    if agent_name in AGENTS:
        return _load_agent(agent_name)
    if AGENTS:
        return _load_agent(next(iter(AGENTS)))
    return "You are an expert football analytics assistant."
