"""
Auto-push updated output CSVs to GitHub after each pipeline run.
Called by Task Scheduler after predict and update_results.
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

V9_DIR = Path(__file__).resolve().parent.parent


def git_push():
    def run(cmd):
        return subprocess.run(cmd, cwd=V9_DIR, capture_output=True, text=True)

    # Stage only output CSVs and models feature importances
    files = [
        "output/bets.csv",
        "output/bets_ledger.csv",
        "output/backtest_results_standard.csv",
        "output/backtest_by_league_standard.csv",
        "output/predictions.csv",
        "output/worldcup_tips.csv",
        "output/worldcup_history.json",
        "output/sharp_tips.csv",
        "output/sharp_history.json",
        "output/live_tips.csv",
        "output/live_games.csv",
        "models/feature_importances_standard.csv",
        "models/feature_importances_newformat.csv",
    ]

    # Only add files that exist
    existing = [f for f in files if (V9_DIR / f).exists()]
    if not existing:
        print("Nothing to push.")
        return

    run(["git", "add"] + existing)

    # Check if there's anything to commit
    status = run(["git", "status", "--porcelain"])
    if not status.stdout.strip():
        print("No changes to push.")
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    run(["git", "commit", "-m", f"auto: update outputs {now}"])
    result = run(["git", "push"])

    if result.returncode == 0:
        print(f"Pushed to GitHub at {now}")
    else:
        print(f"Push failed: {result.stderr}")


if __name__ == "__main__":
    git_push()
