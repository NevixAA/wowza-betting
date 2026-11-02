"""v9 main -> slim dashboard: a push to main sends repository_dispatch 'v9-main-updated' to
NevixAA/wowza-dashboard (.github/workflows/dashboard_sync_dispatch.yml), for every kind of file the
slim repo copies as code (deploy/dashboard_manifest.json 'code')."""
import fnmatch
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows" / "dashboard_sync_dispatch.yml"


def _push_paths(text: str) -> tuple[list[str], list[str]]:
    on = text.split("\njobs:", 1)[0]
    push = on.split("push:", 1)[1].split("workflow_dispatch", 1)[0]
    branches = re.findall(r"branches:\s*\[([^\]]*)\]", push)[0]
    paths = re.findall(r"-\s*'([^']+)'", push)
    return [b.strip() for b in branches.split(",")], paths


def test_push_to_main_dispatches_v9_main_updated_to_the_dashboard_repo():
    text = WF.read_text(encoding="utf-8")
    branches, _ = _push_paths(text)
    assert branches == ["main"]
    assert "https://api.github.com/repos/NevixAA/wowza-dashboard/dispatches" in text
    assert '\\"event_type\\":\\"v9-main-updated\\"' in text
    assert "secrets.DASHBOARD_DISPATCH_TOKEN" in text          # GITHUB_TOKEN cannot dispatch cross-repo


def test_every_shipped_code_file_triggers_the_dispatch():
    _, paths = _push_paths(WF.read_text(encoding="utf-8"))
    code = [e["path"] for e in json.loads((ROOT / "deploy" / "dashboard_manifest.json")
                                          .read_text(encoding="utf-8"))["code"]]
    # GitHub path filters: '**' matches across '/', fnmatch's '*' does too, so '**.py' ~ '*.py'.
    missed = [p for p in code if not any(fnmatch.fnmatch(p, g.replace("**", "*")) for g in paths)]
    assert missed == []
