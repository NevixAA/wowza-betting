"""Private price store LIVE mode (scripts/price_store_live.py): synthetic repos over file://, no
network, no real prices. A local bare "store" plays NevixAA/wowza-exec@af-prices-live and a local
"public" repo plays the workspace of a job."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import price_store_live as psl  # noqa: E402
import price_store_sync as pss  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")

ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
       "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
PREFIX = "wowza-betting/main/"
QUOTES = "output/book_quotes/2026-11-24.csv"
HIST = "odds_history_v9.json"


def git(*a, cwd=None, check=True):
    p = subprocess.run(["git", *a], cwd=cwd, capture_output=True, env={**os.environ, **ENV})
    if check and p.returncode:
        raise AssertionError(p.stderr.decode())
    return p.stdout.decode().strip()


def bare(path: Path) -> Path:
    git("init", "--quiet", "--bare", str(path))
    git("config", "uploadpack.allowFilter", "true", cwd=path)
    git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=path)
    return path


def push_files(remote: Path, branch: str, files: dict, work: Path) -> str:
    if not work.exists():
        git("clone", "--quiet", remote.as_uri(), str(work))
        git("config", "core.autocrlf", "false", cwd=work)
        git("checkout", "--quiet", "-B", branch, cwd=work)
    else:
        git("fetch", "--quiet", "origin", cwd=work)
        git("reset", "--quiet", "--hard", f"origin/{branch}", cwd=work)
    for rel, data in files.items():
        p = work / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    git("add", "-A", cwd=work)
    git("commit", "--quiet", "-m", "x", cwd=work)
    git("push", "--quiet", "origin", f"HEAD:{branch}", cwd=work)
    return git("rev-parse", "HEAD", cwd=work)


def store_file(store: Path, rel: str) -> bytes:
    return subprocess.run(["git", "show", f"af-prices-live:{rel}"], cwd=store, capture_output=True, check=True).stdout


@pytest.fixture()
def env(tmp_path):
    store = bare(tmp_path / "store.git")
    push_files(store, "af-prices-live", {
        ".price_store": b"marker\n",
        PREFIX + QUOTES: b"fixture_id,market,odds\n1,ou25,1.11\n",          # synthetic, not a real price
        PREFIX + HIST: b'{"a": 1, "b": 2}\n',
        PREFIX + "output/bets.csv": b"fixture_id,odds\n1,1.11\n",           # class B: not live here
    }, tmp_path / "s_work")
    cfg = {"store": {"repo": "x", "branch": "af-prices-live"}, "sources": [["wowza-betting", "main"]],
           "rules": {"wowza-betting": {"A_O": [r"^output/book_quotes/", r"^odds_history_v9\.json$"],
                                        "B": [r"^output/bets\.csv$"]}},
           "live": {"wowza-betting": ["A_O"]}}
    cfgp = tmp_path / "cfg.json"
    cfgp.write_text(json.dumps(cfg))
    return dict(tmp=tmp_path, store=store, cfg=cfgp)


def workspace(env, name: str, files: dict | None = None) -> Path:
    ws = env["tmp"] / name
    git("init", "--quiet", "-b", "main", str(ws))
    git("config", "core.autocrlf", "false", cwd=ws)
    (ws / "README.md").write_text("x\n")
    for rel, data in (files or {}).items():
        (ws / rel).parent.mkdir(parents=True, exist_ok=True)
        (ws / rel).write_bytes(data)
    git("add", "-A", cwd=ws)
    git("commit", "--quiet", "-m", "init", cwd=ws)
    return ws


def run_open(env, ws: Path, live="1", tag="a", extra=()):
    return psl.main(["open", "--repo", "wowza-betting", "--workspace", str(ws), "--store-dir", str(env["tmp"] / f"store_{tag}"),
                     "--store-url", env["store"].as_uri(), "--live", live, "--manifest", str(env["tmp"] / f"man_{tag}.json"),
                     "--config", str(env["cfg"]), *extra])


def run_push(env, ws: Path, tag="a"):
    return psl.main(["push", "--producer", "test", "--workspace", str(ws), "--manifest", str(env["tmp"] / f"man_{tag}.json"),
                     "--config", str(env["cfg"]), "--attempts", "3"])


def test_open_places_live_files_only_and_push_without_change_is_a_noop(env):
    ws = workspace(env, "ws")
    assert run_open(env, ws) == 0
    assert (ws / QUOTES).read_bytes() == store_file(env["store"], PREFIX + QUOTES)
    assert (ws / HIST).read_bytes() == store_file(env["store"], PREFIX + HIST)
    assert not (ws / "output/bets.csv").exists()                          # class B is not live
    head = git("rev-parse", "af-prices-live", cwd=env["store"])
    assert run_push(env, ws) == 0
    assert git("rev-parse", "af-prices-live", cwd=env["store"]) == head


def test_two_runs_appending_to_one_file_both_land(env):
    a, b = workspace(env, "wa"), workspace(env, "wb")
    assert run_open(env, a, tag="a") == 0 and run_open(env, b, tag="b") == 0
    with open(a / QUOTES, "ab") as f:
        f.write(b"2,ou25,1.22\n")
    with open(b / QUOTES, "ab") as f:
        f.write(b"3,btts,1.33\n")
    (b / "output/book_quotes/2026-11-25.csv").write_bytes(b"fixture_id,market,odds\n4,ou25,1.44\n")
    assert run_push(env, a, "a") == 0
    assert run_push(env, b, "b") == 0                                    # base moved: merged, not overwritten
    got = store_file(env["store"], PREFIX + QUOTES).splitlines()
    assert got[0] == b"fixture_id,market,odds" and sorted(got[1:]) == [b"1,ou25,1.11", b"2,ou25,1.22", b"3,btts,1.33"]
    assert store_file(env["store"], PREFIX + "output/book_quotes/2026-11-25.csv").endswith(b"4,ou25,1.44\n")


def test_json_three_way_merge_and_removed_lines(env):
    a, b = workspace(env, "wa"), workspace(env, "wb")
    run_open(env, a, tag="a"), run_open(env, b, tag="b")
    (a / HIST).write_text('{"a": 1, "b": 2, "c": 3}\n')
    (b / HIST).write_text('{"a": 9, "b": 2}\n')
    (b / QUOTES).write_bytes(b"fixture_id,market,odds\n")                # rolled out by run b
    assert run_push(env, a, "a") == 0 and run_push(env, b, "b") == 0
    assert json.loads(store_file(env["store"], PREFIX + HIST)) == {"a": 9, "b": 2, "c": 3}
    assert store_file(env["store"], PREFIX + QUOTES) == b"fixture_id,market,odds\n"


def test_unmergeable_file_keeps_both_versions_and_goes_red(env):
    a, b = workspace(env, "wa"), workspace(env, "wb")
    run_open(env, a, tag="a"), run_open(env, b, tag="b")
    with open(a / QUOTES, "ab") as f:
        f.write(b"2,ou25,1.22\n")
    (b / QUOTES).write_bytes(b"fixture_id,market,odds,book\n1,ou25,1.11,x\n")   # schema change
    assert run_push(env, a, "a") == 0
    assert run_push(env, b, "b") == 1
    tree = git("ls-tree", "-r", "--name-only", "af-prices-live", cwd=env["store"]).splitlines()
    kept = [t for t in tree if t.startswith("_conflicts/") and t.endswith(QUOTES)]
    assert kept and store_file(env["store"], kept[0]).endswith(b"2,ou25,1.22\n")


def test_deleted_live_file_is_never_removed_from_the_store(env):
    ws = workspace(env, "ws")
    run_open(env, ws)
    (ws / QUOTES).unlink()
    assert run_push(env, ws) == 0
    assert store_file(env["store"], PREFIX + QUOTES).startswith(b"fixture_id")


def test_hook_keeps_live_files_out_of_a_public_commit(env):
    ws = workspace(env, "ws")
    run_open(env, ws)
    (ws / "output/other.csv").write_text("x\n")
    git("add", "-f", QUOTES, HIST, "output/other.csv", cwd=ws)
    git("commit", "--quiet", "-m", "job", cwd=ws)
    committed = git("show", "--name-only", "--format=", "HEAD", cwd=ws).splitlines()
    assert committed == ["output/other.csv"]
    assert (ws / QUOTES).exists()                                      # the workspace copy stays for the job


def test_rollback_mode_uses_public_files_and_only_fills_missing_ones(env):
    pub = workspace(env, "pub", {QUOTES: b"fixture_id,market,odds\n"})
    assert run_open(env, pub, live="0", tag="p") == 0                  # public copy tracked: no clone, no-op
    assert json.loads((env["tmp"] / "man_p.json").read_text())["mode"] == "off"
    assert not (pub / HIST).exists()
    gone = workspace(env, "gone")                                       # after the cut-over, rolled back
    assert run_open(env, gone, live="0", tag="g") == 0
    assert (gone / HIST).exists() and (gone / QUOTES).exists()
    assert not (Path(git("rev-parse", "--git-common-dir", cwd=gone)) / "hooks" / "pre-commit").exists() or \
        psl.HOOK_TAG not in (gone / ".git/hooks/pre-commit").read_text()
    with open(gone / QUOTES, "ab") as f:
        f.write(b"2,ou25,1.22\n")
    head = git("rev-parse", "af-prices-live", cwd=env["store"])
    assert run_push(env, gone, "g") == 0                                # rollback: the public commit is the record
    assert git("rev-parse", "af-prices-live", cwd=env["store"]) == head


def test_merge_lines_is_three_way():
    base = b"h\n1\n2\n"
    ours = b"h\n2\n3\n"                     # removed 1, added 3
    theirs = b"h\n1\n2\n4\n"                # someone added 4
    assert psl.merge_lines(base, ours, theirs) == b"h\n2\n4\n3\n"
    with pytest.raises(psl.MergeConflict):
        psl.merge_lines(base, b"x\n", theirs)


def test_sync_leaves_live_files_alone_on_main(env, monkeypatch):
    cfg = pss.load_config(env["cfg"])
    monkeypatch.setenv("PRICE_STORE_LIVE", "1")
    assert not pss.selected(cfg, "wowza-betting", QUOTES, "main")
    assert pss.selected(cfg, "wowza-betting", QUOTES, "data-truth-100")
    assert pss.selected(cfg, "wowza-betting", "output/bets.csv", "main")
    monkeypatch.setenv("PRICE_STORE_LIVE", "0")
    assert pss.selected(cfg, "wowza-betting", QUOTES, "main")


def test_repo_config_declares_live_classes_that_exist():
    cfg = psl.load()
    for repo, classes in (cfg.get("live") or {}).items():
        assert repo in cfg["rules"]
        for c in classes:
            assert c in cfg["rules"][repo]
    for repo, classes in (cfg.get("public_retired") or {}).items():
        assert set(classes) <= set((cfg.get("live") or {}).get(repo, []))


#: jobs that commit to the public repo but never read or write a live price file (reason each)
NO_PRICE_JOBS = {
    "af_history_extend.yml:extend": "API-Football fixture statistics backfill + team cards (no odds)",
    "af_usage_monitor.yml:monitor": "API call counter",
    "live_scanner.yml:live": "in-play scanner: own model fair odds only",
    "fantasy_refresh.yml:fantasy": "FPL data",
    "daily_injuries_lineups.yml:collect": "injuries / lineups / football-data raw / side stats collectors",
    "oneoff_dnp_recheck.yml:recheck": "one-off DNP void re-check (player results)",
    "data_truth_collect.yml:collect": "runs on data-truth-100, which keeps its own files",
    "oddsapi_history_probe.yml:probe": "runs on data-truth-100; credit plan counts only",
    "oddsapi_book_close.yml:capture": "sparse checkout; writes its own private store branch only",
}


def test_every_job_that_commits_opens_the_store_first():
    """A writer that ran without its history would write a truncated file: every job that pushes to
    the public repo opens the live store before its work (or is listed above with the reason it
    touches no price file), and every job that opens it with the write key pushes it back last."""
    import yaml
    wf_dir = ROOT / ".github" / "workflows"
    missing, bad = [], []
    for wf in sorted(wf_dir.glob("*.yml")):
        doc = yaml.safe_load(wf.read_text(encoding="utf-8"))
        for name, job in (doc.get("jobs") or {}).items():
            steps = [s for s in job.get("steps", []) if isinstance(s, dict)]
            runs = "\n".join(str(s.get("run", "")) for s in steps)
            if "price_store_live.sh open" in runs:
                k = next(i for i, s in enumerate(steps) if "price_store_live.sh open" in str(s.get("run", "")))
                ck = [i for i, s in enumerate(steps) if str(s.get("uses", "")).startswith("actions/checkout")]
                assert ck and ck[0] < k, f"{wf.name}:{name} opens the store before its checkout"
                if "PRICE_STORE_KEY" in (steps[k].get("env") or {}):
                    # the push follows the job's last public commit (final read-only steps such as
                    # update_results' CLV guard or a capture's NEAR-run chain may come after it)
                    pushes = [i for i, s in enumerate(steps) if "price_store_live.sh push" in str(s.get("run", ""))]
                    commits = [i for i, s in enumerate(steps)
                               if any(x in str(s.get("run", "")) for x in ("git push", "sync-push", "git_push"))]
                    if not pushes or (commits and pushes[-1] < commits[-1]) \
                            or str(steps[pushes[-1]].get("if", "")) != "${{ !cancelled() }}":
                        bad.append(f"{wf.name}:{name}")
                continue
            if "git push" in runs or "sync-push" in runs or "git_push" in runs:
                if f"{wf.name}:{name}" not in NO_PRICE_JOBS:
                    missing.append(f"{wf.name}:{name}")
    assert missing == [], missing
    assert bad == [], bad
