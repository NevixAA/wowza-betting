"""Private price store mirror (scripts/price_store_sync.py): synthetic repos only, no network,
no real prices. Each test builds a local "public" repo and a local "store" repo over file://."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import price_store_sync as pss  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")

ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
       "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}


def git(*a, cwd=None):
    import os
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True,
                          env={**os.environ, **ENV}).stdout.decode().strip()


def bare(path: Path) -> Path:
    git("init", "--quiet", "--bare", str(path))
    git("config", "uploadpack.allowFilter", "true", cwd=path)
    git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=path)
    return path


def commit_files(remote: Path, branch: str, files: dict, work: Path, remove=()):
    if not work.exists():
        git("init", "--quiet", "-b", branch, str(work))
        git("config", "core.autocrlf", "false", cwd=work)
        git("remote", "add", "origin", remote.as_uri(), cwd=work)
    for rel, data in files.items():
        p = work / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    for rel in remove:
        git("rm", "--quiet", rel, cwd=work)
    git("add", "-A", cwd=work)
    git("commit", "--quiet", "-m", "x", cwd=work)
    git("push", "--quiet", "origin", f"HEAD:{branch}", cwd=work)
    return git("rev-parse", "HEAD", cwd=work)


@pytest.fixture()
def env(tmp_path):
    pub = bare(tmp_path / "wowza-betting.git")
    store = bare(tmp_path / "store.git")
    commit_files(store, "af-prices-live", {".price_store": b"marker\n", ".gitattributes": b"* -text\n"}, tmp_path / "s_work")
    files = {
        "output/book_quotes/2026-11-17.csv": b"fixture_id,market,odds\n1,ou25,9.99\n",   # class A (synthetic)
        "output/bets.csv": b"fixture_id,odds\n1,9.99\r\n",                               # class B, CRLF kept
        "output/fd_history.parquet": b"PAR1notreally",                                  # class F: never selected
        "output/league_market_gate_research.csv": b"avg_odds\n1.5\n",                   # class C: never selected
        "README.md": b"hi\n",
    }
    head = commit_files(pub, "main", files, tmp_path / "p_work")
    cfg = {"store": {"repo": "x", "branch": "af-prices-live"}, "sources": [["wowza-betting", "main"]],
           "rules": {"wowza-betting": {"A_O": [r"^output/book_quotes/"], "B": [r"^output/bets\.csv$"]}}}
    cfgp = tmp_path / "cfg.json"
    cfgp.write_text(json.dumps(cfg))
    base = ["--store-url", store.as_uri(), "--public-url", (tmp_path / "{repo}.git").as_uri().replace("%7B", "{").replace("%7D", "}"),
            "--config", str(cfgp)]
    return dict(tmp=tmp_path, pub=pub, store=store, head=head, base=base)


def store_tree(store: Path, branch="af-prices-live"):
    out = git("ls-tree", "-r", branch, cwd=store)
    return {l.split("\t", 1)[1]: l.split()[2] for l in out.splitlines()}


def test_rules_compile_and_exclude_class_c_and_f():
    cfg = pss.load_config()
    assert pss.selected(cfg, "wowza-betting", "output/book_quotes/2026-11-17.csv")
    assert pss.selected(cfg, "wowza-betting", "output/bets_ledger.csv")
    assert pss.selected(cfg, "wowzaV9-Pro", "data/season_2026_27/scout_odds/dt=2026-11-17/part-0.parquet")
    assert pss.selected(cfg, "wowza_v11", "output/v11_shadow_log.csv")
    assert not pss.selected(cfg, "wowza-betting", "output/fd_history.parquet")
    assert not pss.selected(cfg, "wowza-betting", "output/league_market_gate_research.csv")
    assert not pss.selected(cfg, "wowza-betting", "src/pipeline.py")
    for repo, r in cfg["rules"].items():
        for p in r["A_O"] + r["B"]:
            re.compile(p)


def test_sync_mirrors_selected_bytes_and_check_is_equal(env, tmp_path):
    rc = pss.main(["sync", "--work", str(tmp_path / "w1"), *env["base"]])
    assert rc == 0
    t = store_tree(env["store"])
    pre = "wowza-betting/main/"
    pub_t = {l.split("\t", 1)[1]: l.split()[2] for l in git("ls-tree", "-r", "main", cwd=env["pub"]).splitlines()}
    assert t[pre + "output/book_quotes/2026-11-17.csv"] == pub_t["output/book_quotes/2026-11-17.csv"]
    assert t[pre + "output/bets.csv"] == pub_t["output/bets.csv"]           # same blob = same bytes (CRLF kept)
    assert pre + "output/fd_history.parquet" not in t
    assert pre + "output/league_market_gate_research.csv" not in t
    rec = json.loads(git("show", "af-prices-live:_sync/wowza-betting__main.json", cwd=env["store"]))
    assert rec["public_commit"] == env["head"]
    assert pss.main(["check", "--sha256", "--work", str(tmp_path / "c1"), *env["base"]]) == 0


def test_sync_follows_append_and_delete_and_is_idempotent(env, tmp_path):
    assert pss.main(["sync", "--work", str(tmp_path / "w1"), *env["base"]]) == 0
    commit_files(env["pub"], "main", {"output/book_quotes/2026-11-18.csv": b"fixture_id,market,odds\n2,ou25,1.11\n",
                                      "output/bets.csv": b"fixture_id,odds\n1,9.99\r\n2,1.11\r\n"},
                 env["tmp"] / "p_work", remove=["output/book_quotes/2026-11-17.csv"])
    before = git("rev-parse", "af-prices-live", cwd=env["store"])
    assert pss.main(["sync", "--work", str(tmp_path / "w1"), *env["base"]]) == 0
    t = store_tree(env["store"])
    assert "wowza-betting/main/output/book_quotes/2026-11-18.csv" in t
    assert "wowza-betting/main/output/book_quotes/2026-11-17.csv" not in t   # mirrored delete (history keeps it)
    mid = git("rev-parse", "af-prices-live", cwd=env["store"])
    assert mid != before
    assert pss.main(["sync", "--work", str(tmp_path / "w2"), *env["base"]]) == 0
    assert git("rev-parse", "af-prices-live", cwd=env["store"]) == mid        # nothing new -> no commit
    assert pss.main(["check", "--sha256", "--work", str(tmp_path / "c2"), *env["base"]]) == 0


def test_check_tolerates_lag_but_flags_a_wrong_store_byte(env, tmp_path):
    assert pss.main(["sync", "--work", str(tmp_path / "w1"), *env["base"]]) == 0
    # lag: public moved on after the last sync -> still EQUAL against the pinned commit
    commit_files(env["pub"], "main", {"output/bets.csv": b"fixture_id,odds\n1,9.99\r\n3,2.22\r\n"}, env["tmp"] / "p_work")
    rep = tmp_path / "r.json"
    assert pss.main(["check", "--work", str(tmp_path / "c1"), "--report", str(rep), *env["base"]]) == 0
    r = json.loads(rep.read_text())[0]
    assert r["status"] == "EQUAL" and r["lag_files_vs_public_head"] == 1
    # tamper with the store copy -> DIFFERENT, exit 1
    sw = env["tmp"] / "s_clone"
    git("clone", "--quiet", "--branch", "af-prices-live", env["store"].as_uri(), str(sw))
    (sw / "wowza-betting/main/output/bets.csv").write_bytes(b"tampered\n")
    git("commit", "--quiet", "-am", "tamper", cwd=sw)
    git("push", "--quiet", "origin", "HEAD:af-prices-live", cwd=sw)
    assert pss.main(["check", "--work", str(tmp_path / "c2"), *env["base"]]) == 1


def test_refuses_a_store_without_marker(env, tmp_path):
    other = bare(tmp_path / "other.git")
    commit_files(other, "af-prices-live", {"x.txt": b"x\n"}, tmp_path / "o_work")
    args = list(env["base"])
    args[1] = other.as_uri()
    assert pss.main(["sync", "--work", str(tmp_path / "w9"), *args]) == 1


def test_hydrate_is_a_noop_without_store_dir(tmp_path, monkeypatch, capsys):
    import price_store_hydrate as hyd
    monkeypatch.delenv("PRICE_STORE_DIR", raising=False)
    assert hyd.main(["--repo", "wowza-betting", "--branch", "main", "--workspace", str(tmp_path)]) == 0
    assert "public paths" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_hydrate_places_store_bytes_and_prunes(env, tmp_path):
    import price_store_hydrate as hyd
    assert pss.main(["sync", "--work", str(tmp_path / "w1"), *env["base"]]) == 0
    sd = tmp_path / "store_co"
    git("clone", "--quiet", "--branch", "af-prices-live", env["store"].as_uri(), str(sd))
    ws = tmp_path / "ws"
    git("-c", "core.autocrlf=false", "clone", "--quiet", "--branch", "main", env["pub"].as_uri(), str(ws))
    git("config", "core.autocrlf", "false", cwd=ws)
    (ws / "output/book_quotes/2026-11-17.csv").unlink()            # reader workspace without the public file
    (ws / "output/book_quotes/stray.csv").write_bytes(b"x\n")       # a selected file the store does not hold
    cfgp = Path(env["base"][-1])
    rep = tmp_path / "h.json"
    assert hyd.main(["--repo", "wowza-betting", "--branch", "main", "--workspace", str(ws), "--store-dir", str(sd),
                     "--config", str(cfgp), "--prune", "--verify", "--report", str(rep)]) == 0
    r = json.loads(rep.read_text())
    assert r["copied"] == 2 and r["pruned"] == 1
    assert r["differs_from_head"] == 0 and r["head_selected_missing"] == 0
    assert git("status", "--porcelain", cwd=ws) == ""                # byte-identical to the public commit
    assert (ws / "output/fd_history.parquet").read_bytes() == b"PAR1notreally"   # class F untouched


def _steps(wf: str):
    import yaml
    doc = yaml.safe_load((ROOT / ".github" / "workflows" / wf).read_text(encoding="utf-8"))
    return doc, next(iter(doc["jobs"].values()))["steps"]


@pytest.mark.parametrize("wf,first_work", [("phase2_retrain.yml", "Plan (dry run, committed data, no fitting)"),
                                           ("props_clean_retrain.yml", "Harness tests (synthetic data)")])
def test_refit_reads_the_store_only_behind_the_gate_and_before_any_work(wf, first_work):
    _, steps = _steps(wf)
    names = [s.get("name", "") for s in steps]
    i = next(k for k, n in enumerate(names) if n.startswith("Read prices from the private store"))
    s = steps[i]
    assert s["if"] == "vars.PRICE_STORE_READ == '1'"            # default off = public files
    assert "secrets.PRICE_STORE_RO_KEY" in s["env"]["PRICE_STORE_RO_KEY"]
    assert "scripts/price_store_read.sh wowza-betting" in s["run"]
    assert i < names.index(first_work)


def test_sync_workflow_is_gated_and_read_test_never_mirrors():
    doc, _ = _steps("price_store_sync.yml")
    mirror, read_test = doc["jobs"]["mirror"], doc["jobs"]["read_test"]
    assert "vars.PRICE_STORE_SYNC == '1'" in mirror["if"] and "github.event_name == 'schedule'" in mirror["if"]
    assert "inputs.mode != 'read-test'" in mirror["if"]
    assert read_test["if"] == "github.event_name == 'workflow_dispatch' && inputs.mode == 'read-test'"
    assert (ROOT / "scripts" / "price_store_read.sh").is_file()
