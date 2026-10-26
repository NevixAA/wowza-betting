"""What the Streamlit dashboard needs from this repo: deploy/dashboard_manifest.json.

    python scripts/dashboard_manifest.py --write     # trace every page (slow, ~3 min) and rewrite
    python scripts/dashboard_manifest.py --check     # trace and fail if the manifest is missing anything
    python scripts/dashboard_manifest.py --static    # fast static check only (what the unit test runs)

WHY. The dashboard deploys from a SLIM generated repo (scripts/build_dashboard_repo.py) that holds
only the files listed here; its data is mirrored at runtime from v9's raw URLs
(dashboard_data/remote_mirror.py). A page that imports a module or reads a file that is not in the
manifest works on a developer checkout and silently breaks the deployment. So:

  CODE = the runtime import graph of app.py + every page (AppTest render, every tab, widget
         interactions) UNION the static import closure (imports inside functions and button branches
         included), plus the non-Python assets the app needs (.streamlit/config.toml).
  DATA = every repo file the pages open / stat / read_parquet while rendering, found with an audit hook
         (sys.addaudithook 'open') + os.stat + pandas/pyarrow wrappers. Files probed but absent today
         are kept as optional (the mirror asks for them; a 404 is fine). Secrets files (.env,
         .api_keys) are never data. A file too big to mirror (or a directory such as
         player_history.parquet) goes to `excluded` with the reason, and the page must cope without it.

The trace runs in a child process because an audit hook can never be removed.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "deploy" / "dashboard_manifest.json"

#: Non-Python files the slim tree must carry.
ASSETS = [".streamlit/config.toml"]
#: The deploy machinery itself (the mirror is imported by app.py; the builder ships for provenance).
ALWAYS_CODE = ["app.py", "dashboard_data/remote_mirror.py"]
#: Never mirrored, whatever reads them.
SECRET_NAMES = {".env", ".api_keys", "secrets.toml"}
#: Paths streamlit/python probe that are not dashboard data.
NOISE_PREFIXES = (".agents", ".claude", ".codex", ".cortex", ".cursor", ".opencode", ".git/", ".streamlit",
                  "deploy/",
                  "dashboard_data/_remote_cache", "pages/pages", "__pycache__")
DATA_EXT = (".csv", ".json", ".parquet", ".txt", ".jsonl", ".xlsx", ".pkl", ".yml", ".yaml", ".md", ".png",
            ".svg", ".jpg", ".toml")
#: Largest single file the mirror will fetch. Above this a page must degrade instead.
MAX_DATA_BYTES = 25_000_000

#: Data the dashboard can reach but the slim deployment deliberately does NOT mirror, and why.
#: The static test accepts literals naming these; the page in question must degrade gracefully.
KNOWN_EXCLUDED = {
    "player_history.parquet": "directory of season parts, ~173 MB, ~1.35 GB in RAM; only the on-demand "
                              "Fantasy 'Load club squads' button reads it (dashboard_fantasy_tools.club_squads), "
                              "which the slim app disables",
}


# ── static analysis ─────────────────────────────────────────────────────────────────────────────
def entry_files(root: Path = ROOT) -> list[str]:
    sys.path.insert(0, str(root))
    try:
        import dashboard_access  # noqa: WPS433
        pages = [p for p, *_ in dashboard_access.PAGES]
    finally:
        sys.path.pop(0)
    return ["app.py"] + pages


def _module_file(root: Path, dotted: str) -> list[Path]:
    """Repo files for a dotted module name: the module (file or package __init__) + parent __init__s."""
    parts = dotted.split(".")
    out = []
    for i in range(1, len(parts) + 1):
        base = root.joinpath(*parts[:i])
        if (base / "__init__.py").is_file():
            out.append(base / "__init__.py")
        elif i == len(parts) and base.with_suffix(".py").is_file():
            out.append(base.with_suffix(".py"))
        elif i < len(parts) and not base.is_dir():
            return []
    return out


def _imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:                                                # noqa: BLE001
        return set()
    mods = set()
    pkg = path.parent
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:                                           # relative import
                base = pkg
                for _ in range(node.level - 1):
                    base = base.parent
                rel = base.relative_to(ROOT).parts if ROOT in base.parents or base == ROOT else ()
                prefix = ".".join(rel)
                name = ".".join(x for x in (prefix, node.module or "") if x)
            else:
                name = node.module or ""
            if name:
                mods.add(name)
                mods.update(f"{name}.{a.name}" for a in node.names)   # "from pkg import submodule"
        elif isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "import_module" and node.args:
            a = node.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                mods.add(a.value)
    return mods


def static_closure(root: Path = ROOT, entries: list[str] | None = None) -> set[str]:
    """Every repo .py file reachable by import statements (top level AND inside functions)."""
    entries = entries or entry_files(root)
    seen: set[Path] = set()
    todo = [root / e for e in entries]
    while todo:
        f = todo.pop()
        if f in seen or not f.is_file():
            continue
        seen.add(f)
        for m in _imports(f):
            for p in _module_file(root, m):
                if p not in seen:
                    todo.append(p)
    return {p.relative_to(root).as_posix() for p in seen}


VIEWER_LAYER = ("app.py", "pages/", "dashboard_", "dashboard_data/", "utils/")


def data_literals(root: Path, code: list[str]) -> set[str]:
    """String constants that look like data file names, in the viewer layer (pages, dashboard_*,
    dashboard_data, utils). Deeper modules (config.py, src/) name hundreds of production files the
    dashboard never touches, so they are left to the trace."""
    out = set()
    for rel in code:
        if not rel.startswith(VIEWER_LAYER) or rel == "dashboard_data/remote_mirror.py":
            continue
        try:
            tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        except Exception:                                            # noqa: BLE001
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                s = node.value.strip()
                if (s.lower().endswith(DATA_EXT) and " " not in s and "{" not in s and "*" not in s
                        and not s.startswith(("http", "/", ".")) and len(s) < 120):
                    out.add(s.replace("\\", "/"))
    return out


#: Search roots for a bare literal name such as "fd_history.parquet" (code writes OUT / "name").
_LITERAL_DIRS = ("", "output/", "config/", "models/", "registry/")


def resolve_literal(lit: str, root: Path = ROOT, tracked: set[str] | None = None) -> list[str]:
    """v9 files a literal data name can refer to. [] means it names no v9 file (a Pro / v11 file read
    over HTTP, a download filename, a workflow mentioned in help text) and is not v9 data."""
    hits = [d + lit for d in _LITERAL_DIRS if (root / (d + lit)).is_file()]
    if not hits and tracked:
        hits = sorted(t for t in tracked if t.endswith("/" + lit))
    return [h for h in hits if not _is_noise(h)]


def unmatched_literals(manifest: dict, root: Path = ROOT) -> list[str]:
    """Literal data names in the viewer layer that resolve to a v9 file the manifest does not carry
    (as data or as a deliberate exclusion). Used by the fast static test."""
    code = [e["path"] for e in manifest["code"]]
    known = {e["path"] for e in manifest["data"]} | {e["path"] for e in manifest.get("excluded", [])}
    bad = []
    for lit in sorted(data_literals(root, code)):
        hits = resolve_literal(lit, root)
        if any(h in known for h in hits):
            continue
        if hits:
            bad.append(f"{lit} -> {hits}")
    return bad


# ── runtime trace (child process) ───────────────────────────────────────────────────────────────
_TRACER = r'''
import json, os, sys, time, traceback
from pathlib import Path
ROOT = Path(sys.argv[1]).resolve(); OUT = Path(sys.argv[2])
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
os.environ["WOWZA_TIER_OVERRIDE"] = "admin"
ROOT_S = str(ROOT).lower()
cur = {"page": "<boot>"}
reads, blocked = {}, []

def _rel(p):
    try:
        if isinstance(p, int): return None
        if isinstance(p, bytes): p = p.decode("utf-8", "ignore")
        a = os.path.abspath(os.fspath(p))
    except Exception:
        return None
    if not a.lower().startswith(ROOT_S): return None
    return a[len(ROOT_S):].lstrip("\\/").replace("\\", "/")

def _note(rel, kind):
    d = reads.setdefault(rel, {"pages": set(), "kinds": set()})
    d["pages"].add(cur["page"]); d["kinds"].add(kind)

def hook(event, args):
    if event == "open":
        rel = _rel(args[0])
        if rel is None: return
        mode = args[1] or "r"
        if not (isinstance(mode, str) and any(c in mode for c in "wax+")):
            _note(rel, "open")
    elif event == "subprocess.Popen":
        # The admin buttons run pipeline.py / live_scanner.py: never during a trace.
        if ".py" in repr(args[1]) and cur["page"] != "<boot>":
            blocked.append([cur["page"], repr(args[1])[:300]])
            raise RuntimeError("subprocess blocked by dashboard_manifest trace")
sys.addaudithook(hook)

_stat = os.stat
def stat(path, *a, **k):
    rel = _rel(path) if not isinstance(path, int) else None
    if rel is not None and not rel.endswith((".py", ".pyc")) and "__pycache__" not in rel:
        _note(rel, "stat")
    return _stat(path, *a, **k)
os.stat = stat

import pandas as pd
_rp = pd.read_parquet
def read_parquet(path, *a, **k):
    rel = _rel(path)
    if rel is not None: _note(rel, "parquet")
    return _rp(path, *a, **k)
pd.read_parquet = read_parquet

from streamlit.testing.v1 import AppTest
import dashboard_access
pages = ["app.py"] + [p for p, *_ in dashboard_access.PAGES]
res = {}
MAXI, MAXO, BUDGET = 60, 5, 300.0
for page in pages:
    cur["page"] = page; t0 = time.time()
    info = {"runs": 0, "exceptions": [], "errors": []}
    try:
        at = AppTest.from_file(str(ROOT / page), default_timeout=240); at.run(); info["runs"] += 1
        info["exceptions"] += [str(e.value)[:300] for e in at.exception]
        n = 0
        for kind in ("selectbox", "radio", "segmented_control", "pills", "checkbox", "toggle", "multiselect", "button"):
            try: count = len(at.get(kind))
            except Exception: continue
            for i in range(count):
                if n >= MAXI or time.time() - t0 > BUDGET: break
                try:
                    w = at.get(kind)[i]
                    if kind in ("selectbox", "radio", "segmented_control", "pills"):
                        for o in list(getattr(w, "options", []) or [])[:MAXO]:
                            try:
                                at.get(kind)[i].set_value(o); at.run(); n += 1
                            except Exception as e:
                                info["errors"].append(f"{kind}[{i}]={o}: {e}"[:200]); continue
                            info["exceptions"] += [str(e.value)[:300] for e in at.exception]
                    elif kind in ("checkbox", "toggle"):
                        w.set_value(not bool(w.value)); at.run(); n += 1
                    elif kind == "multiselect":
                        if w.options: w.set_value(list(w.options)[:3]); at.run(); n += 1
                    else:
                        w.click(); at.run(); n += 1
                    info["exceptions"] += [str(e.value)[:300] for e in at.exception]
                except Exception as e:
                    info["errors"].append(f"{kind}[{i}]: {e}"[:200])
        info["runs"] += n
    except Exception as e:
        info["errors"].append("RUN FAILED: " + "".join(traceback.format_exception_only(type(e), e))[:500])
    info["seconds"] = round(time.time() - t0, 1)
    res[page] = info
    print(f"{page}: runs={info['runs']} exceptions={len(info['exceptions'])} {info['seconds']}s", flush=True)

mods, third = [], set()
for name, m in list(sys.modules.items()):
    f = getattr(m, "__file__", None)
    if not f: continue
    rel = _rel(f)
    if rel is not None and ".venv" not in rel:
        mods.append(rel)
    elif "site-packages" in f:
        third.add(name.split(".")[0])
OUT.write_text(json.dumps({"pages": res, "modules": sorted(set(mods)), "third_party": sorted(third),
    "reads": {k: {"pages": sorted(v["pages"]), "kinds": sorted(v["kinds"])} for k, v in reads.items()},
    "blocked": blocked}, indent=1, default=str), encoding="utf-8")
'''


def run_trace(root: Path = ROOT) -> dict:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        script, out = Path(td) / "trace.py", Path(td) / "trace.json"
        script.write_text(_TRACER, encoding="utf-8")
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "WOWZA_SLIM": "0"}
        r = subprocess.run([sys.executable, str(script), str(root), str(out)], env=env, timeout=3600)
        if r.returncode != 0 or not out.exists():
            raise SystemExit(f"trace failed (exit {r.returncode})")
        return json.loads(out.read_text(encoding="utf-8"))


def _git(root: Path, *args) -> str:
    git = os.getenv("WOWZA_GIT", "git")
    try:
        return subprocess.run([git, *args], cwd=root, capture_output=True, text=True, encoding="utf-8",
                              timeout=120).stdout
    except Exception:                                                # noqa: BLE001
        return ""


def _is_noise(rel: str) -> bool:
    parts = rel.split("/")
    # Streamlit probes dot-directories (.claude/, .agents/, .git, pages/.cursor ...) for its own config.
    # .github/ is not noise: it is handled as an exclusion in build_manifest.
    dotted = any(c.startswith(".") and not (i == 0 and c == ".github") for i, c in enumerate(parts))
    return (not rel or dotted or rel.startswith(NOISE_PREFIXES) or "/__pycache__/" in rel
            or parts[-1] in SECRET_NAMES or rel.endswith((".py", ".pyc")) or "/skills/" in rel)


def build_manifest(trace: dict, root: Path = ROOT) -> dict:
    tracked = set(_git(root, "ls-files", "-z").split("\0")) - {""}
    code = set(ALWAYS_CODE) | set(ASSETS) | static_closure(root)
    code |= {m for m in trace["modules"] if m.endswith(".py")}
    data, excluded = [], []
    for rel, v in sorted(trace["reads"].items()):
        if _is_noise(rel) or rel in code:
            continue
        p = root / rel
        if rel.startswith(".github/"):
            # Only src.research_lock's live check reads the workflows (Overview health). The slim app shows
            # the lock result CI recorded in output/system_health.json instead (src.system_health).
            if p.is_file():
                excluded.append({"path": rel, "bytes": p.stat().st_size,
                                 "reason": "workflow file: read only by the live research-lock check, which the "
                                           "slim app replaces with the CI-recorded result"})
            continue
        if p.is_dir():
            if any(k == "parquet" for k in v["kinds"]):
                excluded.append({"path": rel, "reason": KNOWN_EXCLUDED.get(rel, "directory read at render"),
                                 "bytes": sum(f.stat().st_size for f in p.rglob("*") if f.is_file())})
            continue
        if not p.exists():
            data.append({"path": rel, "bytes": None, "optional": True, "pages": v["pages"]})
            continue
        size = p.stat().st_size
        if rel in KNOWN_EXCLUDED or size > MAX_DATA_BYTES:
            excluded.append({"path": rel, "bytes": size, "reason": KNOWN_EXCLUDED.get(rel, "too big to mirror")})
            continue
        if tracked and rel not in tracked:
            excluded.append({"path": rel, "bytes": size, "reason": "not committed to v9 (raw URL cannot serve it)"})
            continue
        data.append({"path": rel, "bytes": size, "pages": v["pages"]})
    # Branches the trace could not reach (radios with format_func, functions behind rarely-true
    # conditions): every literal data name in the viewer layer that resolves to a committed v9 file.
    have = {e["path"] for e in data} | {e["path"] for e in excluded}
    for lit in sorted(data_literals(root, sorted(code))):
        for rel in resolve_literal(lit, root, tracked):
            if rel in have or rel in code:
                continue
            size = (root / rel).stat().st_size
            if tracked and rel not in tracked:
                continue
            if rel in KNOWN_EXCLUDED or size > MAX_DATA_BYTES:
                excluded.append({"path": rel, "bytes": size, "reason": KNOWN_EXCLUDED.get(rel, "too big to mirror")})
            else:
                data.append({"path": rel, "bytes": size, "pages": ["<static literal: " + lit + ">"]})
            have.add(rel)
    data.sort(key=lambda e: e["path"])
    code_e = [{"path": c, "bytes": (root / c).stat().st_size} for c in sorted(code) if (root / c).is_file()]
    return {
        "about": "Files the Wowza dashboard needs. GENERATED by scripts/dashboard_manifest.py --write; consumed by "
                 "scripts/build_dashboard_repo.py (code) and dashboard_data/remote_mirror.py (data). Do not hand-edit.",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "v9_commit": _git(root, "rev-parse", "HEAD").strip() or None,
        "code": code_e,
        "data": data,
        "excluded": excluded,
        "third_party_imported": trace.get("third_party", []),
        "totals": {"code_files": len(code_e), "code_bytes": sum(e["bytes"] for e in code_e),
                   "data_files": len(data), "data_bytes": sum(e["bytes"] or 0 for e in data)},
        "trace": {p: {k: i[k] for k in ("runs", "seconds")} | {"exceptions": len(i["exceptions"])}
                  for p, i in trace["pages"].items()},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--static", action="store_true")
    a = ap.parse_args(argv)
    cur = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else None
    if a.static:
        if cur is None:
            print("no manifest"); return 1
        have = {e["path"] for e in cur["code"]}
        miss_code = sorted(static_closure() - have)
        bad = unmatched_literals(cur)
        print(f"static: {len(miss_code)} code file(s) missing {miss_code}; {len(bad)} unmatched data literal(s) {bad}")
        return 1 if (miss_code or bad) else 0
    trace = run_trace()
    new = build_manifest(trace)
    errs = {p: i["exceptions"] for p, i in trace["pages"].items() if i["exceptions"]}
    if trace["blocked"]:
        print("blocked subprocesses (expected for admin buttons):", trace["blocked"])
    if errs:
        print("page exceptions during trace:", json.dumps(errs, ensure_ascii=False)[:2000])
    if a.write:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(new, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        t = new["totals"]
        print(f"wrote {MANIFEST}: {t['code_files']} code files ({t['code_bytes'] / 1e6:.2f} MB), "
              f"{t['data_files']} data files ({t['data_bytes'] / 1e6:.2f} MB), {len(new['excluded'])} excluded")
        return 0
    have_code = {e["path"] for e in (cur or {}).get("code", [])}
    have_data = {e["path"] for e in (cur or {}).get("data", [])} | {e["path"] for e in (cur or {}).get("excluded", [])}
    miss_c = sorted({e["path"] for e in new["code"]} - have_code)
    miss_d = sorted({e["path"] for e in new["data"]} - have_data)
    print(f"check: missing code {miss_c}; missing data {miss_d}")
    return 1 if (miss_c or miss_d) else 0


if __name__ == "__main__":
    sys.exit(main())
