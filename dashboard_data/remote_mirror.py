"""Runtime data mirror for the SLIM dashboard deployment (owner-approved 2026-10-26).

WHY. Streamlit Community Cloud could no longer finish a fresh clone of the v9 repo (~710 MB of
history + ~619 MB working tree): the app log looped "Cloning repository..." every ~6.5 minutes.
The dashboard now deploys from a small GENERATED repo (NevixAA/wowza-dashboard, built by
scripts/build_dashboard_repo.py) that carries the dashboard CODE only. The DATA it renders -
the committed v9 output files listed in deploy/dashboard_manifest.json - is downloaded at runtime
from v9's public raw URLs into the SAME relative paths, so every page keeps reading local paths
exactly as it does in a full checkout. Nothing in the pages changes.

    from dashboard_data import remote_mirror
    remote_mirror.ensure_fresh()      # app.py, before navigation; cheap on every rerun

MODES
  * Not slim (developer checkout, the old full-repo deployment): every function is a no-op.
  * Slim (env WOWZA_SLIM=1, or the marker file deploy/SLIM exists; WOWZA_SLIM=0 forces it off):
      - first call in a process with no synced data yet: BLOCKS (up to COLD_BUDGET_S) on one full
        sync, because config.py and the pages read their files at import time;
      - afterwards: at most one sync per TTL (default 10 min, env WOWZA_MIRROR_TTL_MIN) per
        process, run in a BACKGROUND thread - a render never waits for it and stale data is served
        meanwhile.

HOW A SYNC STAYS CHEAP (raw.githubusercontent.com and the GitHub API are both rate-limited, and
Streamlit Cloud's egress IPs are shared with other tenants):
  1. One GitHub API call asks for main's current commit SHA. Unchanged since the last sync -> done,
     zero downloads. (Best effort only: the anonymous API allows 60 calls/hour per IP.)
  2. SHA moved and the compare API answers -> download only the manifest files that changed.
  3. API unavailable -> sweep every manifest file with a conditional GET (If-None-Match on the
     recorded ETag); unchanged files answer 304 with no body.
Downloads are pinned to one commit SHA when it is known, so a sync is a consistent snapshot.
Each file is written to a temp file and atomically renamed; on any failure the old copy stays.

FRESHNESS. Data freshness shown on the pages still comes from file CONTENT (dashboard_data.core),
never from mtime. What this module records - when it last synced, at which SHA, each file's ETag -
lives in deploy/.mirror_sync.json and is reported by status(); it is never used to judge whether
the DATA is current.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
MANIFEST = DEPLOY / "dashboard_manifest.json"
SLIM_MARKER = DEPLOY / "SLIM"
SYNC_FILE = DEPLOY / ".mirror_sync.json"

REPO = os.getenv("WOWZA_MIRROR_REPO", "NevixAA/wowza-betting")
BRANCH = os.getenv("WOWZA_MIRROR_BRANCH", "main")
RAW = f"https://raw.githubusercontent.com/{REPO}"
API = f"https://api.github.com/repos/{REPO}"
UA = {"User-Agent": "wowza-dashboard-mirror"}

TTL_S = float(os.getenv("WOWZA_MIRROR_TTL_MIN", "10")) * 60
COLD_BUDGET_S = float(os.getenv("WOWZA_MIRROR_COLD_BUDGET_S", "120"))
WORKERS = int(os.getenv("WOWZA_MIRROR_WORKERS", "8"))
TIMEOUT_S = 30

_LOCK = threading.Lock()
_STATE: dict = {"last_start": None, "thread": None, "last_result": None}


# ── mode + manifest ───────────────────────────────────────────────────────────────────────────
#: Present in a FULL v9 checkout, never in the slim tree. Mirroring into a full checkout would overwrite
#: tracked output files with main's copies, so slim mode is impossible there whatever the env says.
FULL_CHECKOUT_SENTINEL = ROOT / ".github" / "workflows" / "predict.yml"


def slim_mode() -> bool:
    if FULL_CHECKOUT_SENTINEL.exists():
        return False
    env = os.getenv("WOWZA_SLIM", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        return True
    return SLIM_MARKER.exists()


def manifest() -> dict:
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except Exception:                                                # noqa: BLE001
        return {}


def data_files() -> list[str]:
    """Relative repo paths the dashboard reads (deploy/dashboard_manifest.json -> data[].path)."""
    out = []
    for e in manifest().get("data", []):
        p = e["path"] if isinstance(e, dict) else str(e)
        if p and not p.startswith("/") and ".." not in Path(p).parts:
            out.append(p.replace("\\", "/"))
    return out


def _read_sync() -> dict:
    try:
        return json.loads(SYNC_FILE.read_text(encoding="utf-8"))
    except Exception:                                                # noqa: BLE001
        return {}


def _write_atomic(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(body)
    os.replace(tmp, path)


# ── network ───────────────────────────────────────────────────────────────────────────────────
def _get(url: str, headers: dict | None = None, timeout: float = TIMEOUT_S):
    """(status, body, headers). 304 and HTTP errors return their status, never raise."""
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, b"", dict(e.headers or {})


def _head_sha() -> str | None:
    try:
        st, body, _ = _get(f"{API}/commits/{BRANCH}", {"Accept": "application/vnd.github.sha"}, timeout=10)
        sha = body.decode().strip() if st == 200 else ""
        return sha if len(sha) == 40 else None
    except Exception:                                                # noqa: BLE001
        return None


def _changed_paths(old: str, new: str) -> set[str] | None:
    """Files changed between two commits, or None when the compare API cannot say."""
    try:
        st, body, _ = _get(f"{API}/compare/{old}...{new}", {"Accept": "application/vnd.github+json"},
                           timeout=15)
        if st != 200:
            return None
        d = json.loads(body)
        files = d.get("files") or []
        if len(files) >= 300:            # the compare API truncates at 300 files: cannot trust it
            return None
        out = set()
        for f in files:
            out.add(f.get("filename", ""))
            if f.get("previous_filename"):
                out.add(f["previous_filename"])
        return out
    except Exception:                                                # noqa: BLE001
        return None


def _fetch_one(rel: str, ref: str, etag: str | None) -> tuple[str, str, str | None]:
    """Download one file. Returns (rel, outcome, etag). Outcomes: updated | unchanged | missing |
    failed. Never raises; on failure the existing local copy is left untouched."""
    dest = ROOT / rel
    headers = {"If-None-Match": etag} if (etag and dest.exists()) else {}
    url = f"{RAW}/{ref}/" + urllib.request.quote(rel)
    try:
        st, body, hdrs = _get(url, headers)
    except Exception:                                                # noqa: BLE001
        return rel, "failed", etag
    if st == 304:
        return rel, "unchanged", etag
    if st == 404:
        return rel, "missing", None
    if st != 200:
        return rel, "failed", etag
    try:
        _write_atomic(dest, body)
    except Exception:                                                # noqa: BLE001
        return rel, "failed", etag
    return rel, "updated", hdrs.get("ETag") or hdrs.get("etag")


# ── sync ──────────────────────────────────────────────────────────────────────────────────────
def sync(force_full: bool = False) -> dict:
    """One mirror pass. Safe to call directly (scripts, tests); returns a summary dict."""
    t0 = time.time()
    files = data_files()
    prev = _read_sync()
    etags: dict = dict(prev.get("etags", {}))
    sha = _head_sha()
    old_sha = prev.get("sha")
    have_all = all((ROOT / f).exists() or prev.get("missing", {}).get(f) for f in files)

    targets = files
    mode = "full"
    if sha and old_sha == sha and have_all and not force_full:
        targets, mode = [], "unchanged-sha"
    elif sha and old_sha and have_all and not force_full:
        changed = _changed_paths(old_sha, sha)
        if changed is not None:
            targets = [f for f in files if f in changed]
            mode = "compare"
    ref = sha or BRANCH

    counts = {"updated": 0, "unchanged": 0, "missing": 0, "failed": 0}
    missing = dict(prev.get("missing", {})) if mode != "full" else {}
    if targets:
        with ThreadPoolExecutor(max_workers=max(1, WORKERS)) as ex:
            for rel, outcome, etag in ex.map(lambda f: _fetch_one(f, ref, etags.get(f)), targets):
                counts[outcome] += 1
                if outcome == "missing":
                    missing[rel] = True
                    etags.pop(rel, None)
                else:
                    missing.pop(rel, None)
                    if etag:
                        etags[rel] = etag
    # A sync that could not reach anything keeps the previous SHA, so the next one retries in full.
    ok = counts["failed"] == 0
    rec = {
        "synced_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) if ok else prev.get("synced_at_utc"),
        "attempted_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sha": (sha if ok and sha else (old_sha if not ok else None)),
        "mode": mode,
        "counts": counts,
        "files": len(files),
        "seconds": round(time.time() - t0, 2),
        "etags": etags,
        "missing": missing,
    }
    try:
        _write_atomic(SYNC_FILE, json.dumps(rec, indent=1).encode("utf-8"))
    except Exception:                                                # noqa: BLE001
        pass
    out = {k: v for k, v in rec.items() if k not in ("etags", "missing")}
    out["missing_files"] = sorted(missing)
    _STATE["last_result"] = out
    return out


def _run_bg() -> None:
    try:
        sync()
    except Exception as e:                                           # noqa: BLE001 - never kill the app
        _STATE["last_result"] = {"error": str(e)[:300]}


def ensure_fresh(ttl_s: float | None = None, cold_budget_s: float | None = None) -> dict:
    """Call on every rerun. No-op outside slim mode. Never raises.

    Returns status(). The first call in a process with no previous sync waits (up to the cold
    budget) for a full sync; later calls start at most one background sync per TTL."""
    if not slim_mode():
        return {"slim": False}
    ttl = TTL_S if ttl_s is None else ttl_s
    budget = COLD_BUDGET_S if cold_budget_s is None else cold_budget_s
    with _LOCK:
        th = _STATE["thread"]
        running = th is not None and th.is_alive()
        last = _STATE["last_start"]
        due = last is None or (time.time() - last) >= ttl
        if due and not running:
            _STATE["last_start"] = time.time()
            th = threading.Thread(target=_run_bg, name="wowza-mirror", daemon=True)
            _STATE["thread"] = th
            th.start()
            running = True
    if running and not _read_sync().get("synced_at_utc"):
        th.join(timeout=budget)          # cold start: the pages need their files to exist at all
    return status()


def status() -> dict:
    prev = _read_sync()
    th = _STATE["thread"]
    return {
        "slim": slim_mode(),
        "synced_at_utc": prev.get("synced_at_utc"),
        "attempted_at_utc": prev.get("attempted_at_utc"),
        "sha": prev.get("sha"),
        "mode": prev.get("mode"),
        "counts": prev.get("counts"),
        "files": prev.get("files"),
        "running": bool(th is not None and th.is_alive()),
    }
