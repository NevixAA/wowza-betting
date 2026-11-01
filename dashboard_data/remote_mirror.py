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

CODE/DATA SKEW (R2 finding 32, 2026-11-01). The slim app's CODE is whatever build was last pushed;
its DATA is v9 main at the mirrored SHA. The two can drift apart (pages changed in v9 but the slim
repo was not rebuilt, or a page needs a file the old build never mirrors). compatibility() compares
  * the build's deploy/BUILD_INFO.json (dashboard build id = digest of the code files, source v9 SHA,
    schema version, the manifest it was built from) with
  * v9's deploy/dashboard_manifest.json AT THE MIRRORED DATA SHA (downloaded by sync() into
    deploy/.v9_manifest_at_data.json),
and returns COMPATIBLE / STALE_BUILD / INCOMPATIBLE / UNKNOWN / NOT_SYNCED / BUILD_INFO_MISSING
(LOCAL_CHECKOUT outside slim mode). Only COMPATIBLE and LOCAL_CHECKOUT are healthy: an unknown or
incompatible schema is never shown as healthy. Bump DASHBOARD_SCHEMA_VERSION whenever a page starts
depending on a data shape older data cannot satisfy.

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
BUILD_INFO = DEPLOY / "BUILD_INFO.json"
V9_MANIFEST_COPY = DEPLOY / ".v9_manifest_at_data.json"
V9_MANIFEST_REL = "deploy/dashboard_manifest.json"

#: The data contract between dashboard code and v9's mirrored files. Written into the manifest by
#: scripts/dashboard_manifest.py and into BUILD_INFO.json by scripts/build_dashboard_repo.py.
DASHBOARD_SCHEMA_VERSION = 1

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
    changed = None
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
    # The DATA side's own manifest, at the same ref, for the code/data compatibility check. Fetched on
    # a full sync, or when the compare API says it changed. Never counted against `ok` (the pages do
    # not read it) and never written over the BUILD's own manifest.
    v9m = dict(prev.get("v9_manifest") or {})
    if mode == "full" or (mode == "compare" and V9_MANIFEST_REL in set(changed or ())):
        _r, outcome, _e = _fetch_manifest_copy(ref)
        v9m = {"outcome": outcome, "ref": ref, "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
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
        "v9_manifest": v9m,
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
        "compat": compatibility(),
    }


# ── code/data compatibility (R2 finding 32) ───────────────────────────────────────────────────
def _fetch_manifest_copy(ref: str) -> tuple[str, str, str | None]:
    """Download v9's dashboard manifest at `ref` into V9_MANIFEST_COPY (never over MANIFEST)."""
    url = f"{RAW}/{ref}/" + urllib.request.quote(V9_MANIFEST_REL)
    try:
        st, body, _h = _get(url)
    except Exception:                                                # noqa: BLE001
        return V9_MANIFEST_REL, "failed", None
    if st != 200:
        return V9_MANIFEST_REL, ("missing" if st == 404 else "failed"), None
    try:
        json.loads(body)
        _write_atomic(V9_MANIFEST_COPY, body)
    except Exception:                                                # noqa: BLE001
        return V9_MANIFEST_REL, "failed", None
    return V9_MANIFEST_REL, "updated", None


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                                # noqa: BLE001
        return None


def _paths(entries) -> set[str]:
    return {(e["path"] if isinstance(e, dict) else str(e)) for e in (entries or [])}


def _code_sig(entries) -> set[tuple]:
    return {((e.get("path"), e.get("bytes")) if isinstance(e, dict) else (str(e), None))
            for e in (entries or [])}


HEALTHY = ("COMPATIBLE", "LOCAL_CHECKOUT")


def compatibility() -> dict:
    """Dashboard build vs mirrored data. Never raises. `healthy` is True only for COMPATIBLE /
    LOCAL_CHECKOUT; everything else carries a human-readable warning."""
    if not slim_mode():
        return {"status": "LOCAL_CHECKOUT", "healthy": True, "schema_version": DASHBOARD_SCHEMA_VERSION,
                "warning": None}
    b = _read_json(BUILD_INFO) or {}
    s = _read_json(SYNC_FILE) or {}
    v = _read_json(V9_MANIFEST_COPY)
    own = _read_json(MANIFEST) or {}
    rec = {
        "dashboard_build_id": b.get("dashboard_build_id"),
        "built_at_utc": b.get("built_at_utc"),
        "source_v9_sha": b.get("v9_head_at_build"),
        "manifest_v9_commit": b.get("v9_commit_of_manifest"),
        "build_schema_version": b.get("schema_version"),
        "code_schema_version": DASHBOARD_SCHEMA_VERSION,
        "mirrored_data_sha": s.get("sha"),
        "data_manifest_v9_commit": (v or {}).get("v9_commit"),
        "data_schema_version": (v or {}).get("schema_version"),
        "last_successful_sync_utc": s.get("synced_at_utc"),
        "last_attempted_sync_utc": s.get("attempted_at_utc"),
    }

    def done(status, warning, **extra):
        rec.update(status=status, healthy=status in HEALTHY, warning=warning, **extra)
        return rec

    if not b:
        return done("BUILD_INFO_MISSING", "deploy/BUILD_INFO.json is missing: cannot tell which code "
                    "build is running. Rebuild the slim repo.")
    if not s.get("synced_at_utc"):
        return done("NOT_SYNCED", "No successful data sync yet: pages may show no data.")
    if v is None:
        return done("UNKNOWN", "v9's dashboard manifest at the mirrored data SHA could not be read, so "
                    "code/data compatibility is UNKNOWN (not assumed healthy).")
    schemas = {rec["build_schema_version"], rec["code_schema_version"], rec["data_schema_version"]}
    if len(schemas) != 1:
        return done("INCOMPATIBLE", f"Dashboard schema mismatch: build {rec['build_schema_version']}, "
                    f"code {rec['code_schema_version']}, data {rec['data_schema_version']}. Pages may "
                    "misread the data. Rebuild and push the slim repo from current v9.")
    data_new = sorted(_paths(v.get("data")) - _paths(own.get("data")))
    data_gone = sorted(_paths(own.get("data")) - _paths(v.get("data")))
    code_diff = sorted({p for p, _ in (_code_sig(v.get("code")) ^ _code_sig(own.get("code")))})
    if data_new or data_gone or code_diff:
        return done("STALE_BUILD", f"The deployed dashboard code is older than v9's dashboard at the "
                    f"mirrored SHA ({len(code_diff)} code file(s) differ, {len(data_new)} data file(s) "
                    f"new, {len(data_gone)} dropped). Rebuild and push the slim repo.",
                    code_files_differ=code_diff[:20], data_files_new=data_new[:20],
                    data_files_dropped=data_gone[:20])
    return done("COMPATIBLE", None)
