"""Read-only access to the PRIVATE price store for admin pages (owner 2026-11-22).

Prices (API-Football / OddsAPI) may not be published (docs/AF_PRICES_PRIVATE_MIGRATION.md). The public repos are
being cut over to the private store NevixAA/wowza-exec, branch af-prices-live, where price_store_sync mirrors each
public file at <repo>/<branch>/<path> (config/price_store_paths.json). An ADMIN page reads a price from there,
server-side, through the GitHub contents API with the read-only Streamlit secret:

    PRICE_STORE_RO_TOKEN = "<fine-grained PAT, Contents: read on wowza-exec only>"

(locally: env PRICE_STORE_RO_TOKEN). The token never reaches the page, a log or an error message, and nothing read
here is ever written to a file. While a public copy still exists it is the fallback; when neither can be read the
page says "odds: private" and why.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATHS = ROOT / "config" / "price_store_paths.json"
API = "https://api.github.com"
SECRET = "PRICE_STORE_RO_TOKEN"
DEFAULT_STORE = ("NevixAA/wowza-exec", "af-prices-live")


class StoreUnavailable(RuntimeError):
    """A store read failed. The message never carries the token."""


def token() -> str:
    tok = None
    try:
        import streamlit as st
        tok = st.secrets.get(SECRET)
    except Exception:                                                  # noqa: BLE001 - no secrets file
        tok = None
    return str(tok or os.getenv(SECRET) or "").strip()


def store() -> tuple[str, str]:
    try:
        s = json.loads(PATHS.read_text(encoding="utf-8")).get("store") or {}
        if s.get("repo") and s.get("branch"):
            return str(s["repo"]), str(s["branch"])
    except (OSError, ValueError):
        pass
    return DEFAULT_STORE


def store_path(repo: str, branch: str, rel: str) -> str:
    """The store layout of price_store_sync: <repo>/<branch with / as __>/<path>."""
    return f"{repo}/{branch.replace('/', '__')}/{rel}"


def get(path: str, *, repo: str | None = None, ref: str | None = None, tok: str | None = None,
        timeout: int = 20) -> bytes:
    """GET one file of a private repo (raw). Raises StoreUnavailable with a token-free reason."""
    tok = token() if tok is None else tok
    if not tok:
        raise StoreUnavailable("token missing")
    srepo, sref = store()
    repo, ref = repo or srepo, ref or sref
    url = (f"{API}/repos/{repo}/contents/{urllib.parse.quote(path)}?ref={urllib.parse.quote(ref, safe='')}")
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github.raw", "Authorization": f"Bearer {tok}",
        "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "wowza-dashboard-admin"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        hint = {401: "token invalid or expired", 403: "token may not read this repo",
                404: "not in the private store yet"}.get(exc.code, f"HTTP {exc.code}")
        raise StoreUnavailable(hint) from None
    except urllib.error.URLError as exc:
        raise StoreUnavailable(f"network error ({exc.reason})") from None
    except Exception as exc:                                           # noqa: BLE001
        raise StoreUnavailable(type(exc).__name__) from None


def public_file_text(source_repo: str, rel: str) -> str:
    """The store copy of a public v9 file: <source_repo>/main/<rel> on the store branch."""
    return get(store_path(source_repo, "main", rel)).decode("utf-8")
