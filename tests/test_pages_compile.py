"""Every dashboard page must at least compile (2026-10-22: a page with a syntax error passed the suite)."""
import py_compile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted(p for p in (ROOT / "pages").glob("*.py")) + [ROOT / "app.py"]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_page_compiles(page):
    py_compile.compile(str(page), doraise=True)
