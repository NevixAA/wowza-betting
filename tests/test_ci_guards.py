"""CI guards (Phase 2, KEY=ci, 2026-11-05): the suite is offline and leaves the repo untouched, and
nothing rewrites the history banks in place.

Owner rule "corrections go beside the originals": output/fd_history.parquet and
output/af_history.parquet are provider banks. A correction goes to a sidecar
(output/corrections/, output/collect/...), never into the bank. The only code allowed to write a bank
is the provider ingestion that builds it; a new writer anywhere else fails here.

tests/conftest.py adds the run-time half: every non-loopback socket is refused, and the session
fails if the run changed any file git can see (both banks included).

    python -m pytest tests/test_ci_guards.py -q
"""
from __future__ import annotations

import ast
import socket
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BANKS = ("fd_history.parquet", "af_history.parquet")
#: the provider ingestion that BUILDS each bank - the only writers allowed. Anything else that
#: corrects a bank row writes a sidecar beside it.
BANK_WRITERS = {
    "src/data_loader.py": "football-data.co.uk download cache (fd_history): provider rows only",
    "scripts/backfill_af_history.py": "API-Football fixture backfill (af_history): provider rows only",
}
_READERS = ("read_parquet", "read_csv", "read_table", "ParquetFile")


def _bank_writes(src: str) -> list[int]:
    """Line numbers of calls that write a bank path: to_parquet / write_bytes / write_text on it, or
    a move / copy / replace onto it. A name counts as a bank path when it is bound to an expression
    naming a bank file (a frame READ from a bank is not a path)."""
    tree = ast.parse(src)
    seg = lambda n: ast.get_source_segment(src, n) or ""
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
            v = n.value
            if isinstance(v, ast.Call) and isinstance(v.func, (ast.Attribute, ast.Name)) and \
                    getattr(v.func, "attr", getattr(v.func, "id", "")) in _READERS:
                continue
            if any(b in seg(v) for b in BANKS):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    if isinstance(t, ast.Name):
                        names.add(t.id)

    def is_bank(node) -> bool:
        return any(b in seg(node) for b in BANKS) or any(
            isinstance(x, ast.Name) and x.id in names for x in ast.walk(node))

    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Attribute):
            continue
        a, args = n.func.attr, n.args
        if a == "to_parquet" and args and is_bank(args[0]):
            out.append(n.lineno)
        elif a in ("write_bytes", "write_text") and is_bank(n.func.value):
            out.append(n.lineno)
        elif a in ("move", "copy", "copy2", "copyfile", "replace", "rename") and len(args) >= 2 \
                and is_bank(args[1]):
            out.append(n.lineno)                                     # shutil / os (src, dst)
        elif a in ("replace", "rename") and len(args) == 1 and is_bank(args[0]):
            out.append(n.lineno)                                     # Path(src).replace(dst)
    return out


def test_the_scanner_sees_every_kind_of_bank_write():
    code = "\n".join([
        'OUT = ROOT / "output" / "fd_history.parquet"',
        "df.to_parquet(OUT)",
        'df.to_parquet(ROOT / "output" / "af_history.parquet", index=False)',
        'os.replace(tmp, OUT)',
        'Path(tmp).replace(OUT)',
        'OUT.write_bytes(b"")',
        'fd = pd.read_parquet(OUT)',                                  # a read, and fd is a frame
        'fd.to_parquet(ROOT / "output" / "corrections" / "x.parquet")',  # a sidecar
    ])
    assert _bank_writes(code) == [2, 3, 4, 5, 6]


def test_only_the_provider_ingestion_writes_a_history_bank():
    found = {}
    from src.integrity.data_truth import production_files           # the rule 2 scanner's file set
    for rel in production_files():
        src = (ROOT / rel).read_text(encoding="utf-8-sig")
        if not any(b in src for b in BANKS):
            continue
        lines = _bank_writes(src)
        if lines:
            found[rel] = lines
    unexpected = {k: v for k, v in found.items() if k not in BANK_WRITERS}
    assert not unexpected, (f"code that rewrites a history bank in place: {unexpected}. A correction "
                            "goes BESIDE the bank (a sidecar), never into it.")
    # the allow-list stays honest: an entry whose writer is gone must be removed
    assert set(found) == set(BANK_WRITERS), sorted(set(BANK_WRITERS) - set(found))


@pytest.mark.parametrize("addr", [("192.0.2.1", 80), ("fantasy.premierleague.com", 443)])
def test_the_suite_is_offline(addr):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match="tests run offline"):
            s.connect(addr)
    finally:
        s.close()


def test_loopback_stays_open_for_local_servers():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        c.connect(srv.getsockname())
    finally:
        c.close()
        srv.close()
