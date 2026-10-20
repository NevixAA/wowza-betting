"""Repair double-encoded (mojibake) names from API-Football.

API-Football intermittently returns a player's name UTF-8-encoded twice for a single fixture:
"Adrià Capdevila" arrives as "AdriÃ\xa0 Capdevila" in some matches and correctly in others
(measured 2026-10-20: 24 of 657,042 player_history rows, rising month by month). The garbled name
then flows into tips, the ledger, CLV bet ids and Telegram.

repair_text() reverses exactly that: re-encode as cp1252 (Latin-1 for the five bytes cp1252 leaves
undefined) and decode as UTF-8. A genuine accented name ("Ângelo", "Saúl Ñíguez") is not valid UTF-8
once re-encoded, so it is returned unchanged - the repair only fires on text that IS mojibake.
"""
from __future__ import annotations


def _to_bytes(s: str) -> bytes | None:
    out = bytearray()
    for ch in s:
        try:
            out += ch.encode("cp1252")
        except UnicodeEncodeError:
            if ord(ch) < 256:                 # 0x81 0x8D 0x8F 0x90 0x9D: cp1252 gaps, kept as Latin-1
                out.append(ord(ch))
            else:
                return None
    return bytes(out)


def repair_text(s):
    """The repaired string, or the input unchanged (also for non-str and pure-ASCII input)."""
    if not isinstance(s, str) or s.isascii():
        return s
    for _ in range(2):                        # a name double-garbled twice is still recovered
        b = _to_bytes(s)
        if b is None:
            break
        try:
            t = b.decode("utf-8")
        except UnicodeDecodeError:
            break
        if t == s:
            break
        s = t
    return s
