"""
Prediction provenance stamping.
===============================
`predictions.csv` is a CURRENT-STATE file: predict overwrites it every 5 minutes. Until now
it carried no record of WHEN it was produced or WHICH code and models produced it, so:

  * the model's path toward kickoff (T-7d, T-3d, T-1h ...) was unreconstructable from the
    file itself — it could only be recovered by walking v9's git history commit by commit;
  * no row could answer "which model version generated this?", which is the first question
    any deployed signal has to be able to answer.

Three additive columns fix both. They change no probability, no tier, no stake and no bet —
every consumer reads this file by column name via pandas, so appending columns is safe.

    generated_at   UTC timestamp of the predict run that wrote the row
    git_sha        v9 code version
    model_sha      short digest over the model .pkl files actually on disk

DESIGN RULE: this must never break predict. A timed-out or crashed predict sends zero tips,
which is far worse than missing provenance. Every failure path still returns the columns, so
the CSV schema stays stable whether stamping succeeded or not.
"""
from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parents[1]
_UNKNOWN = "unknown"

_cached_git: str | None = None
_cached_models: str | None = None


def _git_sha() -> str:
    global _cached_git
    if _cached_git is not None:
        return _cached_git
    sha = os.getenv("GITHUB_SHA", "").strip()
    if sha:
        _cached_git = sha[:12]
        return _cached_git
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=_BASE, capture_output=True, text=True, timeout=10,
        )
        _cached_git = out.stdout.strip() if out.returncode == 0 else _UNKNOWN
    except Exception:
        _cached_git = _UNKNOWN
    return _cached_git


def _model_sha() -> str:
    """Digest over the model files present, so a retrain changes the stamp.

    Hashes file size + mtime rather than contents: the .pkl files are large and predict runs
    every 5 minutes, so reading them all would add avoidable I/O to the hot path. Size+mtime
    is enough to detect "the models changed", which is what the stamp is for.
    """
    global _cached_models
    if _cached_models is not None:
        return _cached_models
    try:
        files = sorted((_BASE / "models").glob("*.pkl"))
        if not files:
            _cached_models = _UNKNOWN
            return _cached_models
        h = hashlib.sha1()
        for f in files:
            st = f.stat()
            h.update(f"{f.name}:{st.st_size}:{int(st.st_mtime)}|".encode())
        _cached_models = h.hexdigest()[:12]
    except Exception:
        _cached_models = _UNKNOWN
    return _cached_models


def stamp(df):
    """Add generated_at / git_sha / model_sha to a predictions frame, in place-ish.

    Returns the frame. Never raises: on any failure the columns are still present, filled
    with 'unknown', so downstream schema expectations hold.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        df["generated_at"] = ts
        df["git_sha"] = _git_sha()
        df["model_sha"] = _model_sha()
    except Exception as e:                                    # pragma: no cover
        log.warning(f"[provenance] stamping failed ({e}); writing placeholders")
        try:
            for col, val in (("generated_at", ts), ("git_sha", _UNKNOWN),
                             ("model_sha", _UNKNOWN)):
                if col not in df.columns:
                    df[col] = val
        except Exception:
            pass
    return df
