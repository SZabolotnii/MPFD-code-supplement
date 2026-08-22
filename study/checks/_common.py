"""Shared helpers for pre-registered checks: result location, seed, JSON writer."""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy

STUDY = Path(__file__).resolve().parents[1]
RESULTS = STUDY / "results"
SEED = 20260821

if str(STUDY) not in sys.path:
    sys.path.insert(0, str(STUDY))


def _json_default(o):
    return o.item() if hasattr(o, "item") else float(o)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=STUDY, text=True
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def write_result(name: str, payload: dict) -> Path:
    """Write results/<name>.json with provenance metadata; returns the path."""
    RESULTS.mkdir(exist_ok=True)
    out = dict(payload)
    out["_meta"] = {
        "check": name,
        "git_sha": _git_sha(),
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=_json_default))
    return path


def load_result(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())
