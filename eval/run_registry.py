#!/usr/bin/env python3
"""Run registry (R10): every run stamps run-id/seed/git-hash/argv into its
metrics JSON and appends one JSONL line to runs/registry.jsonl, so results
are comparable across runs instead of silently overwritten.

Usage:
    from eval.run_registry import new_run, record
    run = new_run(argv=None, seed=seed)
    ... metrics["run"] = run ...
    record(metrics, name="gcbd_benchmark")
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
import uuid
from pathlib import Path


def git_commit(cwd: str | None = None) -> str | None:
    """Short git HEAD hash, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
            text=True, cwd=cwd or str(Path(__file__).resolve().parent.parent),
            timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def new_run(argv: list[str] | None = None, seed: int | None = None,
            extra: dict | None = None) -> dict:
    """Stamp a fresh run. `extra` carries caller-specific context."""
    run = {"run_id": uuid.uuid4().hex[:12],
           "timestamp_utc": datetime.datetime.now(
               datetime.timezone.utc).isoformat(timespec="seconds"),
           "git_commit": git_commit(),
           "argv": list(sys.argv[1:] if argv is None else argv),
           "seed": seed}
    if extra:
        run.update(extra)
    return run


def record(metrics: dict, name: str,
           registry: str = "runs/registry.jsonl") -> str:
    """Append one JSONL summary line; returns the registry path."""
    Path(registry).parent.mkdir(parents=True, exist_ok=True)
    line = {"name": name, "run": metrics.get("run", {}),
            "summary": {k: metrics.get(k) for k in
                        ("accuracy", "roc_auc_ovr", "roc_auc_ovr_macro",
                         "dhw_baseline_accuracy", "ece") if k in metrics}}
    with open(registry, "a") as f:
        f.write(json.dumps(line, default=str) + "\n")
    return registry
