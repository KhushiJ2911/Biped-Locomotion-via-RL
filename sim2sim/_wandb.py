"""Optional Weights & Biases logging.

Every entry point here is failure-tolerant on purpose. Training runs here are
multi-hour; losing one to an expired token, an offline laptop, or a transient
wandb outage would be a far worse outcome than losing the curves. Nothing in
this module is allowed to raise into the caller.
"""
from __future__ import annotations

import os

DEFAULT_PROJECT = "biped-sim2sim"


def available() -> tuple[bool, str]:
    """Is wandb both installed and authenticated?

    Returns (ok, reason). Authentication is checked without a network call, so
    this stays fast and works offline.
    """
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False, "wandb not installed (pip install wandb)"

    if os.environ.get("WANDB_API_KEY"):
        return True, "using WANDB_API_KEY"
    netrc = os.path.expanduser("~/.netrc")
    if os.path.exists(netrc):
        try:
            with open(netrc) as fh:
                if "api.wandb.ai" in fh.read():
                    return True, "using ~/.netrc credentials"
        except OSError:
            pass
    return False, "not logged in -- run: wandb login"


def init(enabled: bool, name: str, config: dict, project: str | None = None,
         group: str | None = None, job_type: str | None = None):
    """Start a run, or return None if logging is off or unavailable."""
    if not enabled:
        return None
    ok, reason = available()
    if not ok:
        print(f"[wandb] logging disabled: {reason}", flush=True)
        return None
    try:
        import wandb

        run = wandb.init(
            project=project or os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT),
            name=name,
            group=group,
            job_type=job_type,
            config=config,
            reinit=True,
        )
        print(f"[wandb] logging to {run.url}", flush=True)
        return run
    except Exception as exc:  # noqa: BLE001 - never break the caller
        print(f"[wandb] init failed, continuing without logging: {exc}", flush=True)
        return None


def log(run, data: dict, step: int | None = None) -> None:
    if run is None:
        return
    try:
        run.log(data, step=step)
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] log failed (ignored): {exc}", flush=True)


def summary(run, data: dict) -> None:
    if run is None:
        return
    try:
        run.summary.update(data)
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] summary failed (ignored): {exc}", flush=True)


def finish(run) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] finish failed (ignored): {exc}", flush=True)
