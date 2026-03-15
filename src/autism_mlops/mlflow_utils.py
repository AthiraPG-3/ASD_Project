from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

import mlflow


def setup_mlflow(experiment_name: str) -> None:
    # If MLFLOW_TRACKING_URI is set, MLflow will use it.
    mlflow.set_experiment(experiment_name)


@contextmanager
def mlflow_run(run_name: str, tags: dict[str, Any] | None = None) -> Iterator[str]:
    with mlflow.start_run(run_name=run_name):
        if tags:
            mlflow.set_tags({k: str(v) for k, v in tags.items()})
        yield mlflow.active_run().info.run_id


def log_git_info() -> None:
    # Best-effort; works only if inside a git repo.
    try:
        import subprocess

        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        mlflow.log_param("git_commit", commit)
    except Exception:
        pass


def log_env_info() -> None:
    for key in ("MLFLOW_TRACKING_URI", "WANDB_PROJECT", "WANDB_MODE"):
        val = os.environ.get(key)
        if val:
            mlflow.log_param(f"env_{key.lower()}", val)
