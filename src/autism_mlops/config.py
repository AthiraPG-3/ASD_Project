from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def scans_dir(self) -> Path:
        return self.data_dir / "scans"

    @property
    def features_path(self) -> Path:
        return self.data_dir / "features" / "features.npz"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def best_model_path(self) -> Path:
        return self.models_dir / "best_model.joblib"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def metrics_path(self) -> Path:
        return self.reports_dir / "metrics.json"

    @property
    def confusion_matrix_path(self) -> Path:
        return self.reports_dir / "confusion_matrix.png"

    @property
    def leaderboard_path(self) -> Path:
        return self.reports_dir / "val_leaderboard.csv"

    @property
    def drift_report_path(self) -> Path:
        return self.reports_dir / "drift_report.html"


def load_params(params_path: str | Path) -> dict[str, Any]:
    params_path = Path(params_path)
    with params_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_root() -> Path:
    # For DVC/scripts, assume repo root = current working directory
    return Path.cwd()

