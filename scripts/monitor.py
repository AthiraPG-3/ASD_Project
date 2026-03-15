from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autism_mlops.config import ProjectPaths, load_params, project_root  # noqa: E402
from autism_mlops.mlflow_utils import log_env_info, log_git_info, mlflow_run, setup_mlflow  # noqa: E402
from autism_mlops.monitoring import generate_drift_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=str, default="params.yaml")
    args = ap.parse_args()

    _ = load_params(args.params)
    paths = ProjectPaths(root=project_root())

    data = np.load(paths.features_path, allow_pickle=False)
    X = data["X"]
    split = np.array([s.lower() for s in data["split"].tolist()], dtype="U")
    feature_names = [str(x) for x in data["feature_names"].tolist()]

    ref_idx = np.where(split == "train")[0]
    cur_idx = np.where(split == "test")[0]
    if ref_idx.size == 0 or cur_idx.size == 0:
        raise ValueError("Need train and test splits to generate drift report")

    setup_mlflow("autism-scan-monitoring")
    tags = {"type": "data_drift", "comparison": "train_vs_test"}
    with mlflow_run("drift_report", tags=tags):
        log_git_info()
        log_env_info()

        mlflow.log_param("ref_rows", int(ref_idx.size))
        mlflow.log_param("cur_rows", int(cur_idx.size))
        mlflow.log_param("n_features", int(X.shape[1]))

        generate_drift_report(
            X_ref=X[ref_idx],
            X_cur=X[cur_idx],
            feature_names=feature_names,
            out_path=paths.drift_report_path,
            metadata={"comparison": "train_vs_test"},
        )
        mlflow.log_artifact(str(paths.drift_report_path), artifact_path="reports")


if __name__ == "__main__":
    main()
