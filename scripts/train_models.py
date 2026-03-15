from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autism_mlops.config import ProjectPaths, load_params, project_root  # noqa: E402
from autism_mlops.mlflow_utils import log_env_info, log_git_info, mlflow_run, setup_mlflow  # noqa: E402
from autism_mlops.plots import save_confusion_matrix  # noqa: E402
from autism_mlops.train import DatasetSplits, build_candidates, compute_metrics, make_pipeline  # noqa: E402


def _load_features_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    data = np.load(path, allow_pickle=False)
    X = data["X"]
    y = data["y"]
    split = data["split"]
    img_path = data["path"]
    feature_names = [str(x) for x in data["feature_names"].tolist()]
    return X, y, split, img_path, feature_names


def _split_arrays(
    X: np.ndarray, y: np.ndarray, split: np.ndarray, feature_names: list[str]
) -> DatasetSplits:
    split = np.array([s.lower() for s in split.tolist()], dtype="U")

    def sel(name: str):
        idx = np.where(split == name)[0]
        if idx.size == 0:
            raise ValueError(f"Split '{name}' not found in features file")
        return X[idx], y[idx]

    X_train, y_train = sel("train")
    X_val, y_val = sel("val")
    X_test, y_test = sel("test")
    return DatasetSplits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
    )


def _score_vector(model, X: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if getattr(proba, "ndim", 0) == 2 and proba.shape[1] == 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        df = model.decision_function(X)
        if getattr(df, "ndim", 0) == 1:
            return df
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=str, default="params.yaml")
    args = ap.parse_args()

    params = load_params(args.params)
    paths = ProjectPaths(root=project_root())

    X, y, split, _img_path, feature_names = _load_features_npz(paths.features_path)
    splits = _split_arrays(X, y, split, feature_names)

    setup_mlflow("autism-scan-classical-ml")
    tags = {"dataset": "scans", "task": "binary_classification"}

    pca_components = int(params["train"].get("pca_components", 0)) or None
    selection_metric = str(params["train"].get("selection_metric", "f1_val"))
    random_seed = int(params["train"].get("random_seed", 42))

    with mlflow_run("train_select", tags=tags) as run_id:
        log_git_info()
        log_env_info()

        mlflow.log_param("n_samples", int(X.shape[0]))
        mlflow.log_param("n_features", int(X.shape[1]))
        mlflow.log_param("pca_components", int(pca_components or 0))
        mlflow.log_param("selection_metric", selection_metric)
        mlflow.log_param("random_seed", random_seed)

        leaderboard_rows: list[dict[str, object]] = []
        best: tuple[str, dict[str, object]] | None = None
        best_value = float("-inf")

        for model_name, model_params in build_candidates(params):
            with mlflow.start_run(run_name=model_name, nested=True):
                mlflow.log_param("model", model_name)
                for k, v in model_params.items():
                    mlflow.log_param(k, v)
                mlflow.log_param("pca_components", int(pca_components or 0))

                pipe = make_pipeline(
                    model_name,
                    model_params,
                    pca_components=pca_components,
                    seed=random_seed,
                )
                pipe.fit(splits.X_train, splits.y_train)

                y_val_pred = pipe.predict(splits.X_val)
                y_val_score = _score_vector(pipe, splits.X_val)
                val_metrics = compute_metrics(splits.y_val, y_val_pred, y_val_score)

                metrics_for_row = {
                    "accuracy_val": float(val_metrics["accuracy"]),
                    "precision_val": float(val_metrics["precision"]),
                    "recall_val": float(val_metrics["recall"]),
                    "f1_val": float(val_metrics["f1"]),
                    "roc_auc_val": float(val_metrics.get("roc_auc", np.nan)),
                }
                mlflow.log_metrics(metrics_for_row)

                leaderboard_rows.append(
                    {
                        "model": model_name,
                        **{f"param_{k}": v for k, v in model_params.items()},
                        **metrics_for_row,
                    }
                )

                metric_value = float(metrics_for_row.get(selection_metric, float("-inf")))
                if metric_value > best_value:
                    best_value = metric_value
                    best = (model_name, dict(model_params))

        if best is None:
            raise RuntimeError("No models trained")

        best_model_name, best_params = best
        mlflow.log_param("best_model", best_model_name)
        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", v)
        mlflow.log_metric("best_val_score", float(best_value))

        leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values(by=selection_metric, ascending=False)
        paths.reports_dir.mkdir(parents=True, exist_ok=True)
        leaderboard_df.to_csv(paths.leaderboard_path, index=False)
        mlflow.log_artifact(str(paths.leaderboard_path), artifact_path="reports")

        # Retrain best on train+val, evaluate on test
        X_tr = np.concatenate([splits.X_train, splits.X_val], axis=0)
        y_tr = np.concatenate([splits.y_train, splits.y_val], axis=0)
        best_model = make_pipeline(
            best_model_name,
            best_params,
            pca_components=pca_components,
            seed=random_seed,
        )
        best_model.fit(X_tr, y_tr)

        y_test_pred = best_model.predict(splits.X_test)
        y_test_score = _score_vector(best_model, splits.X_test)
        test_metrics = compute_metrics(splits.y_test, y_test_pred, y_test_score)
        mlflow.log_metrics({f"{k}_test": float(v) for k, v in test_metrics.items()})

        save_confusion_matrix(
            splits.y_test,
            y_test_pred,
            out_path=paths.confusion_matrix_path,
            labels=("control", "autism"),
            title="Test Confusion Matrix",
        )
        mlflow.log_artifact(str(paths.confusion_matrix_path), artifact_path="reports")

        paths.models_dir.mkdir(parents=True, exist_ok=True)
        from joblib import dump

        dump(best_model, paths.best_model_path)
        mlflow.log_artifact(str(paths.best_model_path), artifact_path="model_files")
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        out = {
            "run_id": run_id,
            "best_model": best_model_name,
            "best_params": best_params,
            "selection_metric": selection_metric,
            "val_best_score": float(best_value),
            "test_metrics": test_metrics,
        }
        paths.metrics_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(paths.metrics_path), artifact_path="reports")

        # Optional W&B mirror (best-effort)
        try:
            if os.environ.get("WANDB_API_KEY"):
                import wandb

                wandb.init(project=os.environ.get("WANDB_PROJECT", "autism-mlops"), reinit=True)
                wandb.config.update({"mlflow_run_id": run_id, **out}, allow_val_change=True)
                wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
                wandb.save(str(paths.best_model_path))
                wandb.save(str(paths.metrics_path))
                wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
