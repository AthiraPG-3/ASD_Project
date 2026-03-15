from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class DatasetSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            pass
    return metrics


def _score_vector(model: Any, X: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        df = model.decision_function(X)
        if getattr(df, "ndim", 0) == 1:
            return df
    return None


def build_candidates(params: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    tp = params["train"]
    candidates: list[tuple[str, dict[str, Any]]] = []

    for k in tp["knn"]["k"]:
        candidates.append(("knn", {"n_neighbors": int(k)}))

    for C in tp["svm"]["C"]:
        candidates.append(("svm_linear", {"kernel": "linear", "C": float(C)}))

    for C in tp["svm"]["C"]:
        for gamma in tp["svm"]["gamma"]:
            candidates.append(("svm_rbf", {"kernel": "rbf", "C": float(C), "gamma": gamma}))

    for C in tp["svm"]["C"]:
        for gamma in tp["svm"]["gamma"]:
            for degree in tp["svm"]["poly_degree"]:
                candidates.append(
                    (
                        "svm_poly",
                        {"kernel": "poly", "C": float(C), "gamma": gamma, "degree": int(degree)},
                    )
                )

    for max_depth in tp["decision_tree"]["max_depth"]:
        for mss in tp["decision_tree"]["min_samples_split"]:
            candidates.append(
                (
                    "decision_tree",
                    {
                        "max_depth": None if max_depth is None else int(max_depth),
                        "min_samples_split": int(mss),
                    },
                )
            )

    for C in tp["logistic_regression"]["C"]:
        candidates.append(
            ("logreg", {"C": float(C), "max_iter": int(tp["logistic_regression"]["max_iter"])})
        )

    for n_estimators in tp["random_forest"]["n_estimators"]:
        for max_depth in tp["random_forest"]["max_depth"]:
            candidates.append(
                (
                    "random_forest",
                    {
                        "n_estimators": int(n_estimators),
                        "max_depth": None if max_depth is None else int(max_depth),
                    },
                )
            )

    return candidates


def make_pipeline(
    model_name: str, model_params: dict[str, Any], *, pca_components: int | None, seed: int
) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if pca_components and pca_components > 0:
        steps.append(("pca", PCA(n_components=pca_components, random_state=seed)))

    if model_name == "knn":
        clf = KNeighborsClassifier(**model_params)
    elif model_name.startswith("svm_"):
        clf = SVC(probability=True, class_weight="balanced", **model_params)
    elif model_name == "decision_tree":
        clf = DecisionTreeClassifier(random_state=seed, class_weight="balanced", **model_params)
    elif model_name == "logreg":
        clf = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            **model_params,
        )
    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
            **model_params,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    steps.append(("clf", clf))
    return Pipeline(steps)


def train_and_select(
    splits: DatasetSplits,
    *,
    params: dict[str, Any],
) -> tuple[Pipeline, pd.DataFrame, dict[str, Any]]:
    tp = params["train"]
    seed = int(tp["random_seed"])
    selection_metric = str(tp.get("selection_metric", "f1_val"))
    pca_components = int(tp.get("pca_components", 0)) or None

    leaderboard_rows: list[dict[str, Any]] = []
    best_key: tuple[str, dict[str, Any]] | None = None
    best_value = -1e18

    for model_name, model_params in build_candidates(params):
        pipe = make_pipeline(model_name, model_params, pca_components=pca_components, seed=seed)
        pipe.fit(splits.X_train, splits.y_train)

        y_val_pred = pipe.predict(splits.X_val)
        y_val_score = _score_vector(pipe, splits.X_val)
        val_metrics = compute_metrics(splits.y_val, y_val_pred, y_val_score)

        row = {
            "model": model_name,
            **{f"param_{k}": v for k, v in model_params.items()},
            "accuracy_val": val_metrics["accuracy"],
            "precision_val": val_metrics["precision"],
            "recall_val": val_metrics["recall"],
            "f1_val": val_metrics["f1"],
            "roc_auc_val": val_metrics.get("roc_auc", np.nan),
        }
        leaderboard_rows.append(row)

        metric_value = float(row.get(selection_metric, -np.inf))
        if metric_value > best_value:
            best_value = metric_value
            best_key = (model_name, model_params)

    if best_key is None:
        raise RuntimeError("No models trained")

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(by=selection_metric, ascending=False)

    # Retrain best on train+val, evaluate on test
    model_name, model_params = best_key
    best_pipe = make_pipeline(model_name, model_params, pca_components=pca_components, seed=seed)
    X_tr = np.concatenate([splits.X_train, splits.X_val], axis=0)
    y_tr = np.concatenate([splits.y_train, splits.y_val], axis=0)
    best_pipe.fit(X_tr, y_tr)

    y_test_pred = best_pipe.predict(splits.X_test)
    y_test_score = _score_vector(best_pipe, splits.X_test)
    test_metrics = compute_metrics(splits.y_test, y_test_pred, y_test_score)

    summary = {
        "best_model": model_name,
        "best_params": model_params,
        "selection_metric": selection_metric,
        "val_best_score": float(best_value),
        "test_metrics": test_metrics,
    }
    return best_pipe, leaderboard, summary
