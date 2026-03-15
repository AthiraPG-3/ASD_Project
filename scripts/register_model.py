from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _default_run_id(metrics_path: Path) -> str:
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found. Provide --run-id explicitly or run the training pipeline first."
        )
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_id = data.get("run_id")
    if not run_id:
        raise ValueError(f"No run_id found in {metrics_path}")
    return str(run_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None, help="MLflow run_id that produced the model artifact.")
    ap.add_argument("--model-name", type=str, default="autism-scan-classifier")
    ap.add_argument("--stage", type=str, default=None, help="Optional stage to transition to (e.g., Staging).")
    ap.add_argument(
        "--artifact-path", type=str, default="model", help="Artifact path used in mlflow.sklearn.log_model"
    )
    ap.add_argument("--metrics-path", type=str, default="reports/metrics.json")
    args = ap.parse_args()

    run_id = args.run_id or _default_run_id(Path(args.metrics_path))
    model_uri = f"runs:/{run_id}/{args.artifact_path}"

    result = mlflow.register_model(model_uri, args.model_name)
    print(f"Registered: name={result.name} version={result.version} uri={model_uri}")

    if args.stage:
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=result.name,
            version=result.version,
            stage=args.stage,
            archive_existing_versions=False,
        )
        print(f"Transitioned to stage: {args.stage}")


if __name__ == "__main__":
    main()
