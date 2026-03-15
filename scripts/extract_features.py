from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autism_mlops.config import ProjectPaths, load_params, project_root  # noqa: E402
from autism_mlops.data import discover_dataset  # noqa: E402
from autism_mlops.features import extract_features, feature_config_from_params, load_image  # noqa: E402


def _process_one(example, cfg):
    img = load_image(example.path, cfg=cfg)
    vec, names = extract_features(img, cfg=cfg)
    return vec, names, str(example.path), example.split, example.label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=str, default="params.yaml")
    args = ap.parse_args()

    params = load_params(args.params)
    paths = ProjectPaths(root=project_root())

    scans_dir = Path(params["data"]["scans_dir"])
    class_to_label = {k: int(v) for k, v in params["data"]["class_names"].items()}
    examples = discover_dataset(scans_dir, class_to_label=class_to_label)

    feat_cfg = feature_config_from_params(params)
    n_jobs = int(params["train"].get("n_jobs", -1))

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_process_one)(ex, feat_cfg) for ex in tqdm(examples, desc="Extracting features")
    )

    X = np.stack([r[0] for r in results], axis=0).astype(np.float32)
    feature_names = results[0][1]
    paths_arr = np.array([r[2] for r in results], dtype="U")
    splits_arr = np.array([r[3] for r in results], dtype="U")
    y = np.array([r[4] for r in results], dtype=np.int64)

    out_path = paths.features_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        split=splits_arr,
        path=paths_arr,
        feature_names=np.array(feature_names, dtype="U"),
    )


if __name__ == "__main__":
    main()

