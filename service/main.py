from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from joblib import load

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autism_mlops.config import load_params  # noqa: E402
from autism_mlops.features import extract_features, feature_config_from_params, load_image  # noqa: E402


app = FastAPI(title="Autism Scan Classifier", version="0.1.0")


def _model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", "models/best_model.joblib"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model_path = _model_path()
    if not model_path.exists():
        raise HTTPException(status_code=500, detail=f"Model not found at {model_path}")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        params = load_params("params.yaml")
        feat_cfg = feature_config_from_params(params)
        img = load_image(tmp_path, cfg=feat_cfg)
        vec, _ = extract_features(img, cfg=feat_cfg)
        X = vec.reshape(1, -1).astype(np.float32)

        model = load(model_path)
        pred = int(model.predict(X)[0])
        score = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.shape[1] == 2:
                score = float(proba[0, 1])
        return {"pred_label": pred, "pred_class": "autism" if pred == 1 else "control", "autism_score": score}
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

