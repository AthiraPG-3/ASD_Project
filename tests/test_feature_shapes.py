from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autism_mlops.features import FeatureConfig, extract_features  # noqa: E402


def test_extract_features_has_stable_length():
    cfg = FeatureConfig(image_size=(64, 64))
    img = np.random.rand(64, 64).astype(np.float32)
    vec1, names1 = extract_features(img, cfg=cfg)
    vec2, names2 = extract_features(img, cfg=cfg)
    assert vec1.shape == vec2.shape
    assert len(names1) == vec1.shape[0]
    assert names1 == names2

