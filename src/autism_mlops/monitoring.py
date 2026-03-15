from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


def _to_frame(X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(X, columns=feature_names)


def generate_drift_report(
    *,
    X_ref: np.ndarray,
    X_cur: np.ndarray,
    feature_names: list[str],
    out_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ref = _to_frame(X_ref, feature_names)
    cur = _to_frame(X_cur, feature_names)

    if metadata:
        for k, v in metadata.items():
            ref[k] = str(v)
            cur[k] = str(v)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    report.save_html(str(out_path))
