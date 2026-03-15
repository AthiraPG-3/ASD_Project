from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    out_path: str | Path,
    labels: tuple[str, str] = ("control", "autism"),
    title: str = "Confusion Matrix",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=list(labels),
        cmap="Blues",
        colorbar=False,
        normalize=None,
    )
    disp.ax_.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
