from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage import exposure
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from skimage.filters import gabor


@dataclass(frozen=True)
class FeatureConfig:
    image_size: tuple[int, int] = (128, 128)
    grayscale: bool = True

    # Preprocessing
    equalize_adapthist: bool = True
    equalize_clip_limit: float = 0.01

    # Features
    intensity_stats: bool = True

    lbp_enabled: bool = True
    lbp_radius: int = 2
    lbp_n_points: int = 16
    lbp_method: str = "uniform"

    glcm_enabled: bool = True
    glcm_levels: int = 16
    glcm_distances: tuple[int, ...] = (1, 2, 3)
    glcm_angles: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
    glcm_props: tuple[str, ...] = (
        "contrast",
        "dissimilarity",
        "homogeneity",
        "ASM",
        "energy",
        "correlation",
    )

    hog_enabled: bool = True
    hog_orientations: int = 9
    hog_pixels_per_cell: tuple[int, int] = (8, 8)
    hog_cells_per_block: tuple[int, int] = (2, 2)

    gabor_enabled: bool = True
    gabor_frequencies: tuple[float, ...] = (0.1, 0.2, 0.3)
    gabor_thetas: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)


def feature_config_from_params(params: dict[str, Any]) -> FeatureConfig:
    fp = params["features"]
    preprocess = fp.get("preprocess", {})
    return FeatureConfig(
        image_size=tuple(fp["image_size"]),
        grayscale=bool(fp.get("grayscale", True)),
        equalize_adapthist=bool(preprocess.get("equalize_adapthist", True)),
        equalize_clip_limit=float(preprocess.get("equalize_clip_limit", 0.01)),
        intensity_stats=bool(fp.get("intensity_stats", True)),
        lbp_enabled=bool(fp.get("lbp", {}).get("enabled", True)),
        lbp_radius=int(fp.get("lbp", {}).get("radius", 2)),
        lbp_n_points=int(fp.get("lbp", {}).get("n_points", 16)),
        lbp_method=str(fp.get("lbp", {}).get("method", "uniform")),
        glcm_enabled=bool(fp.get("glcm", {}).get("enabled", True)),
        glcm_levels=int(fp.get("glcm", {}).get("levels", 16)),
        glcm_distances=tuple(fp.get("glcm", {}).get("distances", [1, 2, 3])),
        glcm_angles=tuple(
            fp.get("glcm", {}).get("angles", [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        ),
        glcm_props=tuple(fp.get("glcm", {}).get("props", ["contrast", "energy", "correlation"])),
        hog_enabled=bool(fp.get("hog", {}).get("enabled", True)),
        hog_orientations=int(fp.get("hog", {}).get("orientations", 9)),
        hog_pixels_per_cell=tuple(fp.get("hog", {}).get("pixels_per_cell", [8, 8])),
        hog_cells_per_block=tuple(fp.get("hog", {}).get("cells_per_block", [2, 2])),
        gabor_enabled=bool(fp.get("gabor", {}).get("enabled", True)),
        gabor_frequencies=tuple(fp.get("gabor", {}).get("frequencies", [0.1, 0.2, 0.3])),
        gabor_thetas=tuple(fp.get("gabor", {}).get("thetas", [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])),
    )


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _normalize_to_unit(img: np.ndarray) -> np.ndarray:
    # Normalize integer images with their dtype range; float images by max if needed.
    if np.issubdtype(img.dtype, np.integer):
        scale = float(np.iinfo(img.dtype).max)
        if scale > 0:
            return img.astype(np.float32) / scale
        return img.astype(np.float32)

    img = img.astype(np.float32)
    maxv = float(np.max(img)) if img.size else 0.0
    if maxv > 1.0:
        img = img / maxv
    return img


def load_image(path: str | Path, *, cfg: FeatureConfig) -> np.ndarray:
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    if cfg.grayscale:
        img = _to_grayscale(img)

    img = cv2.resize(img, cfg.image_size, interpolation=cv2.INTER_AREA)
    img = _normalize_to_unit(img)

    if cfg.equalize_adapthist:
        img = exposure.equalize_adapthist(img, clip_limit=cfg.equalize_clip_limit).astype(np.float32)

    return img


def _intensity_stats(img: np.ndarray) -> tuple[np.ndarray, list[str]]:
    flat = img.ravel()
    eps = 1e-8
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    mn = float(np.min(flat))
    mx = float(np.max(flat))
    median = float(np.median(flat))
    p10, p25, p75, p90 = [float(x) for x in np.percentile(flat, [10, 25, 75, 90])]

    hist, _ = np.histogram(flat, bins=32, range=(0.0, 1.0), density=True)
    hist = hist + eps
    ent = float(-(hist * np.log(hist)).sum())

    feats = np.array([mean, std, mn, mx, median, p10, p25, p75, p90, ent], dtype=np.float32)
    names = [
        "int_mean",
        "int_std",
        "int_min",
        "int_max",
        "int_median",
        "int_p10",
        "int_p25",
        "int_p75",
        "int_p90",
        "int_entropy",
    ]
    return feats, names


def _lbp_hist(img: np.ndarray, *, cfg: FeatureConfig) -> tuple[np.ndarray, list[str]]:
    lbp = local_binary_pattern(img, cfg.lbp_n_points, cfg.lbp_radius, method=cfg.lbp_method)

    if cfg.lbp_method == "uniform":
        n_bins = int(cfg.lbp_n_points + 2)
    else:
        n_bins = int(lbp.max() + 1)

    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    hist = hist.astype(np.float32)
    names = [f"lbp_{i}" for i in range(n_bins)]
    return hist, names


def _glcm_features(img: np.ndarray, *, cfg: FeatureConfig) -> tuple[np.ndarray, list[str]]:
    levels = cfg.glcm_levels
    q = np.clip((img * (levels - 1)).round(), 0, levels - 1).astype(np.uint8)

    glcm = graycomatrix(
        q,
        distances=list(cfg.glcm_distances),
        angles=list(cfg.glcm_angles),
        levels=levels,
        symmetric=True,
        normed=True,
    )

    feats: list[float] = []
    names: list[str] = []
    for prop in cfg.glcm_props:
        vals = graycoprops(glcm, prop=prop)
        for di, d in enumerate(cfg.glcm_distances):
            for ai, _a in enumerate(cfg.glcm_angles):
                feats.append(float(vals[di, ai]))
                names.append(f"glcm_{prop}_d{d}_a{ai}")

    return np.array(feats, dtype=np.float32), names


def _hog_features(img: np.ndarray, *, cfg: FeatureConfig) -> tuple[np.ndarray, list[str]]:
    vec = hog(
        img,
        orientations=cfg.hog_orientations,
        pixels_per_cell=cfg.hog_pixels_per_cell,
        cells_per_block=cfg.hog_cells_per_block,
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True,
    ).astype(np.float32)
    names = [f"hog_{i}" for i in range(vec.shape[0])]
    return vec, names


def _gabor_stats(img: np.ndarray, *, cfg: FeatureConfig) -> tuple[np.ndarray, list[str]]:
    feats: list[float] = []
    names: list[str] = []
    for fi, freq in enumerate(cfg.gabor_frequencies):
        for ti, theta in enumerate(cfg.gabor_thetas):
            real, imag = gabor(img, frequency=freq, theta=theta)
            mag = np.sqrt(real**2 + imag**2)
            feats.append(float(np.mean(mag)))
            feats.append(float(np.std(mag)))
            names.append(f"gabor_f{fi}_t{ti}_mean")
            names.append(f"gabor_f{fi}_t{ti}_std")
    return np.array(feats, dtype=np.float32), names


def extract_features(img: np.ndarray, *, cfg: FeatureConfig) -> tuple[np.ndarray, list[str]]:
    parts: list[np.ndarray] = []
    names: list[str] = []

    if cfg.intensity_stats:
        f, n = _intensity_stats(img)
        parts.append(f)
        names.extend(n)

    if cfg.lbp_enabled:
        f, n = _lbp_hist(img, cfg=cfg)
        parts.append(f)
        names.extend(n)

    if cfg.glcm_enabled:
        f, n = _glcm_features(img, cfg=cfg)
        parts.append(f)
        names.extend(n)

    if cfg.hog_enabled:
        f, n = _hog_features(img, cfg=cfg)
        parts.append(f)
        names.extend(n)

    if cfg.gabor_enabled:
        f, n = _gabor_stats(img, cfg=cfg)
        parts.append(f)
        names.extend(n)

    if not parts:
        raise ValueError("No features enabled in FeatureConfig")

    return np.concatenate(parts, axis=0), names
