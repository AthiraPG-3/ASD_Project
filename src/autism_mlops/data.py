from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageExample:
    path: Path
    split: str
    label: int
    class_name: str


def _iter_images(dir_path: Path) -> list[Path]:
    if not dir_path.exists():
        return []
    return [p for p in dir_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def discover_dataset(
    scans_dir: str | Path,
    *,
    class_to_label: dict[str, int],
    split_names: tuple[str, ...] = ("train", "val", "test"),
) -> list[ImageExample]:
    scans_dir = Path(scans_dir)

    # allow either "val" or "validation" on disk
    split_aliases = {"validation": "val", "valid": "val"}

    examples: list[ImageExample] = []
    for split in split_names:
        split_dir = scans_dir / split
        if not split_dir.exists() and split == "val":
            for alt in ("validation", "valid"):
                alt_dir = scans_dir / alt
                if alt_dir.exists():
                    split_dir = alt_dir
                    break

        for class_name, label in class_to_label.items():
            class_dir = split_dir / class_name
            for img_path in _iter_images(class_dir):
                examples.append(
                    ImageExample(
                        path=img_path,
                        split=split_aliases.get(split_dir.name, split),
                        label=int(label),
                        class_name=class_name,
                    )
                )

    if not examples:
        raise FileNotFoundError(
            f"No images found under {scans_dir}. Expected structure like "
            f"{scans_dir}/train/<class>/*.png"
        )

    return examples

