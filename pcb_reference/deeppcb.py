from dataclasses import dataclass
from pathlib import Path

import cv2


CLASSES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]
VAL_GROUP = "group77000"


@dataclass(frozen=True)
class Sample:
    key: str
    split: str
    inspection_path: Path
    template_path: Path
    annotation: Path

    def read_images(self):
        inspection = cv2.imread(str(self.inspection_path), cv2.IMREAD_GRAYSCALE)
        template = cv2.imread(str(self.template_path), cv2.IMREAD_GRAYSCALE)
        if inspection is None or template is None or inspection.shape != template.shape:
            raise ValueError(f"Invalid image pair for {self.key}")
        return inspection, template


def iter_samples(source, partition):
    for row in (source / f"{partition}.txt").read_text().splitlines():
        if not row.strip():
            continue
        stem, annotation = map(Path, row.split())
        split = partition
        if partition == "trainval":
            split = "val" if stem.parts[0] == VAL_GROUP else "train"
        key = stem.stem
        yield Sample(key, split, source / stem.with_name(f"{key}_test.jpg"),
                     source / stem.with_name(f"{key}_temp.jpg"), source / annotation)


def read_boxes(path, width, height):
    for row in path.read_text().splitlines():
        if not row.strip():
            continue
        left, top, right, bottom, category = map(int, row.split())
        if not (0 <= left < right <= width and 0 <= top < bottom <= height and 1 <= category <= len(CLASSES)):
            raise ValueError(f"Invalid box in {path}: {row}")
        yield left, top, right, bottom, category
