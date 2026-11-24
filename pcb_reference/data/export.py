from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import yaml

from .deeppcb import CLASSES, iter_samples, read_boxes
from ..methods import make_input
from ..methods.aligned.registration import translate


def convert_labels(path, width, height):
    labels = []
    for left, top, right, bottom, category in read_boxes(path, width, height):
        values = ((left + right) / (2 * width), (top + bottom) / (2 * height),
                  (right - left) / width, (bottom - top) / height)
        labels.append(f"{category - 1} " + " ".join(f"{value:.8f}" for value in values))
    return "\n".join(labels) + ("\n" if labels else "")


def export_sample(sample, output, arm, val_shift):
    inspection, template = sample.read_images()
    if sample.split == "val" and any(val_shift):
        template = translate(template, *val_shift)
    image = make_input(inspection, template, arm)
    target = output / "images" / sample.split / f"{sample.key}.png"
    if not cv2.imwrite(str(target), image):
        raise OSError(f"Could not save {target}")
    height, width = inspection.shape
    labels = convert_labels(sample.annotation, width, height)
    (output / "labels" / sample.split / f"{sample.key}.txt").write_text(labels, encoding="utf-8")


def prepare(source, destination, arm, val_shift):
    if destination.exists():
        raise FileExistsError(f"Dataset already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    seen = set()
    with TemporaryDirectory(dir=destination.parent) as staging:
        output = Path(staging) / "dataset"
        for split in ("train", "val", "test"):
            for kind in ("images", "labels"):
                (output / kind / split).mkdir(parents=True)
        for partition in ("trainval", "test"):
            for sample in iter_samples(source, partition):
                if sample.key in seen:
                    raise ValueError(f"Duplicate sample {sample.key}")
                seen.add(sample.key)
                export_sample(sample, output, arm, val_shift)
                counts[sample.split] += 1
        if counts != {"train": 900, "val": 100, "test": 500}:
            raise ValueError(f"DeepPCB split differs from protocol: {counts}")
        settings = {"path": destination.as_posix(), "train": "images/train", "val": "images/val",
                    "test": "images/test", "names": CLASSES}
        (output / "data.yaml").write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
        output.rename(destination)
    print(f"{arm}  train={counts['train']}  val={counts['val']}  test={counts['test']}  {destination}")
