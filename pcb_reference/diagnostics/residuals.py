import csv
from collections import defaultdict

import cv2
import numpy as np

from ..data.deeppcb import iter_samples, read_boxes
from ..methods import make_input
from ..methods.aligned.registration import translate


def measure(source, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    summaries = defaultdict(list)
    preview_key = None
    for sample in iter_samples(source, "trainval"):
        if sample.split != "val":
            continue
        key = sample.key
        if preview_key is None:
            preview_key = key
        inspection, template = sample.read_images()
        height, width = inspection.shape
        box_mask = np.zeros(inspection.shape, dtype=bool)
        for left, top, right, bottom, _ in read_boxes(sample.annotation, width, height):
            box_mask[top:bottom + 1, left:right + 1] = True
        interior = np.zeros_like(box_mask)
        interior[16:-16, 16:-16] = True
        for dx in (0, 2):
            shifted = translate(template, dx, 0)
            panels = [inspection]
            for arm in ("raw", "aligned", "ring"):
                residual = make_input(inspection, shifted, arm)[:, :, 0]
                panels.append(residual)
                background = float(residual[interior & ~box_mask].mean())
                foreground = float(residual[interior & box_mask].mean())
                rows.append((key, dx, arm, background, foreground))
                summaries[dx, arm].append((background, foreground))
            if key == preview_key:
                preview = cv2.hconcat(panels)
                if not cv2.imwrite(str(output / f"{key}_dx{dx}.png"), preview):
                    raise OSError("Could not write comparison image")
    with (output / "paired.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("image", "dx", "arm", "outside_box_mean", "inside_box_mean"))
        writer.writerows(rows)
    for (dx, arm), values in summaries.items():
        background, foreground = np.mean(values, axis=0)
        print(f"dx={dx} {arm:7s} n={len(values)} outside={background:.3f} inside={foreground:.3f}")
