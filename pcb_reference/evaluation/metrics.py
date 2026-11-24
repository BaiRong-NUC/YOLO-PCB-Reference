import io
from collections import Counter
from contextlib import redirect_stdout

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..data.deeppcb import CLASSES


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def iou(box, others):
    left_top = np.maximum(box[:2], others[:, :2])
    right_bottom = np.minimum(box[2:4], others[:, 2:4])
    intersection = np.maximum(right_bottom - left_top, 0).prod(axis=1)
    other_area = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return intersection / np.maximum(area(box) + other_area - intersection, 1e-10)


def match_image(truth, detections, threshold):
    truth = np.asarray(truth, dtype=float).reshape(-1, 5)
    matched = np.zeros(len(truth), dtype=bool)
    retained = sorted((row for row in detections if row[4] >= threshold),
                      key=lambda row: -row[4])
    correct = []
    for detection in retained:
        candidates = np.flatnonzero(~matched & (truth[:, 4] == detection[5]))
        if candidates.size:
            overlaps = iou(detection, truth[candidates])
            best = overlaps.argmax()
            if overlaps[best] >= 0.5:
                matched[candidates[best]] = True
                correct.append(True)
                continue
        correct.append(False)
    return matched, retained, correct


def counts(truth, predictions, threshold, small_area):
    per_image = {}
    per_class = {name: Counter(tp=0, fp=0, gt=0) for name in CLASSES}
    for key, boxes in truth.items():
        matched, retained, correct = match_image(boxes, predictions[key], threshold)
        small = np.array([area(box) <= small_area for box in boxes], dtype=bool)
        per_image[key] = {"tp": int(matched.sum()), "fp": len(retained) - sum(correct),
                          "gt": len(boxes), "small_tp": int((small & matched).sum()),
                          "small_gt": int(small.sum())}
        for box, hit in zip(boxes, matched):
            category = per_class[CLASSES[box[4]]]
            category["gt"] += 1
            category["tp"] += int(hit)
        for detection, hit in zip(retained, correct):
            per_class[CLASSES[int(detection[5])]]["fp"] += int(not hit)
    totals = Counter()
    for row in per_image.values():
        totals.update(row)
    totals.update({"images": len(truth)})
    summary = dict(totals, recall=totals["tp"] / max(totals["gt"], 1),
                   fp_image=totals["fp"] / len(truth),
                   small_recall=totals["small_tp"] / max(totals["small_gt"], 1))
    return summary, per_class, per_image


def calibrate(truth, predictions):
    grid = {}
    for idx in range(1, 20):
        threshold = idx / 20
        summary, _, _ = counts(truth, predictions, threshold, 0)
        grid[threshold] = 2 * summary["tp"] / max(summary["gt"] + summary["tp"] + summary.get("fp", 0), 1)
    return max(grid, key=lambda threshold: (grid[threshold], threshold)), grid


def average_precision(truth, predictions):
    images, labels, detections = [], [], []
    for image_id, (key, boxes) in enumerate(truth.items(), 1):
        images.append({"id": image_id})
        for box in boxes:
            left, top, right, bottom, category = box
            labels.append({"id": len(labels) + 1, "image_id": image_id, "category_id": category + 1,
                           "bbox": [left, top, right-left, bottom-top], "area": area(box), "iscrowd": 0})
        for left, top, right, bottom, score, category in predictions[key]:
            detections.append({"image_id": image_id, "category_id": int(category) + 1, "score": score,
                               "bbox": [left, top, right-left, bottom-top]})
    if not detections:
        return 0.0, 0.0, {name: 0.0 for name in CLASSES}
    with redirect_stdout(io.StringIO()):
        dataset = COCO()
        dataset.dataset = {"images": images, "annotations": labels,
                           "categories": [{"id": idx, "name": name} for idx, name in enumerate(CLASSES, 1)]}
        dataset.createIndex()
        evaluator = COCOeval(dataset, dataset.loadRes(detections), "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    per_class = {}
    for idx, name in enumerate(CLASSES):
        valid = precision[:, :, idx]
        per_class[name] = float(valid[valid >= 0].mean()) if (valid >= 0).any() else None
    return float(evaluator.stats[0]), float(evaluator.stats[1]), per_class
