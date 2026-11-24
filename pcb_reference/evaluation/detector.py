import argparse
import csv
import gzip
import hashlib
import json
from time import perf_counter

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .. import ROOT
from ..data.deeppcb import CLASSES, iter_samples, read_boxes
from ..methods import ARMS, make_input
from ..methods.aligned.registration import translate
from .metrics import area, average_precision, calibrate, counts


SOURCE = ROOT / "data/DeepPCB/PCBData"
SHIFTS = [(0, 0)] + [(dx, dy) for distance in (1, -1, 2, -2, 4, -4)
                      for dx, dy in ((distance, 0), (0, distance))]


def ground_truth(samples):
    return {sample.key: [[*coords, category - 1] for *coords, category in
                         read_boxes(sample.annotation, 640, 640)] for sample in samples}


def prepared(pair, arm, shift):
    inspection, template = pair
    if arm != "single" and any(shift):
        template = translate(template, *shift)
    image = make_input(inspection, template, arm)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image


def infer(model, images, device):
    results = model.predict(images, imgsz=640, conf=0.001, max_det=300,
                            device=device, half=False, verbose=False, save=False)
    return [result.boxes.data.cpu().tolist() for result in results]


def prediction_file(model, samples, arm, shift, device, path, weight_hash):
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            saved = json.load(stream)
        if saved["weights_sha256"] != weight_hash or list(saved["boxes"]) != [sample.key for sample in samples]:
            raise ValueError(f"Evaluation cache differs from requested run: {path}")
        return saved["boxes"]
    boxes = {}
    for start in range(0, len(samples), 8):
        batch = samples[start:start + 8]
        images = [prepared(sample.read_images(), arm, shift) for sample in batch]
        boxes.update(zip((sample.key for sample in batch), infer(model, images, device), strict=True))
    temp = path.with_suffix(".tmp")
    with gzip.open(temp, "wt", encoding="utf-8", compresslevel=1) as stream:
        json.dump({"weights_sha256": weight_hash, "boxes": boxes}, stream, separators=(",", ":"))
    temp.replace(path)
    return boxes


def latency(model, samples, arm, device):
    pairs = [sample.read_images() for sample in samples[:55]]
    times = []
    for idx, (inspection, template) in enumerate(pairs):
        pair = inspection, translate(template, 2, 0)
        if device != "cpu":
            torch.cuda.synchronize(device)
        start = perf_counter()
        infer(model, [prepared(pair, arm, (0, 0))], device)
        if device != "cpu":
            torch.cuda.synchronize(device)
        if idx >= 5:
            times.append((perf_counter() - start) * 1000)
    return {"n": len(times), "p50_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def write_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def evaluate(seed, device):
    test = list(iter_samples(SOURCE, "test"))
    development = list(iter_samples(SOURCE, "trainval"))
    val = [sample for sample in development if sample.split == "val"]
    train = [sample for sample in development if sample.split == "train"]
    small = float(np.percentile([area(box) for boxes in ground_truth(train).values() for box in boxes], 25))
    truth = ground_truth(test)
    destination = ROOT / "outputs" / f"evaluation_s{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    summary_rows, class_rows, timings = [], [], {}
    for arm in ARMS:
        run = ROOT / "runs" / f"{arm}_s{seed}"
        with (run / "results.csv").open(encoding="utf-8-sig") as stream:
            epochs = list(csv.DictReader(stream))
        if len(epochs) != 30 or int(epochs[-1]["epoch"]) != 30:
            raise ValueError(f"Incomplete training: {run}")
        weights = run / "weights/best.pt"
        weight_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
        model = YOLO(str(weights))
        if list(model.names.values()) != CLASSES:
            raise ValueError(f"Unexpected class map: {model.names}")
        folder = destination / arm
        folder.mkdir(exist_ok=True)
        if arm == "single":
            validation = prediction_file(model, val, arm, (0, 0), device, folder / "val.json.gz", weight_hash)
            threshold, grid = calibrate(ground_truth(val), validation)
            (destination / "operating_point.json").write_text(json.dumps(
                {"threshold": threshold, "small_area": small, "validation_images": len(val),
                 "validation_f1": grid, "source_weights_sha256": weight_hash}, indent=2), encoding="utf-8")
        timings[arm] = latency(model, test, arm, device)
        for dx, dy in SHIFTS:
            suffix = f"dx{dx}_dy{dy}"
            if arm != "single" or (dx, dy) == (0, 0):
                predictions = prediction_file(model, test, arm, (dx, dy), device,
                                              folder / f"{suffix}.json.gz", weight_hash)
                totals, by_class, by_image = counts(truth, predictions, threshold, small)
                ap, ap50, class_ap = average_precision(truth, predictions)
            metadata = {"arm": arm, "dx": dx, "dy": dy}
            summary_rows.append(dict(metadata, **totals, ap=ap, ap50=ap50))
            for name, stats in by_class.items():
                class_rows.append(dict(metadata, category=name, **stats, ap=class_ap[name],
                                       recall=stats["tp"] / max(stats["gt"], 1)))
            if (dx, dy) == (2, 0):
                write_table(folder / "primary_by_image.csv", [dict(image=key, **row) for key, row in by_image.items()])
            print(arm, suffix, f"AP={ap:.4f} FP/image={totals['fp_image']:.3f}", flush=True)
        write_table(destination / "conditions.csv", summary_rows)
        write_table(destination / "classes.csv", class_rows)
        (destination / "latency.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate paired DeepPCB detectors at a locked operating point.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    evaluate(args.seed, args.device)
