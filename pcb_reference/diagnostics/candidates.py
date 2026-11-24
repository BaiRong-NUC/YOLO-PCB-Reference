import argparse
import json

import torch
from ultralytics import YOLO
from ultralytics.utils.ops import xywh2xyxy

from .. import ROOT
from ..data.deeppcb import CLASSES, iter_samples
from ..evaluation.detector import SOURCE, ground_truth, prepared, write_table
from ..evaluation.metrics import iou, match_image
from .trace import tensor_input


def trace_failures(seed):
    folder = ROOT / "outputs" / f"evaluation_s{seed}"
    cases = json.loads((folder / "paired.json").read_text())["first_examples"]
    threshold = json.loads((folder / "operating_point.json").read_text())["threshold"]
    samples = [sample for sample in iter_samples(SOURCE, "test") if sample.key in cases.values()]
    truth = ground_truth(samples)
    rows = []
    torch.set_num_threads(2)
    for arm in ("aligned", "ring", "solid"):
        detector = YOLO(str(ROOT / "runs" / f"{arm}_s{seed}/weights/best.pt"))
        for sample in samples:
            image = prepared(sample.read_images(), arm, (2, 0))
            result = detector.predict(image, device="cpu", imgsz=640, conf=0.001, iou=0.7, verbose=False)[0]
            network = detector.predictor.model.model
            if network.model[-1].end2end:
                raise ValueError("This diagnostic expects the dense pre-NMS branch")
            with torch.inference_mode():
                dense = network(tensor_input(image))[0][0].T
                boxes = xywh2xyxy(dense[:, :4]).numpy()
                scores = dense[:, 4:].numpy()
            hits, _, _ = match_image(truth[sample.key], result.boxes.data.tolist(), threshold)
            for idx, (box, hit) in enumerate(zip(truth[sample.key], hits)):
                overlaps = iou(box, boxes)
                eligible = scores[overlaps >= 0.50]
                best = eligible.max(axis=0) if len(eligible) else scores[0] * 0
                rows.append({"arm": arm, "image": sample.key, "gt_index": idx, "gt_class": CLASSES[box[4]],
                             "dense_iou50_candidates": len(eligible), "max_iou": float(overlaps.max()),
                             "best_correct_score": float(best[box[4]]), "best_class": CLASSES[best.argmax()],
                             "best_score": float(best.max()), "matched_after_nms": bool(hit)})
    write_table(folder / "candidate_trace.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    trace_failures(parser.parse_args().seed)
