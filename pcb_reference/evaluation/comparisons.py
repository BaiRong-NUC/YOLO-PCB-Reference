import argparse
import csv
import gzip
import json

import cv2
import numpy as np

from .. import ROOT
from ..data.deeppcb import CLASSES, iter_samples
from ..methods import make_input
from ..methods.aligned.registration import translate
from .detector import SOURCE, ground_truth, write_table
from .metrics import match_image


def load_rows(folder, arm):
    with (folder / arm / "primary_by_image.csv").open(encoding="utf-8") as stream:
        return {row.pop("image"): {key: int(value) for key, value in row.items()}
                for row in csv.DictReader(stream)}


def bootstrap(first, second, keys):
    values = np.asarray([[second[key][field] - first[key][field] for field in ("fp", "small_tp")]
                         + [first[key]["small_gt"]] for key in keys])
    rng = np.random.default_rng(0)
    sampled = rng.integers(len(keys), size=(2000, len(keys)))
    totals = values[sampled].sum(axis=1)
    intervals = np.percentile(np.column_stack((totals[:, 0] / len(keys),
                                               totals[:, 1] / np.maximum(totals[:, 2], 1))), (2.5, 97.5), axis=0)
    return {"images": len(keys), "small_gt": int(values[:, 2].sum()),
            "delta_fp_image": float(values[:, 0].mean()), "fp_ci": intervals[:, 0].tolist(),
            "delta_small_recall": float(values[:, 1].sum() / max(values[:, 2].sum(), 1)),
            "small_recall_ci": intervals[:, 1].tolist()}


def caption(panel, text):
    space = cv2.copyMakeBorder(panel, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    cv2.putText(space, text, (9, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (30, 30, 30), 1, cv2.LINE_AA)
    return space


def draw_box(canvas, box, color, label):
    left, top, right, bottom = map(round, box[:4])
    cv2.rectangle(canvas, (left, top), (right, bottom), color, 2)
    cv2.putText(canvas, label, (max(0, min(left, 550)), max(14, top-4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)


def gallery(folder, sample, threshold, detections, truth, kind):
    inspection, template = sample.read_images()
    shifted = translate(template, 2, 0)
    upper, lower = [], []
    gt = cv2.cvtColor(inspection, cv2.COLOR_GRAY2BGR)
    for box in truth:
        draw_box(gt, box, (240, 160, 30), CLASSES[box[4]])
    upper.append(caption(gt, f"{sample.key} | GT ({len(truth)})"))
    lower.append(caption(cv2.cvtColor(shifted, cv2.COLOR_GRAY2BGR), "Template shifted +2 px"))
    for arm in ("aligned", "ring", "solid"):
        matched, retained, correct = match_image(truth, detections[arm][sample.key], threshold)
        panel = cv2.cvtColor(inspection, cv2.COLOR_GRAY2BGR)
        for box, hit in zip(truth, matched):
            if not hit:
                draw_box(panel, box, (255, 80, 210), "FN " + CLASSES[box[4]])
        for box, hit in zip(retained, correct):
            draw_box(panel, box, (70, 205, 20) if hit else (15, 60, 245),
                     f"{CLASSES[int(box[5])]} {box[4]:.2f}")
        upper.append(caption(panel, f"{arm} | TP {sum(correct)}  FP {len(retained)-sum(correct)}  FN {sum(~matched)}"))
        residual = make_input(inspection, shifted, arm)[:, :, 0]
        lower.append(caption(cv2.cvtColor(residual, cv2.COLOR_GRAY2BGR), f"{arm} residual [0,255]"))
    path = folder / f"{kind}_{sample.key}.png"
    if not cv2.imwrite(str(path), cv2.vconcat([cv2.hconcat(upper), cv2.hconcat(lower)])):
        raise OSError(path)


def compare(seed):
    folder = ROOT / "outputs" / f"evaluation_s{seed}"
    threshold = json.loads((folder / "operating_point.json").read_text())["threshold"]
    samples = list(iter_samples(SOURCE, "test"))
    truth = ground_truth(samples)
    keys = list(truth)
    rows = {arm: load_rows(folder, arm) for arm in ("raw", "aligned", "ring", "solid", "single")}
    intervals = {f"{new}_minus_{old}": bootstrap(rows[old], rows[new], keys)
                 for new, old in (("raw", "single"), ("aligned", "raw"), ("ring", "aligned"), ("ring", "solid"))}
    distribution = []
    with_predictions = {}
    for arm in ("aligned", "ring", "solid"):
        with gzip.open(folder / arm / "dx2_dy0.json.gz", "rt") as stream:
            with_predictions[arm] = json.load(stream)["boxes"]
    examples = {}
    for sample in samples:
        errors = {}
        for arm in ("aligned", "ring", "solid"):
            row = rows[arm][sample.key]
            errors[arm] = row["fp"] + row["gt"] - row["tp"]
        delta = errors["ring"] - errors["aligned"]
        kind = "improved" if delta < 0 else "worse" if delta > 0 else "tied"
        distribution.append(dict(image=sample.key, **errors, outcome=kind))
        if kind not in examples:
            gallery(folder, sample, threshold, with_predictions, truth[sample.key], kind)
            examples[kind] = sample.key
    write_table(folder / "paired_errors.csv", distribution)
    result = {"bootstrap": intervals, "first_examples": examples,
              "outcomes": {kind: sum(row["outcome"] == kind for row in distribution)
                           for kind in ("improved", "tied", "worse")}}
    (folder / "paired.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    compare(parser.parse_args().seed)
