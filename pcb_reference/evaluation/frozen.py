import argparse
import gzip
import hashlib
import json

import torch
from ultralytics import YOLO

from .. import ROOT
from ..data.deeppcb import iter_samples
from .comparisons import bootstrap
from .detector import SOURCE, ground_truth, prediction_file, write_table
from .metrics import average_precision, counts


def compare_inputs(seed, device):
    destination = ROOT / "outputs" / f"evaluation_s{seed}"
    operating_point = json.loads((destination / "operating_point.json").read_text())
    folder = destination / "frozen_aligned"
    folder.mkdir(exist_ok=True)
    samples = list(iter_samples(SOURCE, "test"))
    truth = ground_truth(samples)
    weights = ROOT / "runs" / f"aligned_s{seed}" / "weights/best.pt"
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    model = YOLO(str(weights))
    results, by_image = [], {}
    for arm in ("aligned", "ring", "solid", "raw"):
        if arm == "aligned":
            with gzip.open(destination / "aligned/dx2_dy0.json.gz", "rt") as stream:
                saved = json.load(stream)
            if saved["weights_sha256"] != digest:
                raise ValueError("Primary predictions came from different weights")
            predictions = saved["boxes"]
        else:
            predictions = prediction_file(model, samples, arm, (2, 0), device, folder / f"{arm}.json.gz", digest)
        total, _, by_image[arm] = counts(truth, predictions, operating_point["threshold"], operating_point["small_area"])
        ap, ap50, _ = average_precision(truth, predictions)
        results.append(dict(input=arm, **total, ap=ap, ap50=ap50))
    write_table(folder / "conditions.csv", results)
    intervals = {name: bootstrap(by_image["aligned"], values, list(truth))
                 for name, values in by_image.items() if name != "aligned"}
    (folder / "paired.json").write_text(json.dumps(intervals, indent=2), encoding="utf-8")
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change only the residual for a frozen aligned detector.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    compare_inputs(args.seed, args.device)
