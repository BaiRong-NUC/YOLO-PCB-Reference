import hashlib
import json
from pathlib import Path

from ultralytics import YOLO

from .. import ROOT
from ..data.deeppcb import CLASSES, iter_samples
from ..evaluation.detector import SOURCE, ground_truth
from ..evaluation.metrics import average_precision


def rescore_public_predictions(path, truth):
    records = json.loads(path.read_bytes())
    boxes = {key: [] for key in truth}
    for record in records:
        key = Path(record["file_name"]).stem
        left, top, width, height = record["bbox"]
        boxes[key].append([left, top, left + width, top + height, record["score"], record["category_id"] - 1])
    ap, ap50, _ = average_precision(truth, boxes)
    return {"boxes": len(records), "coco_ap50": ap50, "coco_ap5095": ap}


def evaluate():
    weights = ROOT / "weights/external_yolo11n.pt"
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != "23e511d0599f351cd115d9711b97371ccc1e8d10773e07de4590ee267901e554":
        raise ValueError(f"Unexpected author checkpoint: {digest}")
    model = YOLO(str(weights))
    if list(model.names.values()) != CLASSES:
        raise ValueError(model.names)
    run = {}

    def record_run(validator):
        run.update(images=validator.seen, directory=validator.save_dir)

    model.add_callback("on_val_end", record_run)
    metrics = model.val(data=str(ROOT / "data/single/data.yaml"), split="test", imgsz=640,
                        batch=16, conf=0.001, iou=0.7, device="cpu", workers=0, plots=False,
                        save_json=True, project=str(ROOT / "outputs/reproduction"), name="yolo11n")
    values = {"sha256": digest, "images": run["images"], "instances": int(metrics.nt_per_class.sum()),
              "precision": metrics.box.mp, "recall": metrics.box.mr,
              "map50": metrics.box.map50, "map5095": metrics.box.map}
    truth = ground_truth(list(iter_samples(SOURCE, "test")))
    values["local_predictions"] = rescore_public_predictions(run["directory"] / "predictions.json", truth)
    published = ROOT / "outputs/reproduction/yolo11n_author_predictions.json"
    if published.exists():
        values["author_predictions"] = rescore_public_predictions(published, truth)
    (run["directory"] / "metrics.json").write_text(json.dumps(values, indent=2), encoding="utf-8")
    print(values)


if __name__ == "__main__":
    evaluate()
