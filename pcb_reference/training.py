from ultralytics import YOLO

from . import ROOT


def train(arm, seed):
    dataset = ROOT / "data" / arm / "data.yaml"
    if not dataset.is_file():
        raise FileNotFoundError(f"Prepare the {arm} dataset first: python prepare.py --arm {arm}")
    weights = ROOT / "weights" / "yolo26s.pt"
    weights.parent.mkdir(exist_ok=True)
    YOLO(str(weights), task="detect").train(
        data=str(dataset),
        project=str(ROOT / "runs"),
        name=f"{arm}_s{seed}",
        epochs=30,
        imgsz=640,
        batch=8,
        optimizer="SGD",
        seed=seed,
        deterministic=True,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        bgr=0.0,
        augmentations=[],
    )
