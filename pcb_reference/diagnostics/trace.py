import json
from copy import deepcopy

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel

from .. import ROOT
from ..data.deeppcb import iter_samples
from ..data.export import convert_labels
from ..methods import ARMS, make_input
from ..methods.aligned.registration import translate


def tensor_input(image):
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
    rgb = np.ascontiguousarray(bgr[:, :, ::-1].transpose(2, 0, 1))
    return torch.from_numpy(rgb).unsqueeze(0).float() / 255


def trace_arm(arm, pretrained, inspection, template, labels):
    torch.manual_seed(0)
    model = DetectionModel(deepcopy(pretrained.yaml), nc=6, verbose=False)
    model.load(pretrained, verbose=False)
    model.args = get_cfg(overrides={"epochs": 30})
    model.train()
    criterion = model.init_criterion()
    shapes, matches = {}, {}

    def capture_shape(layer, inputs, output):
        shapes[f"{layer.i}_{type(layer).__name__}"] = list(output.shape)

    def capture_assignments(name):
        def hook(module, inputs, outputs):
            foreground = outputs[3]
            matches[name] = {"positive_anchors": int(foreground.sum()),
                             "matched_gt": int(torch.unique(outputs[4][foreground]).numel())}
        return hook

    handles = [model.model[idx].register_forward_hook(capture_shape)
               for idx in (0, 4, 6, 10, 13, 16, 19, 22)]
    handles.extend(getattr(criterion, name).assigner.register_forward_hook(capture_assignments(name))
                   for name in ("one2many", "one2one"))
    image = tensor_input(make_input(inspection, template, arm))
    targets = torch.tensor([[float(value) for value in row.split()] for row in labels.splitlines()])
    batch = {"img": image, "batch_idx": torch.zeros(len(targets)), "cls": targets[:, :1],
             "bboxes": targets[:, 1:]}
    outputs = model(image)
    losses, items = criterion(outputs, batch)
    losses.sum().backward()
    grad = float(model.model[0].conv.weight.grad.norm())
    if not torch.isfinite(losses).all() or not np.isfinite(grad) or grad <= 0:
        raise ValueError(f"Invalid training chain in {arm}")
    for handle in handles:
        handle.remove()
    with torch.no_grad():
        decoded, _ = model.eval()(image)
    return {"input": list(image.shape), "layers": shapes, "branches": {
        key: {part: list(branch[part].shape) for part in ("boxes", "scores")}
        for key, branch in outputs.items()}, "criterion": type(criterion).__name__,
        "matches": matches, "weighted_total": float(losses.detach().sum()),
        "one2one_loss": {key: float(value) for key, value in items.items()},
        "stem_gradient_norm": grad, "decoded": list(decoded.shape)}


def run(output):
    torch.set_num_threads(2)
    pairs = iter_samples(ROOT / "data/DeepPCB/PCBData", "trainval")
    sample = next(sample for sample in pairs if sample.split == "val")
    inspection, template = sample.read_images()
    labels = convert_labels(sample.annotation, *inspection.shape[::-1])
    pretrained = YOLO(str(ROOT / "weights/yolo26s.pt")).model
    result = {"key": sample.key, "labels": labels.splitlines(), "dx": 2, "dy": 0,
              "arms": {arm: trace_arm(arm, pretrained, inspection, translate(template, 2, 0), labels)
                       for arm in ARMS}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    run(ROOT / "outputs/trace/forward.json")
