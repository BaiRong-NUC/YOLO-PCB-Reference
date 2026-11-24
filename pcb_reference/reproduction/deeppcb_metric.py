"""Score HBB submissions with the original DeepPCB matching and AP functions."""

import argparse
import gzip
import json
import sys
from lib2to3.refactor import RefactoringTool, get_fixers_from_package
from types import ModuleType
from zipfile import ZipFile, ZIP_DEFLATED

from .. import ROOT
from ..data.deeppcb import CLASSES, iter_samples
from ..evaluation.detector import SOURCE, ground_truth
from ..methods import ARMS


def load_evaluator(directory):
    sys.path.insert(0, str(ROOT / "outputs/reproduction/polygon/Lib/site-packages"))
    fixers = [fixer for fixer in get_fixers_from_package("lib2to3.fixes")
              if not fixer.endswith(("fix_unicode", "fix_import"))]
    converter = RefactoringTool(fixers)
    for name in ("rrc_evaluation_funcs", "script"):
        path = directory / f"{name}.py"
        source = path.read_text(encoding="utf-8")
        if name == "script":
            source = source.replace("'([0-9]+).txt' #", "'([0-9]+).txt', #")
            source = source.replace("'([0-9]+).txt'#", "'([0-9]+).txt', #")
        code = str(converter.refactor_string(source + "\n", str(path)))
        module = ModuleType(name)
        module.unicode = lambda text, encoding: text
        sys.modules[name] = module
        exec(compile(code, str(path), "exec"), module.__dict__)
    return module


def prepare_gt(original, destination, truth):
    with ZipFile(original) as src, ZipFile(destination, "w", ZIP_DEFLATED) as out:
        if set(src.namelist()) != {f"{key}.txt" for key in truth}:
            raise ValueError("Official archive and test split disagree")
        for name in src.namelist():
            rows = []
            boxes = []
            for line in src.read(name).decode("utf-8-sig").splitlines():
                if line.strip():
                    left, top, right, bottom, category = map(int, line.split())
                    boxes.append([left, top, right, bottom, category - 1])
                    rows.append(f"{left},{top},{right},{bottom},{CLASSES[category-1]}")
            if sorted(boxes) != sorted(truth[name[:-4]]):
                raise ValueError(f"Official archive annotations disagree: {name}")
            out.writestr(name, "\n".join(rows))


def submit(predictions, destination, threshold):
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for key, boxes in predictions.items():
            rows = [",".join(map(str, (round(left), round(top), round(right), round(bottom),
                                       f"{score:.8f}", CLASSES[int(category)])))
                    for left, top, right, bottom, score, category in boxes if score >= threshold]
            archive.writestr(f"{key}.txt", "\n".join(rows))


def evaluate(seed, oracle_only):
    directory = ROOT / "data/DeepPCB/evaluation"
    destination = ROOT / "outputs/reproduction" / f"official_s{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    truth = ground_truth(list(iter_samples(SOURCE, "test")))
    gt = destination / "gt_ltrb.zip"
    prepare_gt(directory / "gt.zip", gt, truth)
    evaluator = load_evaluator(directory)
    params = dict(evaluator.default_evaluation_params(), LTRB=True)

    def score(boxes, name, threshold):
        pred = destination / f"{name}.zip"
        submit(boxes, pred, threshold)
        evaluator.validate_data(str(gt), str(pred), params)
        return evaluator.evaluate_method(str(gt), str(pred), params)["method"]

    perfect = {key: [[*coords, 1.0, category] for *coords, category in boxes] for key, boxes in truth.items()}
    results = {"oracle": score(perfect, "oracle", 0)}
    if not all(value == 1 for value in results["oracle"].values()):
        raise ValueError(results)
    if not oracle_only:
        folder = ROOT / "outputs" / f"evaluation_s{seed}"
        threshold = json.loads((folder / "operating_point.json").read_text())["threshold"]
        for arm in ARMS:
            with gzip.open(folder / arm / "dx0_dy0.json.gz", "rt") as stream:
                boxes = json.load(stream)["boxes"]
            results[arm] = {"ranked": score(boxes, arm + "_all", 0.001),
                            "fixed": score(boxes, arm + "_fixed", threshold)}
    path = destination / ("oracle.json" if oracle_only else "metrics.json")
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--oracle-only", action="store_true")
    args = parser.parse_args()
    evaluate(args.seed, args.oracle_only)
