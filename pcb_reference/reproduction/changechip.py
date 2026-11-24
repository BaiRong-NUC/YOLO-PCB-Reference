"""Run the authors' ChangeChip pipeline on complete CD-PCB pairs under Python 3.12."""

import argparse
import csv
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from .. import ROOT


def load_upstream(repository, deps):
    sys.path[:0] = [str(repository), str(deps)]
    import scipy.misc
    from sklearn.cluster import KMeans
    from skimage.util import img_as_ubyte

    scipy.misc.imsave = lambda path, values: Image.fromarray(img_as_ubyte(values)).save(path)
    scipy.misc.imread = lambda path: np.asarray(Image.open(path))
    scipy.misc.imresize = lambda values, size: np.asarray(Image.fromarray(values).resize(size[::-1]))
    if not hasattr(cv2, "xfeatures2d"):
        cv2.xfeatures2d = SimpleNamespace(SIFT_create=cv2.SIFT_create)
    import PCA_Kmeans
    from main import main

    PCA_Kmeans.KMeans = partial(KMeans, n_init=10)
    return main


def score_masks(pair, folder):
    accepted = np.atleast_1d(np.loadtxt(folder / "accepted_classes.csv", delimiter=","))
    clusters = np.loadtxt(folder / "clustering_data.csv", delimiter=",")
    mask = np.isin(clusters, accepted)
    return score_prediction(pair, mask)


def score_prediction(pair, mask):
    gt = cv2.imread(str(pair / "GT.JPG"))
    gt = cv2.resize(gt, mask.shape[::-1], interpolation=cv2.INTER_AREA)
    row = {"pair": pair.name, "selected": int(mask.sum()), "pixels": mask.size}
    for name, foreground in (("author_white", np.all(gt == 255, axis=-1)),
                             ("dark_annotation", np.all(gt < 128, axis=-1))):
        tp = int((mask & foreground).sum())
        row.update({f"{name}_tp": tp, f"{name}_gt": int(foreground.sum()),
                    f"{name}_precision": tp / max(int(mask.sum()), 1),
                    f"{name}_recall": tp / max(int(foreground.sum()), 1)})
    return row


def run(repository, source, output, rescore):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from threadpoolctl import threadpool_limits

    main = load_upstream(repository, ROOT / "outputs/reproduction/deps")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair in sorted(source.glob("pair_*"), key=lambda path: int(path.name.split("_")[-1])):
        if not all((pair / name).is_file() for name in ("input.JPG", "reference.JPG", "GT.JPG")):
            continue
        folder = output / pair.name
        np.random.seed(0)
        cv2.setRNGSeed(0)
        if not rescore:
            try:
                with threadpool_limits(limits=2):
                    main(str(folder), str(pair / "input.JPG"), str(pair / "reference.JPG"),
                         16, 5, 3, 9, False, True, True, 0.2, False)
            except SystemExit as error:
                if error.code != 0:
                    raise
                (folder / "accepted_classes.csv").write_text("", encoding="utf-8")
        rows.append(score_masks(pair, folder))
        with (output / "per_pair.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        print(rows[-1], flush=True)
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=ROOT / "data/ChangeChip")
    parser.add_argument("--source", type=Path, default=ROOT / "data/cdpcb/CD-PCB")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reproduction/changechip")
    parser.add_argument("--rescore", action="store_true")
    args = parser.parse_args()
    run(args.repository.resolve(), args.source, args.output, args.rescore)
