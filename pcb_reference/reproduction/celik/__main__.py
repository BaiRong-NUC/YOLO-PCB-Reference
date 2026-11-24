"""Celik (2009), Eqs. 1–6, evaluated after the same PCB preprocessing as ChangeChip."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from ... import ROOT
from ..changechip import load_upstream, score_prediction


def predict(inspection, template):
    intensity = cv2.cvtColor(inspection, cv2.COLOR_BGR2GRAY)
    reference = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    difference = cv2.absdiff(intensity, reference)
    height, width = difference.shape
    block = 4
    if not difference.any():
        return np.zeros_like(difference, dtype=bool)
    samples = sliding_window_view(difference, (block, block))[::block, ::block].reshape(-1, block**2)
    projection = PCA(n_components=3, svd_solver="full").fit(samples.astype(np.float64))
    padded = np.pad(difference, ((1, 2), (1, 2)), mode="edge")
    windows = sliding_window_view(padded, (block, block))
    features = np.empty((height, width, 3), dtype=np.float64)
    for start in range(0, height, 64):
        stripe = windows[start:start + 64]
        features[start:start + 64] = projection.transform(stripe.reshape(-1, block**2)).reshape(-1, width, 3)
    labels = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(features.reshape(-1, 3))
    counts = np.bincount(labels, minlength=2)
    if counts.min() == 0:
        return np.zeros_like(difference, dtype=bool)
    means = np.bincount(labels, weights=difference.ravel(), minlength=2) / counts
    return (labels == means.argmax()).reshape(height, width)


def paired_input(pair, scratch, observed):
    import global_variables
    from light_differences_elimination import light_diff_elimination
    from registration import homography

    global_variables.init(str(scratch), False)
    inspection, template = [cv2.imread(str(pair / filename)) for filename in ("input.JPG", "reference.JPG")]
    size = tuple(int(0.2 * 0.5 * (inspection.shape[axis] + template.shape[axis])) for axis in (1, 0))
    inspection, template = [cv2.resize(image, size, interpolation=cv2.INTER_AREA) for image in (inspection, template)]
    cv2.setRNGSeed(0)
    registered, _, blank = homography(False, inspection, template, None)
    inspection[blank] = 0
    registered[blank] = 0
    corrected = light_diff_elimination(inspection, registered)
    filename = "image2_registered_histogram_matched.jpg"
    if (scratch / filename).read_bytes() != (observed / filename).read_bytes():
        raise ValueError(f"{pair.name}: preprocessed template differs from archived ChangeChip output")
    return inspection, corrected


def main():
    load_upstream(ROOT / "data/ChangeChip", ROOT / "outputs/reproduction/deps")
    observed = ROOT / "outputs/reproduction/changechip"
    destination = ROOT / "outputs/reproduction/celik"
    with (observed / "per_pair.csv").open(encoding="utf-8") as stream:
        recorded = list(csv.DictReader(stream))
    rows = []
    destination.mkdir(exist_ok=True)
    with TemporaryDirectory() as temporary, threadpool_limits(limits=2):
        for actual in recorded:
            pair = ROOT / "data/cdpcb/CD-PCB" / actual["pair"]
            images = paired_input(pair, Path(temporary), observed / pair.name)
            prediction = predict(*images)
            row = score_prediction(pair, prediction)
            for field in ("pixels", "author_white_gt", "dark_annotation_gt"):
                if row[field] != int(actual[field]):
                    raise ValueError(f"{pair.name}: inconsistent {field}")
            rows.append(row)
            print(f"{pair.name}: {row['selected']} predicted pixels", flush=True)
    with (destination / "per_pair.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
