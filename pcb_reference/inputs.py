import cv2
import numpy as np

from .registration import align_template


ARMS = ["single", "raw", "aligned", "ring"]


def ring_mean(values):
    outer = cv2.boxFilter(values, cv2.CV_32F, (15, 15), normalize=False)
    inner = cv2.boxFilter(values, cv2.CV_32F, (5, 5), normalize=False)
    return (outer - inner) / 200


def make_input(inspection, template, arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown input arm: {arm}")
    if arm == "single":
        return inspection
    raw = cv2.absdiff(inspection, template)
    residual = raw
    if arm in {"aligned", "ring"}:
        aligned = align_template(inspection, template)
        corrected = cv2.absdiff(inspection, aligned)
        if arm == "ring":
            residual = np.where(ring_mean(raw) - ring_mean(corrected) > 2.0, corrected, raw)
        else:
            residual = corrected
    return cv2.merge([residual, template, inspection])
