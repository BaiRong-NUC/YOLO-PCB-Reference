import cv2
import numpy as np

from ..aligned import corrected_difference
from ..raw import raw_difference


def ring_mean(values):
    outer = cv2.boxFilter(values, cv2.CV_32F, (15, 15), normalize=False)
    inner = cv2.boxFilter(values, cv2.CV_32F, (5, 5), normalize=False)
    return (outer - inner) / 200


def gated_difference(inspection, template, neighborhood_mean=ring_mean):
    raw = raw_difference(inspection, template)
    corrected = corrected_difference(inspection, template)
    return np.where(neighborhood_mean(raw) - neighborhood_mean(corrected) > 2.0, corrected, raw)
