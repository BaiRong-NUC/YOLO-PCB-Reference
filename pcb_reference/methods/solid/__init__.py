import cv2

from ..ring import gated_difference


def solid_mean(values):
    return cv2.boxFilter(values, cv2.CV_32F, (15, 15))


def solid_difference(inspection, template):
    return gated_difference(inspection, template, solid_mean)
