import cv2

from .registration import align_template


def corrected_difference(inspection, template):
    return cv2.absdiff(inspection, align_template(inspection, template))
