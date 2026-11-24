import cv2
import numpy as np


def translate(image, dx, dy):
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    height, width = image.shape
    return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


def align_template(inspection, template):
    shift, response = cv2.phaseCorrelate(template.astype(np.float32), inspection.astype(np.float32))
    if response > 0 and max(map(abs, shift)) <= 8:
        return translate(template, round(shift[0]), round(shift[1]))
    return template
