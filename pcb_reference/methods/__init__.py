import cv2

from .aligned import corrected_difference
from .raw import raw_difference
from .ring import gated_difference
from .single import single_input
from .solid import solid_difference


ARMS = ("single", "raw", "aligned", "ring", "solid")
DIFFERENCES = {"raw": raw_difference, "aligned": corrected_difference,
               "ring": gated_difference, "solid": solid_difference}


def make_input(inspection, template, arm):
    if arm == "single":
        return single_input(inspection, template)
    if arm not in DIFFERENCES:
        raise ValueError(f"Unknown input arm: {arm}")
    return cv2.merge([DIFFERENCES[arm](inspection, template), template, inspection])
