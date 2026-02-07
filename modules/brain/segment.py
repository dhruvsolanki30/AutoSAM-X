# Segmentation module
import numpy as np

def segment_with_sam(image, box):
    """
    Placeholder for SAM segmentation using box prompt.
    """
    x1, y1, x2, y2 = box
    mask = np.zeros_like(image)

    mask[y1:y2, x1:x2] = 1
    return mask

