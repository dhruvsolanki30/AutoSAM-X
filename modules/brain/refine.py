# Refinement module
import cv2

def refine_mask(mask):
    """
    Simple morphological refinement
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined = cv2.morphologyEx(mask.astype('uint8'), cv2.MORPH_CLOSE, kernel)
    return refined

