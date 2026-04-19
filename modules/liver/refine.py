import cv2
import numpy as np

def refine_mask(mask):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined = cv2.morphologyEx(mask.astype('uint8') * 255, cv2.MORPH_CLOSE, kernel)

    return refined.astype(bool)