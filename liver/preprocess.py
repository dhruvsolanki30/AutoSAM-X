import numpy as np
import cv2

def preprocess_slice(slice_img):

    slice_img = slice_img.astype(np.float32)

    lower, upper = -100, 200
    slice_img = np.clip(slice_img, lower, upper)

    slice_img = (slice_img - lower) / (upper - lower)
    slice_img = (slice_img * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    processed = clahe.apply(slice_img)

    return processed