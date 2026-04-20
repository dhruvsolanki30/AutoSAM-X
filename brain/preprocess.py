import numpy as np
import cv2

def preprocess_slice(slice_img):
    """
    Normalize MRI slice and apply CLAHE
    """

    slice_img = slice_img.astype(np.float32)

    slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)

    slice_img = (slice_img * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    processed = clahe.apply(slice_img)

    return processed