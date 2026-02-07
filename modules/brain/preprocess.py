# Preprocessing module
import numpy as np

def normalize_mri(image):
    """
    Normalize MRI image to 0–1 range
    """
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
    return image

