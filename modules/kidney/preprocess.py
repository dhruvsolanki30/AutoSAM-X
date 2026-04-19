import numpy as np

def preprocess_slice(slice_img):
    slice_img = slice_img.astype(np.float32)

    if slice_img.max() != slice_img.min():
        slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min())
    else:
        slice_img = np.zeros_like(slice_img)

    slice_img = (slice_img * 255).astype(np.uint8)

    return slice_img