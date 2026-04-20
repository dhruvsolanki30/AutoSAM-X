import os
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor

# Get project root directory (brain/ -> AutoSAM-X/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SAM model path
SAM_PATH = os.path.join(BASE_DIR, "sam_vit_b.pth")

# Load SAM model
sam = sam_model_registry["vit_b"](checkpoint=SAM_PATH)
predictor = SamPredictor(sam)

def segment_with_sam(image, box):
    """
    SAM segmentation using bounding box
    """

    if box is None:
        return np.zeros_like(image)

    # SAM expects 3-channel image
    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    predictor.set_image(img)

    input_box = np.array(box)

    masks, scores, logits = predictor.predict(
        box=input_box,
        multimask_output=False
    )

    return masks[0].astype(np.uint8) * 255