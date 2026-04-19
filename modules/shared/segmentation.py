"""
Shared SAM (Segment Anything Model) segmentation utilities
"""
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import torch

# Path to SAM model
SAM_CHECKPOINT = "sam_vit_b.pth"

# Initialize SAM globally
_sam = None
_predictor = None


def initialize_sam():
    """Initialize SAM model globally (only once)"""
    global _sam, _predictor
    
    if _predictor is not None:
        return _predictor
    
    print("Initializing SAM model...")
    
    _sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    _sam.to(device)
    _predictor = SamPredictor(_sam)
    
    return _predictor


def segment_with_bbox(image, bbox):
    """
    Segment an object in image using bounding box prompt
    
    Args:
        image: RGB image (H x W x 3)
        bbox: Bounding box [x1, y1, x2, y2]
    
    Returns:
        mask: Binary segmentation mask
    """
    predictor = initialize_sam()
    
    # Convert bbox to numpy array
    box = np.array(bbox)
    
    # Set image for SAM
    predictor.set_image(image)
    
    # Predict mask from bounding box
    masks, scores, logits = predictor.predict(
        box=box,
        multimask_output=False
    )
    
    mask = masks[0]
    
    return mask


def segment_with_point(image, point, label=1):
    """
    Segment an object using a point prompt
    
    Args:
        image: RGB image (H x W x 3)
        point: Point coordinates [x, y]
        label: Foreground (1) or background (0)
    
    Returns:
        mask: Binary segmentation mask
    """
    predictor = initialize_sam()
    
    # Convert to numpy arrays
    input_point = np.array([point])
    input_label = np.array([label])
    
    # Set image for SAM
    predictor.set_image(image)
    
    # Predict mask from point
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )
    
    mask = masks[0]
    
    return mask
