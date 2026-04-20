"""
Shared visualization utilities for all pathologies
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2


def visualize_pathology_analysis(image, bbox, mask, analysis_text, title="Analysis"):
    """
    Visualize pathology detection with 3-panel layout
    
    Args:
        image: Original CT slice (RGB)
        bbox: Bounding box [x1, y1, x2, y2] or None
        mask: Segmentation mask or None
        analysis_text: Dict with analysis information to display
        title: Title of the analysis
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Panel 1: Original image
    ax[0].imshow(image)
    ax[0].set_title("Original CT Slice")
    ax[0].axis("off")
    
    # Panel 2: Detection (with bounding box)
    img_bbox = image.copy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_bbox, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_bbox, "Detection", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    ax[1].imshow(img_bbox)
    ax[1].set_title("Detection")
    ax[1].axis("off")
    
    # Panel 3: Segmentation with overlay + text info
    img_mask = image.copy()
    if mask is not None:
        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 0] = mask * 255  # Red channel
        img_mask = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
    
    # Add analysis text as overlay
    text_y = 30
    for key, value in analysis_text.items():
        text = f"{key}: {value}"
        cv2.putText(img_mask, text, (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        text_y += 30
    
    ax[2].imshow(img_mask)
    ax[2].set_title("Segmentation + Analysis")
    ax[2].axis("off")
    
    plt.tight_layout()
    plt.show()


def draw_analysis_overlay(image, mask, info_dict):
    """
    Draw analysis text directly on image
    
    Args:
        image: Image to draw on
        mask: Segmentation mask
        info_dict: Dictionary of key-value pairs to display
    
    Returns:
        annotated_image: Image with text overlay
    """
    img = image.copy()
    
    # Draw mask contours
    if mask is not None and mask.any():
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    
    # Draw info text
    text_y = 40
    for key, value in info_dict.items():
        text = f"{key}: {value}"
        cv2.putText(img, text, (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        text_y += 35
    
    return img
