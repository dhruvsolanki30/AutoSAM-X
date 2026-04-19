"""
Shared mask refinement and post-processing utilities
"""
import cv2
import numpy as np


def refine_mask(mask, open_kernel=5, close_kernel=5, blur_kernel=5):
    """
    Refine segmentation mask using morphological operations
    
    Args:
        mask: Binary mask (boolean or uint8)
        open_kernel: Kernel size for morphological opening
        close_kernel: Kernel size for morphological closing
        blur_kernel: Kernel size for Gaussian blur
    
    Returns:
        refined_mask: Cleaned up binary mask (uint8)
    """
    # Convert boolean mask to uint8
    mask = mask.astype(np.uint8) * 255
    
    # Remove small noise
    kernel_open = np.ones((open_kernel, open_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    # Fill small holes
    kernel_close = np.ones((close_kernel, close_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Smooth edges
    mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
    
    # Convert back to binary
    mask = (mask > 127).astype(np.uint8)
    
    return mask


def compute_mask_metrics(mask):
    """
    Compute metrics from a segmentation mask
    
    Returns:
        dict: area, perimeter, circularity, centroid
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    metrics = {
        'area': 0,
        'perimeter': 0,
        'circularity': 0,
        'centroid': (0, 0)
    }
    
    if len(contours) == 0:
        return metrics
    
    # Get largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Area (number of pixels)
    area = cv2.contourArea(contour)
    metrics['area'] = int(area)
    
    # Perimeter
    perimeter = cv2.arcLength(contour, True)
    metrics['perimeter'] = perimeter
    
    # Circularity (4π * area / perimeter²)
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)
        metrics['circularity'] = circularity
    
    # Centroid
    M = cv2.moments(contour)
    if M['m00'] > 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        metrics['centroid'] = (cx, cy)
    
    return metrics
