import numpy as np
import cv2


def analyze_liver_abnormality(image, liver_mask):
    """
    Option-2: ROI-based anomaly detection inside segmented liver

    Parameters:
        image       : preprocessed CT slice (uint8)
        liver_mask  : binary liver mask (0 or 255)

    Returns:
        anomaly_mask, stats
    """

    if liver_mask is None or np.sum(liver_mask) == 0:
        return np.zeros_like(image), {}

    # ---------------------------------------------------------
    # 1. Extract liver ROI
    # ---------------------------------------------------------

    roi_pixels = image[liver_mask > 0]

    mean = np.mean(roi_pixels)
    std = np.std(roi_pixels)

    # ---------------------------------------------------------
    # 2. Detect abnormal pixels (Z-score threshold)
    # ---------------------------------------------------------

    # Tumor-like regions often differ significantly in intensity
    threshold_low = mean - 1.5 * std
    threshold_high = mean + 1.5 * std

    anomaly_mask = np.zeros_like(image, dtype=np.uint8)

    anomaly_mask[(image < threshold_low) & (liver_mask > 0)] = 255
    anomaly_mask[(image > threshold_high) & (liver_mask > 0)] = 255

    # ---------------------------------------------------------
    # 3. Remove noise with morphology
    # ---------------------------------------------------------

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel)
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel)

    # ---------------------------------------------------------
    # 4. Keep only significant regions
    # ---------------------------------------------------------

    contours, _ = cv2.findContours(
        anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    clean_mask = np.zeros_like(anomaly_mask)

    min_area = 200  # Ignore tiny noise regions

    for c in contours:
        if cv2.contourArea(c) > min_area:
            cv2.drawContours(clean_mask, [c], -1, 255, -1)

    anomaly_mask = clean_mask

    # ---------------------------------------------------------
    # 5. Compute statistics
    # ---------------------------------------------------------

    liver_area = np.sum(liver_mask > 0)
    anomaly_area = np.sum(anomaly_mask > 0)

    ratio = anomaly_area / (liver_area + 1e-8)

    if ratio < 0.01:
        risk = "Low"
    elif ratio < 0.05:
        risk = "Medium"
    else:
        risk = "High"

    stats = {
        "liver_area": int(liver_area),
        "anomaly_area": int(anomaly_area),
        "ratio": float(ratio),
        "risk": risk,
    }

    return anomaly_mask, stats