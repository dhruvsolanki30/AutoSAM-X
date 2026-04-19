import cv2
import numpy as np
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modules.brain.preprocess import preprocess_slice
from modules.brain.refine import refine_mask
from modules.brain.detect import create_brain_mask


def get_stroke_box(image):
    """
    Get bounding box for stroke detection using brain-localized hypodense regions.
    """
    brain_mask = create_brain_mask(image)
    brain_pixels = image[brain_mask > 0]

    if brain_pixels.size == 0:
        return None

    mean_intensity = np.mean(brain_pixels)
    std_intensity = np.std(brain_pixels)
    threshold_value = int(np.clip(mean_intensity - std_intensity * 0.7, 10, 110))

    _, dark = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, brain_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=2)
    dark = cv2.erode(dark, kernel, iterations=1)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 250:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < 20 or h < 20 or w * h < 600:
            continue

        aspect_ratio = float(w) / max(h, 1)
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
            continue

        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        region_mean = np.mean(image[mask > 0])
        if region_mean > mean_intensity - std_intensity * 0.25:
            continue

        candidates.append((contour, area))

    if not candidates:
        return None

    best_contour, area = max(candidates, key=lambda x: x[1])
    x, y, w, h = cv2.boundingRect(best_contour)
    return [x, y, x + w, y + h], area


def segment_stroke(image, box):
    """
    Create segmentation mask for stroke using localized low-intensity region detection.
    """
    if box is None:
        return np.zeros_like(image)

    x1, y1, x2, y2 = box
    roi = image[y1:y2, x1:x2]

    mean_roi = np.mean(roi)
    std_roi = np.std(roi)
    threshold_value = int(np.clip(mean_roi - std_roi * 0.45, 10, 120))

    _, mask1 = cv2.threshold(roi, threshold_value, 255, cv2.THRESH_BINARY_INV)
    mask2 = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 4
    )

    combined_mask = cv2.bitwise_and(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        filtered = np.zeros_like(combined_mask)
        cv2.drawContours(filtered, contours, -1, 255, -1)
        combined_mask = filtered

    full_mask = np.zeros_like(image)
    full_mask[y1:y2, x1:x2] = combined_mask
    return full_mask


def run_stroke(image_path):
    """
    Stroke detection with robust brain-aware segmentation.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not load image")

    processed = preprocess_slice(img)
    result = get_stroke_box(processed)

    if result is None:
        box = None
        stroke_area = 0
        confidence = 0.0
        segmentation = np.zeros_like(processed)
    else:
        box, area = result
        segmentation = segment_stroke(processed, box)
        refined = refine_mask(segmentation)
        stroke_area = np.sum(refined > 0)

        if stroke_area > 0:
            brain_mask = create_brain_mask(processed)
            brain_pixels = processed[brain_mask > 0]
            region_mean = np.mean(processed[refined > 0]) if np.any(refined > 0) else 0
            contrast = max(0.0, np.mean(brain_pixels) - region_mean)
            confidence = min(0.96, 0.35 + contrast / 120.0 + min(area / 14000.0, 0.4))
        else:
            confidence = 0.0

    if 'stroke_area' not in locals():
        refined = refine_mask(segmentation)
        stroke_area = np.sum(refined > 0)
        confidence = 0.0

    if stroke_area > 100:
        detection = "Stroke Detected"
        if stroke_area < 1000:
            severity = "Mild"
        elif stroke_area < 5000:
            severity = "Moderate"
        else:
            severity = "Severe"
    else:
        detection = "No Stroke Detected"
        severity = "None"
        confidence = 0.0

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Detection")
    plt.imshow(img, cmap="gray")
    if box is not None:
        x1, y1, x2, y2 = box
        plt.gca().add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor="yellow", linewidth=2, fill=False)
        )
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Segmentation")
    plt.imshow(img, cmap="gray")
    if stroke_area > 0:
        plt.imshow(refined, cmap="Blues", alpha=0.6)
    plt.axis("off")
    plt.tight_layout()

    output_filename = f"stroke_{int(time.time())}.png"
    save_path = os.path.join("static", "outputs", output_filename)
    return_path = f"/static/outputs/{output_filename}"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return {
        "output_image": return_path,
        "detection": detection,
        "confidence": round(confidence, 2),
        "tumor_area": int(stroke_area),
        "severity": severity
    }

    # Preprocess
    processed = preprocess_slice(img)

    # Get stroke bounding box
    result = get_stroke_box(processed)

    if result is None:
        box = None
        area = 0
        confidence = 0.0
        segmentation = np.zeros_like(processed)
    else:
        box, area = result
        confidence = min(0.95, area / 15000)  # Confidence based on area

        # Advanced segmentation
        segmentation = segment_stroke(processed, box)

    # Refinement
    refined = refine_mask(segmentation)

    # Calculate final area
    stroke_area = np.sum(refined > 0)

    # Determine detection result
    if stroke_area > 100:
        detection = "Stroke Detected"
        if stroke_area < 1000:
            severity = "Mild"
        elif stroke_area < 5000:
            severity = "Moderate"
        else:
            severity = "Severe"
    else:
        detection = "No Stroke Detected"
        severity = "None"
        confidence = 0.0

    # Visualization
    plt.figure(figsize=(12, 4))

    # Original
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    # Detection
    plt.subplot(1, 3, 2)
    plt.title("Detection")
    plt.imshow(img, cmap="gray")

    if box is not None:
        x1, y1, x2, y2 = box
        plt.gca().add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                edgecolor="yellow",
                linewidth=2,
                fill=False
            )
        )

    plt.axis("off")

    # Segmentation
    plt.subplot(1, 3, 3)
    plt.title("Segmentation")
    plt.imshow(img, cmap="gray")

    if stroke_area > 0:
        plt.imshow(refined, cmap="Blues", alpha=0.6)

    plt.axis("off")
    plt.tight_layout()

    # Save output
    output_filename = f"stroke_{int(time.time())}.png"
    save_path = os.path.join("static", "outputs", output_filename)
    return_path = f"/static/outputs/{output_filename}"

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return {
        "output_image": return_path,
        "detection": detection,
        "confidence": round(confidence, 2),
        "tumor_area": int(stroke_area),
        "severity": severity
    }