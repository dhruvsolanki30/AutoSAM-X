import cv2
import os
import numpy as np


def save_final_output(original, processed, mask, result, option):

    os.makedirs("results", exist_ok=True)

    # Convert original to uint8 if needed
    if original.max() > 255:
        original = (original / original.max() * 255).astype(np.uint8)

    # Save Input Image
    cv2.imwrite("results/input.png", original)

    # Save Processed Image
    cv2.imwrite("results/processed.png", processed)

    # Save Segmented Mask
    cv2.imwrite("results/segmented.png", mask * 255)

    # Create overlay
    overlay = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    disease_colors = {
        "tumor": (0, 0, 255),           # Red
        "stone": (255, 255, 0),         # Yellow
        "cyst": (0, 255, 0),            # Green
        "atrophy": (255, 0, 0),         # Blue
        "hydronephrosis": (0, 255, 255) # Cyan
    }

    color = disease_colors.get(option, (0, 0, 255))

    overlay[mask > 0] = color

    cv2.imwrite("results/overlay.png", overlay)

    # Save classification results
    with open("results/final_result.txt", "w") as f:

        f.write("Kidney Analysis Result\n")
        f.write("=====================\n\n")

        for key, value in result.items():
            f.write(f"{key}: {value}\n")