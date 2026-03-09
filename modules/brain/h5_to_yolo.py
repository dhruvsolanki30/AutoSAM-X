import os
import h5py
import numpy as np
import cv2

# Get project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Paths
INPUT_DIR = os.path.join(BASE_DIR, "data", "brats_h5")
IMG_DIR = os.path.join(BASE_DIR, "data", "yolo", "images")
LABEL_DIR = os.path.join(BASE_DIR, "data", "yolo", "labels")

# Ensure output folders exist
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)


def convert():

    for file in os.listdir(INPUT_DIR):

        if not file.endswith(".h5"):
            continue

        path = os.path.join(INPUT_DIR, file)

        # Load H5 data
        with h5py.File(path, "r") as f:
            image = np.array(f["image"])
            mask = np.array(f["mask"])

        # Skip slices without tumor
        if np.sum(mask) == 0:
            continue

        # Binary tumor mask
        tumor = (mask > 0).astype(np.uint8)

        # Ensure mask is single channel
        if len(tumor.shape) == 3:
            tumor = tumor[:, :, 0]

        tumor = (tumor * 255).astype(np.uint8)

        # Find tumor contours
        contours, _ = cv2.findContours(
            tumor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        # Largest tumor region
        c = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(c)

        H, W = tumor.shape

        # Convert to YOLO format
        x_center = (x + w / 2) / W
        y_center = (y + h / 2) / H
        width = w / W
        height = h / H

        name = file.replace(".h5", "")

        # Normalize image
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image = (image * 255).astype(np.uint8)

        # Save image
        cv2.imwrite(os.path.join(IMG_DIR, name + ".png"), image)

        # Save YOLO label
        with open(os.path.join(LABEL_DIR, name + ".txt"), "w") as f:
            f.write(f"0 {x_center} {y_center} {width} {height}")

    print("YOLO dataset created successfully!")


if __name__ == "__main__":
    convert()