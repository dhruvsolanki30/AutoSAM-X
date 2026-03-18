import os
import time
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import cv2
from bs4 import BeautifulSoup

from modules.brain.preprocess import preprocess_slice
from modules.brain.detect import detect_tumor
from modules.brain.segment import segment_with_sam
from modules.brain.refine import refine_mask


# ------------------ LOAD IMAGE ------------------

def load_mri(image_path):

    # -------- NIFTI MRI --------
    if image_path.endswith(".nii") or image_path.endswith(".nii.gz"):

        nii = nib.load(image_path)
        volume = nii.get_fdata()

        slice_img = volume[:, :, volume.shape[2] // 2]

        return slice_img

    # -------- NORMAL IMAGE --------
    elif image_path.endswith((".jpg", ".png", ".jpeg")):

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("Image could not be loaded")

        return img

    # -------- HTML FILE --------
    elif image_path.endswith(".html"):

        with open(image_path, "r", encoding="utf-8") as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")

        img_tag = soup.find("img")

        if img_tag is None:
            raise ValueError("No <img> tag found in HTML")

        img_src = img_tag.get("src")

        img_path = os.path.join(os.path.dirname(image_path), img_src)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("Image inside HTML could not be loaded")

        return img

    else:
        raise ValueError("Unsupported file format")


# ------------------ PIPELINE ------------------

def run_pipeline(image_path):

    print("Using MRI:", image_path)

    slice_img = load_mri(image_path)

    # ------------------ PREPROCESS ------------------
    processed = preprocess_slice(slice_img)

    # ------------------ YOLO DETECTION ------------------
    result = detect_tumor(processed)

    if result is None:
        box = None
        confidence = 0
        print("No tumor detected by YOLO")
    else:
        box, confidence = result
        print("YOLO detected tumor with confidence:", round(confidence, 2))

    # ------------------ SAM SEGMENTATION ------------------
    if box is not None:
        segmentation = segment_with_sam(processed, box)
    else:
        segmentation = np.zeros_like(processed)

    # ------------------ REFINEMENT ------------------
    refined = refine_mask(segmentation)

    # ------------------ TUMOR ANALYSIS ------------------
    tumor_area = np.sum(refined > 0)

    if tumor_area > 0:
        detection = "Tumor Detected"
    else:
        detection = "No Tumor Detected"

    if tumor_area < 1000:
        severity = "Low"
    elif tumor_area < 3000:
        severity = "Medium"
    else:
        severity = "High"

    print("Detection:", detection)
    print("Area:", tumor_area)
    print("Severity:", severity)

    # ------------------ VISUALIZATION ------------------
    plt.figure(figsize=(12, 4))

    # ORIGINAL
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(slice_img, cmap="gray")
    plt.axis("off")

    # YOLO DETECTION
    plt.subplot(1, 3, 2)
    plt.title("Detection")
    plt.imshow(slice_img, cmap="gray")

    if box is not None:
        x1, y1, x2, y2 = box
        plt.gca().add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                edgecolor="red",
                linewidth=2,
                fill=False
            )
        )

    plt.axis("off")

    # SEGMENTATION
    plt.subplot(1, 3, 3)
    plt.title("Segmentation")
    plt.imshow(slice_img, cmap="gray")

    if tumor_area > 0:
        plt.imshow(refined, cmap="jet", alpha=0.5)



    plt.axis("off")

    plt.tight_layout()

    # ------------------ SAVE OUTPUT ------------------

    # UNIQUE filename (prevents overwrite)
    output_filename = f"result_{int(time.time())}.png"

    # SYSTEM PATH (for saving)
    save_path = os.path.join("static", "outputs", output_filename)

    # WEB PATH (for frontend display)
    return_path = f"/static/outputs/{output_filename}"

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    # ------------------ RETURN RESULT ------------------
    return {
        "output_image": return_path,
        "detection": detection,
        "confidence": round(confidence, 2),
        "tumor_area": int(tumor_area),
        "severity": severity
    }