import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from preprocess import preprocess_slice
from detect import detect_tumor
from segment import segment_with_sam
from refine import refine_mask


# ------------------ LOAD MRI ------------------

def load_mri(image_path):
    nii = nib.load(image_path)
    return nii.get_fdata()


# ------------------ PIPELINE ------------------

def run_pipeline(image_path=None):

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    images_dir = os.path.join(BASE_DIR, "data", "brain", "images")

    if image_path is None:
        for f in os.listdir(images_dir):
            if f.endswith((".nii", ".nii.gz")):
                image_path = os.path.join(images_dir, f)
                break

    print("Using MRI:", image_path)

    volume = load_mri(image_path)

    # Take middle slice
    slice_img = volume[:, :, volume.shape[2] // 2]

    # ------------------ PREPROCESS ------------------

    processed = preprocess_slice(slice_img)

    # ------------------ YOLO DETECTION ------------------

    box = detect_tumor(processed)

    # ------------------ SAM SEGMENTATION ------------------

    segmentation = segment_with_sam(processed, box)

    # ------------------ REFINEMENT ------------------

    refined = refine_mask(segmentation)

    # ------------------ TUMOR ANALYSIS ------------------

    tumor_area = np.sum(refined > 0)

    if tumor_area > 0:
        print("Tumor detected")
    else:
        print("No tumor detected")

    print("Area:", tumor_area, "pixels")

    # Severity estimation
    if tumor_area < 1000:
        severity = "Low"
    elif tumor_area < 3000:
        severity = "Medium"
    else:
        severity = "High"

    print("Severity:", severity)

    # ------------------ VISUALIZATION ------------------

    plt.figure(figsize=(12, 4))

    # Original MRI
    plt.subplot(1, 3, 1)
    plt.title("Original MRI Slice")
    plt.imshow(slice_img, cmap="gray")
    plt.axis("off")

    # YOLO Detection
    plt.subplot(1, 3, 2)
    plt.title("YOLO Tumor Detection")
    plt.imshow(slice_img, cmap="gray")

    if box:
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

    # SAM Segmentation
    plt.subplot(1, 3, 3)
    plt.title("SAM Tumor Segmentation")
    plt.imshow(slice_img, cmap="gray")
    plt.imshow(refined, cmap="jet", alpha=0.5)

    # Overlay tumor analysis text
    plt.text(
        10,
        20,
        f"Area: {tumor_area} px\nSeverity: {severity}",
        color="white",
        fontsize=12,
        bbox=dict(facecolor="black", alpha=0.6)
    )

    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ------------------ ENTRY ------------------

if __name__ == "__main__":
    run_pipeline()