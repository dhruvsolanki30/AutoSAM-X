import os
import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ------------------ LOAD MRI ------------------
def load_mri(image_path):
    nii = nib.load(image_path)
    return nii.get_fdata()


# ------------------ PREPROCESS ------------------
def preprocess_slice(slice_img):
    slice_img = slice_img.astype(np.float32)
    slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min())
    slice_img = (slice_img * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(slice_img)


# ------------------ CAMOUFLAGE-AWARE TUMOR DETECTION ------------------
def detect_tumor(img):
    """
    Detects bright abnormal regions (tumor-like)
    """
    # Use top intensity percentile (tumors are bright)
    thresh = np.percentile(img, 98)

    mask = np.zeros_like(img, dtype=np.uint8)
    mask[img >= thresh] = 255

    # Clean mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# ------------------ BOUNDING BOX ------------------
def extract_bounding_box(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Keep only reasonable-sized components
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if 100 < area < 5000:   # tumor-sized region
            valid.append(c)

    if not valid:
        return None

    largest = max(valid, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, x + w, y + h)


# ------------------ SEGMENTATION ------------------
def segment_from_box(shape, box):
    seg = np.zeros(shape, dtype=np.uint8)
    if box is None:
        return seg

    x1, y1, x2, y2 = box
    seg[y1:y2, x1:x2] = 255
    return seg


# ------------------ PIPELINE ------------------
def run_pipeline(image_path=None):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    images_dir = os.path.join(BASE_DIR, "data", "brain", "images")

    if image_path is None:
        for f in os.listdir(images_dir):
            if f.endswith((".nii", ".nii.gz")):
                image_path = os.path.join(images_dir, f)
                break

    volume = load_mri(image_path)
    slice_img = volume[:, :, volume.shape[2] // 2]

    processed = preprocess_slice(slice_img)

    tumor_mask = detect_tumor(processed)
    box = extract_bounding_box(tumor_mask)

    segmentation = segment_from_box(slice_img.shape, box)

    # ------------------ VISUALIZATION ------------------
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original MRI Slice")
    plt.imshow(slice_img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Automatic Detection (Tumor Bounding Box)")
    plt.imshow(slice_img, cmap="gray")
    if box:
        x1, y1, x2, y2 = box
        plt.gca().add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                          edgecolor="red", linewidth=2, fill=False)
        )
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Automatic Tumor Segmentation")
    plt.imshow(slice_img, cmap="gray")
    plt.imshow(segmentation, cmap="jet", alpha=0.5)
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    run_pipeline()
