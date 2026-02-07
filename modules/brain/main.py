import os
import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_mri(image_path):
    """Load NIfTI MRI image"""
    nii = nib.load(image_path)
    img = nii.get_fdata()
    return img


def preprocess_slice(slice_img):
    """Normalize + enhance contrast"""
    slice_img = slice_img.astype(np.float32)
    slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min())
    slice_img = (slice_img * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(slice_img)
    return enhanced


def simple_detection(slice_img):
    """Simple threshold-based detection (temporary placeholder for YOLO)"""
    _, mask = cv2.threshold(slice_img, 180, 255, cv2.THRESH_BINARY)
    return mask


def visualize(original, processed, mask):
    """Display results"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original MRI Slice")
    plt.imshow(original, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Preprocessed")
    plt.imshow(processed, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Detected Region")
    plt.imshow(original, cmap="gray")
    plt.imshow(mask, cmap="jet", alpha=0.5)
    plt.axis("off")

    plt.show()


def run_pipeline(image_path=None):
    # Resolve absolute path safely
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    images_dir = os.path.join(BASE_DIR, "data", "brain", "images")

    # If no path provided, prefer sample.nii, else pick first .nii/.nii.gz found
    if image_path is None:
        candidate = os.path.join(images_dir, "sample.nii")
        if os.path.exists(candidate):
            image_path = candidate
        else:
            for fname in os.listdir(images_dir):
                if fname.lower().endswith((".nii", ".nii.gz")):
                    image_path = os.path.join(images_dir, fname)
                    break

    if image_path is None or not os.path.exists(image_path):
        raise FileNotFoundError(f"No NIfTI file found in {images_dir}. Expected sample.nii or any .nii/.nii.gz.")

    # Load MRI volume
    volume = load_mri(image_path)

    # Take middle slice
    slice_index = volume.shape[2] // 2
    slice_img = volume[:, :, slice_index]

    # Preprocess
    processed = preprocess_slice(slice_img)

    # Detect (placeholder logic)
    mask = simple_detection(processed)

    # Visualize
    visualize(slice_img, processed, mask)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MRI pipeline")
    parser.add_argument("image", nargs="?", help="Path to NIfTI image (optional)")
    args = parser.parse_args()

    run_pipeline(args.image)
