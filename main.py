import os
import nibabel as nib
import numpy as np
import cv2

from preprocess import preprocess_slice
from detect import detect_objects
from segment import segment_with_sam
from refine import refine_mask


# ---------------- AUTO PATH SETUP ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.join(BASE_DIR, "data", "liver", "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------- LOAD CT ----------------

def load_ct(path):
    nii = nib.load(path)
    return nii.get_fdata()


# ---------------- VISUALIZATION ----------------

def visualize(image, liver_masks, tumor_masks, save_path):

    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for mask in liver_masks:
        img[mask] = img[mask] * 0.5 + np.array([0, 255, 0]) * 0.5

    for mask in tumor_masks:
        img[mask] = [255, 0, 0]

    cv2.imwrite(save_path, img)


# ---------------- PIPELINE ----------------

def run_pipeline():

    # ---- Find first CT ----
    file_path = None
    for f in os.listdir(INPUT_PATH):
        if f.endswith((".nii", ".nii.gz")):
            file_path = os.path.join(INPUT_PATH, f)
            break

    if file_path is None:
        raise ValueError(f"❌ No NIfTI file found in {INPUT_PATH}")

    print("Using CT:", file_path)

    volume = load_ct(file_path)

    tumor_slices = []
    liver_areas = []

    print("[INFO] Processing slices...")

    start = int(volume.shape[2] * 0.3)
    end   = int(volume.shape[2] * 0.8)

    for i in range(start, end):
        print(f"[INFO] Slice {i}")

        slice_img = volume[:, :, i]

        processed = preprocess_slice(slice_img)

        liver_det, tumor_det, tumor_found = detect_objects(processed)

        if len(liver_det) == 0:
            continue

        if not tumor_found:
            continue

        liver_det = liver_det[:1]
        tumor_det = tumor_det[:2]

        tumor_slices.append(i)

        liver_masks_raw = segment_with_sam(processed, liver_det)
        tumor_masks_raw = segment_with_sam(processed, tumor_det)

        liver_masks = [refine_mask(m) for _, m in liver_masks_raw]
        tumor_masks = [refine_mask(m) for _, m in tumor_masks_raw]

        liver_area = sum(np.sum(mask) for mask in liver_masks)
        liver_areas.append(liver_area)

        save_path = os.path.join(OUTPUT_DIR, f"slice_{i}.png")
        visualize(processed, liver_masks, tumor_masks, save_path)

    # ---------------- FINAL OUTPUT ----------------

    if len(tumor_slices) == 0:
        print("\n❌ No tumor detected")
    else:
        print("\n✅ Tumor detected")
        print("Slices:", tumor_slices)
        print("Saved at:", OUTPUT_DIR)

    if len(liver_areas) > 0:
        avg_liver_area = np.mean(liver_areas)

        print("\n[INFO] Average Liver Area:", int(avg_liver_area))

        if avg_liver_area > 50000:
            print("⚠ Possible Hepatomegaly (Enlarged Liver)")
        else:
            print("✔ Liver size within normal range")


# ---------------- MAIN ----------------

if __name__ == "__main__":
    run_pipeline()