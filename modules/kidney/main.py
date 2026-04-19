import nibabel as nib
import numpy as np
from tqdm import tqdm

from .preprocess import preprocess_slice
from .detect import KidneyDetector
from .segment import SamSegmenter
from .refine import Refiner
from .classify import classify
from .visualize import save_final_output


YOLO_PATH = "modules/kidney/weights/yolo_kidney.pt"
SAM_PATH = "modules/kidney/weights/sam_vit_b.pth"
UNET_PATH = "modules/kidney/weights/unet_kidney.pth"


def run_pipeline(file_path, option):

    print("Loading models...")

    detector = KidneyDetector(YOLO_PATH)
    segmenter = SamSegmenter(SAM_PATH)
    refiner = Refiner(UNET_PATH)

    print("Loading CT volume...")

    volume = nib.load(file_path).get_fdata()
    depth = volume.shape[2]

    best_score = 0
    best_original = None
    best_processed = None
    best_mask = None
    best_result = None

    print("Processing slices...")

    for i in tqdm(range(0, depth, 5)):

        slice_img = volume[:, :, i]

        # Preprocess
        processed = preprocess_slice(slice_img)

        # Detect kidney region
        box = detector.detect(processed)

        if box is None:
            continue

        # SAM segmentation
        sam_mask = segmenter.segment(processed, box)

        # U-Net refinement
        refined_mask = refiner.refine(processed)

        # Disease classification + mask
        result, disease_mask = classify(processed, refined_mask, option)

        score = np.sum(disease_mask)

        # Select best slice
        if score > best_score:

            best_score = score
            best_original = slice_img
            best_processed = processed
            best_mask = disease_mask
            best_result = result

    # If nothing detected
    if best_original is None:

        print("No abnormal region detected")

        best_original = volume[:, :, depth // 2]
        best_processed = preprocess_slice(best_original)
        best_mask = np.zeros_like(best_processed)

        best_result = {
            "Diagnosis": "Healthy",
            "Details": "No abnormality detected"
        }

    print("Saving final output...")

    save_final_output(
        best_original,
        best_processed,
        best_mask,
        best_result,
        option
    )

    print("Pipeline finished.")

    return best_result