# Main module
import nibabel as nib
import matplotlib.pyplot as plt

from preprocess import normalize_mri
from detect import detect_tumor
from segment import segment_with_sam
from refine import refine_mask

def run_pipeline(image_path):
    # Load MRI
    img = nib.load(image_path).get_fdata()
    img_slice = img[:, :, img.shape[2] // 2]

    # Preprocess
    img_norm = normalize_mri(img_slice)

    # Detect
    box = detect_tumor(img_norm)

    # Segment
    mask = segment_with_sam(img_norm, box)

    # Refine
    refined = refine_mask(mask)

    # Display
    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1)
    plt.title("MRI Slice")
    plt.imshow(img_norm, cmap='gray')

    plt.subplot(1,3,2)
    plt.title("Initial Mask")
    plt.imshow(mask, cmap='jet')

    plt.subplot(1,3,3)
    plt.title("Refined Mask")
    plt.imshow(refined, cmap='jet')

    plt.show()

if __name__ == "__main__":
    run_pipeline("data/brain/images/sample.nii")

