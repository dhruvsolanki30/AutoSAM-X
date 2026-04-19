import os
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor

# ---------------- AUTO PATH ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAM_PATH = os.path.join(BASE_DIR, "models", "sam_vit_b_01ec64.pth")

if not os.path.exists(SAM_PATH):
    raise FileNotFoundError(f"❌ SAM model not found at: {SAM_PATH}")

# ---------------- LOAD MODEL ----------------

print("[INFO] Loading SAM model...")
sam = sam_model_registry["vit_b"](checkpoint=SAM_PATH)
predictor = SamPredictor(sam)

# ---------------- SEGMENT ----------------

def segment_with_sam(image, detections):

    orig_h, orig_w = image.shape

    # 🔥 Resize for speed
    image_resized = cv2.resize(image, (512, 512))
    img = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)

    predictor.set_image(img)

    results = []

    # Precompute scale (better performance)
    scale_x = 512 / orig_w
    scale_y = 512 / orig_h

    for cls, conf, box in detections:

        x1, y1, x2, y2 = box

        box_scaled = [
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y)
        ]

        masks, _, _ = predictor.predict(
            box=np.array(box_scaled),
            multimask_output=False
        )

        mask = cv2.resize(
            masks[0].astype(np.uint8),
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        )

        results.append((cls, mask))

    return results