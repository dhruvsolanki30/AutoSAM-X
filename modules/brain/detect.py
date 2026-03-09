from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("runs/detect/train/weights/best.pt")

def detect_tumor(image):

    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    results = model(img)

    # YOLO detection
    if len(results[0].boxes) > 0:

        box = results[0].boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)

        return [x1, y1, x2, y2]

    # -------- FALLBACK DETECTION --------

    thresh = np.percentile(image, 98)

    mask = np.zeros_like(image, dtype=np.uint8)
    mask[image >= thresh] = 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(c)

    return [x, y, x + w, y + h]