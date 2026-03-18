from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("runs/detect/train/weights/best.pt")

CONF_THRESHOLD = 0.6
MIN_BOX_AREA = 200


def create_brain_mask(image):
    _, mask = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def detect_tumor(image):

    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    results = model(img, verbose=False)

    # ---------------- YOLO DETECTION ----------------

    if len(results[0].boxes) > 0:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        best_idx = np.argmax(confs)
        confidence = confs[best_idx]

        if confidence >= CONF_THRESHOLD:

            x1, y1, x2, y2 = map(int, boxes[best_idx])

            area = (x2 - x1) * (y2 - y1)

            if area > MIN_BOX_AREA:

                brain_mask = create_brain_mask(image)
                region = brain_mask[y1:y2, x1:x2]

                # ensure detection inside brain
                if np.mean(region) > 50:
                    return [x1, y1, x2, y2], float(confidence)

    # -------- FALLBACK DETECTION --------

    thresh = np.percentile(image, 98)

    mask = np.zeros_like(image, dtype=np.uint8)
    mask[image >= thresh] = 255

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(c)

    # ignore very thin edge detections
    if w < 20 or h < 20:
        return None

    return [x, y, x + w, y + h], 0.5