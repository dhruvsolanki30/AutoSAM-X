from ultralytics import YOLO
import cv2

model = YOLO("models/best.pt")

def detect_objects(image, conf_thresh=0.25):

    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    results = model(img, conf=conf_thresh, verbose=False)[0]

    liver = []
    tumors = []

    if results.boxes is not None:

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            if cls == 0:  # liver
                liver.append((cls, conf, xyxy))

            elif cls == 1:  # tumor
                tumors.append((cls, conf, xyxy))

            # cls == 2 (vessel) → IGNORE

    tumor_found = len(tumors) > 0

    return liver, tumors, tumor_found