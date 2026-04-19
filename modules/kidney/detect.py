import numpy as np
from ultralytics import YOLO

class KidneyDetector:

    def __init__(self, weight_path):
        self.model = YOLO(weight_path)

    def detect(self, image):

        img = np.stack([image, image, image], axis=-1)

        results = self.model(img, verbose=False)

        if len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()

        best_box = boxes[scores.argmax()]

        return best_box