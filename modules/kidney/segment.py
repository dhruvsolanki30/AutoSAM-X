import numpy as np
from segment_anything import sam_model_registry, SamPredictor

class SamSegmenter:

    def __init__(self, weight_path):

        sam = sam_model_registry["vit_b"](checkpoint=weight_path)
        self.predictor = SamPredictor(sam)

    def segment(self, image, box):

        img = np.stack([image, image, image], axis=-1)

        self.predictor.set_image(img)

        masks, _, _ = self.predictor.predict(
            box=np.array(box),
            multimask_output=False
        )

        return masks[0].astype(np.uint8)