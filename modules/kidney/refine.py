import torch
import cv2
import numpy as np
from monai.networks.nets import UNet

class Refiner:

    def __init__(self, weight_path):

        self.model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16,32,64,128),
            strides=(2,2,2),
        )

        self.model.load_state_dict(torch.load(weight_path, map_location="cpu"))
        self.model.eval()

    def refine(self, image):

        img = cv2.resize(image,(256,256))

        img = torch.tensor(img/255.0).float().unsqueeze(0).unsqueeze(0)

        pred = self.model(img).detach().numpy()[0,0]

        pred = (pred > 0.5).astype(np.uint8)

        refined = cv2.resize(pred,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)

        return refined