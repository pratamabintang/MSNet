import torch
import numpy as np
import cv2
from configs.config import config


class RGBPreprocessor:

    def __init__(self):
        pass

    def __call__(self, rgb_img):
        if not isinstance(rgb_img, torch.Tensor):
            rgb_img = torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0

        return torch.clamp(rgb_img, 0.0, 1.0)


class EventPreprocessor:

    def __init__(self):
        pass

    def __call__(self, event_input):
        if isinstance(event_input, str) or isinstance(event_input, type(None)):
            if event_input is None:
                return torch.zeros((1, config.IMG_SIZE[0], config.IMG_SIZE[1]), dtype=torch.float32)

            evt = cv2.imread(event_input, cv2.IMREAD_GRAYSCALE)
            if evt is None:
                return torch.zeros((1, config.IMG_SIZE[0], config.IMG_SIZE[1]), dtype=torch.float32)

            evt = evt.astype(np.float32) / 255.0
            tensor = torch.from_numpy(evt).unsqueeze(0)  # [1, H, W]

        elif isinstance(event_input, np.ndarray):
            evt = event_input.astype(np.float32)
            if evt.max() > 1.0:
                evt /= 255.0

            if len(evt.shape) == 2:
                tensor = torch.from_numpy(evt).unsqueeze(0)
            else:
                tensor = torch.from_numpy(evt).permute(2, 0, 1)  # [H,W,C] -> [C,H,W]

        elif isinstance(event_input, torch.Tensor):
            tensor = event_input
        else:
            raise TypeError(f"Unknown event input type: {type(event_input)}")

        return tensor