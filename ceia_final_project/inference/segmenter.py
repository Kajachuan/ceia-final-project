import torch
import numpy as np
from PIL import Image
from torch.nn.functional import sigmoid
from torchvision.transforms.v2.functional import normalize, to_image, to_dtype
from ..constants import MEAN, STD
from ..modules import LightningSegmentation

class Segmenter:
    def __init__(self, checkpoint_path, threshold=0.5, device=None):
        self.module = LightningSegmentation.load_from_checkpoint(checkpoint_path)
        self.model = self.module.model
        self.model.eval()

        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def segment(self, input_path, output_path=None, normalize_input=True):
        with Image.open(input_path) as input:
            input = to_image(input)
            input = to_dtype(input, torch.float, scale=True)
            img = input.to(self.device).float()

            if normalize_input:
                img = normalize(img, mean=MEAN, std=STD)

            is_single_image = img.dim() == 3  # (C, H, W)

            if is_single_image:
                img = img.unsqueeze(0)  # (1, C, H, W)

            with torch.no_grad():
                output = self.model(img)
                output = sigmoid(output)
                pred_mask = (output > self.threshold).float()

            pred_mask = pred_mask.cpu()

            if is_single_image:
                pred_mask = pred_mask.squeeze(0).squeeze(0)  # (H, W)
            else:
                pred_mask = pred_mask.squeeze(1)  # (B, H, W)

            if output_path:
                mask = Image.fromarray((pred_mask.numpy() * 255).astype(np.uint8))
                mask.save(output_path)

            return input, pred_mask