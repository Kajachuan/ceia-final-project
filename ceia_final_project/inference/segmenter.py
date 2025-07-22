import torch
from torch.nn.functional import sigmoid
from torchvision.transforms.v2.functional import normalize
from ..constants import MEAN, STD
from ..modules import LightningSegmentation

class Segmenter:
    def __init__(self, checkpoint_path, threshold=0.5, normalize_input=True, device=None):
        self.module = LightningSegmentation.load_from_checkpoint(checkpoint_path)
        self.model = self.module.model
        self.model.eval()

        self.threshold = threshold
        self.normalize_input = normalize_input
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, image):
        image = image.to(self.device).float()

        if self.normalize_input:
            image = normalize(image, mean=MEAN, std=STD)

        is_single_image = image.dim() == 3  # (C, H, W)

        if is_single_image:
            image = image.unsqueeze(0)  # (1, C, H, W)

        with torch.no_grad():
            output = self.model(image)
            output = sigmoid(output)
            pred_mask = (output > self.threshold).float()

        pred_mask = pred_mask.cpu()

        if is_single_image:
            return pred_mask.squeeze(0).squeeze(0)  # (H, W)
        else:
            return pred_mask.squeeze(1)  # (B, H, W)
