import random
import torchvision.transforms.v2.functional as F
from torchvision.transforms import v2

class SegmentationTransform:
    def __init__(self, subset):
        self.color_jitter = v2.ColorJitter(brightness=0.2, contrast=0.2)
        self.gaussian_noise = v2.GaussianNoise()
        self.gaussian_blur = v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1))
        self.subset = subset

    def __call__(self, img, mask):
        if self.subset == 'train':
            i, j, h, w = v2.RandomCrop.get_params(img, output_size=(256, 256))
            img = F.crop(img, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)

            img = self.color_jitter(img)

            angle = random.randrange(-180, 180)
            img = F.rotate(img, angle)
            mask = F.rotate(mask, angle)

            if random.random() > 0.5:
                img = F.hflip(img)
                mask = F.hflip(mask)

            if random.random() > 0.5:
                img = F.vflip(img)
                mask = F.vflip(mask)

            if random.random() > 0.5:
              img = self.gaussian_noise(img)

            if random.random() > 0.5:
              img = self.gaussian_blur(img)

        img = F.normalize(img, mean=[0.3555, 0.3374, 0.2294], std=[0.1790, 0.1094, 0.0960])

        return img, mask