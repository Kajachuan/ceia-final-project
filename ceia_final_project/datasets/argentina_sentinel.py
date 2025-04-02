import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as F

class ArgentinaSentinel2Dataset(Dataset):
    def __init__(self, path, subset, transform=None, samples_per_tile=10):
        images_path = f'{path}/{subset}/tiles'
        masks_path = f'{path}/{subset}/labels/raster'

        tiles_names = sorted(os.listdir(images_path))

        init_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True)
        ])

        self.subset = subset
        self.images = list(map(lambda filename: init_transform(Image.open(f'{images_path}/{filename}')), tiles_names))
        self.masks = list(map(lambda filename: init_transform(Image.open(f'{masks_path}/{filename}')), tiles_names))

        if self.subset in ['valid', 'test']:
            num_crops = np.sqrt(samples_per_tile).astype(int)
            _, height, width = self.images[0].shape
            crop_size = 256
            positions = np.linspace(0, height - crop_size, num_crops).astype(int)

            self.images_crops = []
            self.masks_crops = []

            for i in range(len(self.images)):
                for y in positions:
                    for x in positions:
                        image_crop = F.crop(self.images[i], y, x, crop_size, crop_size)
                        self.images_crops.append(image_crop)

                        mask_crop = F.crop(self.masks[i], y, x, crop_size, crop_size)
                        self.masks_crops.append(mask_crop)

            self.images = self.images_crops
            self.masks = self.masks_crops

        self.samples_per_tile = samples_per_tile
        self.transform = transform

    def __len__(self):
        return len(self.images) * self.samples_per_tile if self.subset == 'train' else len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx % len(self.images)]
        mask = self.masks[idx % len(self.masks)]

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask