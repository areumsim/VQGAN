
#%%
import cv2

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets.folder import DatasetFolder, default_loader

from pycocotools.coco import COCO
from PIL import Image

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from torchvision.io import read_image

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from einops import rearrange


def make_transforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # -1과 1 사이의 값으로 정규화
            transforms.Resize((96, 96)),
        ]
    )

# 결과가 255 - 안으로
def make_augmentation(height=96, width=96):
    return iaa.Sequential(
        [
            iaa.Resize((0.4, 0.5)),
            iaa.SomeOf(
                1,
                [
                    iaa.AdditiveLaplaceNoise(scale=(0, 0.05 * 255)),
                    iaa.Fliplr(0.5),
                    iaa.Add(50, per_channel=True),
                    iaa.Sharpen(alpha=0.5),
                ],
            ),
            iaa.Resize({"height": height, "width": width}),
        ]
    )

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

class ImageNetAugmentation(DatasetFolder):
    def __init__(
        self,
        root: str,
        transform = None,
        target_transform = None,
        loader = default_loader,
        is_valid_file= None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.seq = make_augmentation()
        self.train_indices = None

    def __getitem__(self, index: int) :
        path, target = self.samples[index]
        
        sample = self.loader(path)
        if self.target_transform is not None:
            sample = self.target_transform(sample)

        if index in self.train_indices:
            sample = self.seq(image= np.array(sample))

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


class ImageNet256(Dataset):
    def __init__(
        self, cfg, split_ratio=0.8, seed=42,
    ):
        super(ImageNet256, self).__init__()

        self.data_dir = cfg['ImageNet256']['data_dir']
        self.split_ratio = split_ratio
        self.seed = seed

        self.transform = make_transforms()

        # 데이터셋 초기화
        # self.dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        self.dataset = ImageNetAugmentation(root=self.data_dir, transform=self.transform)
        self._create_data_loaders()
        
        # # 데이터 augmentation
        # self.seq = make_augmentation()


    def _create_data_loaders(self):
        torch.manual_seed(self.seed)  # For PyTorch
        random.seed(self.seed)        # For Python's random module
        np.random.seed(self.seed)     # For NumPy if used

        # 데이터셋 크기 및 분할
        dataset_size = len(self.dataset)
        train_size = int(dataset_size * self.split_ratio)
        valid_size = dataset_size - train_size
        
        # 분할에 사용할 generator 생성
        generator = torch.Generator().manual_seed(self.seed)  # 동일한 seed를 사용하여 generator 생성
        self.train_dataset, self.valid_dataset = random_split(self.dataset, [train_size, valid_size], generator=generator)
    
        self.train_dataset.dataset.train_indices = self.train_dataset.indices
        
    def get_datasets(self):
        return self.train_dataset, self.valid_dataset

    # def __getitem__(self, idx):
    #     image, label = self.dataset[idx]
        
    #     if idx in self.train_indices:
    #         image = np.array(image)
    #         img_augmented = self.seq(image=image)
    #         img_augmented = transforms.ToPILImage()(img_augmented)  # Convert back to PIL Image to use torchvision transforms
    #         img_augmented = self.transform(img_augmented)
        
    #     return img_augmented, label

#%%
if __name__ == "__main__":
    from einops import rearrange, asnumpy
    import yaml
    
    def show_originalimage(image) :
        image = torch.clamp(image, -1, 1)
        img = image.cpu().numpy().copy()
        img *= np.array([0.5, 0.5, 0.5])[:,None,None]
        img += np.array([0.5, 0.5, 0.5])[:,None,None]

        img = rearrange(img, "c h w -> h w c")
        img = img * 255
        img = img.astype(np.uint8)
        return img


    with open("config.yaml", "r") as stream:
        cfg = yaml.safe_load(stream)

    imageNet256 = ImageNet256(cfg['data'])
    train, valid = imageNet256.get_datasets()

    original_img = train[0][0] # (3, 256, 256)
    original_img = valid[0][0] # (3, 256, 256)

    original_img = show_originalimage(original_img)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')  # 축 없애기

    # train_loader = DataLoader(train, batch_size=cfg['train_params']['batch_size'])
    
    # augmented_images, _ = next(iter(train_loader))
    # augmented_img = show_originalimage(augmented_images[0])
    # plt.imshow(augmented_img)
    # plt.axis('off')

    ### Show original image.###
    # idx = 2
    # original_img = train[idx][0] # (3, 256, 256)
    # original_img = show_originalimage(original_img)
    
    # augmented_img = show_originalimage(augmented_images[idx])

    # plt.subplot(1, 2, 1)  # 1행 2열의 첫 번째 위치에 서브플롯 생성
    # plt.imshow(original_img)
    # plt.title('Original Image')
    # plt.axis('off')  # 축 없애기

    # # 증강된 이미지 출력
    # plt.subplot(1, 2, 2)  # 1행 2열의 두 번째 위치에 서브플롯 생성
    # plt.imshow(augmented_img)
    # plt.title('Augmented Image')
    # plt.axis('off')  # 축 없애기

