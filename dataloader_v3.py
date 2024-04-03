import cv2

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from pycocotools.coco import COCO
from PIL import Image

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from torchvision.io import read_image
from torchvision import transforms

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from einops import rearrange

class ImagetNetDataset(Dataset):
    # def __init__(
    #     self, cfg, transform=None, visualize=False, collate_fn=None
    # ):
    def __init__(
        self, data_dir, split, transform=None, visualize=False, collate_fn=None
    ):
        super(ImagetNetDataset, self).__init__()
        self.data_dir = data_dir
        self.image_set = split

        self.image_folrder = os.path.join(self.data_dir, self.image_set)
        self.anno_file = os.path.join(
            self.data_dir.replace("/", "\\"),
            "annotations",
            "instances_" + self.image_set + ".json",
        )

        
        self.transform = transform
        self.visualize = visualize

        self.getMask = True

        # self.seq = iaa.Sequential(
        #     [
        #         iaa.Resize((0.4, 0.5)),
        #         iaa.SomeOf(
        #             1,
        #             [
        #                 iaa.AdditiveLaplaceNoise(scale=(0, 0.05 * 255)),
        #                 iaa.Fliplr(0.5),
        #                 iaa.Add(50, per_channel=True),
        #                 iaa.Sharpen(alpha=0.5),
        #             ],
        #         ),
        #         iaa.Resize(
        #             {"height": self.height, "width": self.width}
        #         ),
                
        #     ]
        # )

    # def __len__(self):
    #     return len(self.image_ids)

    def load_image(self, image_idx):  #
        image_info = self.coco.loadImgs(self.image_ids[image_idx])[0]
        image_file_path = os.path.join(self.image_folrder, image_info["file_name"])
        image = Image.open(image_file_path).convert("RGB")

        image = np.array(image)
        return image

    def __getitem__(self, idx):  
        image = self.load_image(idx)

        # Augment images.
        # image = self.seq(image=image)

        if self.transform is not None:
            image = self.transform(image)

        return image


if __name__ == "__main__":
    from einops import rearrange, asnumpy
    import yaml
        
    def show_originalimage(image) :
        img = image.cpu().numpy().copy()
        img *= np.array([0.5, 0.5, 0.5])[:,None,None]
        img += np.array([0.5, 0.5, 0.5])[:,None,None]

        img = rearrange(img, "c h w -> h w c")
        img = img * 255
        img = img.astype(np.uint8)
        return img


    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # -1과 1 사이의 값으로 정규화
        ]
    )

    
    img_dir = 'C:/Users/wolve/arsim/autoencoder/ImageNet_256/' 
    train = ImagetNetDataset(img_dir, 'train', transform=transform)
    valid = ImagetNetDataset(img_dir, 'valid', transform=transform)
    test = ImagetNetDataset(img_dir, 'test', transform=transform)
    
    img = train[0] # torch.Size([3, 320, 320]) / (19, 5)
    
    ### Show image.###
    img = show_originalimage(img)

    cv2.imshow("", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    # plt.imshow(img)
    # plt.clf()