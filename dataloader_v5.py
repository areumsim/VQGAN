import cv2

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

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
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # ImageNet
        ]
    )


############ ImageNet_256 
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
        self.dataset = datasets.ImageFolder(root=img_dir, transform=transform)
        self._create_data_loaders()
        

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

        # #### indices 저장 (seed가 정해져 있지 않으면, indices를 저장해야함)
        # # Assuming train_dataset and valid_dataset are already created
        # train_indices = train.indices
        # valid_indices = valid.indices

        # # Save indices to disk
        # torch.save(train_indices, './train_indices.pt')
        # torch.save(valid_indices, './valid_indices.pt')

        # ### Load indices from disk
        # train_indices = torch.load('train_indices.pt')
        # valid_indices = torch.load('valid_indices.pt')

        # ### Recreate the datasets using Subset
        # from torch.utils.data import Subset
        # train_dataset = Subset(dataset, train_indices)
        # valid_dataset = Subset(dataset, valid_indices)

    def get_datasets(self):
        return self.train_dataset, self.valid_dataset
    
    # def get_loaders(self):
    #     train_loader = DataLoader(self.train_dataset, batch_size=cfg['train_params']['batch_size'], shuffle=True)
    #     valid_loader = DataLoader(self.valid_dataset, batch_size=cfg['train_params']['batch_size'], shuffle=True)
    #     return train_loader, valid_loader


############ STL10 (96x96x3)
class STL10(Dataset):
    def __init__(self, cfg):
        super(STL10, self).__init__()
        self.data_dir = cfg['STL10']['data_dir']
        self.transform = make_transforms()

  
    def get_datasets(self):
        self.train = datasets.STL10(root=self.data_dir, split='train', download=True, transform=transform) #5000
        self.test = datasets.STL10(root=self.data_dir, split='test', download=True, transform=transform)   #8000

        return self.train, self.test
    
############  ImageNet_1k_256 (확인X, 느림, 256x256x3)
class ImageNet_1k_256(Dataset):
    def __init__(self, cfg):
        super(ImageNet_1k_256, self).__init__()
        self.data_dir = cfg['STL10']['data_dir']
        self.transform = make_transforms()
  
    def get_datasets(self):
        self.train = ImagetNetDataset(self.data_dir, 'train', transform=transform)
        self.valid = ImagetNetDataset(self.data_dir, 'train', transform=transform)
        return self.train, self.valid
    

class ImagetNetDataset(Dataset):
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
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # -1과 1 사이의 값으로 정규화
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 정규화
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