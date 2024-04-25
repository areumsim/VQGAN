
#%%
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
        ]
    )

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
                iaa.Resize(
                    {"height": height, "width": height}
                ),
                
            ]
        )

#     sometimes = lambda aug: iaa.Sometimes(0.3, aug)

#     return iaa.Sequential(
#     [
#         iaa.Fliplr(0.5), # horizontally flip 50% of all images
#         iaa.Flipud(0.2), # vertically flip 20% of all images

#         # crop some of the images by 0-10% of their height/width
#         sometimes(iaa.Crop(percent=(0, 0.1))),

#         # Apply affine transformations to some of the images
#         # - scale to 80-120% of image height/width (each axis independently)
#         # - translate by -20 to +20 relative to height/width (per axis)
#         # - rotate by -45 to +45 degrees
#         # - shear by -16 to +16 degrees
#         # - order: use nearest neighbour or bilinear interpolation (fast)
#         # - mode: use any available mode to fill newly created pixels
#         #         see API or scikit-image for which modes are available
#         # - cval: if the mode is constant, then use a random brightness
#         #         for the newly created pixels (e.g. sometimes black,
#         #         sometimes white)
#         sometimes(iaa.Affine(
#             scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#             translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#             rotate=(-45, 45),
#             shear=(-16, 16),
#             order=[0, 1],
#             cval=(0, 255),
#             mode=ia.ALL
#         )),

#         iaa.SomeOf((0, 5),
#             [
#                 # Convert some images into their superpixel representation,
#                 # sample between 20 and 200 superpixels per image, but do
#                 # not replace all superpixels with their average, only
#                 # some of them (p_replace).
#                 sometimes(
#                     iaa.Superpixels(
#                         p_replace=(0, 1.0),
#                         n_segments=(20, 200)
#                     )
#                 ),

#                 # Blur each image with varying strength using
#                 # gaussian blur (sigma between 0 and 3.0),
#                 # average/uniform blur (kernel size between 2x2 and 7x7)
#                 # median blur (kernel size between 3x3 and 11x11).
#                 iaa.OneOf([
#                     iaa.GaussianBlur((0, 3.0)),
#                     iaa.AverageBlur(k=(2, 7)),
#                     iaa.MedianBlur(k=(3, 11)),
#                 ]),

#                 # Sharpen each image, overlay the result with the original
#                 # image using an alpha between 0 (no sharpening) and 1
#                 # (full sharpening effect).
#                 iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

#                 # Same as sharpen, but for an embossing effect.
#                 iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

#                 # Search in some images either for all edges or for
#                 # directed edges. These edges are then marked in a black
#                 # and white image and overlayed with the original image
#                 # using an alpha of 0 to 0.7.
#                 sometimes(iaa.OneOf([
#                     iaa.EdgeDetect(alpha=(0, 0.7)),
#                     iaa.DirectedEdgeDetect(
#                         alpha=(0, 0.7), direction=(0.0, 1.0)
#                     ),
#                 ])),

#                 # Add gaussian noise to some images.
#                 # In 50% of these cases, the noise is randomly sampled per
#                 # channel and pixel.
#                 # In the other 50% of all cases it is sampled once per
#                 # pixel (i.e. brightness change).
#                 iaa.AdditiveGaussianNoise(
#                     loc=0, scale=(0.0, 0.05*255), per_channel=0.5
#                 ),

#                 # Either drop randomly 1 to 10% of all pixels (i.e. set
#                 # them to black) or drop them on an image with 2-5% percent
#                 # of the original size, leading to large dropped
#                 # rectangles.
#                 iaa.OneOf([
#                     iaa.Dropout((0.01, 0.1), per_channel=0.5),
#                     iaa.CoarseDropout(
#                         (0.03, 0.15), size_percent=(0.02, 0.05),
#                         per_channel=0.2
#                     ),
#                 ]),

#                 # Invert each image's channel with 5% probability.
#                 # This sets each pixel value v to 255-v.
#                 iaa.Invert(0.05, per_channel=True), # invert color channels

#                 # Add a value of -10 to 10 to each pixel.
#                 iaa.Add((-10, 10), per_channel=0.5),

#                 # Change brightness of images (50-150% of original value).
#                 iaa.Multiply((0.5, 1.5), per_channel=0.5),

#                 # Improve or worsen the contrast of images.
#                 iaa.LinearContrast((0.5, 2.0), per_channel=0.5),

#                 # Convert each image to grayscale and then overlay the
#                 # result with the original with random alpha. I.e. remove
#                 # colors with varying strengths.
#                 iaa.Grayscale(alpha=(0.0, 1.0)),

#                 # In some images move pixels locally around (with random
#                 # strengths).
#                 sometimes(
#                     iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
#                 ),

#                 # In some images distort local areas with varying strength.
#                 sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
#             ],
#             # do all of the above augmentations in random order
#             random_order=True
#         )
#     ],
#     # do all of the above augmentations in random order
#     random_order=True
# )


############ ImageNet_256
class ImageNet256Dataset(Dataset):
    def __init__(self, indices, imagenet256):
        self.indices = indices
        self.imagenet256 = imagenet256

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img, label = self.imagenet256.dataset[self.indices[index]]
        img_array = np.array(img)

        # 이미지에 augmentation 적용
        img_augmented = self.imagenet256.seq(image=img_array)
        img_tensor = transforms.ToTensor()(img_augmented)

        return img_tensor, label

class ImageNet256(Dataset):
    def __init__(
        self, cfg, split_ratio=0.8, seed=42,
    ):
        super(ImageNet256, self).__init__()

        self.data_dir = cfg['ImageNet256']['data_dir']
        self.split_ratio = split_ratio
        self.seed = seed

        self.transform = make_transforms()

        self.height = 96
        self.width = 96

        # 데이터셋 초기화
        self.dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        self._create_data_loaders()
        
        # 데이터 augmentation
        self.seq = make_augmentation(self.height, self.width)


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

    # def get_loaders(self):
    #     train_loader = DataLoader(self.train_dataset, batch_size=cfg['train_params']['batch_size'], shuffle=True)
    #     valid_loader = DataLoader(self.valid_dataset, batch_size=cfg['train_params']['batch_size'], shuffle=True)
    #     return train_loader, valid_loader

    # def get_datasets(self):
    #     return self.train_dataset, self.valid_dataset
    
    # def __getitem__(self, index):
    #     img, label = self.dataset[index]  # 원본 데이터셋에서 이미지와 라벨 가져오기
    #     img_array = np.array(img)

    #     # 이미지에 augmentation 적용
    #     img_augmented = self.seq(image=img_array)
    #     img_tensor = transforms.ToTensor()(img_augmented)

    #     return img_tensor, label
    
    def get_datasets(self):
        train_dataset = ImageNet256Dataset(self.train_dataset.indices, self)
        valid_dataset = ImageNet256Dataset(self.valid_dataset.indices, self)
        return train_dataset, valid_dataset




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

    train_loader = DataLoader(train, batch_size=cfg['train_params']['batch_size'])
    
    augmented_images, _ = next(iter(train_loader))
    # augmented_img = show_originalimage(augmented_images[0])
    # plt.imshow(augmented_img)
    # plt.axis('off')

    ### Show original image.###
    idx = 2
    original_img = train[idx][0] # (3, 256, 256)
    original_img = show_originalimage(original_img)
    
    augmented_img = show_originalimage(augmented_images[idx])

    plt.subplot(1, 2, 1)  # 1행 2열의 첫 번째 위치에 서브플롯 생성
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')  # 축 없애기

    # 증강된 이미지 출력
    plt.subplot(1, 2, 2)  # 1행 2열의 두 번째 위치에 서브플롯 생성
    plt.imshow(augmented_img)
    plt.title('Augmented Image')
    plt.axis('off')  # 축 없애기

