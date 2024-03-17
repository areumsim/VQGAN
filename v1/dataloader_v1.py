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

## pycocotools  --
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  showAnns   - Display the specified annotations.

 
def make_transforms(image_set):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

class COCODataset(Dataset):
    def __init__(
        self, cfg, transform=None, visualize=False, collate_fn=None
    ):
        super(COCODataset, self).__init__()
        self.cfg = cfg

        self.data_dir = cfg["data_dir"]
        self.image_set = cfg["train_set"]

        self.width = cfg["image_width"]
        self.height = cfg["image_height"]

        # self.n_class = cfg["n_class"] # real_class + no_obj


        self.image_folrder = os.path.join(self.data_dir, self.image_set)
        self.anno_file = os.path.join(
            self.data_dir.replace("/", "\\"),
            "annotations",
            "instances_" + self.image_set + ".json",
        )

        self.transform = make_transforms(self.image_set)
        self.visualize = visualize

        self.getMask = True

        self.coco = COCO(self.anno_file)
        self.image_ids = self.coco.getImgIds()
        # self.load_classes()  # read class information

        self.seq = iaa.Sequential(
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
                    {"height": self.height, "width": self.width}
                ),
                
            ]
        )

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_idx):  #
        image_info = self.coco.loadImgs(self.image_ids[image_idx])[0]
        image_file_path = os.path.join(self.image_folrder, image_info["file_name"])
        image = Image.open(image_file_path).convert("RGB")

        image = np.array(image)
        return image

    def __getitem__(self, idx):  # 인덱스에 접근할 때 호출
        image = self.load_image(idx)

        # https://imgaug.readthedocs.io/
        # # iaa.Resize(0.5) -> resize size 직접입력
        # 비율로 resize하면 float으로 나와서 나중에 mask를 못 씌움
        # resize_ratio = 0.5
        # resize_height, resize_width = int(image.shape[0] * resize_ratio), int(
        #     image.shape[1] * resize_ratio
        # )

        # Augment images.
        image = self.seq(image=image)

        if self.transform is not None:
            image = self.transform(image)

        return image



def show_originalimage(image) :
    img = image.cpu().numpy().copy()
    # img *= np.array([0.229, 0.224, 0.225])[:,None,None]
    # img += np.array([0.485, 0.456, 0.406])[:,None,None]
    img *= np.array([0.5, 0.5, 0.5])[:,None,None]
    img += np.array([0.5, 0.5, 0.5])[:,None,None]

    img = rearrange(img, "c h w -> h w c")
    img = img * 255
    img = img.astype(np.uint8)
    return img


if __name__ == "__main__":
    from einops import rearrange, asnumpy
    import yaml
    
    with open("config.yaml", "r") as stream:
            cfg = yaml.safe_load(stream)
            
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    coco_train = COCODataset(
        cfg['data']
    )

    ### Show image. tensor to cv2.imshow (numpy) ###
    img = coco_train[0] # torch.Size([3, 320, 320]) / (19, 5)
    
    ## Assuming CHW format, convert to HWC
    img = rearrange(img, "c h w -> h w c")

    ## denormalize :  IMAGENET 형식으로 normalize 된 경우
    IMAGENET_MEAN, IMAGENET_STD = torch.tensor([0.485, 0.456, 0.406]), torch.tensor(
        [0.229, 0.224, 0.225]
    )
    ## tensor -> np.uint8
    img = (
        asnumpy(torch.clip(255.0 * (img * IMAGENET_STD + IMAGENET_MEAN), 0, 255))
        .astype(np.uint8)
        .copy()
    )

    polygons, colors = [], []
    for ann in labels:
        bbox_x1, bbox_y1, bbox_x2, bbox_y2, label = ann.astype(int)
        c = (
            (np.random.random((1, 3)) * 0.6 * 255 + 0.4 * 255)
            .astype(np.uint8)
            .tolist()[0]
        )
        img = cv2.rectangle(
            img, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), color=c, thickness=2
        )

    cv2.imshow("", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ### end. Show image. tensor to cv2.imshow (numpy) ###