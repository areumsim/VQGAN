#%%
import torch
from torchvision import datasets
import numpy as np

## downloading CIFAR10 data from torchvision.datasets
## reference: https://pytorch.org/vision/stable/datasets.html
img_dir = 'C:/Users/wolve/arsim/autoencoder/CIFAR10/'
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

#%%
import torch
from torchvision import datasets
from torchvision import transforms

img_dir = 'C:/Users/wolve/arsim/autoencoder/CIFAR10/'
train = datasets.CIFAR10(root=img_dir, train=True, download=True, transform=transform)
test = datasets.CIFAR10(root=img_dir, train=False, download=True, transform=transform)
data_loader_tr = torch.utils.data.DataLoader(train,
                                        batch_size=cfg['train_params']['batch_size'],
                                        shuffle=True)
data_loader_ts = torch.utils.data.DataLoader(train,
                                        batch_size=cfg['train_params']['batch_size'],
                                        shuffle=True)
images, labels = next(iter(data_loader_tr))
img_tr = images[0]  # 3, 32,32 