import os
from matplotlib import pyplot as plt
import numpy as np
import lightning as L
import yaml
import wandb

from dataloader_v3 import ImagetNetDataset
from torchvision import datasets
from torchvision import transforms

import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader, random_split

from LitConvAutoEncoder import LitAutoEncoder
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import random

if __name__ == "__main__":
    #######   wandb   ######
    with open("prj_idx.txt", "r") as f:
        prj_idx = f.read()
    prj_idx = int(prj_idx)
    with open("prj_idx.txt", "w") as f:
        f.write(str(prj_idx+1))

    wandb_logger = WandbLogger(project="autoencoder-pytorch", name=f"v3_ImageNet_vq_{prj_idx}")
    
    with open("config.yaml", "r") as stream:
        cfg = yaml.safe_load(stream)


    ###### dataload  ######
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # -1과 1 사이의 값으로 정규화
            transforms.Resize((96, 96)),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # ImageNet
        ]
    )
    
    ############ STL10 (96x96x3)
    # img_dir = 'C:/Users/wolve/arsim/autoencoder/STL10/' 
    # train = datasets.STL10(root=img_dir, split='train', download=True, transform=transform) #5000
    # test = datasets.STL10(root=img_dir, split='test', download=True, transform=transform)   #8000

    # train_loader = torch.utils.data.DataLoader(train,
    #                                         batch_size=cfg['train_params']['batch_size'],
    #                                         shuffle=True)
    # valid_loader = torch.utils.data.DataLoader(train,
    #                                         batch_size=cfg['train_params']['batch_size'],
    #                                         shuffle=True)
    
    ############ ImageNet_1k_256 (확인X, 느림, 256x256x3)
    # img_dir = 'C:/Users/wolve/arsim/autoencoder/ImageNet_1k_256/' 
    # train = ImagetNetDataset(img_dir, 'train', transform=transform)
    # valid = ImagetNetDataset(img_dir, 'train', transform=transform)
    # train_loader = torch.utils.data.DataLoader(train, batch_size=cfg['train_params']['batch_size'], shuffle=True)
    # valid_loader = torch.utils.data.DataLoader(valid, batch_size=cfg['train_params']['batch_size'], shuffle=True)
    # # image = train[0]

    ############ ImageNet 256×256 : ImageFolder 사용
    torch.manual_seed(42)  # For PyTorch
    random.seed(42)        # For Python's random module
    np.random.seed(42)     # For NumPy if used

    img_dir = 'C:/Users/wolve/arsim/autoencoder/ImageNet_256/' 
    dataset = datasets.ImageFolder(root=img_dir, transform=transform)

    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8) 
    valid_size = dataset_size - train_size
    print(f"Dataset size: {dataset_size}, Train size: {train_size}, Valid size: {valid_size}")

    # 데이터셋 분할에 사용할 generator 생성
    generator = torch.Generator().manual_seed(42)  # 동일한 seed를 사용하여 generator 생성
    train, valid = random_split(dataset, [train_size, valid_size], generator=generator)

    # #### indices 저장
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


    train_loader = DataLoader(train, batch_size=cfg['train_params']['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid, batch_size=cfg['train_params']['batch_size'], shuffle=True)
    image = train[0][0] # (3, 256, 256)
    # label = train[0][1] # 
    ###########################

    batch_size = cfg['train_params']["batch_size"]
    num_epoch = cfg['train_params']["num_epoch"]

    model_save_path = cfg['model_save_path']
    os.makedirs(model_save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(dirpath=model_save_path)

    autoencoder = LitAutoEncoder(cfg)
    

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    # precision='16-mixed' : 연산 속도/메모리 사용량 향상 - 16비트 부동소수점 사용...
    # val_check_interval=0.5 : 1 epoch당 2번 validation 실행 (0.5 epoch마다)
    trainer = L.Trainer(max_epochs=num_epoch,
                        devices='auto', accelerator='gpu', precision='16-mixed',
                        log_every_n_steps=16, 
                        limit_val_batches=50,
                        logger=wandb_logger,
                        callbacks=[checkpoint_callback],
                        val_check_interval=1/5)
    
    trainer.fit(model=autoencoder,
                train_dataloaders=train_loader,val_dataloaders=valid_loader)
                # ckpt_path='v3_model_save\epoch=72369-step=2315841.ckpt')
    

    