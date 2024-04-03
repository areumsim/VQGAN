import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import optim, nn
import lightning as L
import yaml
import wandb

from torchvision import datasets
from torchvision import transforms

from LitConvAutoEncoder import LitAutoEncoder
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

if __name__ == "__main__":
    #######   wandb   ######
    wandb_logger = WandbLogger(project="autoencoder-pytorch", name="resnet_1")
    
    with open("config.yaml", "r") as stream:
        cfg = yaml.safe_load(stream)


    ###### dataload & backbond - coco ######
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # -1과 1 사이의 값으로 정규화
        ]
    )

    img_dir = 'C:/Users/wolve/arsim/autoencoder/STL10/' 
    train = datasets.STL10(root=img_dir, split='train', download=True, transform=transform) #5000
    test = datasets.STL10(root=img_dir, split='test', download=True, transform=transform)   #8000

    train_loader = torch.utils.data.DataLoader(train,
                                            batch_size=cfg['train_params']['batch_size'],
                                            shuffle=True)
    valid_loader = torch.utils.data.DataLoader(train,
                                            batch_size=cfg['train_params']['batch_size'],
                                            shuffle=True)
    ###########################

    batch_size = cfg['train_params']["batch_size"]
    num_epoch = cfg['train_params']["num_epoch"]

    model_save_path = cfg['model_save_path']

    checkpoint_callback = ModelCheckpoint(dirpath=model_save_path)

    autoencoder = LitAutoEncoder(cfg)
    

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    # precision='16-mixed' : 연산 속도/메모리 사용량 향상 - 16비트 부동소수점 사용...
    # val_check_interval=0.5 : 1 epoch당 2번 validation 실행 (0.5 epoch마다)
    trainer = L.Trainer(limit_train_batches=batch_size, max_epochs=num_epoch,
                        devices='auto', accelerator='gpu', precision='16-mixed',
                        check_val_every_n_epoch=10, log_every_n_steps=16, 
                        limit_val_batches=50,
                        logger=wandb_logger,callbacks=[checkpoint_callback])
    
    trainer.fit(model=autoencoder,
                train_dataloaders=train_loader,val_dataloaders=valid_loader,
                ckpt_path='model_save\epoch=19999-step=640001.ckpt')
    

    