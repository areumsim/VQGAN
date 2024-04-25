import os
from matplotlib import pyplot as plt
import numpy as np
import lightning as L
import yaml
import wandb

from dataloader_v5 import ImageNet256
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
    # with open("prj_idx.txt", "w") as f:
    #     f.write(str(prj_idx+1))
    
    prj_idx = prj_idx.split("-")
    prj_idx = prj_idx[0] + str(int(prj_idx[1])+1)
    # with open("prj_idx.txt", "w") as f:
    #     f.write(str(prj_idx))

    wandb_logger = WandbLogger(project="autoencoder-pytorch", name=f"v5_{prj_idx}")
    
    ### config
    with open("config.yaml", "r") as stream:
        cfg = yaml.safe_load(stream)
    
    ###### dataload  ######
    imageNet256 = ImageNet256(cfg['data'])
    train, valid = imageNet256.get_datasets()

    train_loader = DataLoader(train, batch_size=cfg['train_params']['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid, batch_size=cfg['train_params']['batch_size'], shuffle=True)
    # image = train[0][0] # (3, 256, 256)
    # ### label = train[0][1]  # no_use
    ###########################

    model_save_path = cfg['model_save_path']
    os.makedirs(model_save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(dirpath=model_save_path)

    # callback for save model every 10000 steps donot overwrite
    setp_checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_path,
        filename='step_{step}',
        every_n_train_steps=25_000,
        save_top_k=-1,
    )
    
    autoencoder = LitAutoEncoder(cfg)

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    # precision='16-mixed' : 연산 속도/메모리 사용량 향상 - 16비트 부동소수점 사용...
    # val_check_interval=0.5 : 1 epoch당 2번 validation 실행 (0.5 epoch마다)
    trainer = L.Trainer(max_epochs=cfg['train_params']["num_epoch"],
                        devices='auto', accelerator='gpu', precision='16-mixed',
                        log_every_n_steps=16, 
                        limit_val_batches=50,
                        logger=wandb_logger,
                        callbacks=[checkpoint_callback, setp_checkpoint_callback],
                        val_check_interval=1/5)
    
    trainer.fit(model=autoencoder,
                train_dataloaders=train_loader,val_dataloaders=valid_loader)
    

    