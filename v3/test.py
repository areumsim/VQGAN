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


    # ###### dataload & backbond - coco ######
    # transform = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # -1과 1 사이의 값으로 정규화
    #     ]
    # )

    # img_dir = 'C:/Users/wolve/arsim/autoencoder/STL10/' 
    # train = datasets.STL10(root=img_dir, split='train', download=True, transform=transform) #5000
    # test = datasets.STL10(root=img_dir, split='test', download=True, transform=transform)   #8000

    # train_loader = torch.utils.data.DataLoader(train,
    #                                         batch_size=cfg['train_params']['batch_size'],
    #                                         shuffle=True)
    # valid_loader = torch.utils.data.DataLoader(train,
    #                                         batch_size=cfg['train_params']['batch_size'],
    #                                         shuffle=True)
    # ###########################

    batch_size = cfg['train_params']["batch_size"]
    num_epoch = cfg['train_params']["num_epoch"]

    model_save_path = cfg['model_save_path']

    checkpoint_callback = ModelCheckpoint(dirpath=model_save_path)

    autoencoder = LitAutoEncoder(cfg)
    ckpt = 'model_save\epoch=8649-step=276801.ckpt'
    autoencoder.load_state_dict(torch.load(ckpt)['state_dict'])
    

    # # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    # # precision='16-mixed' : 연산 속도/메모리 사용량 향상 - 16비트 부동소수점 사용...
    # # val_check_interval=0.5 : 1 epoch당 2번 validation 실행 (0.5 epoch마다)
    # trainer = L.Trainer(limit_train_batches=batch_size, max_epochs=num_epoch,
    #                     devices='auto', accelerator='gpu', precision='16-mixed',
    #                     check_val_every_n_epoch=10, log_every_n_steps=16, 
    #                     limit_val_batches=50,
    #                     logger=wandb_logger,callbacks=[checkpoint_callback])
    
    # trainer.fit(model=autoencoder,
    #             train_dataloaders=train_loader,val_dataloaders=valid_loader,
    #             ckpt_path='model_save\epoch=8299-step=265601.ckpt')
    import cv2
    from einops import rearrange
    import torch
    im = cv2.imread(r"king.jpg")
    im = cv2.resize(im, (96, 96))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    im = rearrange(torch.tensor(im)/255, 'h w c -> 1 c h w').sub(0.5).div(0.5)

    z = autoencoder.encoder(im)
    recon = autoencoder.decoder(z)

    cv2.imwrite('tst.png',((recon.flip(1)+1)/2)[0].permute(1, 2, 0).detach().numpy()*255)
    

    