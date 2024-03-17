import os
from einops import rearrange
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import optim, nn
import lightning as L
import wandb
from Encoder import Encoder
from Decoder import Decoder


# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

        self.criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, _ = batch

        z = self.encoder(x)
        x_hat = self.decoder(z)

        loss = self.criterion(x_hat, x)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

        
    def validation_step(self, batch, batch_idx) :
        x, _ = batch
        
        z = self.encoder(x)
        x_hat = self.decoder(z)

        loss = self.criterion(x_hat, x)

        self.val_output = [x, x_hat]
        return loss


    def on_validation_epoch_end(self):
        img_true = show_originalimage(self.val_output[0][0])
        img_predict = show_originalimage(self.val_output[1][0])
        concatenated_image = np.concatenate((img_true, img_predict), axis=1)
        
        ## log images to wandb
        wandb.log({
            'test images': wandb.Image(img_true),
            'test predict': wandb.Image(img_predict)
        })
        wandb.log({
            'test/predict images': wandb.Image(concatenated_image),
        })

        n_epoch = self.current_epoch
        plt.imshow(concatenated_image)
        plt.savefig(f"./result_image/combined_image_valid_e{n_epoch}.png")
        plt.clf()   
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg['train_params']["learning_rate"], weight_decay=self.cfg['train_params']['weight_decay'])
        return optimizer


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