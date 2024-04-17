import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import optim, nn
from torch.nn import functional as F
import lightning as L
import wandb
from Encoder import Encoder
from Decoder import Decoder

from einops import rearrange

from VectorQuantizer import NearestEmbed, VectorQuantizer2
# from VectorQuantizer import VectorQuantizer

# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

        # TODO : cfg
        # num_embeddings, emb_dim
        self.hidden = 6     # 96/16 =
        k = 2048
        self.emb_dim = 128  # vqgan : n_z
        # self.vq_emb = NearestEmbed(k, self.emb_dim)
        self.vq_emb = VectorQuantizer2(k, self.emb_dim, beta=0.25)
        
        # self.criterion = nn.MSELoss()
        self.mseLoss = nn.MSELoss()
        self.vq_coef = 1.
        # self.comit_coef = 0.25
        # self.criterion = self.loss_function
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, _ = batch

        z_e = self.encoder(x)

        ### vq
        # z_q, _ = self.vq_emb(z_e, weight_sg=True)
        # emb, _ = self.vq_emb(z_e.detach())
        z_q, vq_loss, _ = self.vq_emb(z_e)
        # z_q = z_e
        # vq_loss = 0

        x_hat = self.decoder(z_q)

        # loss, recon_loss, vq_loss, commit_loss = self.criterion(x, x_hat, z_e, emb) 
        mse = self.mseLoss(x, x_hat)
        l1 = F.l1_loss(x, x_hat)
        recon_loss = mse + l1

        loss = recon_loss + self.vq_coef*vq_loss

        # Logging to TensorBoard (if installed) by default
        self.log("tr loss", loss)
        self.log_dict({"[tr] recon. loss": mse,
                "[tr] vq_loss" : vq_loss,
                "[tr] l1_loss" : l1})
        return loss

        
    def validation_step(self, batch, batch_idx) :
        x, _ = batch
        
        z_e = self.encoder(x)

        # z_q, _ = self.vq_emb(z_e, weight_sg=True)
        # emb, _ = self.vq_emb(z_e.detach()) # z_e.detach(), weight_sg=False -> weight만 업데이트, z_e는 업데이트 안함 
        z_q, vq_loss, _ = self.vq_emb(z_e)
        # z_q = z_e
        x_hat = self.decoder(z_q)

        # vq_loss = 0

        self.val_output = [x, x_hat]

        # loss = self.criterion(x, x_hat, z_e, emb) 
        
        # loss, recon_loss, vq_loss, commit_loss = self.criterion(x, x_hat, z_e, emb) 
        mse = self.mseLoss(x, x_hat)
        l1 = F.l1_loss(x, x_hat)
        recon_loss = mse + l1
        loss = recon_loss + self.vq_coef*vq_loss
        self.log("ts loss", loss)
        self.log_dict({"[ts] recon. loss": mse,
                       "[ts] vq_loss" : vq_loss,
                "[tr] l1_loss" : l1})
        return loss


    def on_validation_epoch_end(self):
        img_true = show_originalimage(self.val_output[0][0])
        img_predict = show_originalimage(self.val_output[1][0])
        concatenated_image = np.concatenate((img_true, img_predict), axis=1)
        
        try:
            wandb.log({
                'test/predict images': wandb.Image(concatenated_image),
            })
        except:
            os.makedirs('./result_image', exist_ok=True)
            n_epoch = self.current_epoch
            plt.imshow(concatenated_image)
            plt.savefig(f"./result_image/combined_image_valid_e{n_epoch}.png")
            plt.clf()

        return None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
         lr=self.cfg['train_params']["learning_rate"],
           weight_decay=self.cfg['train_params']['weight_decay'])
        return optimizer

    # def loss_function(self, x, recon_x, z_e, emb):
        
    #     self.mse = self.mseLoss(recon_x, x) # F.mse_loss
    #     self.vq_loss = self.mseLoss(emb, z_e.detach())     
    #     self.commit_loss = self.mseLoss(z_e, emb.detach())    

    #     # [VQVQE] β ranging from 0.1 to 2.0. We use β = 0.25
    #     loss = self.mse + self.vq_coef*self.vq_loss + self.comit_coef*self.commit_loss
    #     return loss, self.mse,  self.vq_loss, self.commit_loss

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
