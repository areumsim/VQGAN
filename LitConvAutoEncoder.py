import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import optim, nn
from torch.nn import functional as F
import lightning as L
import wandb
# from Encoder import Encoder
# from Decoder import Decoder
# from VectorQuantizer import VectorQuantizer
from VQGAN import VQGAN
from Discriminator import Discriminator
import lpips

from einops import rearrange
from torch_ema import ExponentialMovingAverage



class LitAutoEncoder(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.hidden = cfg['model_params']['hidden'] # 96/16 
        k = cfg['model_params']['num_embeddings']
        self.emb_dim = cfg['model_params']['embeddings_dim']    # vqgan : n_z

        self.vqgan = VQGAN(cfg)
        self.ema = ExponentialMovingAverage(self.vqgan.vq_emb.parameters(), decay=0.995)    # 안씀
        self.ema.to(self.device)

        # self.mseLoss = nn.MSELoss() #
        self.vq_coef = cfg['model_params']['loss_vq_coef'] # 1
        
        self.perceptual_loss = lpips.LPIPS(net=cfg["model_params"]["perceptual_model"]).to(self.device)
        self.perceptual_loss_factor = cfg['model_params']['perceptual_loss_factor']
        self.rec_loss_factor = cfg['model_params']['rec_loss_factor']
        self.rec_l2_factor = cfg['model_params']['rec_l2_factor']
        self.disc_factor = cfg['model_params']['disc_factor']
        
        self.discriminator = Discriminator(cfg)

        self.automatic_optimization = False # 안씀
        self.disc_on = 0

    def forward(self, x):
        return self.vqgan(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        opt_vq, opt_disc = self.optimizers()

        ## generator    ===========================
        x, _ = batch
        x_hat, vq_codebook_loss = self.vqgan(x)

        ### VQ Loss  
        perceptual_loss = self.perceptual_loss(x, x_hat).mean()    # Perceptural loss로 변경!
        rec_loss = F.l1_loss(x, x_hat)  # l1 loss
        l2_loss = F.mse_loss(x, x_hat)  # l2 loss
        perceptual_rec_loss = self.perceptual_loss_factor * perceptual_loss + self.rec_loss_factor * rec_loss + self.rec_l2_factor * l2_loss
        
        vq_loss = perceptual_rec_loss + self.vq_coef * vq_codebook_loss

        disc_fake = self.discriminator(x_hat)
        ### Discriminator를 속이기 위한 Loss (generator)
        g_loss = -torch.mean(disc_fake)

        # eq. 6
        # _lambda = self.vqgan.calculate_lambda(perceptual_rec_loss, gan_loss) # λ : adopt_weight
        total_loss =  vq_loss + self.disc_factor*g_loss*self.disc_on

        ### Optimize vq ###
        opt_vq.zero_grad()
        self.manual_backward(total_loss)
        opt_vq.step()
        # self.ema.update()

        disc_loss = 0
        if self.disc_on != 0:
            ## discriminator    ===========================
            # loss,  gan_loss = self._calculate_loss_withD(x, x_hat)
            disc_real = self.discriminator(x)
            disc_fake = self.discriminator(x_hat.detach()) # x_hat은 generator의 결과이므로, generator의 gradient를 계산하지 않기 위해 detach()함수 사용
            
            d_loss_real = torch.mean(F.relu(1.0 - disc_real))
            d_loss_fake = torch.mean(F.relu(1.0 + disc_fake))
            
            ### discriminator loss : fake/real 구분
            disc_loss = (d_loss_real + d_loss_fake)*self.disc_on*0.5

            ### Optimize discriminator ###
            opt_disc.zero_grad()
            self.manual_backward(disc_loss)
            opt_disc.step()

        ### Logging to TensorBoard (if installed) by default
        self.log_dict({
                "[tr] perceptual_loss" : perceptual_loss,
                "[tr] l1_loss" : rec_loss,
                "[tr] l2_loss" : l2_loss,
                "[tr] vq_codebook_loss" : vq_codebook_loss,
                "[tr] g_loss" : g_loss,
                "[tr] discriminator_loss" : disc_loss})
        
        # apply schduler
        sch_g, sch_d = self.lr_schedulers()
        sch_g.step()
        sch_d.step()

        ## 학습 초반에 discriminator 학습 X (generator만 학습)
        # current step
        n_step = self.global_step
        if n_step >= 150_001:
            self.disc_on = 0.8

        return None

        
    def validation_step(self, batch, batch_idx) :
        x, _ = batch
        #with self.ema.average_parameters():
        x_hat, vq_codebook_loss = self.vqgan(x)
        self.val_output = [x, x_hat]

        ### VQ Loss  
        perceptual_loss = self.perceptual_loss(x, x_hat).mean()    # Perceptural loss로 변경!   #LPIPS
        rec_loss = F.l1_loss(x, x_hat)  # l1 loss
        # if torch.isnan(rec_loss):
        #     print('nan')

        self.log_dict({
                "[ts] perceptual_loss" : perceptual_loss,
                "[ts] l1_loss" : rec_loss,
                "[ts] vq_codebook_loss" : vq_codebook_loss,
                })
        
        # current epoch
        n_epoch = self.current_epoch
        if n_epoch >= 10:
            self.lambda_ = 1
        
        return None

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
            plt.close()

        return None

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(),
    #      lr=self.cfg['train_params']["learning_rate"],
    #        weight_decay=self.cfg['train_params']['weight_decay'])
    #     return optimizer

        
    def configure_optimizers(self):
        opt_vq = torch.optim.AdamW(
            list(self.vqgan.encoder.parameters())
            + list(self.vqgan.decoder.parameters())
            + list(self.vqgan.vq_emb.parameters()),
            lr=self.cfg['train_params']["learning_rate"],
            betas=self.cfg['train_params']["betas"],
            weight_decay=self.cfg['train_params']['weight_decay'])

        opt_disc = torch.optim.AdamW(self.discriminator.parameters(),
         lr=self.cfg['train_params']["learning_rate"],
                     betas=self.cfg['train_params']["betas"],
           weight_decay=self.cfg['train_params']['weight_decay'])
        
        #learningratelinearly warminguptoapeakvalueof1×10−4over50,000stepsandthendecayingto5×10−5overthe remaining450,000stepswithacosineschedule.
        scheduler_g = WarmupCosineDecayLR(opt_vq, warmup_steps=50000, decay_steps=450000, peak_lr=1e-4, final_lr=4e-6)
        scheduler_d = WarmupCosineDecayLR(opt_disc, warmup_steps=50000, decay_steps=450000, peak_lr=1e-4, final_lr=4e-6)

        return [opt_vq, opt_disc], [scheduler_g, scheduler_d]

import math
class WarmupCosineDecayLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, decay_steps, peak_lr, final_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.peak_lr = peak_lr
        self.final_lr = final_lr
        super(WarmupCosineDecayLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr = (self.peak_lr / self.warmup_steps) * self.last_epoch
        elif self.last_epoch < self.warmup_steps + self.decay_steps:
            lr = self.final_lr + 0.5 * (self.peak_lr - self.final_lr) * (
                1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / self.decay_steps))
        else:
            lr = self.final_lr  # Keep the learning rate constant after decay period
        return [lr for _ in self.base_lrs]

def show_originalimage(image) :
    image = torch.clamp(image, -1, 1)
    img = image.cpu().numpy().copy()
    img *= np.array([0.5, 0.5, 0.5])[:,None,None]
    img += np.array([0.5, 0.5, 0.5])[:,None,None]

    img = rearrange(img, "c h w -> h w c")
    img = img * 255
    img = img.astype(np.uint8)
    return img
