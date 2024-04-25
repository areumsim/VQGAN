import torch
from torch import nn, optim

from Encoder import Encoder
from Decoder import Decoder

from VectorQuantizer import VectorQuantizer
from vqtorch.nn import VectorQuant
from einops import rearrange

class VQGAN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

        self.emb_dim = cfg['model_params']['embeddings_dim']
        k = cfg['model_params']['num_embeddings']
        self.vq_emb = VectorQuantizer(k, self.emb_dim, beta=cfg['model_params']['loss_vqgan_beta'])

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_dict, _ = self.vq_emb(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, vq_dict

    # TODO, ref : https://github.com/Shubhamai/pytorch-vqgan/blob/main/vqgan/vqgan.py
    def calculate_lambda(self, perceptual_rec_loss, gan_loss):
        """Calculating lambda shown in the eq. 7 of the paper

        Args:
            perceptual_rec_loss (torch.Tensor): Perceptual reconstruction loss.
            gan_loss (torch.Tensor): loss from the GAN discriminator.
        """
        last_layer = list(self.decoder.children())[-1]
        last_layer_weight = last_layer.weight

        # Because we have multiple loss functions in the networks, retain graph helps to keep the computational graph for backpropagation
        # https://stackoverflow.com/a/47174709
        perceptual_rec_loss_grads = torch.autograd.grad(
            outputs=perceptual_rec_loss, inputs=last_layer_weight, retain_graph=True    #,allow_unused=True
        )[0]
        gan_loss_grads = torch.autograd.grad(
            gan_loss, last_layer_weight, retain_graph=True
        )[0]

        lmda = torch.norm(perceptual_rec_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        lmda = torch.clamp(
            lmda, 0, 1e4
        ).detach()  # Here, we are constraining the value of lambda between 0 and 1e4,

        return 0.8 * lmda  # Note: not sure why we are multiplying it by 0.8... ?

    # @staticmethod
    # def adopt_weight(
    #     gan_factor: float, i: int, threshold: int, value: float = 0.0
    # ) -> float:
    #     """Starting the discrimator later in training, so that our model has enough time to generate "good-enough" images to try to "fool the discrimator".

    #     To do that, we before eaching a certain global step, set the discriminator factor by `value` ( default 0.0 ) .
    #     This discriminator factor is then used to multiply the discriminator's loss.

    #     Args:
    #         gan_factor (float): This value is multiple to the discriminator's loss.
    #         i (int): The current global step
    #         threshold (int): The global step after which the `gan_factor` value is retured.
    #         value (float, optional): The value of discriminator factor before the threshold is reached. Defaults to 0.0.

    #     Returns:
    #         float: The discriminator factor.
    #     """

    #     if i < threshold:
    #         gan_factor = value

    #     return gan_factor