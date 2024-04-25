# ref. https://github.com/Shubhamai/pytorch-vqgan/blob/main/vqgan/discriminator.py
import torch
import torch.nn as nn
import torch.nn.functional as F


from einops import rearrange
from common_block import Basicblock_Upsampling, conv1x1, conv3x3
# spectral normalization
from torch.nn.utils import spectral_norm


# TODO, 
class Discriminator(nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()

        # Configuration parameters from cfg dictionary
        self.image_channels = cfg['Discriminator_params']['image_channels']
        self.num_filters_last = cfg['Discriminator_params']['num_filters_last']
        self.n_layers = cfg['Discriminator_params']['n_layers']

        # Construct the layers of the discriminator
        self.model = nn.Sequential(*self.build_layers())

    def build_layers(self):
        """ Helper function to construct layers of the discriminator """
        layers = [
            spectral_norm(nn.Conv2d(self.image_channels, self.num_filters_last, 4, 2, 1)),
            nn.LeakyReLU(0.2),
        ]
        num_filters_mult = 1

        # Adding multiple layers based on the configuration
        for i in range(1, self.n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2**i, 8)
            stride = 2 if i < self.n_layers else 1

            layers.extend([
                spectral_norm(nn.Conv2d(
                    self.num_filters_last * num_filters_mult_last,
                    self.num_filters_last * num_filters_mult,
                    4,
                    stride,
                    1,
                    bias=False
                )),
                # nn.BatchNorm2d(self.num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ])

        # Final convolution layer to output a single channel
        layers.append(spectral_norm(nn.Conv2d(self.num_filters_last * num_filters_mult, 1, 4, 1, 1)))
        return layers

    def forward(self, x):
        """ Forward pass of the discriminator """
        return self.model(x)


# class  Discriminator(nn.Module):
#     """  PatchGAN Discriminator
#     Args:
#         image_channels (int): Number of channels in the input image.
#         num_filters_last (int): Number of filters in the last layer of the discriminator.
#         n_layers (int): Number of layers in the discriminator.

#     """
#     def __init__(self, cfg):
#         super(Discriminator, self).__init__()

#         self.image_channels = cfg['Discriminator_params']['image_channels']
#         self.num_filters_last = cfg['Discriminator_params']['num_filters_last']
#         self.n_layers = cfg['Discriminator_params']['n_layers']

#         layers = [
#             nn.Conv2d(self.image_channels, self.num_filters_last, 4, 2, 1),
#             nn.LeakyReLU(0.2),
#         ]
#         num_filters_mult = 1

#         for i in range(1, self.n_layers + 1):
#             num_filters_mult_last = num_filters_mult
#             num_filters_mult = min(2**i, 8)
#             layers += [
#                 nn.Conv2d(
#                     self.num_filters_last * num_filters_mult_last,
#                     self.num_filters_last * num_filters_mult,
#                     4,
#                     2 if i < self.n_layers else 1,
#                     1,
#                     bias=False,
#                 ),
#                 nn.BatchNorm2d(self.num_filters_last * num_filters_mult),
#                 nn.LeakyReLU(0.2, True),
#             ]

#         layers.append(nn.Conv2d(self.num_filters_last * num_filters_mult, 1, 4, 1, 1))
#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.model(x)
            
