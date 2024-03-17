import torch
import torch.nn as nn
import torch.nn.functional as F


from einops import rearrange

from resnet import BasicBlock


class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()

        self.block1 = BasicBlock(3, 16,stride=2, downsample=nn.AvgPool2d(2, 2))  
        self.block2 = BasicBlock(16, 64,stride=2, downsample=nn.AvgPool2d(2, 2))  
        self.block3 = BasicBlock(64, 128,stride=2, downsample=nn.AvgPool2d(2, 2))  
        self.block4 = BasicBlock(128, 256,stride=2, downsample=nn.AvgPool2d(2, 2))  

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        return x
    




# if __name__ == "__main__":
#     import yaml

#     with open("C:/Users/wolve/arsim/autoencoder/config.yaml", "r") as stream:
#         cfg = yaml.safe_load(stream)
#     cfg = cfg["model_params"]

#     image = torch.randn(3, 320, 320)
#     # image = rearrange(image, "c W H -> 1 W H c")
#     image = rearrange(image, "c W H -> 1 c W H")

#     convAutoencoder = ConvAutoencoder(cfg)
#     image_hat = convAutoencoder(image)
#     print(image_hat)  # torch.Size([b, 3, 320, 320])

