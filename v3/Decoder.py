import torch
import torch.nn as nn
import torch.nn.functional as F


from einops import rearrange

from resnet import Basicblock_Upsampling


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()

        self.upsampling = nn.Upsample(scale_factor=2, mode="nearest" )

        self.t_black1 = Basicblock_Upsampling(256, 128)
        self.t_black2 = Basicblock_Upsampling(128, 64)
        self.t_black3 = Basicblock_Upsampling(64, 16)
        self.t_black4 = Basicblock_Upsampling(16, 3, tanh_layer=True)

    def forward(self, x):
        
        x = self.t_black1(x)
        x = self.t_black2(x)
        x = self.t_black3(x)
        x = self.t_black4(x)

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

