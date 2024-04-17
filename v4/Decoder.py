import torch
import torch.nn as nn
import torch.nn.functional as F


from einops import rearrange

from resnet import Basicblock_Upsampling, conv1x1, conv3x3

class  Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()

        # self.upsampling = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv1 = conv1x1(128, 128, bias=True) #stride=1

        self.residual1 = Basicblock_Upsampling(128, 128, upsample=False)
        ### self.non_local = NonLocalBlock(256, 256, 256) # attention layer
        self.residual2 = Basicblock_Upsampling(128, 128, upsample=False)

        # self.res_upsample1 = nn.Sequential(
        #     Basicblock_Upsampling(256, 128, upsample=False),
        #     Basicblock_Upsampling(128, 128, upsample=True)
        # )
        self.res_upsample2 = nn.Sequential(
            Basicblock_Upsampling(128, 128, upsample=False),
            Basicblock_Upsampling(128, 64, upsample=True, rescale=True)
        )
        self.res_upsample3 = nn.Sequential(
            Basicblock_Upsampling(64, 64, upsample=False),
            Basicblock_Upsampling(64, 32, upsample=True, rescale=True)
        )
        self.res_upsample4 = nn.Sequential(
            Basicblock_Upsampling(32, 32, upsample=False),
            Basicblock_Upsampling(32, 16, upsample=True,  isLast_layer=True, rescale=True)
        )

        self.group_norm = nn.GroupNorm(4, 16)
        self.swish = nn.SiLU()
        self.conv2 = conv1x1(16, 3, bias=True) # 굳이 공간 정보 안섞어도되서 3x3 안씀, 3x3은 패딩도 들어가고
        
        self.tanh = nn.Tanh()

    def forward(self, x):
        
        x = self.swish(self.conv1(x))
        x = self.residual1(x)
        ### self.non_local
        x = self.residual2(x)

        # x = self.res_upsample1(x)
        x = self.res_upsample2(x)
        x = self.res_upsample3(x)
        x = self.res_upsample4(x)

        x = self.group_norm(x)
        x = self.swish(x)
        x = self.conv2(x)
        x = self.tanh(x)

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

