import torch
import torch.nn as nn
import torch.nn.functional as F


from einops import rearrange

from resnet import BasicBlock, conv1x1, conv3x3

### 구현된 코드는 라이브러리 함수 썼는데, 아래는 직접 구현한 코드
### https://github.com/Shubhamai/pytorch-vqgan/blob/main/vqgan/common.py
# class Swish(nn.Module):
#     """Swish activation function first introduced in the paper https://arxiv.org/abs/1710.05941v2
#     It has shown to be working better in many datasets as compares to ReLU.
#     """

#     def __init__(self) -> None:
#         super().__init__()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:

#         return x * torch.sigmoid(x)


# class GroupNorm(nn.Module):
#     """Group Normalization is a method which normalizes the activation of the layer for better results across any batch size.
#     Note : Weight Standardization is also shown to given better results when added with group norm

#     Args:
#         in_channels (int): Number of channels in the input tensor.
#     """

#     def __init__(self, in_channels: int) -> None:
#         super().__init__()

#         # num_groups is according to the official code provided by the authors,
#         # eps is for numerical stability
#         # i think affine here is enabling learnable param for affine trasnform on calculated mean & standard deviation
#         self.group_norm = nn.GroupNorm(
#             num_groups=32, num_channels=in_channels, eps=1e-06, affine=True
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.group_norm(x)

class Downsample(nn.Module):
    def __init__(self, in_channels, stride=2):
        super(Downsample, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, bias=False, padding=0)

    def forward(self, x):
        pad = (0,1,0,1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()

        self.conv1 = conv3x3(3, 16, bias=True) # 이미지의 공간적인 정보를 섞고, 채널을 증가 시켜서 -> 이미지 정보 확장

        self.res_downsample1 = nn.Sequential(
            BasicBlock(16, 32, stride=1, downsample=None),
            BasicBlock(32, 32, stride=2, downsample=nn.AvgPool2d(2, 2))
            # BasicBlock(32, 32, stride=2, downsample=Downsample(32))
        )

        self.res_downsample2 = nn.Sequential(
            BasicBlock(32, 64, stride=1, downsample=None),
            BasicBlock(64, 64, stride=2, downsample=nn.AvgPool2d(2, 2))
        )

        self.res_downsample3 = nn.Sequential(
            BasicBlock(64, 128, stride=1, downsample=None),
            BasicBlock(128, 128, stride=2, downsample=nn.AvgPool2d(2, 2))
        )

        # self.res_downsample4 = nn.Sequential(
        #     BasicBlock(128, 256, stride=1, downsample=None),
        #     BasicBlock(256, 256, stride=2, downsample=Downsample(256))
        # )
        


        self.residual5 = BasicBlock(128, 128, stride=1, downsample=None)  
        ### self.non_local = NonLocalBlock(256, 256, 256) # attention layer
        self.residual6 = BasicBlock(128, 128, stride=1, downsample=None)  

        self.group_norm = nn.GroupNorm(32, 128) # num_groups(임의의 수) , num_channels
        self.swish = nn.SiLU()
        #TODO, config , self.emb_dim = 256
        self.conv2 = conv1x1(128, 128, bias=True) # 채널수만! , self.emb_dim = 256


    def forward(self, x):
        #stem
        x = self.swish(self.conv1(x))
        
        #residual blocks
        x = self.res_downsample1(x)
        x = self.res_downsample2(x)
        x = self.res_downsample3(x)
        # x = self.res_downsample4(x)

        # mid_block
        x = self.residual5(x)
        ### self.non_local
        x = self.residual6(x, last_layer=True)

        # output layer
        x = self.group_norm(x)
        x = self.swish(x)
        x = self.conv2(x)

        return x
    

        # x = self.block1(x)
        # x = self.block2(x)
        # x = self.block3(x)
        # x = self.block4(x, last_layer=True)


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

