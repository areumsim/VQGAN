import torch
import torch.nn as nn
import torch.nn.functional as F


from einops import rearrange

from common_block import BasicBlock, conv1x1, conv3x3, NonLocalBlock

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
        # self.non_local = NonLocalBlock(256, 256, 256) # attention layer # TODO  
        self.residual6 = BasicBlock(128, 128, stride=1, downsample=None)  

        self.group_norm = nn.GroupNorm(32, 128) # num_groups(임의의 수-그룹수 , num_channels)
        self.swish = nn.SiLU()

        self.emb_dim = cfg['model_params']['embeddings_dim'] # 128
        self.conv2 = conv1x1(128, self.emb_dim, bias=False) # 채널수만! 


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
        # x = self.non_local(x) # TODO
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

