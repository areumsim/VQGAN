
## ref. https://github.com/pytorch/vision/blob/6db1569c89094cf23f3bc41f79275c45e9fcb3f3/torchvision/models/resnet.py#L124

import torch
import torch.nn as nn

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

# 이미지의 공간적인 정보를 섞고, 채널을 증가 시켜서 -> 이미지 정보 확장
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


# 채널수만 변경! (공간적인 정보는 그대로)
def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.SiLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        
        self.rescale = False
        if inplanes != planes:
            self.conv3 = conv1x1(inplanes, planes)
            self.rescale = True
            

    def forward(self, x, last_layer=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        if self.rescale:
            identity = self.conv3(identity)
        out += identity
        if not last_layer:
            out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)


        out += identity
        out = self.relu(out)

        return out


class Basicblock_Upsampling(nn.Module):
    def __init__(self, inplanes, planes, norm_layer=None, tanh_layer=False, upsample=True, isLast_layer=False, rescale=False):
        super(Basicblock_Upsampling, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        if tanh_layer :
            self.last_layer = nn.Tanh()
        else :
            self.last_layer = nn.SiLU()

        self.upsampling = nn.Upsample(scale_factor=2, mode="nearest")
        self.relu = nn.SiLU()

        self.conv1 = conv3x3(inplanes, planes) #stride=1
        self.bn1 = norm_layer(planes)
        
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.resclae = rescale
        if rescale:
            self.conv3 = conv1x1(inplanes, planes)

        self.isUpsampling = upsample

        self.isLast_layer = isLast_layer

    def forward(self, x):
        if self.isUpsampling :
            identity = self.upsampling(x)
        else:   
            identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.isUpsampling :
            out = self.upsampling(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.resclae:
            identity = self.conv3(identity)

        out += identity

        if not self.isLast_layer:
            out = self.last_layer(out)
        
        return out


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=128, patch_size=4, stride=2, emb_size=128, img_size=32):
        super().__init__()

        ### 이미지를 패치사이즈로 나누고 flatten : B*C*H*W -> B*N*(P*P*C) 
        # 이미지에서는 그냥 안하고, conv 하고 .. 
        # Stride 크기가 Kernel Size와 동일하면, 겹치지 않음 
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=stride),
            Rearrange('b e (h) (w) -> b (h w) e')
        )

        N  = int((img_size/(patch_size-stride)-1))**2   # 총 patch 수
        # N = (patches.shape[1]) # (img_size//patch_size)**2 
        self.positions = nn.Parameter(torch.randn(N, emb_size))
    
    def forward(self, x):
        b,_,_,_ = x.shape
        
        x = self.projection(x)
        
        positions = repeat(self.positions, 'n d -> b n d', b=b)
        x+= positions
        return x


### Batch matrix multiplication (input과 mat2에 모두 batch가 있을 때 사용 )
# torch.bmm(input, mat2, *, deterministic=False, out=None) : [B, N, M] x [B, M, P] = [B, N, P]
# torch.einsum('bnm, bmp->bnp', x, y) : [B, N, M] x [B, M, P] = [B, N, P]

class MHAttention(nn.Module):
    def __init__(self, d_model, n_hidden, n_head, **kwargs):
        super(MHAttention, self).__init__()
        self.n_hidden = n_hidden

        # QW를 하려고 FC Layer , bias를 없애야함 wx+b 형태를 이용(b는 안씀)
        self.FC_Q = nn.Linear(d_model, n_hidden * n_head, bias=False)
        self.FC_K = nn.Linear(d_model, n_hidden * n_head, bias=False)
        self.FC_V = nn.Linear(d_model, n_hidden * n_head, bias=False)

        self.FC = nn.Linear((n_head * n_hidden), d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        q = self.FC_Q(q)
        k = self.FC_K(k)
        v = self.FC_V(v)

        #   b  len n_hidden*n_head
        q = rearrange(q, "b l (c h) -> b l c h", c=self.n_hidden)
        k = rearrange(k, "b l (c h) -> b l c h", c=self.n_hidden)
        v = rearrange(v, "b l (c h) -> b l c h", c=self.n_hidden)

        x = torch.einsum("b l d h, b j d h -> b h l j", q, k) / (self.n_hidden) ** 0.5

        if mask is not None:
            x = x + mask

        x = torch.softmax(x, dim=-1)  # -1로 해야, 마지막 차원에서 합해서 1
        x = torch.einsum("b h l j, b j d h -> b l d h", x, v)
        x = rearrange(x, "b l d h -> b l (d h)")

        x = self.FC(x)

        return x

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, n_hidden, n_head=16 ):
        super(NonLocalBlock, self).__init__()
        # self.in_channels = in_channels # d_model , 입출력차원
        self.in_channels = in_channels # 입출력차원
        self.model = 128 # 100
        self.n_head = n_head

        self.project = PatchEmbedding()
        self.mtAttention = MHAttention(self.model, n_hidden, n_head)

        self.FC = nn.Linear(self.model, self.in_channels, bias=False)

    def forward(self, x):
        input = x
        x = self.project(input)

        x = self.mtAttention(x, x, x)
        # x = torch.concat(x, -1)
        x = self.FC(x)
    
        return x


class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)

class NonLocalBlock_ref(nn.Module):
    def __init__(self, channels):
        super(NonLocalBlock_, self).__init__()
        self.in_channels = channels

        self.gn = GroupNorm(channels)
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        input = x
        h_ = self.gn(input)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        v = v.reshape(b, c, h*w)

        attn = torch.bmm(q, k)
        attn = attn * (int(c)**(-0.5))
        attn = torch.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w)

        return x + A


if __name__ == "__main__":
    NonLocalBlock_ = NonLocalBlock(128, 256) # attention layer    
    # NonLocalBlock_ = NonLocalBlock_ref(128) # attention layer    

    x = torch.randn(1, 128, 32, 32)
    out = NonLocalBlock_(x)
    print(out.shape)  # torch.Size([1, 9, 64, 64]) ([1, 128, 32, 32]) 

    print("")


