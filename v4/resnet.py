
## ref. https://github.com/pytorch/vision/blob/6db1569c89094cf23f3bc41f79275c45e9fcb3f3/torchvision/models/resnet.py#L124

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


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


if __name__ == "__main__":
    Basicblock_Upsampling_Net = Basicblock_Upsampling(3, 9)
    Basicblock_Upsampling_Net2 = Basicblock_Upsampling(3, 9, upsample=False)


    x = torch.randn(1, 3, 32, 32)
    out = Basicblock_Upsampling_Net(x)
    print(out.shape)  # torch.Size([1, 9, 64, 64])

    out = Basicblock_Upsampling_Net2(x)
    print(out.shape)  # torch.Size([1, 9, 32, 32])

    print("")

