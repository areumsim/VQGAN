import torch
import torch.nn as nn
import torch.nn.functional as F


from einops import rearrange


class ConvAutoencoder(nn.Module):
    def __init__(self, cfg):
        super(ConvAutoencoder, self).__init__()
        
        ### Encoder
        # (in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        self.pool = nn.AvgPool2d(2, 2)

        ### Decoder
        # self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        self.upsampling = nn.Upsample(scale_factor=2, mode="nearest" )
        self.t_conv1 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.t_conv2 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.t_conv3 = nn.Conv2d(64, 16, 3, stride=1, padding=1)
        self.t_conv4 = nn.Conv2d(16, 3, 3, stride=1, padding=1)
        
        #TODO 
        # https://www.kaggle.com/code/anirbanmalick/image-colorization-pytorch-conv-autoencoder
        # self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        ### Encoder
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        latent_vector = x

        ### Decoder
        x = self.upsampling(x)
        x = F.relu(self.t_conv1(x))
        x = self.upsampling(x)
        x = F.relu(self.t_conv2(x))
        x = self.upsampling(x)
        x = F.relu(self.t_conv3(x))
        x = self.upsampling(x)
        # x = F.sigmoid(self.t_conv4(x))
        x = F.tanh(self.t_conv4(x))

        return x
    
    




if __name__ == "__main__":
    import yaml

    with open("C:/Users/wolve/arsim/autoencoder/config.yaml", "r") as stream:
        cfg = yaml.safe_load(stream)
    cfg = cfg["model_params"]

    image = torch.randn(3, 320, 320)
    # image = rearrange(image, "c W H -> 1 W H c")
    image = rearrange(image, "c W H -> 1 c W H")

    convAutoencoder = ConvAutoencoder(cfg)
    image_hat = convAutoencoder(image)
    print(image_hat)  # torch.Size([b, 3, 320, 320])

