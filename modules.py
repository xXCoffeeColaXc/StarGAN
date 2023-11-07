import torch.nn as nn
import torch
import config
import numpy as np
from utils import label2onehot

##############################
#           Blocks           #
##############################

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
        )


    def forward(self, x):
        return x + self.conv(x)
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, leaky=False) -> None:
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01)
            if leaky
            else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

##############################
#         Generator          #
##############################

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64, c_dim=3, repeat_num=6) -> None:
        super(Generator, self).__init__()

        # in_channels = channels + c_dim (domain added in channel)
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels + c_dim, features, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(features, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        # Downsampling: 64-128-256
        self.down1 = ConvBlock(features    , features * 2)
        self.down2 = ConvBlock(features * 2, features * 4)
        self.down3 = ConvBlock(features * 4, features * 8)

        # Bottleneck
        bottleneck_layers = [ResidualBlock(features * 8, features * 8) for _ in range(repeat_num)]
        self.bottleneck = nn.Sequential(*bottleneck_layers)

        # Upsampling
        self.up1 = ConvBlock(features * 8    , features * 4, down=False)
        self.up2 = ConvBlock(features * 4 * 2, features * 2, down=False)
        self.up3 = ConvBlock(features * 2 * 2, features    , down=False)

        self.final_up = nn.Sequential(
            nn.Conv2d(features, in_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        )

    def forward(self, x, c):
        # Concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        # Downsampling
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3) 

        # Bottleneck
        bottle = self.bottleneck(d4)

        # Upsampling with skip connections
        up1 = self.up1(bottle)
        up2 = self.up2(torch.cat([up1, d3], dim=1))  # Skip connection from down3
        up3 = self.up3(torch.cat([up2, d2], dim=1))  # Skip connection from down2

        # Final up (no skip connection because initial_down has different dimensions)
        out = self.final_up(up3)
        return out

##############################
#        Discriminator       #
##############################

class Discriminator(nn.Module):
    def __init__(self, image_size=128, in_channels=3, features=64, c_dim=3, repeat_num=6) -> None:
        super(Discriminator, self).__init__()
        
        layers = []
        for i in range(1, repeat_num):
            if i==1:
                layers.append(ConvBlock(in_channels, features, leaky=True))
            layers.append(ConvBlock(features, features*2, leaky=True))
            features = features*2

        self.conv_layers = nn.Sequential(*layers)

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.out_src = nn.Conv2d(features, 1, kernel_size=3, stride=1, padding=1, bias=False) # 2x2x1 patch
        self.out_cls = nn.Conv2d(features, c_dim, kernel_size=kernel_size, stride=1, padding=0, bias=False) #1x1x4

        
    def forward(self, x):
        h = self.conv_layers(x)
        out_src = self.out_src(h)
        out_cls = self.out_cls(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

    

    