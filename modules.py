import torch.nn as nn
import torch
import config
import numpy as np
from utils import label2onehot

##############################
#           Blocks           #
##############################
class SelfAttention(nn.Module):
    def __init__(self, channels, size) -> None:
        """
        Args:
            channels: Channel dimension.
            size: Current image resolution.
        """
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(ResidualBlock, self).__init__()

       
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),  # NOTE GroupNorm
            nn.ReLU(inplace=True), # NOTE GELU
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),  # NOTE GroupNorm
        )


    def forward(self, x):
        return x + self.conv(x) # F.gelu(x + self.conv(x))
    

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

        # TODO create depth parameter for controlling down and upsamling layers depth
        # Downsampling: 64-128-256-512
        self.down1 = ConvBlock(features    , features * 2)
        # NOTE sa1
        self.down2 = ConvBlock(features * 2, features * 4)
        # NOTE sa2
        self.down3 = ConvBlock(features * 4, features * 8)
        # NOTE sa3

        # Bottleneck
        bottleneck_layers = [ResidualBlock(features * 8, features * 8) for _ in range(repeat_num)]
        self.bottleneck = nn.Sequential(*bottleneck_layers)

        # Upsampling
        self.up1 = ConvBlock(features * 8    , features * 4, down=False)
        # NOTE sa4
        self.up2 = ConvBlock(features * 4 * 2, features * 2, down=False)
        # NOTE sa5
        self.up3 = ConvBlock(features * 2 * 2, features    , down=False)
        # NOTE sa6

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
    def __init__(self, image_size=128, in_channels=3, features=64, c_dim=3, repeat_num=4) -> None:
        super(Discriminator, self).__init__()
       
        layers = []
        current_features = in_channels
        for i in range(repeat_num - 1):
            next_features = features * (2 ** i)
            layers.append(ConvBlock(current_features, next_features, leaky=True))
            current_features = next_features 

        self.conv_layers = nn.Sequential(*layers)

        kernel_size = image_size // (2 ** (repeat_num - 1))

        # Output layer for real/fake discrimination (PatchGAN)
        self.out_src = nn.Conv2d(current_features, 1, kernel_size=3, stride=1, padding=1, bias=False)

        # Output layer for class prediction (c_dim outputs, each a 1x1 patch)
        self.out_cls = nn.Conv2d(current_features, c_dim, kernel_size=kernel_size, stride=1, padding=0, bias=False)

    def forward(self, x):
        h = self.conv_layers(x)
        out_src = self.out_src(h)
        out_cls = self.out_cls(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

    

    