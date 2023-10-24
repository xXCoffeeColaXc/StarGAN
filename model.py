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
    def __init__(self, in_channels=3, feautues=64, c_dim=3, repeat_num=6) -> None:
        super(Generator, self).__init__()

        # in_channels = channels + c_dim (domain added in channel)
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels + c_dim, feautues, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(feautues, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        # Downsampling: 64-128-256
        self.down1 = ConvBlock(feautues, feautues*2)
        self.down2 = ConvBlock(feautues*2, feautues*4)

        # Bottleneck
        bottleneck_layers = []
        for i in range(repeat_num):
            bottleneck_layers.append(ResidualBlock(feautues*4, feautues*4))
        self.bottleneck = nn.Sequential(*bottleneck_layers)

        # Upsampling
        self.up1 = ConvBlock(feautues*4, feautues*2, down=False)
        self.up2 = ConvBlock(feautues*2, feautues, down=False)

        self.final_up = nn.Sequential(
            nn.Conv2d(feautues, in_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        )

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        bottle = self.bottleneck(d3)
        up1 = self.up1(bottle)
        up2 = self.up2(up1)
        out = self.final_up(up2)
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
    

if __name__ == "__main__":
    x = torch.randn(4,3,128,128)
    
    label = [0,1,0,2]
    label_org = torch.FloatTensor(label)

    #label_org = torch.randint(0, 3, (4,))

    print(f" label: {label_org}")
    print(label_org.shape)

    # Generate target domain labels randomly.
    rand_idx = torch.randperm(label_org.size(0))
    label_trg = label_org[rand_idx]

    #c_org = label_org.clone()
    c_org = label2onehot(label_org, 4)
    c_trg = label2onehot(label_trg, 4)

    #print(f"onehot vector: {c_org}")
    #print(c_org.shape)

    x_real = x.to(config.DEVICE)           # Input images.
    c_org = c_org.to(config.DEVICE)             # Original domain labels.
    c_trg = c_trg.to(config.DEVICE)             # Target domain labels.
    label_org = label_org.to(config.DEVICE)     # Labels for computing classification loss.
    label_trg = label_trg.to(config.DEVICE)     # Labels for computing classification loss.

    #print(c_org.size(0))
    #print(c_org.size(1))

    #print(x.size(2))
    #print(x.size(3))

    c = c_org.view(c_org.size(0), c_org.size(1), 1, 1)
    c = c.repeat(1, 1, x_real.size(2), x_real.size(3))
    x = torch.cat([x_real, c], dim=1)

    #print(f"c: {c[2]}")
    #print(c.shape)

    #print(x.shape)

    model = Generator(in_channels=config.CHANNEL_IMG, feautues=64, c_dim=config.NUM_DOMAINS)
    model = model.to(config.DEVICE)

    disc = Discriminator(image_size=config.IMAGE_SIZE, in_channels=config.CHANNEL_IMG, features=64, c_dim=config.NUM_DOMAINS)
    disc = disc.to(config.DEVICE)
    print(disc)

    # PatchGAN 2x2
    src, cls = disc(x_real)
   
    print(src)
    print(cls)

   

    #preds = model(x_real, c_org)
    #print(preds.shape)
    #print(model)    


    