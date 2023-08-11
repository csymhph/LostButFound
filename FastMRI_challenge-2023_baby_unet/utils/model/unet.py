import torch
from torch import nn
from torch.nn import functional as F


class Unet(nn.Module):

    def __init__(self, in_chans, out_chans, drop_prob = 0.0):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        
        self.first_block = ConvBlock(in_chans, 64, drop_prob)
        self.down1 = Down(64, 128, drop_prob)
        self.down2 = Down(128, 256, drop_prob)
        self.down3 = Down(256, 512, drop_prob)
        self.down4 = Down(512, 1024, drop_prob)
        self.up1 = Up(1024, 512, drop_prob)
        self.up2 = Up(512, 256, drop_prob)
        self.up3 = Up(256, 128, drop_prob)
        self.up4 = Up(128, 64, drop_prob)
        
        self.last_block = nn.Conv2d(64, out_chans, kernel_size=1)
        self.dropout = drop_prob

    def norm(self, x):
        b, h, w = x.shape
        x = x.view(b, h * w)
        mean = x.mean(dim=1).view(b, 1, 1)
        std = x.std(dim=1).view(b, 1, 1)
        x = x.view(b, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def forward(self, input):
        input, mean, std = self.norm(input)
        input = input.unsqueeze(1)
        
        d1 = self.first_block(input)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        m0 = self.down4(d4)
        u1 = self.up1(m0, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        output = self.last_block(u4)
        output = output.squeeze(1)
        output = self.unnorm(output, mean, std)

        return output


class ConvBlock(nn.Module):

    def __init__(self, in_chans, out_chans, drop_prob):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, x):
        return self.layers(x)


class Down(nn.Module):

    def __init__(self, in_chans, out_chans, drop_prob):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_chans, out_chans, drop_prob)
        )

    def forward(self, x):
        return self.layers(x)


class Up(nn.Module):

    def __init__(self, in_chans, out_chans, drop_prob):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.up = nn.ConvTranspose2d(in_chans, in_chans // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_chans, out_chans, drop_prob)

    def forward(self, x, concat_input):
        x = self.up(x)
        concat_output = torch.cat([concat_input, x], dim=1)
        return self.conv(concat_output)