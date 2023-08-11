import torch
from torch import nn
from torch.nn import functional as F


class Unet(nn.Module):

    def __init__(self, in_chans, out_chans, drop_prob = 0.0):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        

        self.first_block = ConvBlock(in_chans, 32, drop_prob)
        self.down1 = Down(32, 64, drop_prob)
        self.down2 = Down(64, 128, drop_prob)
        self.down3 = Down(128, 256, drop_prob)
        #self.down4 = Down(256, 512, drop_prob)
        self.up1 = Up(256, 128, drop_prob)
        self.up2 = Up(128, 64, drop_prob)
        self.up3 = Up(64, 32, drop_prob)
        #self.up4 = Up(64, 32, drop_prob)
        
        self.last_block = nn.Conv2d(32, out_chans, kernel_size=1)
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
        #d4 = self.down3(d3)
        m0 = self.down3(d3)
        u1 = self.up1(m0, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        #u4 = self.up4(u3, d1)
        output = self.last_block(u3)
        output = output.squeeze(1)
        output = self.unnorm(output, mean, std)

        return output


class ConvBlock(nn.Module): #ResBlock이지만 수정의 편의를 위해 이름을 유지

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
        self.relu = nn.ReLU()

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
        self.relu = nn.ReLU()
        self.s = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=1, padding=0, stride=1),
            nn.MaxPool2d(2)
        )
    def forward(self, x):
        output = self.layers(x)
        s = self.s(x)
        output = output + s
        return output


class Up(nn.Module):

    def __init__(self, in_chans, out_chans, drop_prob):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.up = nn.ConvTranspose2d(in_chans, in_chans // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_chans, out_chans, drop_prob)
        self.s = nn.Conv2d(out_chans, out_chans, kernel_size=1, padding=0, stride=1)
        self.attention = DualAttention(in_chans, 8, out_chans)
        
    def forward(self, x, concat_input):
        x = self.up(x)
        concat_input = self.attention(concat_input)
        concat_output = torch.cat([concat_input, x], dim=1)
        s = self.s(x)
        output = self.conv(concat_output) + s
        return output


class SpatialAttention(nn.Module):
    def __init__(self, in_chans, int_chans, out_chans, kernel_size=3):
        super().__init__()
        self.in_chans = in_chans
        self.int_chans = int_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.Conv2d(1, int_chans, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(int_chans, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )
       

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        #max_out, _ = torch.max(x, dim=1, keepdim=True)
        w = self.layers(avg_out)
        output = w.repeat(1, self.out_chans, 1, 1)
        return output

class ChannelAttention(nn.Module):
    def __init__(self, in_chans, reduction_ratio=16):
        super().__init__()
        self.in_chans = in_chans
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #self.max_pool = nn.AdaptiveMaxPool2d(1)
        int_chans = max(in_chans // reduction_ratio, 1)
        self.int_chans = int_chans
       

        self.fc1 = nn.Linear(in_chans // 2, int_chans, bias=False)
        self.fc2 = nn.Linear(int_chans, in_chans // 2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()

        avg_out = self.avg_pool(x).view(batch, channels)
        avg_out = self.fc2(self.relu(self.fc1(avg_out))).view(batch, channels, 1, 1)

        #max_out = self.max_pool(x).view(batch, channels)
        #max_out = self.fc2(self.relu(self.fc1(max_out))).view(batch, channels, 1, 1)

        scale = self.sigmoid(avg_out)
        return x * scale

class DualAttention(nn.Module):
    def __init__(self, in_chans, int_chans, out_chans, kernel_size = 3, reduction_ratio=16):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.spatial_attention = SpatialAttention(in_chans, int_chans, out_chans)
        self.channel_attention = ChannelAttention(in_chans, reduction_ratio)

    def forward(self, x):
        out_sa = self.spatial_attention(x)
        out_ca = self.channel_attention(x)
        return x * out_sa * out_ca