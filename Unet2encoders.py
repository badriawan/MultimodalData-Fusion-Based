import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // reduction),
            nn.ReLU(),
            nn.Linear(in_ch // reduction, in_ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg, max_], dim=1)
        return x * torch.sigmoid(self.conv(attn))


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

class Encoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        return e1, e2, e3

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(128 + 128, 128)  # 256 → 128

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(64 + 64, 64)     # 128 → 64

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x, skip2, skip1):
        x = self.up2(x)
        x = self.dec2(torch.cat([x, skip2], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, skip1], dim=1))

        return torch.sigmoid(self.out(x))




class UNetDualEncoderFusion(nn.Module):
    def __init__(self, use_attention=True):
        super().__init__()
        self.use_attention = use_attention

        # Encoder A: RGB + NIR (2 channel)
        self.encoder_a = Encoder(in_ch=2)

        # Encoder B: Thermal (1 channel)
        self.encoder_b = Encoder(in_ch=1)

        self.attn = CBAM(256) if use_attention else nn.Identity()
        self.decoder = Decoder()

    def forward(self, x):
        nir = x[:, 0:1]
        therm = x[:, 1:2]
        rgb = x[:, 2:3]

        # Encoder A (RGB + NIR)
        e1a, e2a, e3a = self.encoder_a(torch.cat([rgb, nir], dim=1))

        # Encoder B (Thermal)
        e1b, e2b, e3b = self.encoder_b(therm)

        # Fusion (deep features)
        fused = torch.cat([e3a, e3b], dim=1)
        fused = self.attn(fused)

        # Skip fusion (concat, bukan add)
        skip2 = torch.cat([e2a, e2b], dim=1)
        skip1 = torch.cat([e1a, e1b], dim=1)

        return self.decoder(fused, skip2, skip1)


