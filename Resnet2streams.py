import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out, inplace=True)

class ResEncoder(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, num_blocks=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(base_ch) for _ in range(num_blocks)])

    def forward(self, x):
        return self.blocks(self.stem(x))

class FusionModule(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, fa, fb):
        return self.fuse(torch.cat([fa, fb], dim=1))

class ResDecoder(nn.Module):
    def __init__(self, base_ch=32, out_ch=1, num_blocks=4):
        super().__init__()
        self.blocks = nn.Sequential(*[ResBlock(base_ch) for _ in range(num_blocks)])
        self.out = nn.Sequential(
            nn.Conv2d(base_ch, out_ch, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.out(self.blocks(x))

class ResNetTwoStreamFusion(nn.Module):
    def __init__(self, in_ch_a=1, in_ch_b=1, out_ch=1, base_ch=32,
                 enc_blocks=4, dec_blocks=4):
        super().__init__()
        self.enc_a = ResEncoder(in_ch_a, base_ch, enc_blocks)
        self.enc_b = ResEncoder(in_ch_b, base_ch, enc_blocks)
        self.fusion = FusionModule(base_ch)
        self.decoder = ResDecoder(base_ch, out_ch, dec_blocks)

    def forward(self, xa, xb):
        fa = self.enc_a(xa)
        fb = self.enc_b(xb)
        fused = self.fusion(fa, fb)
        return self.decoder(fused)
