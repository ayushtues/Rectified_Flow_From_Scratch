""" Full assembly of the parts to form the complete network """

from .unet_parts import *

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.ModuleList([
            DoubleConv(1, 64),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            Down(512, 1024)])

    def forward(self, x):
        for down in self.encoder:
            x = down(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(Decoder, self).__init__()
        self.decoder = nn.ModuleList([
            Up(1024, 512, bilinear),
            Up(512, 256, bilinear),
            Up(256, 128, bilinear),
            Up(128, 64, bilinear),
            OutConv(64, n_channels)])

    def forward(self, x):
        for up in self.decoder:
            x = up(x)
        return x



class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.encoder = Encoder()
        self.decoder = Decoder(n_channels, bilinear)

    def forward(self, x):
        z = self.encoder(x)
        x1 = self.decoder(z)
        return x1
