import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, norm='batch', add_dropout=False):
        super(DoubleConv, self).__init__()
        modules = [nn.Conv2d(in_ch, out_ch, 3, padding=1),
                   nn.BatchNorm2d(out_ch) if norm == 'batch' else nn.InstanceNorm2d(out_ch),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(out_ch, out_ch, 3, padding=1),
                   nn.BatchNorm2d(out_ch) if norm == 'batch' else nn.InstanceNorm2d(out_ch),
                   nn.ReLU(inplace=True)]
        if add_dropout:
            modules.append(nn.Dropout2d(p=0.3, inplace=False))
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm='batch', add_dropout=False):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch, norm=norm, add_dropout=add_dropout)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm='batch', add_dropout=False):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, norm=norm, add_dropout=add_dropout)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, norm='batch', upsample='bilinear', add_dropout=False):
        super(Up, self).__init__()
        if upsample == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif upsample == 'conv':
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        else:
            raise NotImplementedError
        self.conv = DoubleConv(in_ch, out_ch, norm=norm, add_dropout=add_dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # print(x1.size())
        # print(x2.size())
        diffY = x1.size()[3] - x2.size()[3]
        diffX = x1.size()[2] - x2.size()[2]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        # print(x2.size())
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self,
                 n_channels=3, n_classes=1, norm='instance', add_down_dropout=True, upsample='conv',
                 add_up_dropout=True):
        super(UNet, self).__init__()
        # TODO: move basic channel size parameters to the constructor function in order experiment using
        #  factories & configs
        self.inc = InConv(n_channels, 32, norm=norm, add_dropout=add_down_dropout)
        self.down1 = Down(32, 64, norm=norm, add_dropout=add_down_dropout)
        self.down2 = Down(64, 128, norm=norm, add_dropout=add_down_dropout)
        self.down3 = Down(128, 256, norm=norm, add_dropout=add_down_dropout)
        self.down4 = Down(256, 256, norm=norm, add_dropout=add_down_dropout)
        self.up1 = Up(512, 128, norm=norm, upsample=upsample, add_dropout=add_up_dropout)
        self.up2 = Up(256, 64, norm=norm, upsample=upsample, add_dropout=add_up_dropout)
        self.up3 = Up(128, 32, norm=norm, upsample=upsample, add_dropout=add_up_dropout)
        self.up4 = Up(64, 32, norm=norm, upsample=upsample, add_dropout=add_up_dropout)
        self.outc = OutConv(32, n_classes)
        # TODO: this is clearly an outdated architecture, try others, se-resnet like, resnext, ... - more efficient ones
        #  use instance norm only for small batch_size, otherwise batch norm is better
        #  have you heard about InplaceABN ? SyncronizedBatchNorm for more than 1 gpu ?

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
