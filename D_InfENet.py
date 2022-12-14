import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale

class ca_layer(nn.Module):
    def __init__(self, channel, reduction=32, bias=True):
        super(ca_layer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class MAM(nn.Module):
    def __init__(
            self, inchannels, kernel_size=3, reduction=32,
            bias=False, bn=False, act=nn.PReLU(), res_scale=1):
        super(MAM, self).__init__()

        ## SA
        self.SA = spatial_attn_layer()

        ## CA
        self.CA = ca_layer(inchannels, reduction, bias=bias)

        self.conv1x1 = nn.Conv2d(inchannels * 2, inchannels, kernel_size=1, bias=bias)

    def forward(self, x):
        sa_branch = self.SA(x)
        ca_branch = self.CA(x)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res


# ###############################################################################################################
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)

        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=3, stride=1, padding=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[0], kernel_size=1)
        self.p3_3 = nn.Conv2d(c3[0], c3[1], kernel_size=3, padding=1)

        self.p4_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_3(self.p3_2(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)

class CDBR(nn.Module):
    def __init__(self, in_channels):
        self.out_channels = in_channels
        super(CDBR, self).__init__()
        self.c_dbr1 = nn.Sequential(nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, dilation=2, padding=2), nn.BatchNorm2d(self.out_channels), nn.ReLU(),)
        self.c_dbr2 = nn.Sequential(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, dilation=2, padding=2),nn.BatchNorm2d(self.out_channels), nn.ReLU(), )
        self.c_dbr3 = nn.Sequential(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, dilation=2, padding=2),
                                    nn.BatchNorm2d(self.out_channels), nn.ReLU(), )
        self.c_dbr4 = nn.Sequential(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, dilation=2, padding=2),
                                    nn.BatchNorm2d(self.out_channels), nn.ReLU(), )
        self.c_dbr5 = nn.Sequential(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, dilation=2, padding=2),
                                    nn.BatchNorm2d(self.out_channels), nn.ReLU(), )
        self.c_dbr6 = nn.Sequential(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, dilation=2, padding=2),
                                    nn.BatchNorm2d(self.out_channels), nn.ReLU(), )
        self.c_dbr7 = nn.Sequential(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, dilation=2, padding=2),
                                    nn.BatchNorm2d(self.out_channels), nn.ReLU(), )
        self.c_dbr8 = nn.Sequential(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, dilation=2, padding=2),
                                    nn.BatchNorm2d(self.out_channels), nn.ReLU(), )
    def forward(self, x):
        x = self.c_dbr1(x)
        x1 = x
        x = self.c_dbr2(x)
        x2 = x
        x = self.c_dbr3(x)
        x = self.c_dbr4(x + x1)
        x = self.c_dbr5(x + x2)
        x = self.c_dbr6(x + x1)
        x = self.c_dbr7(x + x2)
        x = self.c_dbr8(x + x1)
        return x

#

class UpperNet(nn.Module):
    def __init__(self):
        super(UpperNet, self).__init__()
        self.cr1 = nn.Sequential(nn.Conv2d(1, 64, 3, 2, 1), nn.LeakyReLU())
        self.cbr1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), )
        self.cbr2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), )
        self.dau1 = MAM(64)
        self.cbr3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), )
        self.cbr4 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), )
        self.dau2 = MAM(64)

        self.cr2 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.LeakyReLU())
        self.cbr5 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), )
        self.cbr6 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), )
        self.dau3 = MAM(128)
        self.cbr7 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), )
        self.cbr8 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), )
        self.dau4 = MAM(128)

        self.trans1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.Tanh())
        self.trans2 = nn.Sequential(nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh())

        self.conv = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1),nn.Conv2d(1, 1, 1),nn.Conv2d(1, 1, 3, 1, 1), nn.PReLU(), )

        self.conv1 = nn.Sequential(nn.Conv2d(1, 1, 1), nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.LeakyReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(128, 128, 1), nn.LeakyReLU())

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.cr1(x)
        x = self.cbr1(x)
        x22 = x
        x2 = self.conv2(x)
        x = self.cbr2(x)
        x = self.dau1(x)
        x3 = x
        x = self.cbr3(x + x22)
        x = self.cbr4(x)
        x = self.dau2(x)

        x = self.cr2(x + x3)
        x4 = x
        x = self.cbr5(x)
        x = self.cbr6(x)
        x = self.dau3(x)
        x5 = self.conv5(x)
        x = self.cbr7(x + x4)
        x = self.cbr8(x)
        x = self.dau4(x)
        x = self.trans1(x + x5)
        x = self.trans2(x + x2)
        x = self.conv(x + x1)
        return x

class LowerNet(nn.Module):
    def __init__(self):
        super(LowerNet, self).__init__()
        self.cr1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1), nn.Conv2d(32, 32, 1), nn.LeakyReLU(),)
        self.inc = nn.Sequential(Inception(32, 8, (16, 28), (16, 28), 8),
                                 Inception(72, 16, (28, 40), (20, 40), 16))
        self.cr2 = nn.Sequential(nn.Conv2d(in_channels=112, out_channels=128, kernel_size=3, stride=2, padding=1), nn.Conv2d(128, 128, 1), nn.LeakyReLU(),)
        self.cdbr = CDBR(in_channels=128)
        self.trans1 = nn.Sequential(nn.ConvTranspose2d(128, 32, 4, 2, 1), nn.Tanh())
        self.trans2 = nn.Sequential(nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Tanh())
        self.conv = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1), nn.Conv2d(1, 1, 1), nn.Conv2d(1, 1, 3, 1, 1), nn.PReLU())

        self.conv112 = nn.Conv2d(32, 112, 1)
        self.conv1 = nn.Sequential(nn.Conv2d(1, 1, 1), nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 1), nn.LeakyReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, 1), nn.LeakyReLU())

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.cr1(x)
        x2 = self.conv2(x)
        x3 = self.conv112(x2)
        x = self.inc(x)
        x = self.cr2(x + x3)
        x4 = self.conv4(x)
        x = self.cdbr(x)
        x = self.trans1(x + x4)
        x = self.trans2(x + x2)
        return self.conv(x + x1)

class Double(nn.Module):
    def __init__(self):
        super(Double, self).__init__()
        self.U = UpperNet()
        self.L = LowerNet()
        self.conv = nn.Sequential(nn.Conv2d(2, 2, 1), nn.Conv2d(2, 1, 3, 1, 1), nn.LeakyReLU())

    def forward(self, x):
        L = self.L(x)
        U = self.U(x)
        Z = torch.cat((L, U), dim=1)
        Z = self.conv(Z)

        return Z


if __name__ == "__main__":
    x = torch.randn(size=(1, 1, 128, 128))
    net = Double()
    y = net(x)
    print((net(x)).shape)