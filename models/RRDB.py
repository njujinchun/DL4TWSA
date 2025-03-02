import torch.nn as nn
import torch


##########################################################################
##---------- Spatial Attention ----------
class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3, bias=False):
        super(spatial_attn_layer, self).__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1) // 2, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_compress = torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
        scale = self.spatial(x_compress)
        return x * scale


##########################################################################
## ------ Channel Attention --------------
class ca_layer(nn.Module):
    def __init__(self, channel, reduction=24, bias=True, act=nn.PReLU()):
        super(ca_layer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                act,
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
##---------- spatial and channel attention module (SCAM) ----------
class SCAM(nn.Module):
    def __init__(self, nf, reduction=24, bias=False, act=nn.PReLU()):
        super(SCAM, self).__init__()

        ## Spatial Attention
        self.SA = spatial_attn_layer()
        ## Channel Attention
        self.CA = ca_layer(nf,reduction, bias=bias, act=act)
        self.conv1x1 = nn.Conv2d(nf*2, nf, kernel_size=1, bias=bias)

    def forward(self, x):
        res = x
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, nf, act=nn.ReLU()):
        super(DenseResidualBlock, self).__init__()

        def block(in_nf):
            layers = [nn.Conv2d(in_nf, nf, 3, 1, 1, bias=True)]
            layers += [nn.BatchNorm2d(nf)]
            layers += [act]
            return nn.Sequential(*layers)

        # each RDB has 4 conv layers, the input and output are concatenated at the end
        self.b1 = block(in_nf=1 * nf)
        self.b2 = block(in_nf=2 * nf)
        self.b3 = block(in_nf=3 * nf)
        self.b4 = block(in_nf=4 * nf)
        self.blocks = [self.b1, self.b2, self.b3, self.b4]

        # Dual Attention Module implemented on both input and output of the residual dense block
        self.dua_in = SCAM(nf,act=act)
        self.dua_res = SCAM(nf,act=act)

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        out = self.dua_in(out) + self.dua_res(x)

        return out


class RRDB(nn.Module):
    """
    RRDB: Residual in Residual Dense Block
    """
    def __init__(self, nf, act=nn.ReLU()):
        super(RRDB, self).__init__()

        # each RRDB has 3 RDBs
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(nf,act=act), DenseResidualBlock(nf,act=act), DenseResidualBlock(nf,act=act)
        )
        # Dual Attention Module implemented on both input and output of the RRDB
        self.dua_in = SCAM(nf,act=act)
        self.dua_res = SCAM(nf,act=act)

    def forward(self, x):
        inputs = x
        out = self.dense_blocks(inputs)
        out = self.dua_in(out) + self.dua_res(x)

        return out


class down_samp(nn.Module):
    """
    down_samp: downsampling module using strided convolution
    """
    def __init__(self, nf=64, act=nn.ReLU()):
        super(down_samp, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nf, 1*nf, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(1*nf),
            act,
        )

    def forward(self, x):
        return self.conv(x)


class up_samp(nn.Module):
    """
    up_samp: upsampling module using transposed convolution
    """
    def __init__(self, nf=64, act=nn.ReLU()):
        super(up_samp, self).__init__()
        self.convT = nn.Sequential(
            nn.ConvTranspose2d(2*nf, nf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(nf),
            act,
        )

    def forward(self, x):
        return self.convT(x)


class Net(nn.Module):
    def __init__(self, in_chan, out_chan, nf=64, ss=True, act=nn.PReLU()):
        super(Net, self).__init__()

        """ ARCHITECTURE 
        in_chan: number of input channels
        out_chan: number of output channels
        nf: number of feature maps
        ss: use series stationarization or not
        act: activation function
        """

        self.ss = ss
        self.features = nn.Sequential()

        # the first convolutional layer, also acts as a down-sampling layer
        self.features.add_module('inc', nn.Conv2d(in_chan, nf, kernel_size=3, stride=2, padding=1, bias=True))

        # the first RRDB block followed by down-sampling
        self.features.add_module('rbE1', RRDB(nf=nf,act=act))
        self.features.add_module('down1', down_samp(nf=nf,act=act))

        # the second RRDB block followed by down-sampling
        self.features.add_module('rbE2', RRDB(nf=nf,act=act))
        self.features.add_module('down2', down_samp(nf=nf,act=act))

        # the central RRDB block
        self.features.add_module('rbC', RRDB(nf=nf,act=act))

        # the first up-sampling block followed by the third RRDB block
        self.features.add_module('up2', up_samp(nf=nf,act=act))
        self.features.add_module('rbD2', RRDB(nf=nf,act=act))

        # the second up-sampling block followed by the fourth RRDB block
        self.features.add_module('up1', up_samp(nf=nf,act=act))
        self.features.add_module('rbD1', RRDB(nf=nf,act=act))

        # the final up-sampling block followed by a 1x1 convolutional layer
        self.features.add_module('up0', up_samp(nf=nf,act=act))
        self.features.add_module('outc', nn.Conv2d(nf, out_chan, kernel_size=1, bias=True))

    def forward(self, x, hist):
        '''
        :param x: the tensor of auxiliary data during the historical period and the target period
        :param hist: the tensor of historical data of the target variable
        :return: the tensor of predicted target variable
        '''
        if self.ss:
            # Series Stationarization by normalization using local mean and variance
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

            meansh = hist.mean(1, keepdim=True).detach()
            hist = hist - meansh
            stdevh = torch.sqrt(
                torch.var(hist, dim=1, keepdim=True, unbiased=False) + 1e-5)
            hist /= stdevh

        b, t, c, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b, t * c, h, w)
        b, t, c, h, w = hist.shape
        hist = hist.permute(0, 2, 1, 3, 4).reshape(b, t * c, h, w)
        x = torch.cat((x, hist), 1)

        x1 = self.features.inc(x)
        x1_ori = x1
        x2 = self.features.rbE1(x1)
        x2_ori = x2
        x3 = self.features.down1(x2)
        x3_ori = x3
        x4 = self.features.rbE2(x3)
        x4_ori = x4
        x5 = self.features.down2(x4)
        x5_ori = x5
        x6 = self.features.rbC(x5)
        x6 = torch.cat((x6, x5_ori), 1)
        x7 = self.features.up2(x6)
        x8 = self.features.rbD2(x7 + x4_ori)
        x8 = torch.cat((x8, x3_ori), 1)
        x9 = self.features.up1(x8)
        x10 = self.features.rbD1(x9 + x2_ori)
        x10 = torch.cat((x10, x1_ori), 1)

        x11 = self.features.up0(x10)
        y = self.features.outc(x11)

        if self.ss:
            ### De-Normalization using local mean and variance of historical data of the target variable
            y = y * \
                (stdevh[:, 0, [-1]].repeat(1, y.shape[1], 1, 1))
            y = y + \
                (meansh[:, 0, [-1]].repeat(1, y.shape[1], 1, 1))

        return y

    def _num_parameters_convlayers(self):
        n_params, n_conv_layers = 0, 0
        for name, param in self.named_parameters():
            if 'conv' in name:
                n_conv_layers += 1
            n_params += param.numel()
        return n_params, n_conv_layers

    def _count_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            print(name)
            print(param.size())
            print(param.numel())
            n_params += param.numel()
            print('num of parameters so far: {}'.format(n_params))

    def reset_parameters(self, verbose=False):
        for module in self.modules():
            # pass self, otherwise infinite loop
            if isinstance(module, self.__class__):
                continue
            if 'reset_parameters' in dir(module):
                if callable(module.reset_parameters):
                    module.reset_parameters()
                    if verbose:
                        print("Reset parameters in {}".format(module))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    x = torch.Tensor(1, 15, 3, 144, 360).to(device)
    hist = torch.Tensor(1, 12, 1, 144, 360).to(device)
    model = Net(57,3).to(device)
    print("number of parameters: {}\nnumber of layers: {}"
          .format(*model._num_parameters_convlayers()))
    print(model)
    yy = model(x,hist)
    print(yy.shape)