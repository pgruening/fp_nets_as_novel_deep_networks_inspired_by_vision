import torch.nn as nn
import torch
from . import fp_blocks
import random
from . import config


class IBlock(nn.Module):
    def __init__(self):
        super(IBlock, self).__init__()
        self.stride = -1
        self.relu_after_shortcut = False
        self.in_dim = -1
        self.out_dim = -1
        self.use_shortcut = True

    @staticmethod
    def create_random_params():
        raise NotImplementedError


class ResidualAdapter(fp_blocks.ResidualAdapter):
    # defined elsewhere
    def __init__(self, block, **kwargs):
        super(ResidualAdapter, self).__init__(block, **kwargs)


class PyramidBasic(IBlock):
    def __init__(self, **kwargs):
        super(PyramidBasic, self).__init__()
        stride = kwargs.get('stride', 1)

        self.bn1 = nn.BatchNorm2d(kwargs['d_in'])
        self.conv1 = conv3x3(
            kwargs['d_in'], kwargs['d_out'], stride
        )
        self.bn2 = nn.BatchNorm2d(kwargs['d_out'])
        self.conv2 = conv3x3(
            kwargs['d_out'], kwargs['d_out']
        )
        self.bn3 = nn.BatchNorm2d(kwargs['d_out'])
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        self.in_dim = kwargs['d_in']
        self.out_dim = kwargs['d_out']

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        return out

    @staticmethod
    def create_random_params():
        return dict()


class PyramidBottleneck(IBlock):
    outchannel_ratio = 1

    def __init__(self, **kwargs):
        stride = kwargs.get('stride', 1)

        super(PyramidBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(kwargs['d_in'])
        self.conv1 = nn.Conv2d(
            kwargs['d_in'], kwargs['d_out'], kernel_size=1, bias=False
        )

        self.bn2 = nn.BatchNorm2d(kwargs['d_out'])
        self.conv2 = nn.Conv2d(
            kwargs['d_out'], (kwargs['d_out'] * 1), kernel_size=3,
            stride=stride,
            padding=1, bias=False
        )

        self.bn3 = nn.BatchNorm2d((kwargs['d_out'] * 1))
        self.conv3 = nn.Conv2d(
            (kwargs['d_out'] * 1),
            kwargs['d_out'] * PyramidBottleneck.outchannel_ratio,
            kernel_size=1, bias=False
        )

        self.bn4 = nn.BatchNorm2d(
            kwargs['d_out'] * PyramidBottleneck.outchannel_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        self.in_dim = kwargs['d_in']
        self.out_dim = kwargs['d_out']

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = self.bn4(out)
        return out

    @staticmethod
    def create_random_params():
        return dict()


class InvertedResidualLog(IBlock):
    def __init__(self, **kwargs):
        super(InvertedResidualLog, self).__init__()

        stride = kwargs.get('stride', 1)

        # expand first with
        q = kwargs.get('q', 6)
        self.expand = nn.Sequential(
            nn.Conv2d(kwargs['d_in'], q * kwargs['d_in'], 1, bias=False),
            nn.Conv2d(q * kwargs['d_in'], q * kwargs['d_in'], 3, groups=q *
                      kwargs['d_in'], bias=False, padding=1, stride=stride)
        )
        self.bn_expd = nn.BatchNorm2d(q * kwargs['d_in'])
        # relu already in loglayer at start

        k = kwargs.get('k', 3)
        self.log = LogLayer(kwargs['d_in'] * q, kwargs['d_out'], k=k)
        self.relu_log = nn.ReLU6(inplace=True)

        self.stride = stride

        self.in_dim = kwargs['d_in']
        self.out_dim = kwargs['d_out']

    def forward(self, x):
        x = self.expand(x)
        x = self.bn_expd(x)

        x = self.log(x)
        x = self.relu_log(x)

        return x

    @staticmethod
    def create_random_params():
        return {
            'k': random.choice([1, 3, 5]),
            'q': random.choice([1, 2, 4, 6])
        }


class FPBlock(fp_blocks.FPL1DWR0U1, IBlock):
    # defined elsewhere
    def __init__(self, **kwargs):
        assert "which_batch_norm" not in kwargs.keys()
        super(FPBlock, self).__init__(
            kwargs['d_in'], kwargs['d_out'], stride=kwargs.get('stride', 1),
            q=kwargs.get('q', 2), k=kwargs.get('k', 3),
            which_batch_norm=kwargs.get('bn_type', config.DEFAULT_BN),
            c1x1_type=kwargs.get('c1x1_type', config.DEFAULT_1X1)
        )
        self.in_dim = kwargs['d_in']
        self.out_dim = kwargs['d_out']
        self.use_shortcut = kwargs.get('use_shortcut', True)

    @staticmethod
    def create_random_params():
        return {
            'q': random.choice([1, 2, 4, 6])
        }


class FPReLUBlock(fp_blocks.FPL1DWR1U1, IBlock):
    # defined elsewhere
    def __init__(self, **kwargs):
        assert "which_batch_norm" not in kwargs.keys()
        super(FPReLUBlock, self).__init__(
            kwargs['d_in'], kwargs['d_out'], stride=kwargs.get('stride', 1),
            q=kwargs.get('q', 2), k=kwargs.get('k', 3),
            which_batch_norm=kwargs.get('bn_type', config.DEFAULT_BN),
            c1x1_type=kwargs.get('c1x1_type', config.DEFAULT_1X1)
        )
        self.in_dim = kwargs['d_in']
        self.out_dim = kwargs['d_out']
        self.use_shortcut = kwargs.get('use_shortcut', True)

    @staticmethod
    def create_random_params():
        return {
            'q': random.choice([1, 2, 4, 6])
        }


class AbsReLUBlock(fp_blocks.AbsReLUBlock, IBlock):
    # defined elsewhere
    def __init__(self, **kwargs):
        assert "which_batch_norm" not in kwargs.keys()
        super(AbsReLUBlock, self).__init__(
            kwargs['d_in'], kwargs['d_out'], stride=kwargs.get('stride', 1),
            q=kwargs.get('q', 2), k=kwargs.get('k', 3),
            which_batch_norm=kwargs.get('bn_type', config.DEFAULT_BN),
            c1x1_type=kwargs.get('c1x1_type', config.DEFAULT_1X1)
        )
        self.in_dim = kwargs['d_in']
        self.out_dim = kwargs['d_out']
        self.use_shortcut = kwargs.get('use_shortcut', True)

    @staticmethod
    def create_random_params():
        return {
            'q': random.choice([1, 2, 4, 6])
        }


class FPBilinBlock(fp_blocks.BilinearFPLayer, IBlock):
    # defined elsewhere
    def __init__(self, **kwargs):
        super(FPBilinBlock, self).__init__(
            kwargs['d_in'], kwargs['d_out'], stride=kwargs.get('stride', 1),
            q=kwargs.get('q', 2)
        )
        self.in_dim = kwargs['d_in']
        self.out_dim = kwargs['d_out']

    @staticmethod
    def create_random_params():
        return {
            'q': random.choice([1, 2, 4, 6])
        }


class FPBilinReLUBlock(fp_blocks.BilinearFPLayerReLU, IBlock):
    # defined elsewhere
    def __init__(self, **kwargs):
        super(FPBilinReLUBlock, self).__init__(
            kwargs['d_in'], kwargs['d_out'], stride=kwargs.get('stride', 1),
            q=kwargs.get('q', 2)
        )
        self.in_dim = kwargs['d_in']
        self.out_dim = kwargs['d_out']

    @staticmethod
    def create_random_params():
        return {
            'q': random.choice([1, 2, 4, 6])
        }


class BasicBlock(IBlock):
    expansion = 1

    def __init__(self, **kwargs):
        super(BasicBlock, self).__init__()

        in_planes = kwargs['d_in']
        planes = kwargs['d_out']
        self.stride = kwargs.get('stride', 1)

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3,
            stride=self.stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu_after_shortcut = True

        self.relu = nn.ReLU()

        self.in_dim = kwargs['d_in']
        self.out_dim = kwargs['d_out']

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # has relu after shortcut
        # out += self.shortcut(x)
        # out = F.relu(out)

        return out

# ----------------------------------------------------------------------------


class InvertedResidual(IBlock):
    def __init__(self, **kwargs):
        inp = kwargs['d_in']
        oup = kwargs['d_out']
        stride = kwargs.get('stride', 1)
        expand_ratio = kwargs.get('q', 6)
        use_shortcut = kwargs.get('use_shortcut', True)

        super(InvertedResidual, self).__init__()
        self.use_shortcut = use_shortcut

        self.in_dim = kwargs['d_in']
        self.out_dim = kwargs['d_out']
        stride = kwargs.get('stride', 1)

        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        self.stride = stride

    def forward(self, x):
        return self.conv(x)

    @staticmethod
    def create_random_params():
        return {
            'q': random.choice([1, 2, 4, 6])
        }


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ConvBNReLU(IBlock):
    def __init__(self, **kwargs):
        super(ConvBNReLU, self).__init__()
        in_planes = kwargs['d_in']
        out_planes = kwargs['d_out']
        kernel_size = kwargs.get('k', 3)
        stride = kwargs.get('stride', 1)
        groups = kwargs.get('groups', 1)
        use_shortcut = kwargs.get('use_shortcut', False)
        padding = (kernel_size - 1) // 2

        self.use_shortcut = use_shortcut
        self.in_dim = in_planes
        self.out_planes = out_planes

        self.f = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                      padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.f(x)


class Dropout(IBlock):
    def __init__(self, **kwargs):
        super(Dropout, self).__init__()
        p = kwargs['p']
        self.use_shortcut = False

        self.f = nn.Dropout(p)

    def forward(self, x):
        return self.f(x)


class Linear(IBlock):
    def __init__(self, **kwargs):
        super(Linear, self).__init__()
        self.in_dim = kwargs['d_in']
        self.out_dim = kwargs['d_out']
        self.use_shortcut = False

        self.f = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x):
        return self.f(x)


class LogLayer(nn.Module):
    def __init__(self, d_in, d_out, k=1, c=.5, log='log2', **kwargs):
        super(LogLayer, self).__init__()

        self.relu = nn.ReLU6(inplace=False)
        self.c = c
        if log == 'log2':
            self.log = torch.log2
            self.exp = fp_blocks.Exp2Layer()

        self.c1 = nn.Conv2d(d_in, d_out, k, bias=False, padding=k // 2,
                            dilation=kwargs.get('dilation', 1),
                            stride=kwargs.get('stride', 1),
                            groups=kwargs.get('groups', 1)
                            )

        self.bn = nn.BatchNorm2d(d_out, affine=True)
        self.bn_end = nn.BatchNorm2d(d_out, affine=False)

    def forward(self, x, get_all=False, **kwargs):
        relu_x = self.relu(x)
        bias_x = relu_x + self.c

        log_x = self.log(bias_x)
        mix_x = self.c1(log_x)

        mix_x_norm = self.bn(mix_x)
        exp_x = self.exp(mix_x_norm)

        exp_x_norm = self.bn_end(exp_x)

        return exp_x_norm
