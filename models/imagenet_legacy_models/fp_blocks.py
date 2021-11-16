import torch.nn as nn
import torch
from . import config
from DLBio.pytorch_helpers import get_device
DEFAULT_BN = config.DEFAULT_BN
DEFAULT_1X1 = config.DEFAULT_1X1


class ResidualAdapter(nn.Module):
    def __init__(self, block, relu_after_shortcut=False, use_1x1=False, in_dim=-1, out_dim=-1):
        super(ResidualAdapter, self).__init__()

        if use_1x1:
            self.conv_for_res = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        else:
            self.conv_for_res = None

        if block.stride > 1:
            self.downsample = nn.AvgPool2d(
                (2, 2), stride=(2, 2), ceil_mode=True)
        else:
            self.downsample = None
        self.block = block

        self.relu = nn.ReLU()
        self.relu_after_shortcut = relu_after_shortcut

    def forward(self, x):
        out = self.block(x)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        if self.conv_for_res is not None:
            # in case residual_channel < shortcut channel
            shortcut = self.conv_for_res(shortcut)

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.zeros(
                batch_size,
                residual_channel - shortcut_channel,
                featuremap_size[0],
                featuremap_size[1]
            )
            if out.is_cuda:
                padding = padding.cuda()

            out = out + torch.cat((shortcut, padding), 1)
        else:
            out = out + shortcut

        # old ResNet Basic Block uses ReLU after shortcut
        if self.relu_after_shortcut:
            return self.relu(out)

        else:
            return out


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class IBase(nn.Module):
    def __init__(self, in_dim, out_dim, k=3, stride=1, **kwargs):
        super(IBase, self).__init__()
        self.upper = nn.Identity()
        self.lower = nn.Identity()

        self.left_dw = None
        self.right_dw = None
        self.stride = stride
        self.which_batch_norm = kwargs.get('which_batch_norm', DEFAULT_BN)

        self.mult = Multiplication()

    def _init_dw(self, dim, k, stride, use_relu, padding=None):
        self.left_dw = self._get_dw(dim, k, stride, use_relu, padding=padding)
        self.right_dw = self._get_dw(dim, k, stride, use_relu, padding=padding)

    def _get_dw(self, dim, k, stride, use_relu, padding=None):
        if padding is None:
            padding = (k - 1) // 2
        else:
            print(f'Use new padding value: {padding}')

        tmp = nn.Sequential()
        if k != 3:
            print(f'using conv with kernel size: {k}')
        tmp.add_module('dw_conv', nn.Conv2d(
            dim, dim, k, groups=dim, padding=padding,
            stride=stride, bias=False)
        )

        if self.which_batch_norm == 'batch':
            print('using batchnorm')
            tmp.add_module('bn', nn.BatchNorm2d(dim))
        elif self.which_batch_norm == 'group':
            print('using group norm')
            tmp.add_module('bn', nn.GroupNorm(dim // 4, dim))
        elif self.which_batch_norm == 'instance':
            print('using instance norm')
            tmp.add_module('bn', nn.InstanceNorm2d(dim))
        elif self.which_batch_norm == 'none':
            print('no normalization')
        elif self.which_batch_norm == 'gdn':
            from pytorch_gdn import GDN
            print('using generalized divisive norm')
            tmp.add_module('gdn', GDN(dim, get_device()))
        else:
            raise ValueError(
                f'unknown normalization type: {self.which_batch_norm}'
            )

        if use_relu:
            tmp.add_module('relu', nn.ReLU())

        return tmp

    def forward(self, x):
        x = self.upper(x)
        x_left = self.left_dw(x)
        x_right = self.right_dw(x)

        x = self.mult(x_left, x_right)

        x = self.lower(x)

        return x


class ILogBase(IBase):

    def _init_dw(self, dim, k, stride, use_relu):
        super(ILogBase, self)._init_dw(dim, k, stride, use_relu)
        self.log = torch.log2
        self.exp = Exp2Layer()
        self.relu = nn.ReLU6()

        self.bn = nn.BatchNorm2d(dim, affine=False)

    def forward(self, x):
        x = self.upper(x)
        x_left = self.left_dw(x)
        x_right = self.right_dw(x)

        x_left = self.relu(x_left) + .5
        x_right = self.relu(x_right) + .5
        x = self.log(x_left) + self.log(x_right)
        x = self.bn(x)
        x = self.exp(x)

        x = self.lower(x)

        return x


class IPoolBase(IBase):
    def _init_dw(self, dim, k, stride, use_relu):
        super(IPoolBase, self)._init_dw(dim, k, stride, use_relu)
        self.pool = nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x):
        x = self.upper(x)
        x_left = self.left_dw(x)
        x_right = self.right_dw(x)
        x_right = -1. * self.pool(x_right)

        x = x_left * x_right

        x = self.lower(x)

        return x


class IBPoolBase(IPoolBase):

    def forward(self, x):
        x = self.upper(x)
        x_left = self.left_dw(x)
        x_left = self.pool(x_left)
        x_right = self.right_dw(x)
        x_right = self.pool(x_right)

        x = x_left * x_right

        x = self.lower(x)

        return x


class IStructBase(IBase):

    def forward(self, x):
        x = self.upper(x)
        fx = self.left_dw(x)
        fy = self.right_dw(x)

        x = torch.relu(fx) * torch.relu(fy) - torch.relu(fx * fy)

        x = self.lower(x)

        return x


class BilinearFPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, k=3, stride=1, **kwargs):
        super(BilinearFPLayer, self).__init__()

        use_relu = kwargs.get('use_relu', False)
        q = kwargs.get('q', 2)
        self.combine = nn.Conv2d(
            q * in_dim, in_dim, 1, groups=in_dim, bias=False)
        self.lower = ConvBNReLU(in_dim, out_dim, 1)

        self.left_dw = None
        self.right_dw = None
        self.stride = stride
        self._init_dw(in_dim, k, stride, use_relu, q)

    def _init_dw(self, dim, k, stride, use_relu, q):
        self.left_dw = self._get_dw(dim, k, stride, use_relu, q=q)
        self.right_dw = self._get_dw(dim, k, stride, use_relu, q=q)

    def _get_dw(self, dim, k, stride, use_relu, q=2):
        tmp = nn.Sequential()
        tmp.add_module('dw_conv', nn.Conv2d(
            dim, dim * q, k, groups=dim, padding=k // 2, stride=stride, bias=False)
        )
        tmp.add_module('bn', nn.BatchNorm2d(dim * q))
        if use_relu:
            tmp.add_module('relu', nn.ReLU())

        return tmp

    def forward(self, x):
        x_left = self.left_dw(x)
        x_right = self.right_dw(x)

        x = x_left * x_right

        x = self.combine(x)
        x = self.lower(x)

        return x


class BilinearFPLayerReLU(BilinearFPLayer):
    def __init__(self, in_dim, out_dim, k=3, stride=1, **kwargs):
        q = kwargs.get('q', 2)
        super(BilinearFPLayerReLU, self).__init__(
            in_dim, out_dim, k=k, stride=stride, use_relu=True, q=q
        )


class AbsReLUBlock(IBase):
    def __init__(self, in_dim, out_dim, k=3, stride=1, **kwargs):
        super(AbsReLUBlock, self).__init__(
            in_dim, out_dim, k=k, stride=stride, **kwargs)
        self.mult = AbsCombination()
        # based on the FP111 layer
        q = kwargs.get('q', 2)
        c1x1_type = kwargs.get('c1x1_type', DEFAULT_1X1)
        self.upper = get_upper(in_dim, int(q * out_dim), type_=c1x1_type)
        self._init_dw(int(q * out_dim), k, stride, use_relu=True)
        self.lower = get_lower(int(q * out_dim), out_dim, type_=c1x1_type)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class MyInvertedResidual(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, q=2, k=3, **kwargs):
        super(MyInvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(round(in_dim * q))

        hdim_quad = hidden_dim // 3
        hdim_lin = hidden_dim - hdim_quad
        hidden_dim = hdim_quad + hdim_lin

        self.lin_1x1 = ConvBNReLU(in_dim, hdim_lin, kernel_size=1)
        self.lin_dwc = nn.Sequential(
            ConvBNReLU(
                hdim_lin, hdim_lin, kernel_size=k,
                stride=stride, groups=hdim_lin),
        )
        self.quad_1x1 = ConvBNReLU(in_dim, hdim_quad, kernel_size=1)
        self.quad_dwc_1 = nn.Sequential(
            ConvBNReLU(
                hdim_quad, hdim_quad, kernel_size=k,
                stride=stride, groups=hdim_quad),
        )
        self.quad_dwc_2 = nn.Sequential(
            ConvBNReLU(
                hdim_quad, hdim_quad, kernel_size=k,
                stride=stride, groups=hdim_quad),
        )

        self.bn = nn.BatchNorm2d(hdim_quad)
        self.use_res_connect = self.stride == 1 and in_dim == out_dim

        self.out = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, x):
        x_lin = self.lin_1x1(x)
        x_lin = self.lin_dwc(x_lin)
        # x = x_lin  # test!
        x_quad = self.quad_1x1(x)
        x_q1 = self.quad_dwc_1(x_quad)
        x_q2 = self.quad_dwc_2(x_quad)
        x_quad = x_q1 * x_q2
        # x_quad = self.bn(x_quad)
        # x_quad = torch.relu(x_quad)

        x = torch.cat([x_lin, x_quad], dim=1)
        return self.out(x)
        # if self.use_res_connect:
        #    return x + self.out(x_lin)
        # else:


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                      padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class Exp2Layer(nn.Module):
    def __call__(self, x):
        return 2.**x

# ----------------------------------------------------------------------------
# Lower, Depth-Wise ReLU, Upper
# at least one of lower/upper needs to be active

# 001


class FPL0DWR0U1(IBase):
    def __init__(self, in_dim, out_dim, k=3, stride=1, which_batch_norm=DEFAULT_BN, c1x1_type=DEFAULT_1X1):
        super(FPL0DWR0U1, self).__init__(
            in_dim, out_dim, k=k, stride=stride, which_batch_norm=which_batch_norm)
        self._init_dw(in_dim, k, stride, use_relu=False)
        self.lower = get_lower(in_dim, out_dim, type_=c1x1_type)

# 011


class FPL0DWR1U1(IBase):
    def __init__(self, in_dim, out_dim, k=3, stride=1, which_batch_norm=DEFAULT_BN, c1x1_type=DEFAULT_1X1):
        super(FPL0DWR1U1, self).__init__(
            in_dim, out_dim, k=k, stride=stride, which_batch_norm=which_batch_norm)
        self._init_dw(in_dim, k, stride, use_relu=True)
        self.lower = get_lower(in_dim, out_dim, type_=c1x1_type)

# 100


class FPL1DWR0U0(IBase):
    def __init__(self, in_dim, out_dim, k=3, stride=1, which_batch_norm=DEFAULT_BN, c1x1_type=DEFAULT_1X1):
        super(FPL1DWR0U0, self).__init__(
            in_dim, out_dim, k=k, stride=stride, which_batch_norm=which_batch_norm)
        self.upper = get_upper(in_dim, out_dim, type_=c1x1_type)
        self._init_dw(out_dim, k, stride, use_relu=False)

# 101


class FPL1DWR0U1(IBase):
    def __init__(self, in_dim, out_dim, k=3, stride=1, q=1, which_batch_norm=DEFAULT_BN, c1x1_type=DEFAULT_1X1):
        super(FPL1DWR0U1, self).__init__(
            in_dim, out_dim, k=k, stride=stride, which_batch_norm=which_batch_norm)
        print(f'using q: {q}')
        self.upper = get_upper(in_dim, int(q * out_dim), type_=c1x1_type)
        self._init_dw(int(q * out_dim), k, stride, use_relu=False)
        self.lower = get_lower(int(q * out_dim), out_dim, type_=c1x1_type)

# 110


class FPL1DWR1U0(IBase):
    def __init__(self, in_dim, out_dim, k=3, stride=1, which_batch_norm=DEFAULT_BN, c1x1_type=DEFAULT_1X1):
        super(FPL1DWR1U0, self).__init__(
            in_dim, out_dim, k=k, stride=stride, which_batch_norm=which_batch_norm)
        self.upper = get_upper(in_dim, out_dim, type_=c1x1_type)
        self._init_dw(out_dim, k, stride, use_relu=True)

# 111


class FPL1DWR1U1(IBase):
    def __init__(self, in_dim, out_dim, k=3, stride=1, q=1, which_batch_norm=DEFAULT_BN, c1x1_type=DEFAULT_1X1):
        super(FPL1DWR1U1, self).__init__(
            in_dim, out_dim, k=k, stride=stride, which_batch_norm=which_batch_norm)
        print(f'using q: {q}')
        self.upper = get_upper(in_dim, int(q * out_dim), type_=c1x1_type)
        self._init_dw(int(q * out_dim), k, stride, use_relu=True)
        self.lower = get_lower(int(q * out_dim), out_dim, type_=c1x1_type)


class LogFPL1DWR0U1(ILogBase):
    def __init__(self, in_dim, out_dim, k=3, stride=1, q=1, which_batch_norm=DEFAULT_BN, c1x1_type=DEFAULT_1X1):
        super(LogFPL1DWR0U1, self).__init__(
            in_dim, out_dim, k=3, stride=stride)
        self.upper = get_upper(in_dim, int(q * out_dim), type_=c1x1_type)
        self._init_dw(int(q * out_dim), k, stride, use_relu=False)
        self.lower = get_lower(int(q * out_dim), out_dim, type_=c1x1_type)


class FPPoolL0DWR0U1(IPoolBase):
    def __init__(self, in_dim, out_dim, k=3, stride=1, **kwargs):
        super(FPPoolL0DWR0U1, self).__init__(
            in_dim, out_dim, k=3, stride=stride)
        self._init_dw(in_dim, k, stride, use_relu=False)
        self.lower = get_lower(in_dim, out_dim, type_=c1x1_type)


class BPoolL0DWR0U1(IBPoolBase):
    def __init__(self, in_dim, out_dim, k=3, stride=1, **kwargs):
        super(BPoolL0DWR0U1, self).__init__(
            in_dim, out_dim, k=k, stride=stride)
        self._init_dw(in_dim, k, stride, use_relu=False)
        self.lower = get_lower(in_dim, out_dim, type_=c1x1_type)


class StructL0DWR0U1(IStructBase):
    def __init__(self, in_dim, out_dim, k=3, stride=1, **kwargs):
        super(StructL0DWR0U1, self).__init__(
            in_dim, out_dim, k=k, stride=stride)
        self._init_dw(in_dim, k, stride, use_relu=False)
        self.lower = get_lower(in_dim, out_dim, type_=c1x1_type)


# DEFAULT  VALUE 101


def get_block(block_type, in_dim, out_dim, k=3, stride=1, q=1, add_res=False, **kwargs):
    blocks_ = {
        'FP001': FPL0DWR0U1,  # no q
        'FP011': FPL0DWR1U1,  # no q
        'FP100': FPL1DWR0U0,  # no q
        'FP101': FPL1DWR0U1,
        'FP110': FPL1DWR1U0,  # no q
        'FP111': FPL1DWR1U1,
        'L_FP101': LogFPL1DWR0U1,
        'Pool001': FPPoolL0DWR0U1,
        'BPool001': BPoolL0DWR0U1,
        'Struct001': StructL0DWR0U1,
        'InvRes': MyInvertedResidual,
        'BilinearFP': BilinearFPLayer,
        'BilinearFPReLU': BilinearFPLayerReLU,
    }
    assert block_type in blocks_.keys()

    block = blocks_[block_type](
        in_dim, out_dim, k=k, stride=stride, q=q, **kwargs
    )
    if add_res:
        if out_dim < in_dim:
            block = ResidualAdapter(
                block, use_1x1=True, in_dim=in_dim, out_dim=out_dim
            )
        else:
            block = ResidualAdapter(block)

    return block


def get_upper(in_dim, out_dim, **kwargs):
    type_ = kwargs.get('type_', DEFAULT_1X1)
    return get_1x1_function(in_dim, out_dim, type_=type_)


def get_lower(in_dim, out_dim, **kwargs):
    type_ = kwargs.get('type_', DEFAULT_1X1)
    return get_1x1_function(in_dim, out_dim, type_=type_)


def get_1x1_function(in_dim, out_dim, type_=DEFAULT_1X1):
    if type_ == 'convbnrelu':
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()

        )
    elif type_ == 'convrelu':
        print('using conv relu')
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.ReLU()
        )
    elif type_ == 'conv':
        print('using conv')
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, bias=False)
        )


class Multiplication(nn.Module):
    def forward(self, x, y):
        return x * y


class AbsCombination(nn.Module):
    def forward(self, a, b):
        # to approximage abs(a+b) - abs(a-b) -> both inputs need to be >= 0
        return torch.relu(a + b) - torch.relu(a - b) - torch.relu(b - a)
