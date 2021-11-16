import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

sys.path.append('..')


class TestResNetBasicBlock(unittest.TestCase):

    def test_same_block_output(self):
        import models.conv_blocks

        # this does not work with stride 2 because the shortcut is different:
        # res adapter uses avgpool reduction
        # Ydelbayev basic block uses sub-sampling (y[...,::2, ::2])
        original_block = BasicBlock(16, 32, stride=2, option='A').eval()
        new_block = models.conv_blocks.BasicBlock(
            16,  32, k=3, stride=2
        ).eval()

        with torch.no_grad():
            # set equal weights for both blocks
            wc1 = original_block.conv1.weight
            new_block.block_with_shortcut.block[0].weight = wc1
            assert new_block.block_with_shortcut.block[0].bias is None

            wc2 = original_block.conv2.weight
            new_block.block_with_shortcut.block[3].weight = wc2
            assert new_block.block_with_shortcut.block[3].bias is None

            for _ in range(10):
                x = torch.rand(1, 16, 16, 16)

                self.assertEqual(
                    torch.abs(original_block(x) - new_block(x)).sum().item(),
                    0.
                )


class TestPyramidNetBasicBlock(unittest.TestCase):
    def test_same_block_output(self):
        import models.conv_blocks

        stride = 2
        # or self.inplanes != int(round(featuremap_dim_1st)) * block.outchannel_ratio:
        if stride != 1:
            downsample = nn.AvgPool2d((2, 2), stride=(2, 2), ceil_mode=True)
        original_block = PyramidBasicBlock(
            16, 32, stride=stride, downsample=downsample
        ).eval()
        new_block = models.conv_blocks.PyramidBasicBlock(
            16,  32, k=3, stride=2
        ).eval()

        with torch.no_grad():
            # set equal weights for both blocks
            wc1 = original_block.conv1.weight
            new_block.block_with_shortcut.block[1].weight = wc1
            assert new_block.block_with_shortcut.block[1].bias is None

            wc2 = original_block.conv2.weight
            new_block.block_with_shortcut.block[4].weight = wc2
            assert new_block.block_with_shortcut.block[4].bias is None

            for _ in range(10):
                x = torch.rand(1, 16, 16, 16)

                self.assertEqual(
                    torch.abs(original_block(x) - new_block(x)).sum().item(),
                    0.
                )


class TestJOVFPReLUBlock(unittest.TestCase):
    def test_same_block_output(self):
        import models.conv_blocks

        original_block = FPReLUBlock(
            d_in=16, d_out=32, stride=2, q=2, k=3, c1x1_type='convbnrelu'
        ).eval()
        new_block = models.conv_blocks._FPBlockJOV(
            16,  32, k=3, stride=2, q=2
        ).eval()

        with torch.no_grad():
            # set equal weights for both blocks
            wc_upper = original_block.upper[0].weight
            new_block.upper[0].weight = wc_upper
            assert new_block.upper[0].bias is None

            wc_lower = original_block.lower[0].weight
            new_block.lower[0].weight = wc_lower
            assert new_block.lower[0].bias is None

            wc_left = original_block.left_dw[0].weight
            new_block.left_dw[0].weight = wc_left
            assert new_block.left_dw[0].bias is None

            wc_right = original_block.right_dw[0].weight
            new_block.right_dw[0].weight = wc_right
            assert new_block.right_dw[0].bias is None

            for _ in range(10):
                x = torch.rand(1, 16, 16, 16)

                self.assertEqual(
                    torch.abs(original_block(x) - new_block(x)).sum().item(),
                    0.
                )


# https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                # this short
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 0, planes//2), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/PyramidNet.py


class PyramidBasicBlock(nn.Module):
    outchannel_ratio = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PyramidBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(torch.FloatTensor(
                batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).fill_(0))
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut

        return out


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# https://github.com/pgruening/feature_products_cvpr/blob/master/models/fp_blocks.py

DEFAULT_1X1 = 'conv'
DEFAULT_BN = 'instance'


class IBlock(nn.Module):
    def __init__(self):
        super(IBlock, self).__init__()
        self.stride = -1
        self.relu_after_shortcut = False
        self.in_dim = -1
        self.out_dim = -1
        self.use_shortcut = True


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


class FPL1DWR1U1(IBase):
    def __init__(self, in_dim, out_dim, k=3, stride=1, q=1, which_batch_norm=DEFAULT_BN, c1x1_type=DEFAULT_1X1):
        super(FPL1DWR1U1, self).__init__(
            in_dim, out_dim, k=k, stride=stride, which_batch_norm=which_batch_norm)
        self.upper = get_upper(in_dim, int(q * out_dim), type_=c1x1_type)
        self._init_dw(int(q * out_dim), k, stride, use_relu=True)
        self.lower = get_lower(int(q * out_dim), out_dim, type_=c1x1_type)


class FPReLUBlock(FPL1DWR1U1, IBlock):
    # defined elsewhere
    def __init__(self, **kwargs):
        assert "which_batch_norm" not in kwargs.keys()
        super(FPReLUBlock, self).__init__(
            kwargs['d_in'], kwargs['d_out'], stride=kwargs.get('stride', 1),
            q=kwargs.get('q', 2), k=kwargs.get('k', 3),
            which_batch_norm=kwargs.get('bn_type', DEFAULT_BN),
            c1x1_type=kwargs.get('c1x1_type', DEFAULT_1X1)
        )
        self.in_dim = kwargs['d_in']
        self.out_dim = kwargs['d_out']
        self.use_shortcut = kwargs.get('use_shortcut', True)


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


if __name__ == '__main__':
    unittest.main()
