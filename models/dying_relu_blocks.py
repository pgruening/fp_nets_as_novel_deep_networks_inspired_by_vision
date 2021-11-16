"""
Essential to reproduce:
exp2
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from . import conv_blocks as cbs


class LeakyReLUMix(nn.Module):
    def __init__(self, in_dim, feat_dim, *, neg_slope, k=1):
        super(LeakyReLUMix, self).__init__()
        mix = nn.Conv2d(
            in_dim, feat_dim, kernel_size=k,
            stride=1, padding=k // 2, bias=False,
        )

        init.kaiming_uniform_(
            mix.weight, a=neg_slope,
            mode='fan_in',
            nonlinearity='leaky_relu'
        )

        self.f = nn.Sequential(
            mix,
            nn.BatchNorm2d(feat_dim),
            nn.LeakyReLU(negative_slope=neg_slope)
        )

    def forward(self, x):
        return self.f(x)


class AssymetricInitMix(nn.Module):
    def __init__(self, in_dim, feat_dim, *, neg_slope):
        super(AssymetricInitMix, self).__init__()

        mix = nn.Conv2d(
            in_dim, feat_dim, kernel_size=1,
            stride=1, padding=0, bias=False,
        )

        # shape: (out, in, 1, 1)
        new_weight = RAI(
            in_dim, feat_dim
        )
        with torch.no_grad():
            mix.weight = torch.nn.Parameter(torch.Tensor(new_weight))

        self.f = nn.Sequential(
            mix,
            nn.BatchNorm2d(feat_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.f(x)


def RAI(fan_in, fan_out):
    # +1 if bias
    W = np.random.randn(fan_out, fan_in, 1, 1) * 0.6007 / fan_in ** 0.5
    for j in range(fan_out):
        k = np.random.randint(0, high=fan_in)
        W[j, k, ...] = np.random.beta(2, 1)

    return W

# --------------------------------------


class DWSLeakyUpper(cbs.DWSBlock):
    def __init__(self, in_dim, out_dim, *, k, stride, q, neg_slope):
        super(DWSLeakyUpper, self).__init__(
            in_dim, out_dim, k=k, stride=stride, q=q
        )

        feat_dim = int(q * out_dim)
        self.block_with_shortcut.block.upper = LeakyReLUMix(
            in_dim, feat_dim, neg_slope=neg_slope
        )


class DWSLeakyBoth(cbs.DWSBlock):
    def __init__(self, in_dim, out_dim, *, k, stride, q, neg_slope):
        super(DWSLeakyBoth, self).__init__(
            in_dim, out_dim, k=k, stride=stride, q=q
        )

        feat_dim = int(q * out_dim)
        self.block_with_shortcut.block.upper = LeakyReLUMix(
            in_dim, feat_dim, neg_slope=neg_slope
        )
        self.block_with_shortcut.block.lower = LeakyReLUMix(
            feat_dim, out_dim, neg_slope=neg_slope
        )


# --------------------------------------


class AbsReLULeakyUpper(cbs.AbsReLUBlock):
    def __init__(self, in_dim, out_dim, *, k, stride, q, neg_slope):
        super(AbsReLULeakyUpper, self).__init__(
            in_dim, out_dim, k=k, stride=stride, q=q
        )

        feat_dim = int(q * out_dim)
        self.block_with_shortcut.block.upper = LeakyReLUMix(
            in_dim, feat_dim, neg_slope=neg_slope
        )


class AbsReLULeakyBoth(cbs.AbsReLUBlock):
    def __init__(self, in_dim, out_dim, *, k, stride, q, neg_slope):
        super(AbsReLULeakyBoth, self).__init__(
            in_dim, out_dim, k=k, stride=stride, q=q
        )

        feat_dim = int(q * out_dim)
        self.block_with_shortcut.block.upper = LeakyReLUMix(
            in_dim, feat_dim, neg_slope=neg_slope
        )
        self.block_with_shortcut.block.lower = LeakyReLUMix(
            feat_dim, out_dim, neg_slope=neg_slope
        )


# --------------------------------------


class FPBlockJOVLeakyUpper(cbs.FPBlockJOV):
    def __init__(self, in_dim, out_dim, *, k, stride, q, neg_slope):
        super(FPBlockJOVLeakyUpper, self).__init__(
            in_dim, out_dim, k=k, stride=stride, q=q
        )

        feat_dim = int(q * out_dim)
        self.block_with_shortcut.block.upper = LeakyReLUMix(
            in_dim, feat_dim, neg_slope=neg_slope
        )


class FPBlockJOVLeakyBoth(cbs.FPBlockJOV):
    def __init__(self, in_dim, out_dim, *, k, stride, q, neg_slope):
        super(FPBlockJOVLeakyBoth, self).__init__(
            in_dim, out_dim, k=k, stride=stride, q=q
        )

        feat_dim = int(q * out_dim)
        self.block_with_shortcut.block.upper = LeakyReLUMix(
            in_dim, feat_dim, neg_slope=neg_slope
        )
        self.block_with_shortcut.block.lower = LeakyReLUMix(
            feat_dim, out_dim, neg_slope=neg_slope
        )


# --------------------------------------


class INetLeakyUpper(cbs.INetBlock):
    def __init__(self, in_dim, out_dim, *, k, stride, neg_slope):
        super(INetLeakyUpper, self).__init__(
            in_dim, out_dim, k=k, stride=stride
        )

        self.block_with_shortcut.block.upper = LeakyReLUMix(
            in_dim, out_dim, neg_slope=neg_slope
        )


class INetLeakyBoth(cbs.INetBlock):
    def __init__(self, in_dim, out_dim, *, k, stride, neg_slope):
        super(INetLeakyBoth, self).__init__(
            in_dim, out_dim, k=k, stride=stride
        )

        self.block_with_shortcut.block.upper = LeakyReLUMix(
            in_dim, out_dim, neg_slope=neg_slope
        )
        self.block_with_shortcut.block.lower = LeakyReLUMix(
            3 * out_dim, out_dim, neg_slope=neg_slope
        )

# --------------------------------------
# --------------------------------------
# --------------------------------------


class DWSAssymUpper(cbs.DWSBlock):
    def __init__(self, in_dim, out_dim, *, k, stride, q, neg_slope):
        super(DWSAssymUpper, self).__init__(
            in_dim, out_dim, k=k, stride=stride, q=q
        )

        feat_dim = int(q * out_dim)
        self.block_with_shortcut.block.upper = AssymetricInitMix(
            in_dim, feat_dim, neg_slope=neg_slope
        )


class DWSAssymBoth(cbs.DWSBlock):
    def __init__(self, in_dim, out_dim, *, k, stride, q, neg_slope):
        super(DWSAssymBoth, self).__init__(
            in_dim, out_dim, k=k, stride=stride, q=q
        )

        feat_dim = int(q * out_dim)
        self.block_with_shortcut.block.upper = AssymetricInitMix(
            in_dim, feat_dim, neg_slope=neg_slope
        )
        self.block_with_shortcut.block.lower = AssymetricInitMix(
            feat_dim, out_dim, neg_slope=neg_slope
        )


# --------------------------------------


class AbsReLUAssymUpper(cbs.AbsReLUBlock):
    def __init__(self, in_dim, out_dim, *, k, stride, q, neg_slope):
        super(AbsReLUAssymUpper, self).__init__(
            in_dim, out_dim, k=k, stride=stride, q=q
        )

        feat_dim = int(q * out_dim)
        self.block_with_shortcut.block.upper = AssymetricInitMix(
            in_dim, feat_dim, neg_slope=neg_slope
        )


class AbsReLUAssymBoth(cbs.AbsReLUBlock):
    def __init__(self, in_dim, out_dim, *, k, stride, q, neg_slope):
        super(AbsReLUAssymBoth, self).__init__(
            in_dim, out_dim, k=k, stride=stride, q=q
        )

        feat_dim = int(q * out_dim)
        self.block_with_shortcut.block.upper = AssymetricInitMix(
            in_dim, feat_dim, neg_slope=neg_slope
        )
        self.block_with_shortcut.block.lower = AssymetricInitMix(
            feat_dim, out_dim, neg_slope=neg_slope
        )


# --------------------------------------


class FPBlockJOVAssymUpper(cbs.FPBlockJOV):
    def __init__(self, in_dim, out_dim, *, k, stride, q, neg_slope):
        super(FPBlockJOVAssymUpper, self).__init__(
            in_dim, out_dim, k=k, stride=stride, q=q
        )

        feat_dim = int(q * out_dim)
        self.block_with_shortcut.block.upper = AssymetricInitMix(
            in_dim, feat_dim, neg_slope=neg_slope
        )


class FPBlockJOVAssymBoth(cbs.FPBlockJOV):
    def __init__(self, in_dim, out_dim, *, k, stride, q, neg_slope):
        super(FPBlockJOVAssymBoth, self).__init__(
            in_dim, out_dim, k=k, stride=stride, q=q
        )

        feat_dim = int(q * out_dim)
        self.block_with_shortcut.block.upper = AssymetricInitMix(
            in_dim, feat_dim, neg_slope=neg_slope
        )
        self.block_with_shortcut.block.lower = AssymetricInitMix(
            feat_dim, out_dim, neg_slope=neg_slope
        )


# --------------------------------------


class INetAssymUpper(cbs.INetBlock):
    def __init__(self, in_dim, out_dim, *, k, stride, neg_slope):
        super(INetAssymUpper, self).__init__(
            in_dim, out_dim, k=k, stride=stride
        )

        self.block_with_shortcut.block.upper = AssymetricInitMix(
            in_dim, out_dim, neg_slope=neg_slope
        )


class INetAssymBoth(cbs.INetBlock):
    def __init__(self, in_dim, out_dim, *, k, stride, neg_slope):
        super(INetAssymBoth, self).__init__(
            in_dim, out_dim, k=k, stride=stride
        )

        self.block_with_shortcut.block.upper = AssymetricInitMix(
            in_dim, out_dim, neg_slope=neg_slope
        )
        self.block_with_shortcut.block.lower = AssymetricInitMix(
            3 * out_dim, out_dim, neg_slope=neg_slope
        )
