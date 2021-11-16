import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conv_blocks import (AbsBabylon, ReLUBabylon, ResidualAdapter,
                                _FPBlockJOV)
from models.dying_relu_blocks import LeakyReLUMix


class MobileV2Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, in_dim, out_dim, expansion, stride):
        super(MobileV2Block, self).__init__()
        self.stride = stride

        planes = expansion * in_dim
        self.conv1 = nn.Conv2d(in_dim, planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, out_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_dim)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_dim),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileV1Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_dim, out_dim, stride):
        super(MobileV1Block, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=3,
                               stride=stride, padding=1, groups=in_dim, bias=False)
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv2 = nn.Conv2d(
            in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class FPBLOCKJOVForV2(nn.Module):
    def __init__(self, in_dim, out_dim, *, k, stride, q, use_1x1):
        super(FPBLOCKJOVForV2, self).__init__()

        fp_block = _FPBlockJOV(in_dim, out_dim, k=k, stride=stride, q=q)
        self.block_with_shortcut = ResidualAdapter(
            fp_block, stride=stride, relu_after_shortcut=False, in_dim=in_dim, out_dim=out_dim, use_1x1=use_1x1
        )

    def forward(self, x):
        return self.block_with_shortcut(x)


class AbsReLUBlockForV2(FPBLOCKJOVForV2):
    def __init__(self, in_dim, out_dim, *, k, stride, q, use_1x1):
        super(AbsReLUBlockForV2, self).__init__(
            in_dim, out_dim, k=k, stride=stride, q=q, use_1x1=use_1x1
        )
        # change the multiplication
        self.block_with_shortcut.block.mult = ReLUBabylon()


class AbsReLULeakyBothForV2(AbsReLUBlockForV2):
    def __init__(self, in_dim, out_dim, *, k, stride, q, neg_slope, use_1x1):
        super(AbsReLULeakyBothForV2, self).__init__(
            in_dim, out_dim, k=k, stride=stride, q=q, use_1x1=use_1x1
        )

        feat_dim = int(q * out_dim)
        self.block_with_shortcut.block.upper = LeakyReLUMix(
            in_dim, feat_dim, neg_slope=neg_slope
        )
        self.block_with_shortcut.block.lower = LeakyReLUMix(
            feat_dim, out_dim, neg_slope=neg_slope
        )


class FPBlockJOVLeakyBothForV2(FPBLOCKJOVForV2):
    def __init__(self, in_dim, out_dim, *, k, stride, q, neg_slope, use_1x1):
        super(FPBlockJOVLeakyBothForV2, self).__init__(
            in_dim, out_dim, k=k, stride=stride, q=q, use_1x1=use_1x1
        )

        feat_dim = int(q * out_dim)
        self.block_with_shortcut.block.upper = LeakyReLUMix(
            in_dim, feat_dim, neg_slope=neg_slope
        )
        self.block_with_shortcut.block.lower = LeakyReLUMix(
            feat_dim, out_dim, neg_slope=neg_slope
        )


# -----------------------------
# -------- Exp 7_1 ------------
# -----------------------------


class FPBlockLinLower(nn.Module):
    def __init__(self, in_dim, out_dim, *, k, stride, q, use_1x1):
        super(FPBlockLinLower, self).__init__()

        fp_block = _FpBlockLinLower(in_dim, out_dim, k=k, stride=stride, q=q)
        self.block_with_shortcut = ResidualAdapter(
            fp_block, stride=stride, relu_after_shortcut=False, in_dim=in_dim, out_dim=out_dim, use_1x1=use_1x1
        )

    def forward(self, x):
        return self.block_with_shortcut(x)


class AbsReLUBlockLinLower(FPBlockLinLower):
    def __init__(self, in_dim, out_dim, *, k, stride, q, use_1x1):
        super(AbsReLUBlockLinLower, self).__init__(
            in_dim, out_dim, k=k, stride=stride, q=q, use_1x1=use_1x1
        )
        self.block_with_shortcut.block.mult = ReLUBabylon()


class _FpBlockLinLower(_FPBlockJOV):
    def __init__(self, in_dim, out_dim, *, k, stride, q):
        super(_FpBlockLinLower, self).__init__(
            in_dim, out_dim, k=k, stride=stride, q=q
        )
        feat_dim = int(q * out_dim)
        self.lower = nn.Sequential(
            nn.Conv2d(feat_dim, out_dim, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim)
        )


# -----------------------------
# -------- Exp 7_5 ------------
# -----------------------------

class AbsReLUMobileV2Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, in_dim, out_dim, expansion, stride):
        super(AbsReLUMobileV2Block, self).__init__()
        self.stride = stride

        planes = expansion * in_dim
        self.conv1 = nn.Conv2d(in_dim, planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.abs_relu_block = ParallelDWAndBabylon(
            planes, kernel_size=3,
            stride=stride, padding=1
        )

        self.conv3 = nn.Conv2d(
            planes, out_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_dim)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_dim),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.abs_relu_block(out)

        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class ParallelDWAndBabylon(nn.Module):
    def __init__(self, dim, *, kernel_size, stride, padding):
        super(ParallelDWAndBabylon, self).__init__()
        self.conv_v = nn.Conv2d(
            dim, dim, kernel_size=kernel_size,
            padding=padding, bias=True, groups=dim, stride=stride
        )
        self.bn_v = nn.BatchNorm2d(dim)

        self.conv_g = nn.Conv2d(
            dim, dim, kernel_size=kernel_size,
            padding=padding, bias=True, groups=dim, stride=stride
        )
        self.bn_g = nn.BatchNorm2d(dim)

        self.abs_babylon = AbsBabylon()

    def change_weights(self, conv, batchnorm, max_angle=.49):
        assert conv.weight.shape[1] == 1
        assert conv.bias is None
        k = conv.kernel_size
        assert k[0] > 1
        if isinstance(k, tuple):
            assert k[0] == k[1]
            k = k[0]
        batchnorm.eval()

        W = conv.weight.detach().cpu().numpy()
        MU = batchnorm.running_mean.detach().cpu().numpy()
        SIGMA = batchnorm.running_var.detach().cpu().numpy()
        GAMMA = batchnorm.weight.detach().cpu().numpy()
        BETA = batchnorm.bias.detach().cpu().numpy()

        dim = W.shape[0]

        new_weights_v = np.zeros(W.shape).astype('float32')
        new_weights_g = np.zeros(W.shape).astype('float32')
        new_bias_v = np.zeros(self.conv_v.bias.shape).astype('float32')
        new_bias_g = np.zeros(self.conv_g.bias.shape).astype('float32')

        for i in range(dim):
            # filter to weight vector
            w = W[i, 0, ...].flatten()
            mu = MU[i]
            sigma = np.sqrt(SIGMA[i] + batchnorm.eps)
            gamma = GAMMA[i]
            beta = BETA[i]

            # zero maps: 0/eps
            if sigma < 1e-9:
                sigma = 1.
                #test = torch.rand(1, conv.in_channels, 32, 32)
                #xx = conv(test.to(conv.weight.device))
                #yy = batchnorm(xx)
                #xxx = 0

            new_w = w * gamma / sigma
            bias = beta - mu * gamma / sigma

            v, g = get_weight_pair(new_w, max_angle)

            new_weights_v[i, 0, ...] = v.reshape(k, k)
            new_weights_g[i, 0, ...] = g.reshape(k, k)
            new_bias_v[i] = .5 * bias
            new_bias_g[i] = .5 * bias

        with torch.no_grad():
            self.conv_v.weight = nn.Parameter(
                torch.from_numpy(new_weights_v)
            )
            self.conv_v.bias = nn.Parameter(
                torch.from_numpy(new_bias_v)
            )
            self.conv_g.weight = nn.Parameter(
                torch.from_numpy(new_weights_g)
            )
            self.conv_g.bias = nn.Parameter(
                torch.from_numpy(new_bias_g)
            )

    def forward(self, x):
        a = F.relu(self.bn_v(self.conv_v(x)))
        b = F.relu(self.bn_g(self.conv_g(x)))
        return self.abs_babylon(a, b)


def get_weight_pair(w, max_angle):
    v = np.random.rand(w.shape[0])

    def sca_prod(w, x):
        return (w * x).sum()

    # create 2d orthonormal basis
    # gram-schmidt, random vector norm. orthogonal to w
    base_y = v - sca_prod(v, w) / sca_prod(w, w) * w
    base_x = w / sca_prod(w, w)

    def R(angle):
        return np.stack([
            np.array([np.cos(angle), -1 * np.sin(angle)]),
            np.array([np.sin(angle), np.cos(angle)])
        ], 0)

    angle = np.random.rand() * np.pi * max_angle
    #angle = np.pi * max_angle

    # w = [1,0 ] = bisector of two vectors v and g
    v_2 = np.matmul(np.array([1, 0]), R(-angle))
    g_2 = np.matmul(np.array([1, 0]), R(+angle))

    # make sure that <w,w> = <v,w> + <g,w>
    v_2 = v_2 / np.linalg.norm(v_2)
    g_2 = g_2 / np.linalg.norm(g_2)

    c = np.linalg.norm(v_2 + g_2)
    v_2 = v_2 / c * sca_prod(w, w)
    g_2 = g_2 / c * sca_prod(w, w)

    # get original Nd vectors from 2d projection
    v = base_x * v_2[0] + base_y * v_2[1]
    g = base_x * g_2[0] + base_y * g_2[1]

    # ensure the norm is the same
    c = np.linalg.norm(v)
    g = c * g / np.linalg.norm(g)

    # test <w,w> == |<v,w> + <g,w>| - |<v,w> - <g,w>| in original space
    s0 = sca_prod(w, w)
    s1 = sca_prod(w, v) + sca_prod(w, g)
    s2 = np.abs(sca_prod(w, v) - sca_prod(w, g))

    return v, g


class MobileV2BlockWithAbs(nn.Module):
    def __init__(self, in_dim, out_dim, expansion, stride):
        super(MobileV2BlockWithAbs, self).__init__()
        self.block = MobileV2Block(in_dim, out_dim, expansion, stride)
        self.abs = Abs()

    def forward(self, x):
        return self.abs(self.block(x))


class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)
