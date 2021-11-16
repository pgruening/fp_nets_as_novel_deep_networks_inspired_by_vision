"""
Essential to reproduce:
exp4
"""


import torch.nn as nn
import torch


from .conv_blocks import ReLUBabylon, ResidualAdapter, FPBlockJOV, Multiplication, AbsBabylon

class RealAbsReLUBlock(FPBlockJOV):
    def __init__(self, in_dim, out_dim, *, k, stride, q):
        super(RealAbsReLUBlock, self).__init__(
            in_dim, out_dim, k=k, stride=stride, q=q
        )
        # change the multiplication
        self.block_with_shortcut.block.mult = AbsBabylon()


class AbsReLUBlockNoNorm(nn.Module):
    def __init__(self, in_dim, out_dim, *, k, stride, q):
        super(AbsReLUBlockNoNorm, self).__init__()
        fp_block = _AbsReLUBlockNoNorm(
            in_dim, out_dim, k=k, stride=stride, q=q)

        self.block_with_shortcut = ResidualAdapter(
            fp_block, stride=stride, relu_after_shortcut=False
        )

    def forward(self, x):
        return self.block_with_shortcut(x)


class _AbsReLUBlockNoNorm(nn.Module):
    def __init__(self, in_dim, out_dim, *, k, stride, q):
        super(_AbsReLUBlockNoNorm, self).__init__()

        feat_dim = int(q * out_dim)
        self.upper = nn.Sequential(
            nn.Conv2d(in_dim, feat_dim, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU()
        )
        self.lower = nn.Sequential(
            nn.Conv2d(feat_dim, out_dim, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

        self.left_dw = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=k, stride=stride,
                      padding=k // 2, bias=False, groups=feat_dim),
            nn.ReLU()
        )
        self.right_dw = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=k, stride=stride,
                      padding=k // 2, bias=False, groups=feat_dim),
            nn.ReLU()
        )

        self.mult = ReLUBabylon()

    def forward(self, x):
        x = self.upper(x)
        x_left = self.left_dw(x)
        x_right = self.right_dw(x)

        x = self.mult(x_left, x_right)

        x = self.lower(x)

        return x
      
class FPBlockJOVNoNorm(AbsReLUBlockNoNorm):
    def __init__(self, in_dim, out_dim, *, k, stride, q):
        super(FPBlockJOVNoNorm, self).__init__(
            in_dim, out_dim, k=k, stride=stride, q=q
        )
        #change the multiplication
        self.block_with_shortcut.block.mult = Multiplication()
