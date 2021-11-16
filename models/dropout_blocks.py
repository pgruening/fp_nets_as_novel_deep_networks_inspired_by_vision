from .dying_relu_blocks import FPBlockJOVLeakyBoth
from .conv_blocks import FPBlockJOV
import torch.nn as nn


class FPBlockJOVLeakyBothDO(FPBlockJOVLeakyBoth):
    def __init__(self, in_dim, out_dim, *, k, stride, q, neg_slope, dropout):
        super(FPBlockJOVLeakyBothDO, self).__init__(
            in_dim, out_dim, k=k, stride=stride, q=q,
            neg_slope=neg_slope
        )
        # this upper module is a class
        self.block_with_shortcut.block.upper.f.add_module(
            'dropout', nn.Dropout2d(dropout)
        )


class FPBlockJOVDO(FPBlockJOV):
    def __init__(self, in_dim, out_dim, *, k, stride, q, dropout):
        super(FPBlockJOVDO, self).__init__(
            in_dim, out_dim, k=k, stride=stride, q=q
        )
        # this upper module is a Sequential
        self.block_with_shortcut.block.upper.add_module(
            'dropout', nn.Dropout2d(dropout)
        )
