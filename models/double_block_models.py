import torch.nn as nn
from . import conv_blocks as cb
from . import dying_relu_blocks as drcb
from .cifar_base_model import BaseModel
from .dying_relu_blocks import LeakyReLUMix


def get_model(model_type, in_dim, out_dim, device, **model_kwargs):
    N = int(model_kwargs['N'][0])
    assert N > 0 and N < 20

    q = float(model_kwargs['q'][0])
    assert q > 0

    if model_type == 'AbsReLU-ALL-DB':
        get_block = get_block_adapter('AbsReLUBlock', q=q)
        model = DoubleBlockBaseModel(
            in_dim, out_dim, block_getter=get_block, N=N,
        )

    elif model_type == 'LeakyBoth-AbsReLU-ALL-DB':
        neg_slope = float(model_kwargs['neg_slope'][0])
        get_block = get_block_adapter(
            'AbsReLULeakyBoth', q=q, neg_slope=neg_slope
        )
        model = DoubleBlockBaseModel(
            in_dim, out_dim, block_getter=get_block, N=N,
        )

    elif model_type == 'DWS-ALL-DB':
        get_block = get_block_adapter('DWSBlock', q=q)
        model = DoubleBlockBaseModel(
            in_dim, out_dim, block_getter=get_block, N=N,
        )

    elif model_type == 'LeakyBoth-Abs-ALL-PyrBack':
        neg_slope = float(model_kwargs['neg_slope'][0])
        get_block = get_block_adapter(
            'LeakyAbsBlockPyrBack', q=q, neg_slope=neg_slope
        )
        model = BaseModel(
            in_dim, out_dim, block_getter=get_block, N=N,
        )
    elif model_type == 'Abs-ALL-PyrBack':
        get_block = get_block_adapter('AbsBlockPyrBack', q=q)
        model = BaseModel(
            in_dim, out_dim, block_getter=get_block, N=N,
        )

    elif model_type == 'LeakyBoth-Abs-ALL-3x3Start':
        neg_slope = float(model_kwargs['neg_slope'][0])
        get_block = get_block_adapter(
            'LeakyBothAbsBlock3x3Start', q=q, neg_slope=neg_slope
        )
        model = BaseModel(
            in_dim, out_dim, block_getter=get_block, N=N,
        )

    elif model_type == 'LeakyBoth-Abs-ALL-3x3End':
        neg_slope = float(model_kwargs['neg_slope'][0])
        get_block = get_block_adapter(
            'LeakyBothAbsBlock3x3End', q=q, neg_slope=neg_slope
        )
        model = BaseModel(
            in_dim, out_dim, block_getter=get_block, N=N,
        )

    elif model_type == 'LeakyBoth-Abs-ALL-3x3Both':
        neg_slope = float(model_kwargs['neg_slope'][0])
        get_block = get_block_adapter(
            'LeakyBothAbsBlock3x3Both', q=q, neg_slope=neg_slope
        )
        model = BaseModel(
            in_dim, out_dim, block_getter=get_block, N=N,
        )

    elif model_type == 'LeakyBoth-Abs-ALL-PyrStart':
        neg_slope = float(model_kwargs['neg_slope'][0])
        get_block = get_block_adapter(
            'LeakyBothAbsBlockPyrStart', q=q, neg_slope=neg_slope
        )
        model = BaseModel(
            in_dim, out_dim, block_getter=get_block, N=N,
        )
    return model


def get_block_adapter(block_type, **kwargs):
    # for most block the bias is automatically set to false
    if block_type == 'AbsReLUBlock':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return cb.AbsReLUBlock(
                in_dim, out_dim, k=k, stride=stride, q=q
            )
    elif block_type == 'DWSBlock':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return cb.DWSBlock(
                in_dim, out_dim, k=k, stride=stride, q=q
            )

    elif block_type == 'AbsReLULeakyBoth':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return drcb.AbsReLULeakyBoth(
                in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope
            )

    elif block_type == 'AbsBlockPyrBack':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return AbsBlockPyrBack(
                in_dim, out_dim, k=k, stride=stride, q=q
            )

    elif block_type == 'LeakyAbsBlockPyrBack':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return LeakyBothAbsBlockPyrBack(
                in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope
            )

    elif block_type == 'LeakyBothAbsBlock3x3Start':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return LeakyBothAbsBlock3x3Start(
                in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope
            )

    elif block_type == 'LeakyBothAbsBlock3x3End':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return LeakyBothAbsBlock3x3End(
                in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope
            )

    elif block_type == 'LeakyBothAbsBlock3x3Both':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return LeakyBothAbsBlock3x3Both(
                in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope
            )

    elif block_type == 'LeakyBothAbsBlockPyrStart':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return LeakyBothAbsBlockPyrStart(
                in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope
            )

    return get_block


class DoubleBlockBaseModel(nn.Module):
    def __init__(self, in_dim, out_dim, *, block_getter, N):
        super(DoubleBlockBaseModel, self).__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_dim, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # create three stacks, all with the same number and type of blocks
        # each first block of a stack may change the number of features and
        # the stride.

        # ---------------- Stack 1 ------------------------------------------
        self.stack1 = nn.Sequential()
        for i in range(N):
            self.stack1.add_module(
                f'block_{i}_i',
                block_getter(16, 16, k=3, stride=1, padding=1, bias=False),
            )
            self.stack1.add_module(
                f'block_{i}_ii',
                block_getter(16, 16, k=3, stride=1, padding=1, bias=False),
            )

        # ---------------- Stack 2 ------------------------------------------
        self.stack2 = nn.Sequential()
        self.stack2.add_module(
            f'block_{0}_i',
            block_getter(16, 32, k=3, stride=2, padding=1, bias=False),
        )
        self.stack2.add_module(
            f'block_{0}_ii',
            block_getter(32, 32, k=3, stride=1, padding=1, bias=False),
        )
        for i in range(1, N):
            self.stack2.add_module(
                f'block_{i}_i',
                block_getter(32, 32, k=3, stride=1, padding=1, bias=False),
            )
            self.stack2.add_module(
                f'block_{i}_ii',
                block_getter(32, 32, k=3, stride=1, padding=1, bias=False),
            )

        # ---------------- Stack 3 ------------------------------------------
        self.stack3 = nn.Sequential()
        self.stack3.add_module(
            f'block_{0}_i',
            block_getter(32, 64, k=3, stride=2, padding=1, bias=False),
        )
        self.stack3.add_module(
            f'block_{0}_ii',
            block_getter(64, 64, k=3, stride=1, padding=1, bias=False),
        )
        for i in range(1, N):
            self.stack3.add_module(
                f'block_{i}_i',
                block_getter(64, 64, k=3, stride=1, padding=1, bias=False),
            )
            self.stack3.add_module(
                f'block_{i}_ii',
                block_getter(64, 64, k=3, stride=1, padding=1, bias=False),
            )

        self.out = nn.Linear(64, out_dim, bias=True)

    def forward(self, x):
        x = self.first_conv(x)

        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        # global average pooling
        z = x.mean([2, 3])
        y = self.out(z)
        return y


class IExtendedFPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, *, k, stride, q, fp_relu_type, neg_slope, lincomb_3x3='none', added_modules='pyr-back'):
        super(IExtendedFPBlock, self).__init__()

        assert fp_relu_type in ['leaky-relu', 'relu']

        # start with fp-block
        feat_dim = int(q * out_dim)

        if lincomb_3x3 == 'none':
            k_upper = 1
            k_lower = 1
        elif lincomb_3x3 == 'first':
            k_upper = 3
            k_lower = 1
        elif lincomb_3x3 == 'last':
            k_upper = 1
            k_lower = 3
        elif lincomb_3x3 == 'both':
            k_upper = 3
            k_lower = 3
        else:
            raise ValueError(f'Unknown lincomb_3x3: {lincomb_3x3}')

        if fp_relu_type == 'leaky-relu':
            self.upper = LeakyReLUMix(
                in_dim, feat_dim, neg_slope=neg_slope, k=k_upper
            )
            self.lower = LeakyReLUMix(
                feat_dim, out_dim, neg_slope=neg_slope, k=k_lower
            )

        else:
            self.upper = NormalReLUMix(in_dim, feat_dim, k=k_upper)
            self.lower = NormalReLUMix(feat_dim, out_dim, k=k_lower)

        self.left_dw = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=k, stride=stride,
                      padding=k // 2, bias=False, groups=feat_dim),
            nn.InstanceNorm2d(feat_dim),
            nn.ReLU()
        )
        self.right_dw = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=k, stride=stride,
                      padding=k // 2, bias=False, groups=feat_dim),
            nn.InstanceNorm2d(feat_dim),
            nn.ReLU()
        )

        self.mult = None

        # PyramidBlock-> 1st: [BN, 3x3, BN, ReLU,] 2nd: [3x3, BN]
        # Add Pyrblock 2nd part

        if added_modules == 'pyr-front':
            # strided conv is done by DWS
            self.first_part = PyramidFront(in_dim, in_dim)
            self.second_part = nn.Identity()
        elif added_modules == 'pyr-back':
            self.first_part = nn.Identity()
            self.second_part = PyramidBack(out_dim, out_dim)
        elif added_modules == 'none':
            self.first_part = nn.Identity()
            self.second_part = nn.Identity()
        else:
            raise ValueError(f'Unknown added_modules: {added_modules}')

    def forward(self, x):
        x = self.first_part(x)

        # --------- FP-computation ----------------------
        x = self.upper(x)
        x_left = self.left_dw(x)
        x_right = self.right_dw(x)

        x = self.mult(x_left, x_right)

        x = self.lower(x)
        # -----------------------------------------------

        x = self.second_part(x)

        return x


class AbsBlockPyrBack(nn.Module):
    def __init__(self, in_dim, out_dim, *, k, stride, q):
        super(AbsBlockPyrBack, self).__init__()

        block = IExtendedFPBlock(
            in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=0.,
            fp_relu_type='relu'
        )
        block.mult = cb.AbsBabylon()

        self.block_with_shortcut = cb.ResidualAdapter(
            block, stride=stride, relu_after_shortcut=False
        )

    def forward(self, x):
        return self.block_with_shortcut(x)


class LeakyBothAbsBlockPyrBack(nn.Module):
    def __init__(self, in_dim, out_dim, *, k, stride, q, neg_slope):
        super(LeakyBothAbsBlockPyrBack, self).__init__()

        block = IExtendedFPBlock(
            in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope,
            fp_relu_type='leaky-relu'
        )
        block.mult = cb.AbsBabylon()

        self.block_with_shortcut = cb.ResidualAdapter(
            block, stride=stride, relu_after_shortcut=False
        )

    def forward(self, x):
        return self.block_with_shortcut(x)


class LeakyBothAbsBlock3x3Start(nn.Module):
    def __init__(self, in_dim, out_dim, *, k, stride, q, neg_slope):
        super(LeakyBothAbsBlock3x3Start, self).__init__()

        block = IExtendedFPBlock(
            in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope,
            fp_relu_type='leaky-relu', lincomb_3x3='first',
            added_modules='none'
        )
        block.mult = cb.AbsBabylon()

        self.block_with_shortcut = cb.ResidualAdapter(
            block, stride=stride, relu_after_shortcut=False
        )

    def forward(self, x):
        return self.block_with_shortcut(x)


class LeakyBothAbsBlock3x3End(nn.Module):
    def __init__(self, in_dim, out_dim, *, k, stride, q, neg_slope):
        super(LeakyBothAbsBlock3x3End, self).__init__()

        block = IExtendedFPBlock(
            in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope,
            fp_relu_type='leaky-relu', lincomb_3x3='last',
            added_modules='none'
        )
        block.mult = cb.AbsBabylon()

        self.block_with_shortcut = cb.ResidualAdapter(
            block, stride=stride, relu_after_shortcut=False
        )

    def forward(self, x):
        return self.block_with_shortcut(x)


class LeakyBothAbsBlock3x3Both(nn.Module):
    def __init__(self, in_dim, out_dim, *, k, stride, q, neg_slope):
        super(LeakyBothAbsBlock3x3Both, self).__init__()

        block = IExtendedFPBlock(
            in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope,
            fp_relu_type='leaky-relu', lincomb_3x3='both',
            added_modules='none'
        )
        block.mult = cb.AbsBabylon()

        self.block_with_shortcut = cb.ResidualAdapter(
            block, stride=stride, relu_after_shortcut=False
        )

    def forward(self, x):
        return self.block_with_shortcut(x)


class LeakyBothAbsBlockPyrStart(nn.Module):
    def __init__(self, in_dim, out_dim, *, k, stride, q, neg_slope):
        super(LeakyBothAbsBlockPyrStart, self).__init__()

        block = IExtendedFPBlock(
            in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope,
            fp_relu_type='leaky-relu', added_modules='pyr-front'
        )
        block.mult = cb.AbsBabylon()

        self.block_with_shortcut = cb.ResidualAdapter(
            block, stride=stride, relu_after_shortcut=False
        )

    def forward(self, x):
        return self.block_with_shortcut(x)


class NormalReLUMix(nn.Module):
    def __init__(self, in_dim, feat_dim, k=1):
        super(NormalReLUMix, self).__init__()
        mix = nn.Conv2d(
            in_dim, feat_dim, kernel_size=k,
            stride=1, padding=k // 2, bias=False,
        )

        self.f = nn.Sequential(
            mix,
            nn.BatchNorm2d(feat_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.f(x)


class PyramidFront(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PyramidFront, self).__init__()
        self.f = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.f(x)


class PyramidBack(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PyramidBack, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, x):
        return self.f(x)
