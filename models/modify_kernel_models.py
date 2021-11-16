"""
Essential to reproduce
exp 3_1
    We want to check the performance of two models with 
        a kernel size other than 3
    Therefore the architecture is pretty similar to 
        already used models but we can change some default parameters
"""
import warnings 
import math

import torch.nn as nn
import torch.nn.init as init

from . import dying_relu_blocks as blocks


def get_model(model_type, in_dim, out_dim, device,*, mix_type, **model_kwargs):
    assert mix_type in [
        'LeakyBoth'
    ]    

    neg_slope = model_kwargs['neg_slope'][0]
    neg_slope = float(neg_slope)    
    
    N = int(model_kwargs['N'][0])
    assert N > 0 and N < 20

    k = int(model_kwargs['k'][0])
    assert k > 1 and k < 9

    q = model_kwargs['q'][0]
    q = float(q)

    if model_type == 'CifarAbsReLU-ALL-VarK':

        get_block = get_block_adapter(
            'AbsReLU' + mix_type, q=q, neg_slope=neg_slope)
        model = BaseModel(
            in_dim, out_dim, block_getter=get_block, N=N, k=k, pyr_weight_init=False
        )
    elif model_type == 'CifarDWS-ALL-VarK':
        
        get_block = get_block_adapter(
            'DWS' + mix_type, q=q, neg_slope=neg_slope)
        model = BaseModel(
            in_dim, out_dim, block_getter=get_block, N=N, k=k, pyr_weight_init=False
        )
    else:
        raise ValueError(f'Unknown Model: {model_type}')

    return model.to(device).eval()


def get_block_adapter(block_type, **kwargs):
    if block_type == 'DWSLeakyBoth':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return blocks.DWSLeakyBoth(
                in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope
            )
    elif block_type == 'AbsReLULeakyBoth': 
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return blocks.AbsReLULeakyBoth(
                in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope
            )
    else:
        raise ValueError(f'Unknown block: {block_type}')

    return get_block



class BaseModel(nn.Module):
    def __init__(self, in_dim, out_dim, *, block_getter, N, k, init_normal=False, pyr_weight_init=False):
        super(BaseModel, self).__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_dim, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # create three stacks, all with the same number and type of blocks
        # each first block of a stack may change the number of features and
        # the stride.

        self.stack1 = nn.Sequential()
        for i in range(N):
            self.stack1.add_module(
                f'block_{i}',
                block_getter(16, 16, k=k, stride=1, padding=1, bias=False),
            )

        self.stack2 = nn.Sequential()
        self.stack2.add_module(
            f'block_{0}',
            block_getter(16, 32, k=k, stride=2, padding=1, bias=False),
        )
        for i in range(1, N):
            self.stack2.add_module(
                f'block_{i}',
                block_getter(32, 32, k=k, stride=1, padding=1, bias=False),
            )

        self.stack3 = nn.Sequential()
        self.stack3.add_module(
            f'block_{0}',
            block_getter(32, 64, k=k, stride=2, padding=1, bias=False),
        )
        for i in range(1, N):
            self.stack3.add_module(
                f'block_{i}',
                block_getter(64, 64, k=k, stride=1, padding=1, bias=False),
            )

        self.out = nn.Linear(64, out_dim, bias=True)

        if init_normal:
            warnings.warn('Overwriting weight init with Kaminig-Normal')
            self.apply(_weights_init)

        if pyr_weight_init:
            warnings.warn(
                'Overwriting weight init with Kaminig-Normal Fan-out(PyramidNet)'
            )
            _pyr_weight_init(self)

    def forward(self, x):
        x = self.first_conv(x)

        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        # global average pooling
        z = x.mean([2, 3])
        y = self.out(z)
        return y





def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


def _pyr_weight_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
