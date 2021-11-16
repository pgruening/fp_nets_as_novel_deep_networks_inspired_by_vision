import torch.nn as nn
import torch.nn.functional as F

from models import conv_blocks, mobile_blocks


def get_model(model_type, in_dim, out_dim, device, **model_kwargs):
    # ----------V1Models----------
    if model_type == 'CifarMobileNetV1':
        get_block = get_block_adapter('MobileV1Block')
        model = MobileNetV1(in_dim, out_dim, block_getter=get_block)

    elif model_type == 'CifarMobileJOVFP-V1':
        q = model_kwargs['q'][0]
        q = float(q)
        k = int(model_kwargs['k'][0])

        get_block = get_block_adapter('FPBlockJOV', q=q, k=k)
        model = MobileNetV1(in_dim, out_dim, block_getter=get_block)

    elif model_type == 'CifarMobileAbsReLU-V1':
        q = model_kwargs['q'][0]
        q = float(q)
        k = int(model_kwargs['k'][0])

        get_block = get_block_adapter('AbsReLUBlock', q=q, k=k)
        model = MobileNetV1(in_dim, out_dim, block_getter=get_block)

    # ----------V2Models----------

    elif model_type == 'CifarMobileNetV2':
        get_block = get_block_adapter('MobileV2Block')
        model = MobileNetV2(in_dim, out_dim, block_getter=get_block)

    elif model_type == 'CifarMobileNetV2Abs':
        get_block = get_block_adapter('MobileV2BlockWithAbs')
        model = MobileNetV2(in_dim, out_dim, block_getter=get_block)

    elif model_type == 'CifarMobileNetV2AbsReLU':
        get_block = get_block_adapter('AbsReLUMobileV2Block')
        model = MobileNetV2(in_dim, out_dim, block_getter=get_block)

    elif model_type == 'CifarMobileJOVFP-V2':
        # raise NotImplementedError(f'model {model_type} is not yet working')
        q = model_kwargs['q'][0]
        q = float(q)
        k = int(model_kwargs['k'][0])

        get_block = get_block_adapter('FPBlockJOV-V2', q=q, k=k)
        model = MobileNetV2(in_dim, out_dim, block_getter=get_block)

    elif model_type == 'CifarMobileAbsReLU-V2':
        q = model_kwargs['q'][0]
        q = float(q)
        k = int(model_kwargs['k'][0])

        get_block = get_block_adapter('AbsReLUBlock-V2', q=q, k=k)
        model = MobileNetV2(in_dim, out_dim, block_getter=get_block)

    # ----------LeakyModels----------

    elif model_type == 'CifarMobileAbsReLU-V2-LeakyBoth':
        q = model_kwargs['q'][0]
        q = float(q)
        k = int(model_kwargs['k'][0])

        neg_slope = model_kwargs["neg_slope"][0]
        neg_slope = float(neg_slope)

        get_block = get_block_adapter(
            'AbsReLULeakyBoth', k=k, q=q, neg_slope=neg_slope)
        model = MobileNetV2(in_dim, out_dim, block_getter=get_block)

    elif model_type == 'CifarMobileJOVFP-V2-LeakyBoth':
        q = model_kwargs['q'][0]
        q = float(q)
        k = int(model_kwargs['k'][0])

        neg_slope = model_kwargs["neg_slope"][0]
        neg_slope = float(neg_slope)

        get_block = get_block_adapter(
            'FPBlockJOVLeakyBoth', k=k, q=q, neg_slope=neg_slope)
        model = MobileNetV2(in_dim, out_dim, block_getter=get_block)

    # ----------LinLowerModels----------

    elif model_type == 'CifarMobileFPLinLower':
        q = model_kwargs['q'][0]
        q = float(q)
        k = int(model_kwargs['k'][0])

        get_block = get_block_adapter('FPBlockLinLower', q=q, k=k)
        model = MobileNetV2(in_dim, out_dim, block_getter=get_block)

    elif model_type == 'CifarMobileAbsReLULinLower':
        q = model_kwargs['q'][0]
        q = float(q)
        k = int(model_kwargs['k'][0])

        get_block = get_block_adapter('AbsReLUBlockLinLower', q=q, k=k)
        model = MobileNetV2(in_dim, out_dim, block_getter=get_block)

    else:
        raise ValueError(f'Unknown model_type: {model_type}')

    return model.to(device).eval()


def get_block_adapter(block_type, **kwargs):

    if block_type == 'MobileV1Block':
        def get_block(in_dim, out_dim, *, stride, expansion):
            return mobile_blocks.MobileV1Block(
                in_dim, out_dim, stride=stride
            )

    elif block_type == 'FPBlockJOV':
        q = kwargs["q"]
        k = kwargs['k']
        assert isinstance(q, (float, int))

        def get_block(in_dim, out_dim, *, stride, expansion):
            return conv_blocks.FPBlockJOV(
                in_dim, out_dim, k=k, stride=stride, q=q
            )

    elif block_type == 'AbsReLUBlock':
        q = kwargs["q"]
        k = kwargs['k']
        assert isinstance(q, (float, int))

        def get_block(in_dim, out_dim, *, stride, expansion):
            return conv_blocks.AbsReLUBlock(
                in_dim, out_dim, k=k, stride=stride, q=q
            )

    elif block_type == 'MobileV2Block':
        def get_block(in_dim, out_dim, *, stride, expansion, use_1x1):
            return mobile_blocks.MobileV2Block(
                in_dim, out_dim, expansion, stride
            )

    elif block_type == 'MobileV2BlockWithAbs':
        def get_block(in_dim, out_dim, *, stride, expansion, use_1x1):
            return mobile_blocks.MobileV2BlockWithAbs(
                in_dim, out_dim, expansion, stride
            )

    elif block_type == 'AbsReLUMobileV2Block':
        def get_block(in_dim, out_dim, *, stride, expansion, use_1x1):
            return mobile_blocks.AbsReLUMobileV2Block(
                in_dim, out_dim, expansion, stride
            )

    elif block_type == 'FPBlockJOV-V2':
        q = kwargs["q"]
        k = kwargs['k']
        assert isinstance(q, (float, int))

        def get_block(in_dim, out_dim, *, stride, expansion, use_1x1):
            return mobile_blocks.FPBLOCKJOVForV2(
                in_dim, out_dim, k=k, stride=stride, q=q, use_1x1=use_1x1
            )

    elif block_type == 'AbsReLUBlock-V2':
        q = kwargs["q"]
        k = kwargs['k']
        assert isinstance(q, (float, int))

        def get_block(in_dim, out_dim, *, stride, expansion, use_1x1):
            return mobile_blocks.AbsReLUBlockForV2(
                in_dim, out_dim, k=k, stride=stride, q=q, use_1x1=use_1x1
            )

    elif block_type == 'FPBlockJOVLeakyBoth':
        q = kwargs["q"]
        k = kwargs['k']
        assert isinstance(q, (float, int))

        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, stride, expansion, use_1x1):
            return mobile_blocks.FPBlockJOVLeakyBothForV2(
                in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope, use_1x1=use_1x1
            )

    elif block_type == 'AbsReLULeakyBoth':
        q = kwargs["q"]
        k = kwargs['k']
        assert isinstance(q, (float, int))

        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, stride, expansion, use_1x1):
            return mobile_blocks.AbsReLULeakyBothForV2(
                in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope, use_1x1=use_1x1
            )

    elif block_type == 'FPBlockLinLower':
        q = kwargs['q']
        k = kwargs['k']
        assert isinstance(q, (float, int))

        def get_block(in_dim, out_dim, *, stride, expansion, use_1x1):
            return mobile_blocks.FPBlockLinLower(
                in_dim, out_dim, k=k, stride=stride, q=q, use_1x1=use_1x1
            )

    elif block_type == 'AbsReLUBlockLinLower':
        q = kwargs['q']
        k = kwargs['k']
        assert isinstance(q, (float, int))

        def get_block(in_dim, out_dim, *, stride, expansion, use_1x1):
            return mobile_blocks.AbsReLUBlockLinLower(
                in_dim, out_dim, k=k, stride=stride, q=q, use_1x1=use_1x1
            )

    else:
        raise(ValueError(f'Unknown block: {block_type}'))

    return get_block


class MobileNetV2(nn.Module):
    # (expansion, out_dim, num_blocks, stride)
    cfg = [(1, 16, 1, 1),
           (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6, 32, 3, 2),
           (6, 64, 4, 2),
           (6, 96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, in_dim, out_dim, *, block_getter):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_dim=32, block_getter=block_getter)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, out_dim)

    def _make_layers(self, in_dim, block_getter):
        layers = []
        for expansion, out_dim, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                use_1x1 = out_dim < in_dim
                layers.append(block_getter(
                    in_dim, out_dim, stride=stride, expansion=expansion, use_1x1=use_1x1
                ))
                in_dim = out_dim
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class MobileNetV1(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2),
           512, 512, 512, 512, 512, (1024, 2), 1024]

    def __init__(self, in_dim, out_dim, *, block_getter):
        super(MobileNetV1, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(
            in_planes=32, block_getter=block_getter)
        self.linear = nn.Linear(1024, out_dim)

    def _make_layers(self, in_planes, block_getter):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(
                block_getter(in_planes, out_planes, stride=stride, expansion=None))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
