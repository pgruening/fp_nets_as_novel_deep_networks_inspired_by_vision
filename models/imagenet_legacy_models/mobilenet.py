import torch.nn as nn
from DLBio.pytorch_helpers import get_num_params
from torchvision.models import mobilenet
from torchvision.models.mobilenetv2 import InvertedResidual
from . import block_architectures as ba

NORMALIZATION = 'instance'
CONV_1X1_TYPE = 'convbnrelu'


def MobileNetLayerStart(in_dim, out_dim, q, add_res=True):
    # load original model
    model = mobilenet.mobilenet_v2(pretrained=False)
    print(f'Basline model num parameters: {get_num_params(model)}')

    # change input and output dimensions
    # both if cases are not True when training ImageNet
    if in_dim != 3:
        input_channel = 32
        tmp = mobilenet.ConvBNReLU(
            in_dim, input_channel, stride=2
        )
        nn.init.kaiming_normal_(tmp.weight, mode='fan_out')
        model.features[0] = tmp

    if out_dim != 1000:
        last_channel = 1280
        tmp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, out_dim),
        )
        # the next two lines were no tested
        nn.init.normal_(tmp[-1].weight, 0, 0.01)
        nn.init.zeros_(tmp[-1].bias)
        model.classifier = tmp

    # for layer 1-5 change the first block
    # change the first layer
    model.features[1] = set_fp_relu(
        32, 16, 3, 1, NORMALIZATION, CONV_1X1_TYPE, add_res, q
    )

    # change all inverted residual blocks with a stride=2
    for i in range(2, len(model.features)):
        module_ = model.features[i]
        if isinstance(module_, InvertedResidual):
            stride = module_.conv[1][0].stride[0]
            if stride > 1:
                # first 1x1 convolution
                d_in = module_.conv[0][0].in_channels
                # batchnorm output
                d_out = module_.conv[3].num_features

                model.features[i] = set_fp_relu(
                    d_in, d_out, 3, stride,
                    NORMALIZATION, CONV_1X1_TYPE, add_res, q
                )
                print(f'change layer {i}')

    print(f'Updated fp-model num parameters: {get_num_params(model)}')

    return model


def set_fp_relu(in_dim, out_dim, kernel_size, stride, bn_type, c1x1_type, add_res, q):
    out = ba.FPReLUBlock(
        d_in=in_dim, d_out=out_dim,
        k=kernel_size, stride=stride,
        bn_type=bn_type,
        c1x1_type=c1x1_type,
        q=q
    )

    if add_res:
        if in_dim > out_dim:
            out = ba.ResidualAdapter(
                out, use_1x1=True, in_dim=in_dim, out_dim=out_dim
            )
        else:
            out = ba.ResidualAdapter(out)

    return out
