import torch.nn as nn
from DLBio.pytorch_helpers import get_num_params
from torchvision.models import resnet

from . import block_architectures as ba

Q_VALUE = .5


def FPResNet50(in_dim, out_dim, bn_type, add_res, c1x1_type, first_block, q=Q_VALUE):
    model = resnet.resnet50(pretrained=False)
    model.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7,
                            stride=2, padding=3, bias=False)

    model.layer1[0] = set_first_block(
        first_block,
        64, 256, 3, 1, bn_type, c1x1_type, add_res, q
    )
    model.layer1[-1] = set_fp_block(
        256, 256, 3, 1, bn_type, c1x1_type, add_res, q
    )

    model.layer2[0] = set_first_block(
        first_block,
        256, 512, 3, 2, bn_type, c1x1_type, add_res, q
    )
    model.layer2[-1] = set_fp_block(
        512, 512, 3, 1, bn_type, c1x1_type, add_res, q
    )

    model.layer3[0] = set_first_block(
        first_block,
        512, 1024, 3, 2, bn_type, c1x1_type, add_res, q
    )
    model.layer3[-1] = set_fp_block(
        1024, 1024, 3, 1, bn_type, c1x1_type, add_res, q
    )

    model.layer4[0] = set_first_block(
        first_block,
        1024, 2048, 3, 2, bn_type, c1x1_type, add_res, q
    )
    model.layer4[-1] = set_fp_block(
        2048, 2048, 3, 1, bn_type, c1x1_type, add_res, q
    )

    model.fc = nn.Linear(2048, out_dim)

    return model


def FPResNet50FixedQ(in_dim, out_dim, bn_type, add_res, c1x1_type, first_block):
    model = resnet.resnet50(pretrained=False)
    print('Baseline:')
    print(get_num_params(model))
    print('---')

    model.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7,
                            stride=2, padding=3, bias=False)

    q_0 = .9
    q_1 = .5

    p_before = get_num_params(model.layer1[0])
    model.layer1[0] = set_first_block(
        first_block,
        64, 256, 3, 1, bn_type, c1x1_type, add_res, q_0
    )
    p_after = get_num_params(model.layer1[0])
    print(f'Layer1/0: {p_before} -> {p_after}')
    print(f'---')

    p_before = get_num_params(model.layer1[-1])
    model.layer1[-1] = set_fp_block(
        256, 256, 3, 1, bn_type, c1x1_type, add_res, q_1
    )
    p_after = get_num_params(model.layer1[-1])
    print(f'Layer1/-1: {p_before} -> {p_after}')
    print(f'---')

    p_before = get_num_params(model.layer2[0])
    model.layer2[0] = set_first_block(
        first_block,
        256, 512, 3, 2, bn_type, c1x1_type, add_res, q_0
    )
    p_after = get_num_params(model.layer2[0])
    print(f'Layer2/0: {p_before} -> {p_after}')
    print(f'---')

    p_before = get_num_params(model.layer2[-1])
    model.layer2[-1] = set_fp_block(
        512, 512, 3, 1, bn_type, c1x1_type, add_res, q_1
    )
    p_after = get_num_params(model.layer2[-1])
    print(f'Layer2/-1: {p_before} -> {p_after}')
    print(f'---')

    p_before = get_num_params(model.layer3[0])
    model.layer3[0] = set_first_block(
        first_block,
        512, 1024, 3, 2, bn_type, c1x1_type, add_res, q_0
    )
    p_after = get_num_params(model.layer3[0])
    print(f'Layer3/0: {p_before} -> {p_after}')
    print(f'---')

    p_before = get_num_params(model.layer3[-1])
    model.layer3[-1] = set_fp_block(
        1024, 1024, 3, 1, bn_type, c1x1_type, add_res, q_1
    )
    p_after = get_num_params(model.layer3[-1])
    print(f'Layer3/-1: {p_before} -> {p_after}')
    print(f'---')

    p_before = get_num_params(model.layer4[0])
    model.layer4[0] = set_first_block(
        first_block,
        1024, 2048, 3, 2, bn_type, c1x1_type, add_res, q_0
    )
    p_after = get_num_params(model.layer4[0])
    print(f'Layer4/0: {p_before} -> {p_after}')
    print(f'---')

    p_before = get_num_params(model.layer4[-1])
    model.layer4[-1] = set_fp_block(
        2048, 2048, 3, 1, bn_type, c1x1_type, add_res, q_1
    )
    p_after = get_num_params(model.layer4[-1])
    print(f'Layer4/-1: {p_before} -> {p_after}')
    print(f'---')

    model.fc = nn.Linear(2048, out_dim)

    return model


def FPResNet50LayerStart(in_dim, out_dim, bn_type, add_res, c1x1_type, first_block, q, k=3):
    model = resnet.resnet50(pretrained=False)

    # change input and output layer according to new dimensions
    model.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7,
                            stride=2, padding=3, bias=False)
    model.fc = nn.Linear(2048, out_dim)
    print('Baseline:')
    print(get_num_params(model))
    print('---')

    if k != 3:
        print(f'Changed FP-block kernel size from 3 to {k}')

    p_before = get_num_params(model.layer1[0])
    model.layer1[0] = set_first_block(
        first_block,
        64, 256, k, 1, bn_type, c1x1_type, add_res, q
    )
    p_after = get_num_params(model.layer1[0])
    print(f'Layer1/0: {p_before} -> {p_after}, {p_after-p_before}')
    print(f'---')

    p_before = get_num_params(model.layer2[0])
    model.layer2[0] = set_first_block(
        first_block,
        256, 512, k, 2, bn_type, c1x1_type, add_res, q
    )
    p_after = get_num_params(model.layer2[0])
    print(f'Layer2/0: {p_before} -> {p_after}, {p_after-p_before}')

    print(f'---')

    p_before = get_num_params(model.layer3[0])
    model.layer3[0] = set_first_block(
        first_block,
        512, 1024, k, 2, bn_type, c1x1_type, add_res, q
    )
    p_after = get_num_params(model.layer3[0])
    print(f'Layer3/0: {p_before} -> {p_after}, {p_after-p_before}')
    print(f'---')

    p_before = get_num_params(model.layer4[0])
    model.layer4[0] = set_first_block(
        first_block,
        1024, 2048, k, 2, bn_type, c1x1_type, add_res, q
    )
    p_after = get_num_params(model.layer4[0])
    print(f'Layer4/0: {p_before} -> {p_after}, {p_after-p_before}')
    print(f'---')

    print('FP-model:')
    print(get_num_params(model))
    print('---')

    return model


def set_first_block(type_, in_dim, out_dim, kernel_size, stride, bn_type, c1x1_type, add_res, q):
    if type_ == 'relu':
        return set_fp_relu(in_dim, out_dim, kernel_size,
                           stride, bn_type, c1x1_type, add_res, q)
    elif type_ == 'fp':
        return set_fp_block(in_dim, out_dim, kernel_size,
                            stride, bn_type, c1x1_type, add_res, q)
    else:
        raise ValueError(f'unknown type: {type_}')


def set_fp_relu(in_dim, out_dim, kernel_size, stride, bn_type, c1x1_type, add_res, q):
    out = ba.FPReLUBlock(
        d_in=in_dim, d_out=out_dim,
        k=kernel_size, stride=stride,
        bn_type=bn_type,
        c1x1_type=c1x1_type,
        q=q
    )

    if add_res:
        out = ba.ResidualAdapter(out)

    return out


def set_fp_block(in_dim, out_dim, kernel_size, stride, bn_type, c1x1_type, add_res, q):
    out = ba.FPBlock(
        d_in=in_dim, d_out=out_dim,
        k=kernel_size, stride=stride,
        bn_type=bn_type,
        c1x1_type=c1x1_type,
        q=q
    )

    if add_res:
        out = ba.ResidualAdapter(out)

    return out
