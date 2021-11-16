from os.path import join

import torch
import torch.nn as nn
import torch.nn.functional as F
from DLBio.helpers import dict_to_options, load_json
from DLBio.kwargs_translator import get_kwargs
from DLBio.pytorch_helpers import load_model_with_opt, walk

from models.cifar_mobilenet_model import get_model as get_mobile_model
from models.mobile_blocks import AbsReLUMobileV2Block, MobileV2Block

MODEL_PATH = 'experiments/exp_7_1_1/exp_data/CifarMobileNetV2_s9'


def get_model(dummy, in_dim, out_dim, device, **model_kwargs):
    max_angle = float(model_kwargs['max_angle'][0])
    run_tests = model_kwargs.get('run_tests', [False])[0]
    model_path = model_kwargs.get('model_path', [MODEL_PATH])[0]

    options = load_json(join(model_path, 'opt.json'))
    original_model = load_model(
        options, device, new_model_path=join(model_path, 'model.pt')
    )

    original_in_dim = original_model.conv1.in_channels
    original_out_dim = original_model.linear.out_features
    assert original_in_dim == in_dim

    new_model = get_mobile_model(
        'CifarMobileNetV2AbsReLU',
        in_dim, original_out_dim, device
    )

    list_o = [
        x for x in walk(original_model) if isinstance(x, MobileV2Block)
    ]
    list_new = [
        x for x in walk(new_model) if isinstance(x, AbsReLUMobileV2Block)
    ]

    assert len(list_o) == len(list_new)

    for mod_o, mod_new in zip(list_o, list_new):

        mod_new.abs_relu_block.change_weights(
            mod_o.conv2, mod_o.bn2, max_angle=max_angle
        )
        new_model.eval().to(device)

        # all other weights are equal
        mod_new.conv1 = mod_o.conv1
        mod_new.conv3 = mod_o.conv3
        mod_new.bn1 = mod_o.bn1
        mod_new.bn3 = mod_o.bn3
        mod_new.shortcut = mod_o.shortcut

        if run_tests:
            test = torch.rand(1, mod_o.conv2.in_channels, 32, 32).to(device)
            x = F.relu(mod_o.bn2(mod_o.conv2(test)))
            y = mod_new.abs_relu_block(test)
            delta = torch.abs(y - x).mean()
            print(delta)

            test = torch.rand(1, mod_o.conv1.in_channels, 32, 32).to(device)
            x = mod_o(test)
            y = mod_new(test)
            delta = torch.abs(y - x).mean()
            print(delta)

            print('---')

    # equal operations for all other modules
    new_model.conv1 = original_model.conv1
    new_model.bn1 = original_model.bn1
    new_model.conv2 = original_model.conv2
    new_model.bn2 = original_model.bn2

    if original_out_dim != out_dim:
        new_model.linear = nn.Linear(
            original_model.linear.in_features,
            out_dim
        )
    else:
        new_model.linear = original_model.linear

    del original_model
    torch.cuda.empty_cache()

    return new_model.eval().to(device)


def load_model(options, device, strict=True, new_model_path=None, map_location=None, from_par_gpu=False):
    # copy from helpers to avoid circular import
    from models.model_getter import get_model

    if isinstance(options, dict):
        options = dict_to_options(options)

    def get_model_fcn(options, device):
        model_kwargs = get_kwargs(options.model_kw)
        return get_model(
            options.model_type,
            options.in_dim,
            options.out_dim,
            device,
            **model_kwargs
        )

    if new_model_path is not None:
        model_path = new_model_path
    else:
        model_path = options.model_path

    return load_model_with_opt(
        model_path,
        options,
        get_model_fcn,
        device,
        strict=strict,
        map_location=map_location,
        from_par_gpu=from_par_gpu
    )
