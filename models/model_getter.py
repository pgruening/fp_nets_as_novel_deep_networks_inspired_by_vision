from . import (cifar_base_model, double_block_models, dropout_models,
               exp4_models, leaky_models, modify_kernel_models, cifar_mobilenet_model, pre_init_absReLU_mobilenet)
from .imagenet_legacy_models import my_resnet, mobilenet
from torchvision.models import mobilenet_v2

EXP0_MODELS = ['CifarResNet', 'CifarPyrResNet', 'CifarJOVFPNet']

EXP1_MODELS = [
    'CifarAbsReLU-LS', 'CifarAbsReLU-LS-DWS', 'CifarAbsReLU-ALL',
    'CifarDWS-ALL', 'CifarINet'
]

EXP2_MODELS = ['LeakyBoth-CifarJOVFPNet']
EXP2_MODELS += ['LeakyUpper-' + x for x in EXP1_MODELS]
EXP2_MODELS += ['LeakyBoth-' + x for x in EXP1_MODELS]
EXP2_MODELS += ['AssymUpper-' + x for x in EXP1_MODELS]
EXP2_MODELS += ['AssymBoth-' + x for x in EXP1_MODELS]

EXP2_2_MODELS = ['CifarJOVFPNetDO', 'LeakyBoth-CifarJOVFPNetDO']

EXP_3_MODELS = ['AbsReLU-ALL-DB', 'LeakyBoth-AbsReLU-ALL-DB', 'DWS-ALL-DB']
EXP_3_3_MODELS = ['LeakyBoth-Abs-ALL-PyrBack', 'Abs-ALL-PyrBack']
EXP_3_4_MODELS = [
    'LeakyBoth-Abs-ALL-3x3Start', 'LeakyBoth-Abs-ALL-3x3End',
    'LeakyBoth-Abs-ALL-3x3Both', 'LeakyBoth-Abs-ALL-PyrStart'
]

EXP3_1_MODELS = ['LeakyBoth-CifarAbsReLU-ALL-VarK',
                 'LeakyBoth-CifarDWS-ALL-VarK']

EXP4_MODELS = ['CifarJOVFPNet-RNBasic', 'CifarAbsReLU-LS-realAbs',
               'CifarAbsReLU-LS-NoNorm', 'CifarJOVFPNet-NoNorm']

EXP7_MODELS = ['CifarMobileNetV1', 'CifarMobileNetV2',
               'CifarMobileJOVFP-V1', 'CifarMobileJOVFP-V2',
               'CifarMobileAbsReLU-V1', 'CifarMobileAbsReLU-V2',
               'CifarMobileJOVFP-V2-LeakyBoth', 'CifarMobileAbsReLU-V2-LeakyBoth',
               'CifarMobileNetV2Abs'
               ]


EXP7_1_MODELS = ['CifarMobileFPLinLower', 'CifarMobileAbsReLULinLower']
EXP_7_5_MODELS = ['PreInitMobileNetAbsReLUV2']

TEST_MODELS = ['AllZero']


def get_model(model_type, input_dim, output_dim, device, **kwargs):
    if model_type in EXP0_MODELS:
        # exp0
        model = cifar_base_model.get_model(
            model_type, input_dim, output_dim, device, **kwargs
        )

    elif model_type in EXP1_MODELS:
        # exp1
        model = cifar_base_model.get_model(
            model_type, input_dim, output_dim, device, **kwargs
        )

    elif model_type in EXP2_MODELS:
        # exp2
        mix_type = model_type.split('-')[0]
        model_type = model_type.split(mix_type + '-')[1]

        model = leaky_models.get_model(
            model_type, input_dim, output_dim, device, mix_type=mix_type, **kwargs
        )

    elif model_type in EXP2_2_MODELS:
        model = dropout_models.get_model(
            model_type, input_dim, output_dim, device, **kwargs
        )

    elif model_type in EXP_3_MODELS:
        model = double_block_models.get_model(
            model_type, input_dim, output_dim, device, **kwargs
        )
    elif model_type in EXP3_1_MODELS:
        mix_type = model_type.split('-')[0]
        model_type = model_type.split(mix_type + '-')[1]

        model = modify_kernel_models.get_model(
            model_type, input_dim, output_dim, device, mix_type=mix_type, **kwargs
        )

    elif model_type in EXP_3_3_MODELS:
        model = double_block_models.get_model(
            model_type, input_dim, output_dim, device, **kwargs
        )

    elif model_type in EXP_3_4_MODELS:
        model = double_block_models.get_model(
            model_type, input_dim, output_dim, device, **kwargs
        )

    elif model_type in EXP4_MODELS:
        model = exp4_models.get_model(
            model_type, input_dim, output_dim, device, **kwargs)

    elif model_type in EXP7_MODELS:
        model = cifar_mobilenet_model.get_model(
            model_type, input_dim, output_dim, device, **kwargs
        )
    elif model_type in EXP7_1_MODELS:
        model = cifar_mobilenet_model.get_model(
            model_type, input_dim, output_dim, device, **kwargs
        )

    elif model_type in EXP_7_5_MODELS:
        model = pre_init_absReLU_mobilenet.get_model(
            model_type, input_dim, output_dim, device, **kwargs
        )

    elif model_type == "fp_resnet_50_layer_start":
        bn_type = kwargs["bn_type"][0]
        add_res = kwargs["add_res"][0]
        c1x1_type = kwargs["c1x1_type"][0]
        first_block = kwargs["first_block"][0]
        q = float(kwargs["q"][0])

        model = my_resnet.FPResNet50LayerStart(
            input_dim, output_dim, bn_type, add_res, c1x1_type, first_block, q
        )

    elif model_type == "mobilenet_layer_start":
        add_res = kwargs["add_res"][0]
        q = float(kwargs["q"][0])

        model = mobilenet.MobileNetLayerStart(
            input_dim, output_dim, q, add_res=add_res
        )

# mobilenet_v2
    elif model_type == "mobilenet_v2":
        pre_trained = kwargs["pre_trained"][0]
        model = mobilenet_v2(pretrained=pre_trained)

    elif 'AllZero: ' in model_type:
        from test_cases import models_for_testing as test_model
        # call this function again to get the actual base model
        base_model_type = model_type.split(': ')[-1]
        model = get_model(
            base_model_type, input_dim, output_dim, device, **kwargs
        )
        model = test_model.set_all_weights_to_zero(model)

    else:
        raise ValueError(f'Unknown modeltype: {model_type}')

    return model.to(device).eval()
