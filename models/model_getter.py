from . import cifar_base_model, exp4_models
from .imagenet_legacy_models import my_resnet, mobilenet
from torchvision.models import mobilenet_v2

EXP0_MODELS = ['CifarResNet', 'CifarPyrResNet', 'CifarJOVFPNet']


EXP4_MODELS = ['CifarJOVFPNet-RNBasic', 'CifarAbsReLU-LS-realAbs',
               'CifarAbsReLU-LS-NoNorm', 'CifarJOVFPNet-NoNorm']

TEST_MODELS = ['AllZero']


def get_model(model_type, input_dim, output_dim, device, **kwargs):
    if model_type in EXP0_MODELS:
        # exp0
        model = cifar_base_model.get_model(
            model_type, input_dim, output_dim, device, **kwargs
        )

    elif model_type in EXP4_MODELS:
        model = exp4_models.get_model(
            model_type, input_dim, output_dim, device, **kwargs)

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
