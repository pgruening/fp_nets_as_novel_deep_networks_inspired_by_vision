''' 
TODO create the models for exp4

See experiments/exp_4/ TODO.md or README.md for details

These Models should test:
* influence of PyramidBLock
* ReluModel w/o normalization
* AbsBabylon instead of ReluBabylon


'''

from . import cifar_base_model 
from . import exp4_blocks


def get_block_adapter(block_type, **kwargs):
    if block_type == 'RealAbsReLUBlock':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return exp4_blocks.RealAbsReLUBlock(
                in_dim, out_dim, k=k, stride=stride, q=q
            )
    elif block_type == 'AbsReLUBlockNoNorm':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return exp4_blocks.AbsReLUBlockNoNorm(
                in_dim, out_dim, k=k, stride=stride, q=q
            )
    elif block_type == 'FPBlockJOVNoNorm':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return exp4_blocks.FPBlockJOVNoNorm(
                in_dim, out_dim, k=k, stride=stride, q=q
            )
    else:
        get_block = cifar_base_model.get_block_adapter(block_type, **kwargs)

    return get_block


def get_model(model_type, in_dim, out_dim, device, **model_kwargs):
    if model_type == 'CifarJOVFPNet-RNBasic':
        # test the impact of a pyramid block
        N = int(model_kwargs['N'][0])
        assert N > 0 and N < 20

        q = model_kwargs['q'][0]
        q = float(q)

        default_block = get_block_adapter('BasicBlock')
        start_block = get_block_adapter('FPBlockJOV', q=q)

        model = cifar_base_model.JOVFPAtStackStart(
            in_dim, out_dim, default_block=default_block, start_block=start_block, N=N
        )
    elif model_type == 'CifarAbsReLU-LS-realAbs':
        # difference between abs and ReLU for Babylon
        N = int(model_kwargs['N'][0])
        assert N > 0 and N < 20

        q = model_kwargs['q'][0]
        q = float(q)

        default_block = get_block_adapter('PyramidBasicBlock')
        start_block = get_block_adapter('RealAbsReLUBlock', q=q)

        model = cifar_base_model.JOVFPAtStackStart(
            in_dim, out_dim, default_block=default_block, start_block=start_block, N=N
        )
    elif model_type == 'CifarAbsReLU-LS-NoNorm':
        #ReLU Model w/o normalization
        N = int(model_kwargs['N'][0])
        assert N > 0 and N < 20

        q = model_kwargs['q'][0]
        q = float(q)

        default_block = get_block_adapter('PyramidBasicBlock')
        start_block = get_block_adapter('AbsReLUBlockNoNorm', q=q)

        model = cifar_base_model.JOVFPAtStackStart(
            in_dim, out_dim, default_block=default_block, start_block=start_block, N=N
        )
    elif model_type == 'CifarJOVFPNet-NoNorm':
        N = int(model_kwargs['N'][0])
        assert N > 0 and N < 20

        q = model_kwargs['q'][0]
        q = float(q)

        default_block = get_block_adapter('PyramidBasicBlock')
        start_block = get_block_adapter('FPBlockJOVNoNorm', q=q)

        model = cifar_base_model.JOVFPAtStackStart(
            in_dim, out_dim, default_block=default_block, start_block=start_block, N=N
        )

    
    else:
        raise ValueError(f'Unknown Model: {model_type}')

    return model.to(device).eval()
