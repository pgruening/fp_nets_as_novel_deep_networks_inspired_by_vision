from . import cifar_base_model as base
from . import dropout_blocks as blocks


def get_model(model_type, in_dim, out_dim, device, **model_kwargs):
    dropout = float(model_kwargs['dropout'][0])
    assert dropout > 0 and dropout < 1.

    N = int(model_kwargs['N'][0])
    assert N > 0 and N < 20

    q = model_kwargs['q'][0]
    q = float(q)

    if model_type == 'CifarJOVFPNetDO':
        default_block = get_block_adapter('PyramidBasicBlock')
        start_block = get_block_adapter(
            'FPBlockJOVDO',
            q=q, dropout=dropout
        )
        model = base.JOVFPAtStackStart(
            in_dim, out_dim, default_block=default_block, start_block=start_block, N=N
        )

    elif model_type == 'LeakyBoth-CifarJOVFPNetDO':
        neg_slope = float(model_kwargs['neg_slope'][0])

        default_block = get_block_adapter('PyramidBasicBlock')
        start_block = get_block_adapter(
            'FPBlockJOVLeakyBothDO',
            q=q, neg_slope=neg_slope, dropout=dropout
        )

        model = base.JOVFPAtStackStart(
            in_dim, out_dim, default_block=default_block, start_block=start_block, N=N
        )

    else:
        raise ValueError(f'Unknown model-type: {model_type}')

    return model


def get_block_adapter(block_type, **kwargs):
    # for most blocks the bias is automatically set to false
    if block_type == 'BasicBlock':
        return base.get_block_adapter(block_type, **kwargs)

    elif block_type == 'PyramidBasicBlock':
        return base.get_block_adapter(block_type, **kwargs)

    elif block_type == 'FPBlockJOVLeakyBothDO':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        dropout = kwargs["dropout"]
        assert isinstance(dropout, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return blocks.FPBlockJOVLeakyBothDO(
                in_dim, out_dim, k=k,
                stride=stride, q=q, neg_slope=neg_slope,
                dropout=dropout
            )

        return get_block

    elif block_type == 'FPBlockJOVDO':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        dropout = kwargs["dropout"]
        assert isinstance(dropout, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return blocks.FPBlockJOVDO(
                in_dim, out_dim, k=k,
                stride=stride, q=q,
                dropout=dropout
            )

        return get_block
