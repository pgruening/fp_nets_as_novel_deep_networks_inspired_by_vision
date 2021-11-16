from . import cifar_base_model as base
from . import dying_relu_blocks as blocks


def get_model(model_type, in_dim, out_dim, device, *, mix_type, **model_kwargs):
    assert mix_type in [
        'LeakyUpper', 'LeakyBoth', 'AssymUpper', 'AssymBoth'
    ]

    neg_slope = model_kwargs['neg_slope'][0]
    neg_slope = float(neg_slope)

    N = int(model_kwargs['N'][0])
    assert N > 0 and N < 20

    if model_type == 'CifarJOVFPNet':
        q = model_kwargs['q'][0]
        q = float(q)

        default_block = get_block_adapter('PyramidBasicBlock')
        start_block = get_block_adapter(
            'FPBlockJOV' + mix_type, q=q, neg_slope=neg_slope)

        model = base.JOVFPAtStackStart(
            in_dim, out_dim, default_block=default_block, start_block=start_block, N=N
        )

    elif model_type == 'CifarAbsReLU-LS':
        q = model_kwargs['q'][0]
        q = float(q)

        default_block = get_block_adapter('PyramidBasicBlock')
        start_block = get_block_adapter(
            'AbsReLU' + mix_type, q=q, neg_slope=neg_slope)

        model = base.JOVFPAtStackStart(
            in_dim, out_dim, default_block=default_block, start_block=start_block, N=N
        )

    elif model_type == 'CifarAbsReLU-LS-DWS':
        q = model_kwargs['q'][0]
        q = float(q)

        default_block = get_block_adapter(
            'DWS' + mix_type, q=q, neg_slope=neg_slope)
        start_block = get_block_adapter(
            'AbsReLU' + mix_type, q=q, neg_slope=neg_slope)

        model = base.JOVFPAtStackStart(
            in_dim, out_dim, default_block=default_block, start_block=start_block, N=N
        )

    elif model_type == 'CifarAbsReLU-ALL':
        q = model_kwargs['q'][0]
        q = float(q)

        get_block = get_block_adapter(
            'AbsReLU' + mix_type, q=q, neg_slope=neg_slope)
        model = base.BaseModel(
            in_dim, out_dim, block_getter=get_block, N=N, pyr_weight_init=False
        )

    elif model_type == 'CifarDWS-ALL':
        q = model_kwargs['q'][0]
        q = float(q)

        get_block = get_block_adapter(
            'DWS' + mix_type, q=q, neg_slope=neg_slope)
        model = base.BaseModel(
            in_dim, out_dim, block_getter=get_block, N=N, pyr_weight_init=False
        )

    elif model_type == 'CifarINet':
        get_block = get_block_adapter('INet' + mix_type, neg_slope=neg_slope)
        model = base.BaseModel(
            in_dim, out_dim, block_getter=get_block, N=N, pyr_weight_init=False
        )
    else:
        raise ValueError(f'Unknown Model: {model_type}')

    return model.to(device).eval()


def get_block_adapter(block_type, **kwargs):
    # for most block the bias is automatically set to false
    if block_type == 'BasicBlock':
        return base.get_block_adapter(block_type, **kwargs)

    elif block_type == 'PyramidBasicBlock':
        return base.get_block_adapter(block_type, **kwargs)


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

    elif block_type == 'DWSLeakyUpper':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return blocks.DWSLeakyUpper(
                in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope
            )

    elif block_type == 'DWSLeakyBoth':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return blocks.DWSLeakyBoth(
                in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope
            )

    elif block_type == 'AbsReLULeakyUpper':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return blocks.AbsReLULeakyUpper(
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

    elif block_type == 'FPBlockJOVLeakyUpper':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return blocks.FPBlockJOVLeakyUpper(
                in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope
            )

    elif block_type == 'FPBlockJOVLeakyBoth':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return blocks.FPBlockJOVLeakyBoth(
                in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope
            )

    elif block_type == 'INetLeakyUpper':
        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return blocks.INetLeakyUpper(
                in_dim, out_dim, k=k, stride=stride, neg_slope=neg_slope
            )

    elif block_type == 'INetLeakyBoth':
        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return blocks.INetLeakyBoth(
                in_dim, out_dim, k=k, stride=stride, neg_slope=neg_slope
            )

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------

    elif block_type == 'DWSAssymUpper':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return blocks.DWSAssymUpper(
                in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope
            )

    elif block_type == 'DWSAssymBoth':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return blocks.DWSAssymBoth(
                in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope
            )

    elif block_type == 'AbsReLUAssymUpper':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return blocks.AbsReLUAssymUpper(
                in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope
            )

    elif block_type == 'AbsReLUAssymBoth':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return blocks.AbsReLUAssymBoth(
                in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope
            )

    elif block_type == 'FPBlockJOVAssymUpper':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return blocks.FPBlockJOVAssymUpper(
                in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope
            )

    elif block_type == 'FPBlockJOVAssymBoth':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return blocks.FPBlockJOVAssymBoth(
                in_dim, out_dim, k=k, stride=stride, q=q, neg_slope=neg_slope
            )

    elif block_type == 'INetAssymUpper':
        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return blocks.INetAssymUpper(
                in_dim, out_dim, k=k, stride=stride, neg_slope=neg_slope
            )

    elif block_type == 'INetAssymBoth':

        neg_slope = kwargs["neg_slope"]
        assert isinstance(neg_slope, float)

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return blocks.INetAssymBoth(
                in_dim, out_dim, k=k, stride=stride, neg_slope=neg_slope
            )

    else:
        raise ValueError(f'Unknown block: {block_type}')

    return get_block
