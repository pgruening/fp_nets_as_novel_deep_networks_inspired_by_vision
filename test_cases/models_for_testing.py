import torch.nn as nn
import torch
from DLBio.pytorch_helpers import walk


def set_all_weights_to_zero(model):
    with torch.no_grad():
        for mod_ in walk(model):
            # conv and linear
            if hasattr(mod_, 'weight') and mod_.weight is not None:
                mod_.weight = nn.Parameter(torch.zeros(*mod_.weight.shape))
            if hasattr(mod_, 'bias') and mod_.bias is not None:
                mod_.bias = nn.Parameter(torch.zeros(*mod_.bias.shape))

            # batchnorm in eval-mode: these are not parameters (no grad)
            if hasattr(mod_, 'running_mean') and mod_.running_mean is not None:
                mod_.running_mean = torch.zeros(*mod_.running_mean.shape)
            if hasattr(mod_, 'running_var') and mod_.running_var is not None:
                mod_.running_var = torch.ones(*mod_.running_var.shape)

    return model
