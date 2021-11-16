import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from DLBio.pt_train_printer import IPrinterFcn
from DLBio.pt_training import ITrainInterface
from DLBio.pytorch_helpers import ActivationGetter, get_device
from DLBio.train_interfaces import (Accuracy, Classification, ErrorRate,
                                    image_counter)

import helpers
from models.conv_blocks import AbsReLUBlock, DWSBlock, FPBlockJOV, INetBlock

USED_BLOCKS = (AbsReLUBlock, FPBlockJOV, DWSBlock, INetBlock)


def get_interface(ti_type, model, device, printer, **kwargs):
    if ti_type == Classification.name:
        return Classification(model, device, printer)
    elif ti_type == Logging.name:
        return Logging(model, device, printer, kwargs['logger_dict'])
    elif ti_type == NoDRClassification.name:
        return NoDRClassification(model, device, printer, kwargs['num_epochs'])
    elif ti_type == ClassAndLogging.name:
        return ClassAndLogging(model, device, printer, kwargs['logger_dict'])
    raise ValueError(f"Unknown ti_type: {ti_type}")


class Logging(ITrainInterface):
    name = 'Logging'

    def __init__(self, model, device, printer, logger_dict):
        self.printer = printer
        self.model = model

        self.functions = logger_dict
        self.printer.dont_print = list(logger_dict.keys())

        self.counters = {
            'num_samples': image_counter
        }

        self.d = device

    def train_step(self, sample):
        images, targets = sample[0].to(self.d), sample[1].to(self.d)
        pred = self.model(images)

        metrics = None
        counters = dict()
        counters.update(
            {k: v(pred, targets) for k, v in self.counters.items()}
        )
        functions = {
            k: f.update(pred, targets) for k, f in self.functions.items()
        }
        return torch.Tensor([0.]).float(), metrics, counters, functions


class NoDRClassification(Classification):
    name = 'NoDRClassification'
    # TODO: assert no resume mode in run training!

    def __init__(self, model, device, printer, num_epochs):
        super(NoDRClassification, self).__init__(model, device, printer)

        re_init_functions = get_reinit_functions(model, num_epochs)
        self.functions.update(
            re_init_functions
        )
        self.printer.dont_print = list(re_init_functions.keys())


def get_reinit_functions(model, num_epochs):
    mod_list = helpers.get_ordered_module_list(
        model, batch_input_size=(1, 3, 32, 32),
        device=get_device(),
        use_only=USED_BLOCKS
    )

    reinit_functions = {}
    for depth, module in mod_list:
        key = f'{module._get_name()}_d{depth}'

        upper = module.block_with_shortcut.block.upper
        act_getter = ActivationGetter(upper)

        upper_conv = upper[0]
        ri_func = ReInitDeadReLUs(
            upper_conv, act_getter, num_epochs=num_epochs
        )
        reinit_functions[key] = ri_func

    return reinit_functions


class ReInitDeadReLUs(IPrinterFcn):
    def __init__(self, module, act_getter, *, num_epochs, init_weight=1., num_restart_calls=2):
        self.act_getter = act_getter
        self.module = module

        self.values = []

        # used to compute the weight for reinit
        self.num_restart_calls = num_restart_calls
        self.thres = 1e-6
        self.ctr = 0.
        self.N = num_epochs
        self.b = init_weight
        # N-1: the weight should be zero in the last epoch
        self.m = -1. * self.b / (self.N - 1)

    def update(self, *args):
        with torch.no_grad():
            self.values.append(
                self.act_getter.out.mean([0, 2, 3]).cpu().numpy()
            )
            self.act_getter.out = None

        return self

    def restart(self):
        self.values = []
        self.ctr += 1. / float(self.num_restart_calls)
        assert self.ctr <= self.N

    def _get_weight(self):
        w = float(self.ctr) * self.m + self.b
        return w

    def __call__(self):
        X = np.stack(self.values, -1).copy()
        X = np.mean(X, -1)

        is_dead = (X < self.thres)

        w = self._get_weight()

        with torch.no_grad():
            new_weights = init.kaiming_uniform_(
                torch.zeros(self.module.weight.shape), a=0.
            ).to(self.module.weight.device)

            self.module.weight[is_dead, :, ...] += (
                w * new_weights[is_dead, :, ...]
            )

        # return the percentage of dead ReLUs
        return is_dead.astype('float32').mean() * 100.


class ClassAndLogging(ITrainInterface):
    name = 'ClassAndLogging'

    def __init__(self, model, device, printer, logger_dict):
        self.printer = printer
        self.model = model
        self.xent_loss = nn.CrossEntropyLoss()
        self.functions = {
            'acc': Accuracy(),
            'er': ErrorRate()
        }

        self.counters = {
            'num_samples': image_counter
        }

        self.functions.update(logger_dict)
        self.printer.dont_print = list(logger_dict.keys())

        self.d = device

    def train_step(self, sample):
        #print('train step')
        images, targets = sample[0].to(self.d), sample[1].to(self.d)
        pred = self.model(images)

        if targets.ndim == 2:
            targets = targets[:, 0]

        loss = self.xent_loss(pred, targets)

        metrics = None

        counters = dict()
        counters.update(
            {k: v(pred, targets) for k, v in self.counters.items()}
        )

        functions = {
            k: f.update(pred, targets) for k, f in self.functions.items()
        }
        return loss, metrics, counters, functions


class Logger(IPrinterFcn):
    """
    A Logger is a printer function that monitors activation values,
    the activations are transformed using _get_transformed_activation
    and stored in self.values. When calling the Logger, the list of values
    is typecasted to an numpy array. On this array, the reduce function is
    applied.

    """

    def __init__(self, name, act_getter, reduce_fcn):
        self.name = name
        self.init_name = name
        self.act_getter = act_getter
        self.reduce = reduce_fcn
        self.values = []

    def update(self, *args):
        if self.act_getter.out is not None:
            if self.act_getter.out.numel() != 0:
                self.values.append(self._get_transformed_activation())
                self.act_getter.out = None
        return self

    def restart(self):
        assert hasattr(self, "init_name"), (
            "In your init function you need to "
            "set the init_name attribute to name. This attribute is needed "
            "for the restart function.")
        self.name = self.init_name
        self.values = []

    def _get_transformed_activation(self):
        return self.act_getter.out.mean().item()

    def __call__(self):
        x = np.array(self.values)
        return self.reduce(x)
