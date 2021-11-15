"""
This experiment derives its hyperparameters from He et al.'s
Deep Residual learning paper:

"We use a weight decay of 0.0001 and momentum of 0.9,
and adopt the weight initialization in [13] and BN [16]
but with no dropout. These models are trained with a
mini-batch size of 128 on two GPUs. We start with a
learning rate of 0.1, divide it by 10 at 32k and 48k
iterations, and terminate training at 64k iterations,
which is determined on a 45k/5k train/val split."

According to the paper, the training should be 164 epochs,
with a change of the learning rate at 82 and 123 epochs.

In the Ydelbayev repo: the networks are trained for 200 epochs:
"
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
milestones=[100, 150], last_epoch=args.start_epoch - 1)"

"""
import copy
from os.path import join

import torch
from DLBio import kwargs_translator, pt_run_parallel
from DLBio.helpers import check_mkdir, load_json, get_subfolders, search_rgx
import glob

try:
    from helpers import get_data_loaders, load_model
    from log_tensorboard import log_tensorboard
except ModuleNotFoundError:
    # in path ./experiments/exp0 -> change to
    from os import chdir
    chdir('../..')
    import sys
    sys.path.append('.')

    from helpers import get_data_loaders, load_model
    from log_tensorboard import log_tensorboard


EXE_FILE = '/nfshome/gruening/my_code/DLBio_repos/fp_net_after_jov/run_training.py'
BASE_FOLDER = '/nfshome/gruening/my_code/DLBio_repos/fp_net_after_jov/experiments/exp_0'
AVAILABLE_GPUS = [0, 1, 2, 3]

SEEDS = [9, 507, 723, 16, 744]

DEFAULT_KWARGS = {
    'comment': 'exp_0: Recreate the JOV Cifar-10 experiments with original ResNet Base architecture.',
    'lr': 0.1,
    'wd': 0.0001,
    'mom': 0.9,
    'bs': 128,
    'opt': 'SGD',

    'train_interface': 'classification',

    # model / ds specific params
    'in_dim': 3,
    'out_dim': 10,

    # scheduling
    'epochs': 200,
    'lr_steps': 0,
    'fixed_steps': [100, 150],

    # dataset
    'dataset': 'cifar_10',

    # model saving
    'sv_int': -1,
    'early_stopping': None,
    'es_metric': 'val_acc',
}


def main():
    def param_generator():
        for p in _param_generator(default_kwargs):
            yield p

    _run(param_generator)


class TrainingProcess(pt_run_parallel.ITrainingProcess):
    def __init__(self, **kwargs):
        self.start_time = -1
        self.device = -1

        self.__name__ = 'Exp0_training_process'
        self.module_name = EXE_FILE
        self.kwargs = kwargs


def _run(param_generator):
    make_object = pt_run_parallel.MakeObject(TrainingProcess)
    pt_run_parallel.run(param_generator(), make_object,
                        available_gpus=AVAILABLE_GPUS
                        )


def run():
    default_kwargs = copy.deepcopy(DEFAULT_KWARGS)
    base_folder = join(BASE_FOLDER, 'exp_data')
    default_kwargs["nw"] = 4

    def param_generator():
        for p in _param_generator(default_kwargs, base_folder, seeds=SEEDS):
            yield p

    _run(param_generator)


def one_epoch_test():
    default_kwargs = copy.deepcopy(DEFAULT_KWARGS)
    default_kwargs['epochs'] = 1
    default_kwargs['lr_steps'] = 0
    default_kwargs['comment'] = "exp_0: .one epoch test for JOV-Cifar 10 recreation"
    default_kwargs['do_overwrite'] = None
    default_kwargs['sv_int'] = 0

    default_kwargs.pop('early_stopping')
    default_kwargs.pop('fixed_steps')

    base_folder = join(BASE_FOLDER, 'one_epoch_test')

    def param_generator():
        for p in _param_generator(default_kwargs, base_folder, seeds=[0, 1]):
            yield p

    _run(param_generator)


def one_run_test():
    default_kwargs = copy.deepcopy(DEFAULT_KWARGS)
    default_kwargs['epochs'] = 1
    default_kwargs['lr_steps'] = 0
    default_kwargs['comment'] = "exp_0: .one epoch test for JOV-Cifar 10 recreation"
    default_kwargs['sv_int'] = 0

    default_kwargs.pop('early_stopping')
    default_kwargs.pop('fixed_steps')

    seed = 0
    N = 3
    model_type = 'CifarJOVFPNet'

    base_folder = join(BASE_FOLDER, 'one_run_test')
    output = copy.deepcopy(default_kwargs)
    model_kw = {'N': [N]}

    if model_type == 'CifarJOVFPNet':
        model_kw['q'] = [2]

    output['model_kw'] = kwargs_translator.to_kwargs_str(model_kw)
    output['folder'] = join(
        base_folder, f'trained_models/{model_type}_N{N}_s{seed}'
    )
    output['model_type'] = model_type
    output['seed'] = seed

    def param_generator():
        yield output

    _run(param_generator)


def check_tensorboard_one_epoch():
    folder_names_ = search_rgx(
        r'(.*)_N\d+_s\d+', join(BASE_FOLDER, 'one_epoch_test', 'trained_models')
    )
    assert folder_names_

    for idx, folder_name in enumerate(folder_names_):
        out_name = join('one_epoch_test', folder_name)
        folder = join(BASE_FOLDER, 'one_epoch_test',
                      'trained_models', folder_name)

        if idx == 0:
            add_images = True
        else:
            add_images = False
        _check_tensorboard(folder, out_name, add_images=add_images)


def _check_tensorboard(folder, out_name, *, add_images):
    tb_out = join(BASE_FOLDER, 'tboard', out_name)
    check_mkdir(tb_out)

    options = load_json(join(folder, 'opt.json'))
    assert options is not None, f'no options at: {folder}'

    model = load_model(options, 'cpu', map_location=torch.device(
        'cpu'), new_model_path=join(folder, 'model.pt')
    )

    if add_images:
        data_loaders = get_data_loaders(options)
    else:
        data_loaders = None

    log_tensorboard(folder, tb_out, data_loaders,
                    model=model, remove_old_events=True,
                    input_shape=(1, 3, 32, 32)
                    )


def _param_generator(default_kwargs, base_folder, seeds=SEEDS):
    for seed in seeds:
        for N in [3, 5, 7, 9]:
            for model_type in ['CifarResNet', 'CifarPyrResNet', 'CifarJOVFPNet']:
                output = copy.deepcopy(default_kwargs)
                model_kw = {'N': [N]}

                if model_type == 'CifarJOVFPNet':
                    model_kw['q'] = [2]

                output['model_kw'] = kwargs_translator.to_kwargs_str(model_kw)
                output['folder'] = join(
                    base_folder, f'trained_models/{model_type}_N{N}_s{seed}'
                )
                output['model_type'] = model_type
                output['seed'] = seed

                yield output


if __name__ == '__main__':
    # one_run_test()
    # one_epoch_test()
    # check_tensorboard_one_epoch()
    run()
    pass
