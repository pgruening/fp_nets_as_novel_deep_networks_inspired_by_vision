import copy
from os.path import join

import torch
import config
from DLBio import kwargs_translator, pt_run_parallel
from DLBio.helpers import check_mkdir, load_json, get_subfolders, search_rgx
from helpers import get_data_loaders, load_model
from log_tensorboard import log_tensorboard

EXE_FILE = 'run_training.py'
BASE_FOLDER = 'experiments/exp_4'
AVAILABLE_GPUS = [0, 1, 2, 3]

SEEDS = [9, 507, 723, 16, 744]
NUM_BLOCKS = [3, 5, 7, 9]


DEFAULT_KWARGS = {
    'comment': 'exp_4: additional models.',
    'lr': 0.1,
    'wd': 0.0001,
    'mom': 0.9,
    'bs': 128,
    'opt': 'SGD',
    'nw': 4,

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

MODELS = ['CifarJOVFPNet-RNBasic', 'CifarAbsReLU-LS-realAbs', 
                    'CifarAbsReLU-LS-NoNorm', 'CifarJOVFPNet-NoNorm']



class TrainingProcess(pt_run_parallel.ITrainingProcess):
    def __init__(self, **kwargs):
        self.start_time = -1
        self.device = -1

        self.__name__ = 'Exp4_training_process'
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
        for p in _param_generator(default_kwargs, base_folder, seeds=SEEDS, num_blocks=NUM_BLOCKS):
            yield p

    _run(param_generator)


def one_run_test():
    default_kwargs = copy.deepcopy(DEFAULT_KWARGS)
    default_kwargs['epochs'] = 1
    default_kwargs['lr_steps'] = 0
    default_kwargs['comment'] = "exp_4: .one run test for PyramidBlock impact"
    default_kwargs['sv_int'] = 0

    default_kwargs.pop('early_stopping')
    default_kwargs.pop('fixed_steps')

    seed = 0
    N = 3
    model_type = MODELS[0]

    base_folder = join(BASE_FOLDER, 'one_run_test')
    output = copy.deepcopy(default_kwargs)
    model_kw = {'N': [N]}

    if model_type == 'CifarJOVFPNet-RNBasic':
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


def one_epoch_test():
    default_kwargs = copy.deepcopy(DEFAULT_KWARGS)
    default_kwargs['epochs'] = 1
    default_kwargs['lr_steps'] = 0
    default_kwargs['comment'] = "exp_4: one epoch test"
    default_kwargs['do_overwrite'] = None
    default_kwargs['sv_int'] = 0

    default_kwargs.pop('early_stopping')
    default_kwargs.pop('fixed_steps')

    base_folder = join(BASE_FOLDER, 'one_epoch_test')

    def param_generator():
        N = 5
        seed = 0
        for p in _param_generator(default_kwargs, base_folder, seeds=[seed], num_blocks=[N]):
            yield p

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


def _param_generator(default_kwargs, base_folder, seeds=SEEDS, num_blocks=NUM_BLOCKS):
    for seed in seeds:
        for N in num_blocks:
            for model_type in MODELS:
                output = copy.deepcopy(default_kwargs)

                model_kw = {'N': [N]}
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
