"""
Using the repositroy: https://github.com/RobustBench/robustbench

Note that for our standardized evaluation of Linf-robustness we use the full 
version of AutoAttack which is slower but more accurate (for that just use 
adversary = AutoAttack(model, norm='Linf', eps=8/255)).
"""
from os.path import join

import torch
from DLBio.helpers import search_rgx
from DLBio import pt_run_parallel

AVAILABLE_GPUS = [0, 1, 2, 3]
#AVAILABLE_GPUS = [0]

BASE_FOLDERS = [
    'experiments/exp_0/exp_data/trained_models',
    'experiments/exp_4/exp_data/trained_models'
]
RGXS = [
    r'(CifarJOVFPNet|CifarPyrResNet|CifarResNet)_N(\d)_s(\d+)',
    r'(CifarJOVFPNet-RNBasic)_N(\d)_s(\d+)'
]

EXE_FILE = 'experiments/exp_8_1/run_bechmark_no_normalization.py'


class TrainingProcess(pt_run_parallel.ITrainingProcess):
    def __init__(self, **kwargs):
        self.start_time = -1
        self.device = -1

        self.__name__ = 'Exp8_training_process'
        self.module_name = EXE_FILE
        self.kwargs = kwargs


def run():
    make_object = pt_run_parallel.MakeObject(TrainingProcess)
    pt_run_parallel.run(param_generator(), make_object,
                        available_gpus=AVAILABLE_GPUS
                        )


def param_generator():
    model_folders = []
    for rgx, base_folder in zip(RGXS, BASE_FOLDERS):
        tmp = search_rgx(rgx, base_folder)
        assert tmp
        model_folders += [join(base_folder, x) for x in tmp]

    for folder in model_folders:
        yield {'folder': folder}


if __name__ == '__main__':
    run()
