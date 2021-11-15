"""
Create databases for all models used in JOV paper.
"""
import subprocess

PATH_TO_EXE = 'run_create_model_database.py'
BATCH_SIZE = 100
NUM_WORKERS = 0

MODELS_TO_EVALUATE = [
    {
        'path_to_model': 'experiments/exp_0/exp_data/trained_models/CifarJOVFPNet_N9_s744',
        'depth_offset': 2,
        'is_imagenet_model': False,
        'from_par_gpu': False,
    },
    {
        'path_to_model': 'experiments/exp_0/exp_data/trained_models/CifarPyrResNet_N9_s744',
        'depth_offset': 2,
        'is_imagenet_model': False,
        'from_par_gpu': False,
    },
    {
        'path_to_model': 'experiments/imagenet_resnet/resnet50_imagenet_layer_start_q1',
        'depth_offset': 2,
        'is_imagenet_model': True,
        'from_par_gpu': True,
    },
]


def run():
    for params in MODELS_TO_EVALUATE:

        call_list = [
            'python', PATH_TO_EXE,
            '--model_folder', params['path_to_model'],
            '--bs', str(BATCH_SIZE),
            '--nw', str(NUM_WORKERS),
            '--depth_offset', str(params['depth_offset'])
        ]

        # add booleans if needed
        if params['is_imagenet_model']:
            call_list.append('--is_imagenet_model')

        if params['from_par_gpu']:
            call_list.append('--from_par_gpu')

        print(call_list)
        subprocess.call(call_list)


if __name__ == '__main__':
    run()
