import subprocess

EXE_FILE = 'experiments/new_JOV_result_plots/run_model_database_plots.py'

MODELS_TO_EVALUATE = [
    {
        'model_name': 'CifarJOVFPNet_N9_s744',
        'num_stacks': 3,
        'deg_of_es_operations': ['upper', 'mult', 'lower', 'res'],
    },
    {
        'model_name': 'CifarPyrResNet_N9_s744',
        'num_stacks': 3,
        'deg_of_es_operations': ['bn1', 'bn2', 'bn3', 'relu', 'res'],
    },
    {
        'model_name': 'resnet50_imagenet_layer_start_q1',
        'num_stacks': 4,
        'deg_of_es_operations': ['upper', 'mult', 'lower', 'res'],
    },
]


def run():
    for params in MODELS_TO_EVALUATE:
        call_list = [
            'python', EXE_FILE,
            '--model_name', params['model_name'],
            '--num_stacks', str(params['num_stacks']),
        ]

        call_list += (
            ['--deg_of_es_operations'] + params['deg_of_es_operations']
        )

        print(call_list)
        subprocess.call(call_list)


if __name__ == '__main__':
    run()
