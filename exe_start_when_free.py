from DLBio.pytorch_helpers import start_when_free

MODULE_PATH = 'experiments/exp_3/exe_train_exp3.py'

start_when_free(
    MODULE_PATH, gpus=[0], mode='any'
)
