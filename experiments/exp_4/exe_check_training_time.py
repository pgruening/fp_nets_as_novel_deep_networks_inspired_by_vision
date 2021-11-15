
import time
import warnings
from os.path import join

import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt

from DLBio.helpers import MyDataFrame, load_json, search_rgx, check_mkdir
from DLBio.kwargs_translator import get_kwargs
from models.conv_blocks import ReLUBabylon
from models.exp4_blocks import AbsBabylon

SEEDS = [9, 507, 723, 16, 744]
NUM_BLOCKS = [3, 5, 7, 9]

MODELS_RGX = r'(CifarAbsReLU-LS|CifarAbsReLU-LS-realAbs)_N(\d)_s(\d+)'


BASE_FOLDER = 'experiments/exp_4/exp_data/trained_models'
BASE_FOLDER_EXP1 = 'experiments/exp_1/exp_data/trained_models'

FOLDERS = [BASE_FOLDER, BASE_FOLDER_EXP1]

IM_OUT_FOLDER = 'experiments/exp_4/time_comparison'
check_mkdir(IM_OUT_FOLDER)


def run():
    folders_ = []
    df = MyDataFrame()
    for f in FOLDERS:
        tmp = search_rgx(MODELS_RGX, f)
        assert tmp
        tmp = [join(f,x) for x in tmp]
        folders_ +=tmp
        for folder in tmp:
            df = update(df, folder)

    df = df.get_df()    

    sns_plot = sns.violinplot(data=df, y="sum_sec", x="N", hue="model_type",
                split=True, inner="quart", linewidth=1
                )
    sns_plot.figure.savefig(join(IM_OUT_FOLDER, 'real_violinplot.png'))
    sns_plot.figure.clf()
    sns_plot2 = sns.boxplot(data=df, y="sum_sec", x="N", hue="model_type")
    sns_plot2.figure.savefig(join(IM_OUT_FOLDER, 'real_boxplot.png'))


def update(df, folder):
    log = load_json(join(folder, 'log.json'))
    if log is None:
        warnings.warn(f'No log-file found for {folder}')
        return df

    opt = load_json(join(folder, 'opt.json'))
    model_kwargs = get_kwargs(opt['model_kw'])
    assert opt is not None
    
    model_specs = load_json(join(folder, 'model_specs.json'))
    assert model_specs is not None

    df.update({
        'num_params': model_specs['num_params'],
        'seed': opt['seed'],
        'N': model_kwargs['N'][-1],
        'model_type': opt['model_type'],
        'sum_sec':int(sum(log['sec']))
    })

    return df


def run_synthetic():
    num_runs = 10000
    x = torch.randn(8, 64, 32, 32)
    y = torch.randn(8, 64, 32, 32)
    blocks = [ReLUBabylon(), AbsBabylon()]
    data = {}
    for block in blocks:
        time_values = []        
        for _ in range(num_runs):
            start_time = time.time()
            block(x,y)
            time_values.append(time.time() - start_time)
        data[str(block)] = time_values
        

    df = pd.DataFrame(data=data)
    
    agg = ['sum', 'mean', 'max', 'min', 'std']


    _, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.xaxis.tick_top()  # Display x-axis ticks on top
    df.plot.box(table=df.agg(agg), ax=ax)
    plt.title(f'time comparison in sec \nfor {num_runs} runs')
    plt.savefig(join(IM_OUT_FOLDER, 'synthetic_boxplot.png'))





if __name__ == '__main__':
    run()
    run_synthetic()
    


    
