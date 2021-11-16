import warnings
from os.path import join, basename
import re

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from DLBio.helpers import MyDataFrame, load_json, search_rgx, check_mkdir
from DLBio.kwargs_translator import get_kwargs


BASE_FOLDER = 'experiments/exp_4/exp_data/trained_models'
RGX = r'(CifarJOVFPNet-RNBasic)_N(\d)_s(\d+)'

IM_OUT_FOLDER = 'experiments/exp_4'

AGG = {
    'min_val_er': ('mean', 'std', 'min'),
    'last_val_er': ('mean', 'std'),
    'num_params': 'first',
}

COLORS = {
    'CifarJOVFPNet': 'g',
    'CifarJOVFPNet-RNBasic': 'lime',
    'CifarPyrResNet': 'k',
    'CifarResNet': 'r',
}


ADD_COMP_MODELS = True


def run():
    folders_ = search_rgx(RGX, BASE_FOLDER)
    assert folders_

    df = MyDataFrame()
    for folder in folders_:
        df = update(df, join(BASE_FOLDER, folder))
    df = df.get_df()

    if ADD_COMP_MODELS:
        exp_0_df = pd.read_csv(
            'experiments/exp_0/all_results.csv'
        )
        df = pd.concat([exp_0_df, df], ignore_index=True)

    df = df.sort_values(by=['N', 'model_type', 'min_val_er'])

    df.to_excel(join(IM_OUT_FOLDER, 'all_results.xlsx'))
    df.to_csv(join(IM_OUT_FOLDER, 'all_results.csv'))
    print(df.head())

    df_grouped = df.groupby(by=['model_type', 'N'], as_index=False).agg(
        AGG).sort_values(by=('min_val_er', 'mean'))
    print(df_grouped)
    df_grouped.to_excel(join(IM_OUT_FOLDER, 'grouped_results.xlsx'))

    create_plot(df_grouped)

    if not ADD_COMP_MODELS:
        return

    # add plot for each question

    plot_q1(df_grouped)


def create_plot(df_grouped, pref=''):
    plt.figure(figsize=(15, 15))
    for key in set(df_grouped['model_type']):
        tmp = df_grouped[df_grouped['model_type'] == key].copy()
        x = tmp[('num_params', 'first')] / 1000
        y = tmp[('min_val_er', 'mean')]
        z = tmp[('min_val_er', 'std')]
        plt.errorbar(x, y, z, linewidth=4, label=key, color=COLORS[key])

    plt.ylabel('Best error +/- std.')
    plt.xlabel('Num. Parameters (k)')
    plt.legend()
    plt.grid()

    plt.savefig(join(IM_OUT_FOLDER, pref + 'num_params_vs_min_val_error.png'))
    plt.savefig(join(IM_OUT_FOLDER, pref + 'num_params_vs_min_val_error.eps'))
    plt.close()


def plot_subset(subset, df_grouped, pref=''):
    where = [
        True if x in subset else False for x in df_grouped['model_type']
    ]
    create_plot(df_grouped[where], pref=pref)


def plot_q1(df_grouped):

    models = ['CifarJOVFPNet', 'CifarJOVFPNet-RNBasic',
              'CifarPyrResNet', 'CifarResNet']
    plot_subset(models, df_grouped, pref='q1_models_')


def update(df, folder):
    log = load_json(join(folder, 'log.json'))
    if log is None:
        warnings.warn(f'No log-file found for {folder}')
        return df

    if 'val_er' not in log.keys():
        warnings.warn(f'No values found for {folder}')
        return df

    opt = load_json(join(folder, 'opt.json'))
    model_kwargs = get_kwargs(opt['model_kw'])
    assert opt is not None

    model_specs = load_json(join(folder, 'model_specs.json'))
    assert model_specs is not None

    df.update({
        'min_val_er': min(log['val_er']),
        'last_val_er': log['val_er'][-1],
        'num_params': model_specs['num_params'],
        'seed': opt['seed'],
        'N': model_kwargs['N'][-1],
        'model_type': opt['model_type']
    })

    return df


if __name__ == '__main__':
    run()
