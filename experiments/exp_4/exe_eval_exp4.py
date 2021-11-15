import warnings
from os.path import join, basename
import re

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from DLBio.helpers import MyDataFrame, load_json, search_rgx, check_mkdir
from DLBio.kwargs_translator import get_kwargs




BASE_FOLDER = 'experiments/exp_4/exp_data/trained_models'
RGX=r'(CifarJOVFPNet-RNBasic|CifarAbsReLU-LS-realAbs|CifarAbsReLU-LS-NoNorm|CifarJOVFPNet-NoNorm)_N(\d)_s(\d+)'

IM_OUT_FOLDER = 'experiments/exp_4'

AGG = {
    'min_val_er': ('mean', 'std', 'min'),
    'last_val_er': ('mean', 'std'),
    'num_params': 'first',
}

COLORS = {
    'CifarJOVFPNet': 'g',
    'CifarJOVFPNet-RNBasic':'lime',  
    'CifarAbsReLU-LS': 'b',  
    'CifarAbsReLU-LS-realAbs' : 'c',
    'CifarAbsReLU-LS-NoNorm' : 'dodgerblue',
    'CifarPyrResNet': 'k',
    'CifarResNet' : 'r',
    'CifarJOVFPNet-NoNorm' : 'olive'
}



ADD_COMP_MODELS = True

BASE_FOLDER_EXP1 = 'experiments/exp_1/exp_data/trained_models'



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

        exp_1_df = pd.read_csv(
            'experiments/exp_1/all_results.csv'
        )
        comp_df = exp_1_df[exp_1_df['model_type']=='CifarAbsReLU-LS'].copy()
        df = pd.concat([comp_df, df], ignore_index=True)
        
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

    plot_q2(df_grouped)

    plot_q3(df_grouped)


def create_plot(df_grouped, pref=''):
    plt.figure(figsize=(15, 15))
    for key in set(df_grouped['model_type']):
        tmp = df_grouped[df_grouped['model_type'] == key].copy()
        x = tmp[('num_params', 'first')]/1000
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

    models = ['CifarJOVFPNet', 'CifarJOVFPNet-RNBasic', 'CifarPyrResNet', 'CifarResNet']
    plot_subset(models, df_grouped, pref='q1_models_')


def plot_q2(df_grouped):
    models = ['CifarAbsReLU-LS', 'CifarAbsReLU-LS-realAbs']
    plot_subset(models, df_grouped, pref='q2_models_')

    models_rgx = r'(CifarAbsReLU-LS|CifarAbsReLU-LS-realAbs)_N(\d)_s(\d+)'
    exp_folders = [BASE_FOLDER, BASE_FOLDER_EXP1]

    out_name = join('curve_comparison', 'relu_vs_abs')
    compare_training_curves(models_rgx, exp_folders, out_name)
     

def plot_q3(df_grouped):
    models = ['CifarAbsReLU-LS', 'CifarAbsReLU-LS-NoNorm', 
                        'CifarPyrResNet', 'CifarJOVFPNet-NoNorm']    
    plot_subset(models, df_grouped, pref='q3_models_')

    rgx = r'(CifarJOVFPNet-NoNorm|CifarAbsReLU-LS-NoNorm)_N(\d)_s(\d+)'

    out_name = join('curve_comparison', 'no_norm_models')
    compare_training_curves(rgx, [BASE_FOLDER],out_name)







def compare_training_curves(rgx, exp_folders, out_name):
    folders_ = []
    for f in exp_folders:
        tmp = search_rgx(rgx, f)
        assert tmp
        tmp = [join(f,x) for x in tmp]
        folders_ +=tmp

    def get_folder_subset(N, seed, rgx, folders_):
        out = {}
        for x in folders_:
            match = re.match(rgx, basename(x))
            mtype = match.group(1)
            n = int(match.group(2))
            s = int(match.group(3))
            if n == N and s==seed:
                if mtype not in out.keys():
                    out[mtype] = [x]
                else:
                    out[mtype].append(x)

        return out

    def load_curve(folder):
        log = load_json(join(folder, 'log.json'))
        assert log is not None
        return log['er']
        

    seeds = [9, 507, 723, 16, 744]
    blocks = [3, 5, 7, 9]

    outdir = join(IM_OUT_FOLDER, out_name)
    check_mkdir(outdir)

    for N in blocks:
        for s in seeds:
            plt.figure()
            plt.title(f'N{N},seed{s}')
            tmp = get_folder_subset(N,s, rgx, folders_)
            for mtype, f_list in tmp.items():
                tmp = []
                for folder in f_list:
                    tmp.append(load_curve(folder))
                y = np.stack(tmp, 0)
                x = np.arange(y.shape[-1])
                plt.errorbar(
                    x, y.mean(0), y.std(0), label=mtype, color=COLORS[mtype]
                )

            plt.ylim([0., 10.])
            plt.grid()
            plt.legend()
            plt.savefig(join(outdir, f'models_N{N}_s{s}.png'))
            plt.close()




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
