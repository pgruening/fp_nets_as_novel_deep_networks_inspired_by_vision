import re
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from datasets.ds_cifar10_compression_test import Cifar10JpegCompression
from DLBio.helpers import (MyDataFrame, check_mkdir, load_json, search_rgx,
                           to_uint8_image)
from DLBio.pytorch_helpers import cuda_to_numpy
from experiments.eval_methods import save_curve_plot
from tqdm import tqdm

RGXS = [r'(CifarJOVFPNet|CifarPyrResNet|CifarResNet|CifarJOVFPNet-RNBasic)_N(\d)_s(\d+)']
BASE_FOLDER = '/nfshome/gruening/my_code/DLBio_repos/fp_net_after_jov/experiments/exp_6'
DATA_FOLDER = join(BASE_FOLDER, 'model_predictions')
MODEL_FOLDERS = [
    'experiments/exp_0/exp_data/trained_models',
    'experiments/exp_4/exp_data/trained_models'
]

EXP_FOLDER = 'experiments'

IM_OUT_FOLDER = join(BASE_FOLDER, 'images')

AGG = {
    'error': ('min', 'std', 'mean', 'max', 'median'),
    'init_error': ('min', 'std', 'mean', 'max', 'median'),
    'nc': ('min', 'std', 'mean', 'max', 'median')
}

COLORS = {
    'CifarJOVFPNet': 'g',
    'CifarPyrResNet': 'k',
    'CifarResNet': 'c',
    'CifarJOVFPNet-RNBasic': 'olive'
}

RENAME = {
    'CifarJOVFPNet': 'FP-net',
    'CifarResNet': 'ResNet',
    'CifarPyrResNet': 'PyrBlockNet',
    'CifarJOVFPNet-RNBasic': 'FP-net (basic)'
}


def get_agg():
    v = ('min', 'std', 'mean', 'max')
    out = dict()
    # for key in ['error', 'init_error', 'nc']:
    for key in ['error', 'init_error']:
        out[key] = v

    for i in range(10):
        out[f'er_{i}'] = v
        out[f'nc_{i}'] = v

    return out


def get_proper_key(key, is_table=False):
    rgx = r'(er|nc)_(\d)'
    match = re.match(rgx, key)
    if not bool(match):
        return key

    idx = int(match.group(2))
    if is_table:
        type_ = {
            'er': 'Error',
            'nc': 'NCC'
        }[match.group(1)]
    else:
        type_ = {
            'er': 'test-error',
            'nc': 'rel. number of class changes'
        }[match.group(1)]

    return f'{type_} Q: {100 - 10*idx}'


def get_ylabel(key):
    rgx1 = r'nc_(\d+)'
    match = re.search(rgx1, key)
    if bool(match):
        q = 100 - 10 * int(match.group(1))
        return f'Perc. changed predictions Q: {q}'

    rgx2 = r'er_(\d+)'
    match = re.search(rgx2, key)
    if bool(match):
        q = 100 - 10 * int(match.group(1))
        return f'Error rate Q: {q}'

    return key


def run_eval():
    df = pd.read_csv(join(BASE_FOLDER, 'eval.csv'))

    df_grouped = df.groupby(
        ['model_type', 'N', 'num_params'], as_index=False
    ).agg(get_agg()).sort_values(('error', 'mean'))

    df_grouped['new_name'] = [
        RENAME[row['model_type'][0]] for _, row in df_grouped.iterrows()
    ]

    check_mkdir(IM_OUT_FOLDER)
    result_out_folder = join(BASE_FOLDER, 'images', 'results')
    check_mkdir(result_out_folder)

    for key in get_agg().keys():
        save_curve_plot(
            df_grouped,
            result_out_folder,
            val_key=key,
            colors_=COLORS, name_key='new_name',
            ylabel=get_ylabel(key)
        )

    print('\n' * 2)
    get_table(df_grouped, 'er')
    print('\n' * 2)
    get_table(df_grouped, 'nc', start=1)
    print('\n' * 2)


def get_table(df_grouped, key, start=0):
    df_grouped = df_grouped[df_grouped['N'] == 9].set_index(
        'model_type').round(2)
    cols = []
    for i in range(start, 10):
        cols.append((f'{key}_{i}', 'mean'))

    tmp = df_grouped.loc[:, cols]
    tmp = tmp.sort_values((f'{key}_{start}', 'mean'))
    tmp.columns = [get_proper_key(x[0], is_table=True) for x in tmp.columns]
    print(tmp.to_markdown())


def create_df():
    df = MyDataFrame()
    files_ = get_data()
    for file in tqdm(files_):
        df = update(df, file)

    df = df.get_df()

    df.to_csv(join(BASE_FOLDER, 'eval.csv'))


def get_data():
    files_ = []
    for rgx in RGXS:
        tmp = search_rgx(rgx, DATA_FOLDER)
        assert tmp
        files_ += [join(DATA_FOLDER, x) for x in tmp]

    return files_


def update(df, file):
    name = file.split('/')[-1].split('.npy')[0]
    rgx = r'(.*)_N(\d)_s(\d+)'
    match = re.match(rgx, name)
    X = np.load(file)

    original_folder = get_original_folder(re.compile(name))
    model_specs = load_json(join(original_folder, 'model_specs.json'))

    tmp = dict()
    tmp.update({
        'name': name,
        'model_type': match.group(1),
        'N': int(match.group(2)),
        'seed': match.group(3),
        'error': compute_error(X),
        'init_error': compute_initial_error(X),
        'num_params': model_specs['num_params'],
    })

    for i in range(10):
        tmp[f'er_{i}'] = compute_error_for_subset(X, i)
        tmp[f'nc_{i}'] = compute_change_prob(X, idx=i)

    df.update(tmp)

    return df


def get_original_folder(rgx):
    for folder in MODEL_FOLDERS:
        tmp = search_rgx(rgx, folder)
        if len(tmp) == 1:
            return join(folder, tmp[0])

    raise ValueError(rgx)


def compute_initial_error(X):
    return 100. * (1. - (X[:, 0] == X[:, 1]).mean())


def compute_error_for_subset(X, idx):
    y = X[:, 0]
    y = y
    X = X[:, 1:]
    acc = (y == X[:, idx]).mean()
    return 100. * (1. - acc)


def compute_error(X):
    y = X[:, 0]
    D = y.reshape(-1, 1) == X[:, 1:]

    return (1. - D.mean()) * 100.


def compute_change_prob(X, idx):
    X = X[:, 1:]
    did_change = X[:, 0] != X[:, idx]
    return did_change.mean() * 100.


def compute_num_changes(X, idx=None):
    # remove label
    if idx is None:
        X = X[:, 1:]
    else:
        X = X[:, [1, idx]]

    mode_count = stats.mode(X, axis=1)[1]
    num_changes = X.shape[1] - mode_count

    return num_changes.mean()


def compute_entropy(X):
    X = X[:, 1:]
    entropy = []

    max_val = -1. * np.log(1. / 10.)

    for i in range(X.shape[0]):
        h, _ = np.histogram(X[i, :], bins=range(10))
        h = h / h.sum()
        entropy.append((-1. * h * np.log2(h + 1e-6)).sum())

    return np.array(entropy).mean() / max_val


def create_data_images():
    folder = join(IM_OUT_FOLDER, 'examples')
    check_mkdir(folder)
    dataset = Cifar10JpegCompression()
    qus = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    for name, i in enumerate([10, 35, 42, 110, 222, 723]):
        _, ax = plt.subplots(2, 5)
        X = dataset[i][0]
        for i, q in enumerate(qus):
            image = cuda_to_numpy(X[i, ...])
            image = to_uint8_image(image)
            y = i // 5
            x = i % 5
            ax[y][x].set_title(q)
            ax[y][x].imshow(image)

            ax[y][x].set_xticks([])
            ax[y][x].set_yticks([])

        path = join(folder, str(name).zfill(2) + '.png')
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


if __name__ == '__main__':
    create_data_images()
    create_df()
    run_eval()
