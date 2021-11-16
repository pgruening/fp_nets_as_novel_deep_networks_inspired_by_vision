"""
Creates plots and tables for the JOV jpeg experiments and saves them to 
OUT_FOLDER.

"""
import re
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from datasets.ds_cifar10_compression_test import Cifar10JpegCompression
from DLBio.helpers import (MyDataFrame, check_mkdir, load_json, search_rgx,
                           to_uint8_image)
from DLBio.pytorch_helpers import cuda_to_numpy
from experiments.eval_methods import save_curve_plot
from mdutils.mdutils import MdUtils
from tqdm import tqdm

DO_CREATE_PLOTS = True

RGXS = [r'(CifarJOVFPNet|CifarPyrResNet|CifarResNet|CifarJOVFPNet-RNBasic)_N(\d)_s(\d+)']
BASE_FOLDER = 'experiments/exp_6'
DATA_FOLDER = join(BASE_FOLDER, 'model_predictions')
MODEL_FOLDERS = [
    'experiments/exp_0/exp_data/trained_models',
    'experiments/exp_4/exp_data/trained_models'
]

ER_RGX = r'er_(\d+)'
NC_RGX = r'nc_(\d+)'


EXP_FOLDER = 'experiments'

OUT_FOLDER = 'experiments/new_JOV_result_plots/jpeg'
check_mkdir(OUT_FOLDER)

AGG = {
    'error': ('min', 'std', 'mean', 'max', 'median'),
    'init_error': ('min', 'std', 'mean', 'max', 'median'),
    'nc': ('min', 'std', 'mean', 'max', 'median')
}

COLORS = {
    'CifarJOVFPNet': 'g',
    'CifarJOVFPNet-RNBasic': 'g',
    'CifarPyrResNet': 'k',
    'CifarResNet': 'k',
}

RENAME = {
    'CifarJOVFPNet': 'FP-net',
    'CifarResNet': 'ResNet',
    'CifarPyrResNet': 'PyrBlockNet',
    'CifarJOVFPNet-RNBasic': 'FP-net (basic)'
}


def run_eval():
    df_grouped = get_grouped_dataframe()
    result_out_folder = join(OUT_FOLDER, 'results')
    check_mkdir(result_out_folder)

    create_tables(df_grouped, get_num_changes_col, 'num_changes')
    create_tables(df_grouped, get_error_col, 'robust_error')

    if DO_CREATE_PLOTS:
        create_plots(
            df_grouped, result_out_folder,
            subset=['CifarResNet', 'CifarJOVFPNet-RNBasic'],
            prefix='basic_block_'
        )
        create_plots(
            df_grouped, result_out_folder,
            subset=['CifarPyrResNet', 'CifarJOVFPNet'],
            prefix='pyr_block_'
        )


def create_tables(df_grouped, get_col_fcn, table_name):
    new_keys_ = ['Model']
    # create a new name
    df_grouped['Model'] = [
        f"{row['new_name'][0]} (N={row['N'][0]})" for _, row in df_grouped.iterrows()
    ]

    df_grouped = df_grouped.sort_values(by=['N', 'new_name'])
    for key in df_grouped.columns:
        # keys look like ('robust_acc_3', 'min'), ('num_params',''), ...
        new_key = get_col_fcn(key[0])
        if new_key is None or key[1] != 'mean':
            continue

        if new_key == '100':
            continue

        # grab the mean value of the key in question and rename it
        df_grouped[new_key] = df_grouped.loc[:, (key, 'mean')]
        new_keys_.append(new_key)

    # grab all the new keys
    table_out = df_grouped.loc[:, new_keys_]
    # col keys are still tuples ('', some_name), change that
    table_out.columns = [''.join(col) for col in table_out.columns.values]
    table_out = table_out.set_index('Model')

    md_file = MdUtils(
        file_name=join(OUT_FOLDER, f'{table_name}.md'),
        title='Cifar-10 adversarial attacks'
    )
    md_file.new_paragraph(table_out.to_markdown())
    md_file.create_md_file()

    with open(join(OUT_FOLDER, f'{table_name}_latex.txt'), 'w') as file:
        file.write(
            table_out.to_latex()
        )


def create_plots(df_grouped, result_out_folder, subset=None, prefix=''):
    if subset is not None:
        where = []
        for mt in list(df_grouped['model_type']):
            if mt in subset:
                where.append(True)
            else:
                where.append(False)
        df_grouped = df_grouped.copy()[where]
        legend_order = [RENAME[x] for x in subset]
    else:
        legend_order = None

    for key in get_agg().keys():
        save_curve_plot(
            df_grouped,
            result_out_folder,
            val_key=key,
            colors_=COLORS, name_key='new_name',
            ylabel=get_ylabel(key),
            pref=prefix,
            legend_order=legend_order
        )


def get_grouped_dataframe():
    df = pd.read_csv(join(BASE_FOLDER, 'eval.csv'))

    df_grouped = df.groupby(
        ['model_type', 'N', 'num_params'], as_index=False
    ).agg(get_agg()).sort_values(('error', 'mean'))

    df_grouped['new_name'] = [
        RENAME[row['model_type'][0]] for _, row in df_grouped.iterrows()
    ]
    return df_grouped


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
    folder = join(OUT_FOLDER, 'examples')
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


def get_error_col(key):
    match = re.match(ER_RGX, key)
    if not bool(match):
        return None
    # TODO: check this
    Q = 100 - 10 * int(match.group(1))
    return f'{Q}'


def get_num_changes_col(key):
    match = re.match(NC_RGX, key)
    if not bool(match):
        return None

    Q = 100 - 10 * int(match.group(1))
    return f'{Q}'


if __name__ == '__main__':
    # create_data_images()
    create_df()
    run_eval()
