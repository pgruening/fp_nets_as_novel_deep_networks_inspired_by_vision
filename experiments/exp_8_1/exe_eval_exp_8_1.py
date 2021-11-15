import re
from os.path import join, splitext

from DLBio.helpers import MyDataFrame, load_json, search_rgx
from DLBio.kwargs_translator import get_kwargs
from experiments.eval_methods import save_curve_plot

BASE_FOLDER = 'experiments/exp_8_1/logs'
RGX = r'(CifarJOVFPNet|CifarPyrResNet|CifarResNet|CifarJOVFPNet-RNBasic)_N(\d)_s(\d+).json'
IM_OUT_FOLDER = 'experiments/exp_8_1'

RA_RGX = r'robust_acc_(\d+)'
NC_RGX = r'num_changes_(\d+)'

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


def get_ylabel(key):
    match = re.match(RA_RGX, key)
    if bool(match):
        eps = int(match.group(1))
        return f'robust acc. eps={eps}/255'

    match = re.match(NC_RGX, key)
    if bool(match):
        eps = int(match.group(1))
        return f'Perc. changed predictions eps={eps}/255'

    raise ValueError(key)


def run():
    df = MyDataFrame()
    model_folders = search_rgx(RGX, BASE_FOLDER)
    for file_path in model_folders:
        file_path = join(BASE_FOLDER, file_path)
        df = update(df, file_path)

    df = df.get_df()
    agg = dict()
    keys_ = get_robust_acc_keys(df.columns) + get_num_changes_keys(df.columns)
    for key in keys_:
        agg[key] = ('mean', 'std', 'max', 'min')

    df_grouped = df.groupby(
        ['num_params', 'model_type', 'N'], as_index=False
    ).agg(agg)
    df_grouped = df_grouped.sort_values(by=('robust_acc_4', 'mean'))
    df_grouped['new_name'] = [
        RENAME[row['model_type'][0]] for _, row in df_grouped.iterrows()
    ]
    for key in keys_:
        save_curve_plot(
            df_grouped, IM_OUT_FOLDER, val_key=key,
            colors_=COLORS, name_key='new_name',
            ylabel=get_ylabel(key)
        )


def update(df, file_path):
    adv_log = load_json(file_path)
    model_folder = adv_log['folder']
    options = load_json(join(model_folder, 'opt.json'))
    model_kwargs = get_kwargs(options['model_kw'])

    log = load_json(join(model_folder, 'log.json'))

    tmp = {
        'original_acc': max(log['val_acc']),
        'seed': options['seed'],
        'model_type': options['model_type'],
        'num_params': adv_log['num_params'],
        'N': model_kwargs['N'][-1],
    }

    for x in get_robust_acc_keys(adv_log.keys()):
        tmp[x] = adv_log[x]

    for x in get_num_changes_keys(adv_log.keys()):
        tmp[x] = adv_log[x]

    df.update(tmp)
    return df


def get_robust_acc_keys(keys):
    ra_keys = [x for x in keys if bool(re.match(RA_RGX, x))]
    return ra_keys


def get_num_changes_keys(keys):
    nc_keys = [x for x in keys if bool(re.match(NC_RGX, x))]
    return nc_keys


if __name__ == '__main__':
    run()
