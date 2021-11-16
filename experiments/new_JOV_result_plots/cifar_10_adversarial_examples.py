"""
Uses the results stored in BASE_FOLDER and creates plots and tables in
OUT_FOLDER.
"""
import re
from os.path import join

from DLBio.helpers import MyDataFrame, check_mkdir, load_json, search_rgx
from DLBio.kwargs_translator import get_kwargs
from experiments.eval_methods import save_curve_plot
from mdutils.mdutils import MdUtils

# this folder contains a set of json files with the estimated accuracy etc
# values
BASE_FOLDER = 'experiments/exp_8_1/logs'
RGX = r'(CifarJOVFPNet|CifarPyrResNet|CifarResNet|CifarJOVFPNet-RNBasic)_N(\d)_s(\d+).json'
OUT_FOLDER = 'experiments/new_JOV_result_plots/adversarial'
check_mkdir(OUT_FOLDER)

RA_RGX = r'robust_acc_(\d+)'
RE_RGX = r'robust_error_(\d+)'
NC_RGX = r'num_changes_(\d+)'

COLORS = {
    'CifarJOVFPNet': 'g',
    'CifarJOVFPNet-RNBasic': 'g',
    'CifarPyrResNet': 'k',
    'CifarResNet': 'k',
}

RENAME = {
    'CifarJOVFPNet': 'FP-net',
    'CifarPyrResNet': 'PyrBlockNet',
    'CifarResNet': 'ResNet',
    'CifarJOVFPNet-RNBasic': 'FP-net (basic)'
}


def run():
    df = get_dataframe()
    df_grouped, keys_ = get_grouped_dataframe(df)

    create_tables(df_grouped, to_nice_name_num_changes_col, 'num_changes')
    #create_tables(df_grouped, to_nice_name_robust_acc_col, 'robust_acc')
    create_tables(df_grouped, to_nice_name_robust_error_col, 'robust_error')

    create_plots(
        df_grouped, keys_,
        subset=['CifarResNet', 'CifarJOVFPNet-RNBasic'],
        prefix='basic_block_'
    )
    create_plots(
        df_grouped, keys_,
        subset=['CifarPyrResNet', 'CifarJOVFPNet'],
        prefix='pyr_block_'
    )


def create_plots(df_grouped, keys_, subset=None, prefix=''):
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

    for key in keys_:
        save_curve_plot(
            df_grouped, OUT_FOLDER, val_key=key,
            colors_=COLORS, name_key='new_name',
            ylabel=get_ylabel(key),
            pref=prefix,
            legend_order=legend_order
        )


def create_tables(df_grouped, get_nice_name_fcn, table_name):
    # Create nice model names
    new_keys_ = ['Model']
    df_grouped['Model'] = [
        f"{row['new_name'][0]} (N={row['N'][0]})" for _, row in df_grouped.iterrows()
    ]

    # create nice key names for values
    df_grouped = df_grouped.sort_values(by=['N', 'new_name'])
    for key in df_grouped.columns:
        # keys look like ('robust_acc_3', 'min'), ('num_params',''), ...
        new_key = get_nice_name_fcn(key[0])
        if new_key is None or key[1] != 'mean':
            continue

        if new_key == '0/255':
            continue

        # grab the mean value (the dataframe is already grouped)
        df_grouped[new_key] = df_grouped.loc[:, (key, 'mean')]
        new_keys_.append(new_key)

    table_out = df_grouped.loc[:, new_keys_]
    # column is still a tuple ('', some_name)
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


def get_dataframe():
    df = MyDataFrame()
    model_folders = search_rgx(RGX, BASE_FOLDER)
    for file_path in model_folders:
        file_path = join(BASE_FOLDER, file_path)
        df = update(df, file_path)

    df = df.get_df()
    return df


def get_grouped_dataframe(df):
    agg = {'original_acc': ('mean', 'std')}
    keys_ = (
        get_robust_acc_keys(df.columns)
        + get_num_changes_keys(df.columns) +
        get_robust_error_keys(df.columns)
    )
    for key in keys_:
        agg[key] = ('mean', 'std', 'max', 'min')

    df_grouped = df.groupby(
        ['num_params', 'model_type', 'N'], as_index=False
    ).agg(agg)
    df_grouped = df_grouped.sort_values(by=('robust_acc_4', 'mean'))
    df_grouped['new_name'] = [
        RENAME[row['model_type'][0]] for _, row in df_grouped.iterrows()
    ]
    return df_grouped, keys_


def update(df, file_path):
    adv_log = load_json(file_path)
    model_folder = adv_log['folder']
    options = load_json(join(model_folder, 'opt.json'))
    model_kwargs = get_kwargs(options['model_kw'])

    log = load_json(join(model_folder, 'log.json'))

    tmp = {
        'original_acc': max(log['val_acc']) * 100.,
        'seed': options['seed'],
        'model_type': options['model_type'],
        'num_params': adv_log['num_params'],
        'N': model_kwargs['N'][-1],
    }

    for x in get_robust_acc_keys(adv_log.keys()):
        tmp[x] = adv_log[x] * 100.

    for x in get_num_changes_keys(adv_log.keys()):
        tmp[x] = adv_log[x] * 100.

    tmp = write_robust_error(tmp, adv_log)

    df.update(tmp)
    return df


def write_robust_error(tmp_dict, adv_log):
    for x in get_robust_acc_keys(adv_log.keys()):
        id_ = int(re.match(RA_RGX, x).group(1))
        new_key = f'robust_error_{id_}'
        tmp_dict[new_key] = (1. - adv_log[x]) * 100.

    return tmp_dict


def get_robust_acc_keys(keys):
    ra_keys = [x for x in keys if bool(re.match(RA_RGX, x))]
    return ra_keys


def get_robust_error_keys(keys):
    re_keys = [x for x in keys if bool(re.match(RE_RGX, x))]
    return re_keys


def get_num_changes_keys(keys):
    nc_keys = [x for x in keys if bool(re.match(NC_RGX, x))]
    return nc_keys


def get_ylabel(key):
    match = re.match(RA_RGX, key)
    if bool(match):
        eps = int(match.group(1))
        return f'Robust acc. (%) eps={eps}/255'

    match = re.match(RE_RGX, key)
    if bool(match):
        eps = int(match.group(1))
        return f'Robust error (%) eps={eps}/255'

    match = re.match(NC_RGX, key)
    if bool(match):
        eps = int(match.group(1))
        return f'Perc. changed predictions eps={eps}/255'

    raise ValueError(key)


def to_nice_name_robust_acc_col(key):
    match = re.match(RA_RGX, key)
    if not bool(match):
        return None

    eps = int(match.group(1))
    return f'{eps}/255'


def to_nice_name_robust_error_col(key):
    match = re.match(RE_RGX, key)
    if not bool(match):
        return None

    eps = int(match.group(1))
    return f'{eps}/255'


def to_nice_name_num_changes_col(key):
    match = re.match(NC_RGX, key)
    if not bool(match):
        return None

    eps = int(match.group(1))
    return f'{eps}/255'


if __name__ == '__main__':
    run()
