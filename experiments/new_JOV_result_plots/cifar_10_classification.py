"""
Creates the Cifar-10 classification plots for the JOV paper and writes them
to OUT_FOLDER

Numbers from Ydelbayev:

Name 	# layers 	# params 	Test err(paper) 	Test err(this impl.)
ResNet20 	20 	0.27M 	8.75% 	8.27%
ResNet32 	32 	0.46M 	7.51% 	7.37%
ResNet44 	44 	0.66M 	7.17% 	6.90%
ResNet56 	56 	0.85M 	6.97% 	6.61%

     model_type min_val_er                 last_val_er           num_params     N
                      mean       std   min        mean       std      first first
8   CifarResNet      8.010  0.202485  7.85       8.132  0.203273   269722.0     3
9   CifarResNet      7.598  0.421331  7.17       7.674  0.474268   464154.0     5
10  CifarResNet      7.108  0.283320  6.92       7.306  0.350542   658586.0     7
11  CifarResNet      6.808  0.240354  6.60       6.934  0.233517   853018.0     9

# The parameter numbers check out.
# Our min N=3 is better by 0.5
# Our min N=5 is better by 0.2
# N=7 and N=9 results are identical.
"""

from os.path import join

from DLBio.helpers import MyDataFrame, load_json, search_rgx, check_mkdir
from DLBio.kwargs_translator import get_kwargs
import matplotlib.pyplot as plt
from experiments.eval_methods import save_curve_plot

BASE_FOLDERS = [
    'experiments/exp_0/exp_data/trained_models',
    'experiments/exp_4/exp_data/trained_models'
]
RGXS = [
    r'(CifarJOVFPNet|CifarPyrResNet|CifarResNet)_N(\d)_s(\d+)',
    r'(CifarJOVFPNet-RNBasic)_N(\d)_s(\d+)'
]

OUT_FOLDER = 'experiments/new_JOV_result_plots/cifar10_classification'
check_mkdir(OUT_FOLDER)

AGG = {
    'min_val_er': ('mean', 'std', 'min', 'max'),
    'last_val_er': ('mean', 'std'),
}

# black for baseline, green for FP-nets
COLORS = {
    'CifarJOVFPNet': 'g',
    'CifarJOVFPNet-RNBasic': 'g',
    'CifarPyrResNet': 'k',
    'CifarResNet': 'k',
}
RENAME = {
    'CifarJOVFPNet': 'FP-net',
    'CifarJOVFPNet-RNBasic': 'FP-net (basic)',
    'CifarResNet': 'ResNet',
    'CifarPyrResNet': 'PyrBlockNet',
}


def run():
    folders_ = get_folders()
    df = get_dataframe(folders_)
    df_grouped = group_dataframe(df)

    # create_plot(df_grouped)
    create_plot(
        df_grouped, subset=['CifarPyrResNet', 'CifarJOVFPNet'],
        prefix='pyr_block_'
    )
    create_plot(
        df_grouped, subset=['CifarResNet', 'CifarJOVFPNet-RNBasic'],
        prefix='basic_block_'
    )


def get_folders():
    out = []
    for rgx, base_folder in zip(RGXS, BASE_FOLDERS):
        folders_ = search_rgx(rgx, base_folder)
        assert folders_
        out += [join(base_folder, x) for x in folders_]

    return out


def get_dataframe(folders_):
    print('Creating Dataframe...')
    df = MyDataFrame()
    for folder in folders_:
        df = update(df, folder)

    df = df.get_df().sort_values(by=['N', 'model_type'])
    df.to_excel(join(OUT_FOLDER, 'classification_results.xlsx'))
    df.to_csv(join(OUT_FOLDER, 'classification_results.csv'))

    print(df.head())
    return df


def group_dataframe(df):
    df_grouped = df.groupby(by=['model_type', 'N', 'num_params'], as_index=False).agg(
        AGG).sort_values(by=('min_val_er', 'mean'))

    # add names you want to see in the plot's legend
    df_grouped['new_name'] = [
        RENAME[row['model_type'][0]] for _, row in df_grouped.iterrows()
    ]

    df_grouped.to_excel(join(OUT_FOLDER, 'grouped_results.xlsx'))

    print(df_grouped)
    return df_grouped


def create_plot(df_grouped, subset=None, prefix=''):
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

    save_curve_plot(
        df_grouped, OUT_FOLDER, val_key='min_val_er',
        colors_=COLORS, name_key='new_name',
        ylabel='Cifar-10 test-error (%)', pref=prefix,
        legend_order=legend_order
    )


def update(df, folder):
    log = load_json(join(folder, 'log.json'))
    assert log is not None

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
