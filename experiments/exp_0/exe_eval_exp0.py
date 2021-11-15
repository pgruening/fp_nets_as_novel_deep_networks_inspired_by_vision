"""
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

from DLBio.helpers import MyDataFrame, load_json, search_rgx
from DLBio.kwargs_translator import get_kwargs
import matplotlib.pyplot as plt
from experiments.eval_methods import save_curve_plot

BASE_FOLDER = 'experiments/exp_0/exp_data/trained_models'
RGX = r'(CifarJOVFPNet|CifarPyrResNet|CifarResNet)_N(\d)_s(\d+)'

IM_OUT_FOLDER = 'experiments/exp_0'

AGG = {
    'min_val_er': ('mean', 'std', 'min', 'max'),
    'last_val_er': ('mean', 'std'),
}

COLORS = {
    'CifarJOVFPNet': 'g',
    'CifarPyrResNet': 'k',
    'CifarResNet': 'c'
}
RENAME = {
    'CifarJOVFPNet': 'FP-net',
    'CifarResNet': 'ResNet',
    'CifarPyrResNet': 'PyrBlockNet',
}


def run():
    folders_ = search_rgx(RGX, BASE_FOLDER)
    assert folders_

    df = MyDataFrame()
    for folder in folders_:
        df = update(df, join(BASE_FOLDER, folder))

    df = df.get_df().sort_values(by=['N', 'model_type'])
    df.to_excel(join(IM_OUT_FOLDER, 'all_results.xlsx'))
    df.to_csv(join(IM_OUT_FOLDER, 'all_results.csv'))
    print(df.head())

    df_grouped = df.groupby(by=['model_type', 'N', 'num_params'], as_index=False).agg(
        AGG).sort_values(by=('min_val_er', 'mean'))
    df_grouped['new_name'] = [
        RENAME[row['model_type'][0]] for _, row in df_grouped.iterrows()
    ]

    print(df_grouped)
    df_grouped.to_excel(join(IM_OUT_FOLDER, 'grouped_results.xlsx'))

    save_curve_plot(
        df_grouped, IM_OUT_FOLDER, val_key='min_val_er',
        colors_=COLORS, name_key='new_name',
        ylabel='Cifar-10 test-error (%)'
    )

    resnet_results = df_grouped[df_grouped['model_type']
                                == 'CifarResNet'].copy().sort_values('N')
    print(resnet_results)


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
