"""
"""
import argparse
import warnings
from os.path import join

import cv2
import matplotlib.pyplot as plt
import model_database_api as api
import numpy as np
import seaborn as sns
from DLBio.helpers import check_mkdir, set_plt_font_size

IM_OUT_FOLDER = join('experiments/new_JOV_result_plots', 'database_results')
DEG_OF_ES_FNAME = 'degree_of_es_activations'
FONT_SIZE = 24


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--num_stacks', type=int, default=None)
    parser.add_argument('--deg_of_es_operations',
                        type=str, nargs='+', default=None
                        )
    parser.add_argument('--save_deg_of_es_actications', action='store_true')

    return parser.parse_args()


def run(options):
    model_name = options.model_name
    if options.num_stacks == 3:
        palette = ["C0", "C1", "C2"]
    elif options.num_stacks == 4:
        palette = ["C0", "C1", "C2", "C3"]
    else:
        raise ValueError(f'Wrong number of layers: {options.num_stacks}')

    operation_keys = options.deg_of_es_operations

    df = get_df(model_name)

    print_num_of_dying_relus(df, model_name)
    plot_angle_distribution(df, model_name, palette=palette)
    plot_entropy_angle_dependency(df, model_name, palette=palette)
    plot_degree_of_endstopping(df, model_name, operation_keys)

    if options.save_deg_of_es_actications:
        save_degree_of_end_stopping_activations(df, model_name, operation_keys)


def get_df(model_name):
    df = api.get_dataframe([
        model_name
    ])

    df['position'] = np.array(df['position']).astype('float32')
    return df


def plot_angle_distribution(df, model_name, palette=["C0", "C1", "C2"]):
    sns_kwargs = {
        'stat': 'probability',
        'multiple': 'dodge',
        'palette': palette,
        'common_norm': False,
        'height': 15,
    }
    set_plt_font_size(FONT_SIZE)
    tmp = df[df['block_id'] == 'fp_block'].copy()
    if tmp.shape[0] == 0:
        print('No fp-blocks')
        return

    sns.displot(tmp, x="kd_angles", hue="depth", **sns_kwargs)
    plt.xlabel('gamma')

    out_name = join(IM_OUT_FOLDER, model_name, f'{model_name}_angle_dist')
    check_mkdir(out_name + '.png')

    plt.savefig(out_name + '.png')
    plt.savefig(out_name + '.pdf')
    plt.close()


def plot_entropy_angle_dependency(df, model_name, palette=["C0", "C1", "C2"]):
    set_plt_font_size(FONT_SIZE)

    # note number of fmaps in upper equal to number of fmaps in mult
    # there is a direct link to a fmap in upper and in mult
    # however this link does not exist anymore in lower

    plt.figure(figsize=(12, 12))
    tmp = df[df['block_id'] == 'fp_block'].copy()
    if tmp.shape[0] == 0:
        print('No fp-blocks')
        return

    tmp = tmp[tmp['ent_per_fmap-mult'] >= 0.]
    assert tmp.shape[0] > 0

    # remove dying ReLUs
    num_before = tmp.shape[0]
    tmp = tmp[tmp['act_per_fmap-upper'] > 1e-6]
    num_after = tmp.shape[0]
    print(f'removed {num_before - num_after} dying ReLU values')

    sns.scatterplot(
        data=tmp, x='kd_angles', y='ent_per_fmap-mult',
                    hue='depth', palette=palette
    )
    plt.xlabel('gamma')
    plt.ylabel('entropy (multiplication)')

    out_name = join(IM_OUT_FOLDER, model_name,
                    f'{model_name}_entropy_vs_angle')
    check_mkdir(out_name + '.png')

    plt.savefig(out_name + '.png')
    plt.savefig(out_name + '.pdf')
    plt.close()


def plot_degree_of_endstopping(df, model_name, op_keys):
    tmp = df[df['position'] == 0].copy()
    tmp = api.get_degree_of_es_analysis(tmp, op_keys)

    sns_kwargs = {
        'stat': "probability",
        'binwidth': .1,
        'height': 12,
        'common_norm': True,
        'palette': ["C0", "C1", "C2"],
        'hue_order': ['1D & 2D', '0D', 'silent'],
        'multiple': "stack",

    }

    set_plt_font_size(FONT_SIZE)
    post = ''
    ylim = .7
    if True:
        for key in op_keys:
            # some operations may have a lot of nans, due to different feature
            # map sizes within one block -> remove those values first:
            num_before = tmp.shape[0]
            where = np.logical_not(np.isnan(tmp[key + '-id_ratio' + post]))
            tmp2 = tmp.copy()[where]
            num_after = tmp2.shape[0]
            print(f'{key}: removed {num_before - num_after} nan values')

            plt.figure(figsize=(12, 12))
            g = sns.displot(
                data=tmp2,
                x=key + '-id_ratio' + post,
                hue=key + '-id2_0' + post,
                **sns_kwargs
            )

            plt.ylim([0, ylim])
            plt.xlim([-1., 1.2])

            new_title = ''
            g._legend.set_title(new_title)

            plt.xlabel('degree of endstopping')
            plt.ylabel('# neurons (%)')
            plt.xticks(np.arange(-.6, 1.2, step=0.2))

            out_name = join(IM_OUT_FOLDER, model_name,
                            f'{model_name}_degree_of_es_' + key)
            check_mkdir(out_name + '.png')

            plt.savefig(out_name + '.png')
            plt.savefig(out_name + '.pdf')
            plt.close()


def save_degree_of_end_stopping_activations(df, model_name, op_keys):
    tmp = df[df['position'] == 0].copy()
    for idx, row in tmp.iterrows():
        for operation in op_keys:
            act = api.get_deg_of_es_activation(row, operation)
            if act is None:
                continue

            # TODO: check pos-neg order is right
            for i, key in enumerate(['pos', 'neg']):
                im_out_path = join(
                    IM_OUT_FOLDER, model_name, DEG_OF_ES_FNAME, operation,
                    str(idx).zfill(3) + f'_{key}.png'
                )
                check_mkdir(im_out_path)
                out = (255. * act[i, ...].copy()).astype('uint8')
                assert cv2.imwrite(im_out_path, out)


def print_num_of_dying_relus(df, model_name, eps=1e-6):
    tmp = df.copy()
    # TODO: at the moment: only fp-block analysis
    tmp = tmp[tmp['block_id'] == 'fp_block']
    if tmp.shape[0] == 0:
        warnings.warn('No dying relu analysis done.')
        return
    tmp = tmp[['block_id', 'depth', 'act_per_fmap-upper']]

    # report number of dying ReLUs in percent (mean over indicator function)
    tmp['is_zero'] = np.array(tmp['act_per_fmap-upper'] < eps).astype('int')
    tmp = tmp.groupby(['block_id', 'depth']).mean()

    out_path = join(IM_OUT_FOLDER, model_name, f'{model_name}_dying_relus.csv')
    check_mkdir(out_path)
    tmp.to_csv(
        out_path
    )


if __name__ == '__main__':
    OPTIONS = get_options()
    run(OPTIONS)
