from os.path import isfile, join

import helpers
import matplotlib.pyplot as plt
from datasets.data_getter import get_data_loaders
from DLBio import pt_training
from DLBio.helpers import check_mkdir, load_json, set_plt_font_size
from DLBio.kwargs_translator import get_kwargs
from DLBio.pytorch_helpers import ActivationGetter, get_device
from models.model_evaluation.loggers import MeanAbsActivationLogger
from train_interfaces import get_interface
import numpy as np


def save_curve_plot(df, out_folder, val_key='min_val_er', colors_=None, pref='', ylim=None, name_key='model_type', ylabel=None, legend_order=None):
    set_plt_font_size(26)
    _, ax = plt.subplots(1, figsize=(15, 15))
    for key in set(df['model_type']):
        tmp = df[df['model_type'] == key].copy()
        tmp = tmp.sort_values(by='N')
        x = np.array(tmp['num_params'] // 1000)
        y = np.array(tmp[(val_key, 'mean')])
        z = np.array(tmp[(val_key, 'std')])
        vmin = np.array(tmp[(val_key, 'min')])
        vmax = np.array(tmp[(val_key, 'max')])

        name = tmp.iloc[0, :][name_key][0]

        # TODO set hue order

        if colors_ is not None:
            plt.errorbar(
                x, y, z, linewidth=4, label=name,
                color=colors_[key], marker='d', markersize=14,
            )
            plt.fill_between(
                x, vmin, vmax,
                color=colors_[key], alpha=0.2
            )
        else:
            plt.errorbar(
                x, y, z, linewidth=4, label=name,
                marker='d', markersize=14,
            )
            plt.fill_between(
                x, vmin, vmax, alpha=0.2
            )

    plt.xlabel('Number of Parameters (k)')
    if ylabel is None:
        plt.ylabel(val_key)
    else:
        plt.ylabel(ylabel)

    plt.legend()

    if legend_order is not None:
        # sort legends according to
        handles, labels = ax.get_legend_handles_labels()
        assert len(labels) == len(legend_order), f'{legend_order} vs. {labels}'
        tmp = [labels.index(x) for x in legend_order]
        new_handles = [handles[i] for i in tmp]
        # sort both labels and handles by labels
        ax.legend(new_handles, legend_order)

    plt.grid()
    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout()
    plt.savefig(join(out_folder, pref + val_key + '.png'))
    plt.savefig(join(out_folder, pref + val_key + '.pdf'))
    plt.close()


def create_dr_data(folder, out_folder, *, used_blocks):
    if not isfile(join(folder, 'model.pt')):
        return None

    model = helpers.load_model(
        join(folder, 'opt.json'), get_device(),
        new_model_path=join(folder, 'model.pt')
    )
    mod_list = helpers.get_ordered_module_list(
        model, batch_input_size=(1, 3, 32, 32),
        device=get_device(),
        use_only=used_blocks
    )

    logger_dict = setup_loggers(mod_list)

    dataloader = get_data_loaders(
        'cifar_10_subset',
        batch_size=128,
        num_workers=0
    )['val']

    log_file = join(out_folder, 'model_eval', 'test_log.json')
    check_mkdir(log_file)

    train_interface = get_interface(
        'Logging', model, get_device(),
        pt_training.get_printer(50, log_file=log_file),
        logger_dict=logger_dict
    )
    optim = pt_training.get_optimizer('SGD', model.parameters(), 1.)

    training = pt_training.Training(
        optim, None, train_interface, printer=train_interface.printer,
        val_data_loader=dataloader, validation_only=True
    )
    training(1)

    opt = load_json(join(folder, 'opt.json'))
    model_kw = get_kwargs(opt['model_kw'])
    return {
        'model_type': opt['model_type'],
        'seed': opt['seed'],
        'N': model_kw['N'][0],
    }


def setup_loggers(module_list):
    logger_dict = {}
    for depth, module in module_list:
        key = f'{module._get_name()}_d{depth}'

        upper = module.block_with_shortcut.block.upper
        act_getter = ActivationGetter(upper)

        logger = MeanAbsActivationLogger(act_getter)
        logger_dict[key] = logger

    return logger_dict
