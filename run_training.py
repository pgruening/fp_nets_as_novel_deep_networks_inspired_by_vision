import argparse
import json
import warnings
from os.path import isdir, join

import torch
import torch.nn as nn
from DLBio import pt_training
from DLBio.helpers import check_mkdir, copy_source, load_json
from DLBio.kwargs_translator import get_kwargs
from DLBio.pt_train_printer import Printer
from DLBio.pytorch_helpers import get_device, get_num_params, save_options

import config
from helpers import get_data_loaders, load_model
from train_interfaces import get_interface


def get_options():
    parser = argparse.ArgumentParser()

    # train hyperparams
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--wd', type=float, default=None)
    parser.add_argument('--mom', type=float, default=None)
    parser.add_argument('--cs', type=int, default=None)
    parser.add_argument('--bs', type=int, default=None)
    parser.add_argument('--opt', type=str, default=None)
    parser.add_argument('--opt_kw', type=str, default=None)
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--folder', type=str, default=None)
    parser.add_argument('--nw', type=int, default=0)

    parser.add_argument('--train_interface', type=str, default=None)

    # model / ds specific params
    parser.add_argument('--in_dim', type=int, default=None)
    parser.add_argument('--out_dim', type=int, default=None)
    parser.add_argument('--model_type', type=str, default=None)
    parser.add_argument('--comment', type=str, default=None)

    # a string that defines parameters of specific models
    # look at DLBio's kwargs_translator for more information
    parser.add_argument('--model_kw', type=str, default=None)

    # scheduling
    parser.add_argument('--epochs', type=int, default=-1)
    parser.add_argument('--sched_kw', type=str, default=None)
    parser.add_argument('--lr_steps', type=int, default=-1)
    parser.add_argument('--fixed_steps', nargs='+', default=None)
    parser.add_argument('--batch_scheduler', nargs='+', default=None)

    # dataset
    parser.add_argument('--split_index', type=int, default=-1)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--ds_kwargs', type=str, default=None)

    # model saving
    parser.add_argument('--sv_int', type=int, default=0)
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--es_get_min', action='store_true')
    parser.add_argument('--es_metric', type=str, default=None)
    parser.add_argument('--es_epoch_thres', type=int, default=None)

    # overwrite existing folder
    parser.add_argument('--do_overwrite', action='store_true')

    # data parallel for imagenet
    parser.add_argument('--use_data_parallel', action='store_true')

    # resume training an already trained model
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)

    # log which operations take up training time
    parser.add_argument('--do_time_logging', action='store_true')

    parser.add_argument('--print_freq', type=int,
                        default=config.PRINT_FREQUENCY)

    return parser.parse_args()


def run(options):
    if options.device is not None:
        pt_training.set_device(options.device)

    device = get_device()

    pt_training.set_random_seed(options.seed)

    folder = options.folder

    if not options.do_overwrite:
        if abort_due_to_overwrite_safety(folder):
            print('Process aborted.')
            return

    check_mkdir(folder)

    if options.comment is None:
        print('You forgot to add a comment to your experiment. Please add something!')
        options.comment = input('Comment: ')

    save_options(join(
        folder, 'opt.json'),
        options)

    copy_source(folder, do_not_copy_folders=config.DO_NOT_COPY)

    _train_model(options, folder, device)


def _train_model(options, folder, device):
    model_out = join(folder, 'model.pt')
    log_file = join(folder, 'log.json')
    check_mkdir(log_file)

    if options.do_time_logging:
        time_log = join(folder, 'time.json')
        time_log_printer = Printer(-1, time_log, resume=options.resume)
    else:
        time_log_printer = None

    model_kwargs = get_kwargs(options.model_kw)

    if options.resume:
        # not implemented
        assert options.train_interface != 'NoDRClassification'

        log = load_json(log_file)
        start_epoch = log['epoch'][-1]
        options.model_path = model_out
        model = load_model(
            options, device, strict=True,
            from_par_gpu=options.use_data_parallel
        )
    else:
        start_epoch = 0
        model = load_model(options, device)

    if options.use_data_parallel:
        model = nn.DataParallel(model)
        print(f'Using {torch.cuda.device_count()} devices.')

    write_model_specs(folder, model)

    optimizer = get_optimizer(options, model)
    scheduler, batch_scheduler = setup_scheduling(options, optimizer)
    print(f'ds_{options.dataset}')

    data_loaders = get_data_loaders(options)

    if options.early_stopping:
        # model can be saved with wrong weights
        assert options.train_interface != 'NoDRClassification'

        get_max = not options.es_get_min
        if not get_max:
            warnings.warn('Early stopping on min-value')

        assert options.sv_int == -1
        if options.es_epoch_thres is None:
            es_epoch_thres = options.epochs
        else:
            es_epoch_thres = options.es_epoch_thres
        early_stopping = pt_training.EarlyStopping(
            options.es_metric, get_max=True, epoch_thres=es_epoch_thres
        )
    else:
        early_stopping = None

    train_interface = get_interface(
        options.train_interface,
        model, device, Printer(options.print_freq, log_file),
        num_epochs=options.epochs
    )

    training = pt_training.Training(
        optimizer, data_loaders['train'], train_interface,
        scheduler=scheduler, printer=train_interface.printer,
        save_path=model_out, save_steps=options.sv_int,
        val_data_loader=data_loaders['val'],
        early_stopping=early_stopping,
        save_state_dict=True,
        test_data_loader=data_loaders['test'],
        batch_scheduler=batch_scheduler,
        start_epoch=start_epoch,
        time_log_printer=time_log_printer
    )

    training(options.epochs)


def write_model_specs(folder, model):

    print(f'#train params: {get_num_params(model, True):,}')

    with open(join(folder, 'model_specs.json'), 'w') as file:
        json.dump({
            'num_trainable': float(get_num_params(model, True)),
            'num_params': float(get_num_params(model, False))
        }, file)


def get_optimizer(options, model):
    optim_kwargs = get_kwargs(options.opt_kw)
    # TODO:
    # translate optime kwargs to proper dictionary. E.g. {'alpha':['0.99']} to
    # {'alpha': 0.99 }

    optimizer = pt_training.get_optimizer(
        options.opt, model.parameters(),
        options.lr,
        momentum=options.mom,
        weight_decay=options.wd,
        **optim_kwargs
    )

    # TODO: if some kw (maybe "use_sam") exists and is set to True, wrap
    # the base optimizer in SAM optimizer
    # TODO: copy the SAM code somewhere, don't forget to add the link to
    # the original code in the comments
    # TODO: translate any sam-specific kwargs from optim_kwargs

    return optimizer


def setup_scheduling(options, optimizer):
    sched_kwargs = get_kwargs(options.sched_kw)
    sched_type = sched_kwargs.get('sched_type', [None])[0]
    if sched_type is not None:
        # TODO: if sched_type == 'Foo': scheduler = 1; elif ...
        scheduler = -1
        raise NotImplementedError

    else:
        if options.lr_steps > 0 or options.fixed_steps is not None:
            scheduler = pt_training.get_scheduler(
                options.lr_steps, options.epochs, optimizer,
                fixed_steps=options.fixed_steps
            )
        else:
            print('no scheduling used')
            scheduler = None

    # it is possible to have batch scheduling and global scheduling
    if options.batch_scheduler is not None:
        batch_scheduler = pt_training.BatchScheduler(
            options.batch_scheduler,
            optimizer, options.lr, False, options.epochs
        )
        print('USING BATCH SCHEDULING')
    else:
        batch_scheduler = None

    return scheduler, batch_scheduler


def abort_due_to_overwrite_safety(folder):
    abort = False
    if isdir(folder):
        print(f'The folder {folder} already exists. Overwrite it?')
        print('Y: overwrite')
        print('Any key: stop')
        char = input('Overwrite?')
        if char != 'Y':
            abort = True

    return abort


if __name__ == "__main__":
    OPTIONS = get_options()
    run(OPTIONS)
