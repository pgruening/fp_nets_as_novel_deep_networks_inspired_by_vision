"""
For a specific model, this module creates a folder structure of files 
containing specific metrics about the model, such as gamma-angles or entropy 
statistics. The database query module creates an API to access these data.

The idea of this approach is to not bother with specific model architectures
and their naming conventions. I.e., we don't want to bother whether in one
model each block is called model.block_1 or model.conv1 and so on.
To achieve this, the entire model-graph is always traversed. A node of this
graph can be a high level structure, e.g. a stack or a block, or a low level
operation such as a ReLU. BlockManagers filter out the interesting nodes, the
ones that are actual basic- or fp-blocks and evaluate them. If you want to 
create a database for a model with a new type of block, implement the 
IBlockManager abstract class.

The folder structure of the database looks like this:

* [model]
  * [position]-[block_type] (see get_block_folder_str)
    *  a set of npy files and json files
 
Files contained:

* meta.json: all metrics of the respective block. Naming convention for logged values: `[layer]-[metric]`.

* [operation]-activation-examples.npy: the activations of a block for a batch of images

# FP-block specific:   
* filters.npy: np.stack([dw1, dw2], 0) both depth wise filters of an fp-block

* lin_comb.npy: weights for upper linear combination

The position counter: always starts at 0. If a block is detected by the 
BlockManager this counter is increased. It is meant to count all the blocks and 
create a unique id for all blocks in a model. Note that this counting 
depends strongly on the BlockManagers you use. If, e.g., you only use an 
FP-block manager all other block, like the PyramidBlock are simply not counted.
The model graph is entirely traversed, including all operations of a model,
e.g. ReLUs, single convolution, or batchnorm layers. The block managers filter
through this list of graph nodes and only evaluate the interesting block nodes.

Naming convention for logger values:
[name of block];[name of metric];[position in model]/[name of operation]
Example:
fp_block;ent_per_fmap;00/upper

"""

import argparse
import json
import re
import shutil
import warnings
from os.path import join

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from DLBio import pt_training
from DLBio.helpers import check_mkdir, load_json, search_rgx
from DLBio.pt_train_printer import Printer
from DLBio.pytorch_helpers import ActivationGetter, cuda_to_numpy, get_device
from torchvision.models.resnet import Bottleneck
from tqdm import tqdm

from datasets.data_getter import get_data_loaders
from helpers import load_model, walk
from models import conv_blocks as cb
from models.imagenet_legacy_models import fp_blocks
from train_interfaces import Logger, get_interface

DATABASE_PATH = 'experiments/new_JOV_result_plots/model_database'
LOG_JSON_NAME = 'loggers.json'
META_NAME = 'meta.json'
BLOCK_INFO_NAME = 'position_info.json'
TRAIN_OPT_NAME = 'train_options.json'

BATCH_SIZE = 100
NUM_WORKERS = 0
Z = .5  # used to determine curvature

# TODO: no global variables
CIFAR_MODE = True  # evaluate a model trained on Cifar-10 (or imagenet?)
ENTROPY_DATASET = None


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', type=str)
    parser.add_argument('--bs', type=int, default=BATCH_SIZE)
    parser.add_argument('--nw', type=int, default=NUM_WORKERS)
    parser.add_argument('--is_imagenet_model', action='store_true')
    parser.add_argument('--from_par_gpu', action='store_true')
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--depth_offset', type=int, default=None)
    return parser.parse_args()


def main(options):
    global CIFAR_MODE
    global ENTROPY_DATASET

    if options.is_imagenet_model:
        CIFAR_MODE = False
        ENTROPY_DATASET = 'im_net_entropy'
    else:
        CIFAR_MODE = True
        ENTROPY_DATASET = 'cifar_10_validation_subset'

    model, model_folder, _ = setup(options)

    copy_training_options(options.model_folder, model_folder)

    block_managers = get_block_managers()

    # run tasks
    if True:
        run_logging(
            model, model_folder, block_managers,
            dataset=ENTROPY_DATASET,
            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
        )
        save_logging_data_to_meta_files(model_folder)

    if True:
        save_block_metrics_to_meta_files(model, model_folder, block_managers)

    # determine degree of end-stopping for first block
    if True:
        if not CIFAR_MODE:
            blocks_to_determine_deg_of_es = [
                get_ith_block(0, model, model_folder, block_managers)
            ]
        else:
            blocks_to_determine_deg_of_es = [
                get_ith_block(0, model, model_folder, block_managers)
            ]
        determine_degree_of_endstopping(
            model, model_folder, block_managers,
            blocks_to_look_at=blocks_to_determine_deg_of_es
        )

    # save activations
    if True:
        save_activations(
            model, model_folder, block_managers, None
        )
    if True:
        create_block_info_file(
            model_folder, block_managers, options.depth_offset
        )

# --------------------------------------------------------------
# ------------------ HIGH-LEVEL TASKS --------------------------
# --------------------------------------------------------------


def get_block_managers():
    if CIFAR_MODE:
        block_managers = [
            FPBlockManager([
                EntropyPerFeatureMapLogger,
                ActivationPerFeatureMapLogger
            ]),
            PyrBlockManager([
                EntropyPerFeatureMapLogger,
                ActivationPerFeatureMapLogger
            ]),

        ]
    else:
        block_managers = [
            FPBlockManager([
                EntropyPerFeatureMapLogger,
                ActivationPerFeatureMapLogger
            ]),
            FPBlockManagerLegacy([
                EntropyPerFeatureMapLogger,
                ActivationPerFeatureMapLogger
            ]),
            BottleneckManager([
                EntropyPerFeatureMapLogger,
                ActivationPerFeatureMapLogger
            ]),
        ]

    return block_managers


def run_logging(model, model_folder, block_managers, *, dataset, batch_size, num_workers=0):
    """Instantiates loggers depending on the logger getters. Runs on epoch
    for the entropy dataset and records a json file logger.json in the folder
    [DATABASE_PATH]/[model_folder].

    Parameters
    ----------
    model: nn.Module
        A pytorch neural network
    model_folder : str
        determines the database folder for the model
    block_managers : list of BlockManagers
        each block managers can contain one or several logger class objects
        that instantiate loggers. Loggers are function that monitor the
        activation of a block and save the results to a json file.
    dataset: str
        which dataset is used to log the activations. ONLY THE VALIDATION
        DATASET IS USED!
    batch_size: int
    num_workers: int

    """
    print('Starting Featuremap logging')
    device = get_device()
    log_path = join(DATABASE_PATH, model_folder, LOG_JSON_NAME)
    check_mkdir(log_path)

    model = model.to(device)

    # all loggers are created here –not when instantiating the train interface.
    loggers = get_all_loggers_for_model(
        model, block_managers, blocks_to_look_at=None
    )
    assert loggers, 'No blocks found!'

    # LoggerCreator is a simple Adapter:
    # train interface calls the creator like this:
    # loggers = logger_creator()(model)

    train_interface = get_interface(
        'ClassAndLogging', model, device, Printer(10, log_path),
        logger_dict=loggers
    )

    # TODO: move entropy dataset to arguments
    data_loader = get_data_loaders(
        dataset, batch_size=batch_size, num_workers=num_workers
    )['val']

    fake_optimizer = pt_training.get_optimizer('SGD', model.parameters(), .1)
    training = pt_training.Training(
        fake_optimizer, None, train_interface,
        scheduler=None, printer=train_interface.printer,
        save_path=None, save_steps=-1,
        val_data_loader=data_loader,
        validation_only=True
    )
    # run for one epoch
    training(1)


def copy_training_options(folder_with_model, model_folder):
    """Copy the opt.json from the training folder of the model
    to the new database folder

    Parameters
    ----------
    folder_with_model : str
        folder with model.pt, log.json, opt.json, created during training
    model_folder : str
        folder name in the database
    """
    options_path = join(folder_with_model, 'opt.json')
    new_path = join(DATABASE_PATH, model_folder, TRAIN_OPT_NAME)
    check_mkdir(new_path)
    shutil.copy(options_path, new_path)


def save_activations(model, model_folder, block_managers, blocks_to_look_at=None):
    """
    Grab an example batch of images from _get_example_batch and save the
    activations of several blocks as a numpy file.
    Iterates to possibly all blocks of a model. If a block has a block
    manager, the save activations function is called.

    Parameters
    ----------
    model_folder : str
    model : nn.Module
    block_managers : list of IBlockManager
    blocks_to_look_at : list of nn.Module, optional
        by default None, look at all modules of a model
    """
    print('Saving activations...')

    kwargs = {
        'images': _get_example_batch(ENTROPY_DATASET).to(get_device())
    }

    block_functions = [
        mngr.save_activation_of_examples for mngr in block_managers
    ]
    _iterate_blocks(
        model_folder, model, block_functions,
        blocks_to_look_at=blocks_to_look_at,
        **kwargs
    )


def determine_degree_of_endstopping(model, model_folder, block_managers, blocks_to_look_at=None):
    """
    For a list of blocks:
    Determines the degree of end-stopping and saves these values to the 
    '[folder]/meta.json' file and saves the activations to an npy file:
    '[folder]/[prefix]-deg_of_es_activations.npy' (see _extract_deg_of_es)

     Parameters
    ----------
    model: nn.Module
        A pytorch neural network
    model_folder : str
    model : nn.Module
    block_managers : list of IBlockManager
    blocks_to_look_at : list of nn.Module, optional
        by default None, look at all modules of a model
    """
    print('Degree of end-stopping...')

    block_functions = [
        mngr.extract_deg_of_es for mngr in block_managers
    ]
    _iterate_blocks(
        model_folder, model, block_functions, blocks_to_look_at=blocks_to_look_at
    )


def save_block_metrics_to_meta_files(model, model_folder, block_managers, blocks_to_look_at=None):
    """
    Extracts metrics derived from the model's weights and saves them to the
    respective meta.json files. The logger meta data are extracted and saved
    in the save_logging_data_to_meta_files

    Iterate al blocks of a model and call the respective 'extract_meta'
    function from the block manager. These functions differe strongly between
    block managers, e.g., the FP-block extracts angle metrics and so on. These
    metrics do not even exist for basic blocks.


    Parameters
    ----------
    model: nn.Module
        A pytorch neural network
    model_folder : str
    block_managers : list of IBlockManager
    blocks_to_look_at : list of nn.Module, optional
        by default None, look at all modules of a model
    """
    print('Block metrics...')
    # block_manager_.extract_meta(block, current_pos, parent_folder)
    block_functions = [
        mngr.extract_meta for mngr in block_managers
    ]
    _iterate_blocks(
        model_folder, model, block_functions, blocks_to_look_at=blocks_to_look_at
    )


def get_ith_block(index, model, model_folder, block_managers, blocks_to_look_at=None):

    block_functions = [
        mngr.save_blocks_to_tmp_var for mngr in block_managers
    ]

    _iterate_blocks(
        model_folder, model, block_functions, blocks_to_look_at=blocks_to_look_at
    )

    merged_data = dict()
    for mngr in block_managers:
        if mngr.tmp is None:
            continue

        for key, value in mngr.tmp.items():
            assert key not in merged_data.keys()
            merged_data[key] = value

        # delete tmp variable
        mngr.tmp = None

    return merged_data[index]


def _iterate_blocks(model_folder, model, block_functions, blocks_to_look_at=None, **kwargs):
    """Iterate all nodes of a model-graph and apply all functions in 
    block_functions to specific blocks. 'block_functions' are functions of 
    BlockManagers and need to follow a specific I/O scheme:
    fcn(model, block, current_pos, parent_folder) -> return Boolean

    Parameters
    ----------
    model_folder : str
    model : nn.Module
    block_functions : list of methods belonging to block_managers
        the function needs to expect this input:
        model, block, current_pos, parent_folder and return a boolean
        True means that the function was applied successfully to this block
    blocks_to_look_at : list of nn.Modules, optional
        by default None which means: look at all modules.

    """
    parent_folder = join(DATABASE_PATH, model_folder)
    if blocks_to_look_at is None:
        # get all children of a model graph
        blocks_to_look_at = walk(model)

    # TODO: why is the position computation needed here?
    current_pos = 0
    for block in tqdm(blocks_to_look_at):
        do_increment = False
        for fcn in block_functions:
            # extraction is done within the function 'fcn', only a boolean is
            # returned to notify whether the counter needs to be increased
            update_happened = fcn(
                model, block, current_pos, parent_folder, **kwargs
            )
            assert update_happened is not None

            if update_happened:
                do_increment = True

        if do_increment:
            current_pos += 1


def get_all_loggers_for_model(model, block_managers, blocks_to_look_at=None):
    """Iterate all blocks of a model and create loggers for all blocks
    depending on the block managers.

    Parameters
    ----------
    model : nn.Module
    block_managers : list of IBlockManager
        block_managers create Logger objects that compute activation statistics
        of a block.
    blocks_to_look_at : list of nn.Module, optional
        by default None, all blocks are considered

    Returns
    -------
    dictionary {str : Logger}
        [description]

    comment:
    THIS IS ESSENTIALLY THE SAME BLOCK ITERATION AS IN _iterate_blocks
    in a next refactoring iteration this should be changed.

    # TODO: write test -> input a model with known number of blocks, is the 
    position counter correct.

    """
    if blocks_to_look_at is None:
        blocks_to_look_at = walk(model)

    loggers = dict()

    current_pos = 0
    for block in tqdm(blocks_to_look_at):
        do_increment = False
        for block_manager_ in block_managers:
            tmp = block_manager_.get_loggers_if_condition_met(
                block, current_pos
            )
            if tmp is not None:
                assert not do_increment, 'Block already found'
                loggers.update(tmp)
                do_increment = True

        if do_increment:
            current_pos += 1

    return loggers


def get_json_meta_key(metric, operation):
    return f'{metric}-{operation}'


def save_logging_data_to_meta_files(model_folder):
    """ For each block at a specific positions create a folder and save a
    meta.json file there. If a file already exists, the file is first wiped
    entirely.

    The dictionary of the meta file is usually a {str : list of floats} pair.

    Parameters
    ----------
    model_folder : str
        Used to determine the parent folder.
    """
    def clear_all_json_files(log, model_folder):
        def clear_json(json_path, *args):
            update_json(json_path, dict(), do_rewrite=True)

        _iterate_logger(log, model_folder, clear_json)

    def save_new_logger_values(log, model_folder):
        def save_new_value(json_path, value, tmp):
            # comes from logger: lists look like this [[x1, x2, ...]]
            key = get_json_meta_key(tmp["metric"], tmp["op"])
            update_json(json_path, {key: value[0]})

        _iterate_logger(log, model_folder, save_new_value)

    def _iterate_logger(log, model_folder, json_function):
        for name, value in log.items():
            tmp = IBlockManager.get_dict_from(name)
            if tmp is None:
                continue

            block_folder = get_block_folder_str(tmp["position"], tmp["block"])
            json_path = _get_path(model_folder, block_folder)

            check_mkdir(json_path)
            json_function(json_path, value, tmp)

    def _get_path(model_folder, block_folder):
        parent_folder = join(DATABASE_PATH, model_folder, block_folder)
        json_path = join(parent_folder, META_NAME)
        return json_path

    # model_folder determines the parent folder of a database for a model
    log_path = join(DATABASE_PATH, model_folder, LOG_JSON_NAME)
    log = load_json(log_path)
    assert log is not None, f"No file found at: {log_path}"

    clear_all_json_files(log, model_folder)
    save_new_logger_values(log, model_folder)


def create_block_info_file(model_folder, block_managers, depth_offset):
    def create_rgx(block_managers):
        # e.g. r'(\d\d\d)-(fp_block|pyrblock)$'
        rgx = '(\d\d\d)-('
        block_names = [type(mngr).name for mngr in block_managers]
        for name in block_names:
            rgx += name + '|'
        rgx += ')$'
        rgx = re.compile(rgx)
        return rgx

    def get_pos_from(name, rgx):
        return int(re.match(rgx, name).group(1))

    def get_block_id_from(name, rgx):
        return re.match(rgx, name).group(2)

    def _check_order(f_names, rgx):
        positions = [get_pos_from(x, rgx) for x in f_names]
        for i in range(max(positions)):
            assert i == positions[i]

    def get_folder_names(model_folder, rgx):
        f_names = search_rgx(rgx, join(DATABASE_PATH, model_folder))
        assert f_names
        f_names = sorted(f_names, key=lambda x: get_pos_from(x, rgx))
        _check_order(f_names, rgx)
        return f_names

    def check_duplicate_names(block_managers):
        names = [mngr.name for mngr in block_managers]
        if len(set(names)) < len(names):
            warnings.warn('Duplicate block manager names.')

    check_duplicate_names(block_managers)

    rgx = create_rgx(block_managers)
    f_names = get_folder_names(model_folder, rgx)
    block_managers = {
        mngr.name: mngr for mngr in block_managers
    }

    # add some option data as well
    options = load_json(join(DATABASE_PATH, model_folder, TRAIN_OPT_NAME))
    model_type = options['model_type']

    assert depth_offset is not None
    depth_counter = depth_offset
    for folder in f_names:
        # create file
        pos = get_pos_from(folder, rgx)
        name = get_block_id_from(folder, rgx)

        path = join(DATABASE_PATH, model_folder, folder, BLOCK_INFO_NAME)
        update_json(path, {
            'position': pos,
            'block_id': name,
            # number of convolutions. I.e., which ResNet layer
            'depth': depth_counter,
            'model_type': model_type
        }, do_rewrite=True)

        # update depth counter
        mngr = block_managers[name]
        assert isinstance(type(mngr).num_conv_layers_in_sequence, int)
        depth_counter += type(mngr).num_conv_layers_in_sequence


# -------------------------------------------------------------------------
# ------------------ INFOS FROM KEYS AND FOLDERS --------------------------
# -------------------------------------------------------------------------


def check_str(x):
    # TODO: this is only used ONCE in query_database and supposedly only for
    # a specific input. Should this function be moved and pruned?
    # might be a folder
    rgx_folder = r'(\d\d\d)-(.*)'
    rgx_logger_meta = r'(.*)-(.*)'
    rgx_deg_of_es = r'(.*)-(id(1|2)_(ctr|hrz))'
    rgx_weight_meta = r'(kd|or)_(.*)'

    match = re.match(rgx_folder, x)
    if bool(match):
        return {
            'position': int(match.group(1)),
            'block_id': match.group(2)
        }

    match = re.match(rgx_deg_of_es, x)
    if bool(match):
        return {
            'operation': match.group(1),
            'metric': match.group(2)
        }

    match = re.match(rgx_logger_meta, x)
    if bool(match):
        return {
            'metric': match.group(1),
            'operation': match.group(2),
        }

    match = re.match(rgx_weight_meta, x)
    if bool(match):
        return {
            'metric': match.group(0),
        }

# -----------------------------------------------------------
# ------------------ BLOCK MANAGERS -------------------------
# -----------------------------------------------------------


class IBlockManager():
    """ 
    A block manager is a container of functions needed to evaluate a block.
    It contains, GetLoggerIfConditionMet functions. Loggers compute statistics
    based on the feature map activations of a block in a model.
    Furthermore, the manager offers extractor functions (extract_[sth])
    that save statistics to the database so that they can be queried later.
    This interface assumes that a number of functions is implemented when 
    inheriting it:
    * extract_meta
    * extract_deg_of_es
    * check
    * _get_logger

    This class is needed, because when evaluating a model (without knowing 
    the naming conventions of a specific model), the model graph is traversed
    in its entirety –including all operations of a model,
    e.g. ReLUs, single convolution, or batchnorm layers. The block managers 
    filter through this list of graph nodes and only evaluate the interesting 
    block nodes.

    """
    name = "None"
    num_conv_layers_in_sequence = None

    def __init__(self, list_cls_log_metrics=list()):
        """Constructor function

        Parameters
        ----------
        list_cls_log_metrics : list of Loggers, optional
            class object to instantiate a logger object for all modules in a 
            model that pass the check-function, by default list()
        """
        self.cls_loggers_ = list_cls_log_metrics

        self.tmp = None

    def get_loggers_if_condition_met(self, block, current_pos):
        """Return a dictionary of Loggers if the block fits this manager (see
        check function).

        Parameters
        ----------
        block : nn.Module
            neural network module that we may want to create a logger for
        current_pos : int
            keep track of the current position within the network

        Returns
        -------
        dict {str: Logger(block)}
            returns a dictionary where the name contains information about the
            name of the block (e.g., FP-block), the metric of the Logger 
            (e.g. Entropy), the position within the network and the kind of 
            operation that is logged (e.g., multiplication in an FP-block).
        """
        if not self.check(block):
            # block does not fit to this Manager
            return None

        # to keep consistent with position counting, rather keep each logger
        # getter in the list, if no metrics are given. return an empty list
        # this triggers the counter increment without creating new loggers
        if not self.cls_loggers_:
            return dict()

        # Go through all LoggerClasses, i.e. different metrics and instantiate
        # logger object that logs the activations of block.
        out = list()
        for cls_logger in self.cls_loggers_:
            out += self._get_logger(cls_logger, current_pos, block)

        return {x.name: x for x in out}

    def _get_logger_prefix(self, cls_logger, current_pos):
        """Prefix to create an unique id for a logger like this:
        [name of manager];[name of logger];[position in the model]
        examples:
        fp_block;ent_per_fmap;01 -> position has two leading zeros

        Parameters
        ----------
        cls_logger : class object of Logger
            can be used to instantiate a logger
        current_pos : int

        Returns
        -------
        string
            [name of manager];[name of logger];[position in the model]
        """
        name = type(self).name + ';' + cls_logger.name
        name += ';' + str(current_pos).zfill(2)
        print(name)
        return name

    @staticmethod
    def get_dict_from(name):
        """Returns meta data from the dictionary of a logger

        Parameters
        ----------
        name : str
            TODO: for example: ???

        Returns
        -------
        dictionary of int and string
            meta data derived from the name
        """
        # for the metrics, but not keys like epoch or lr, 'val_' is written in
        # front of it
        name = name.split('val_')[-1]
        split = name.split(';')
        # values like epoch an lr do not have an ';' in their name. For a real
        # metric, we assume the split length to be three
        if len(split) != 3:
            return None

        out = {
            'block': split[0],
            'metric': split[1],
        }

        tmp = split[2].split('/')
        position = int(tmp[0])
        op = tmp[1]

        out.update({
            'position': position,
            'op': op
        })

        return out

    def save_blocks_to_tmp_var(self, model, block, current_pos, parent_folder):
        if self.check(block):
            if self.tmp is None:
                self.tmp = {}

            assert current_pos not in self.tmp.keys()
            self.tmp[current_pos] = block

            return True
        return False

    def extract_meta(self, model, block, current_pos, parent_folder):
        """This function is normally used to extract meta data from a model's
        weights and saves them to a meta.json file. 
        A Block manager may overwrite this function. If not, the function 
        simply returns a True if the input block passes the check function of 
        the block. The positions counter is increased, but no meta.json is 
        altered.

        Parameters
        ----------
        block : nn.Module
        current_pos : int
        parent_folder : str

        Returns
        -------
        boolean
            Returns the check result for the block. Used to increase the
            position counter.
        """
        return self.check(block)

    def extract_deg_of_es(self, model, block, current_pos, parent_folder):
        """In most cases, calls the _extract_deg_of_es several times for
        different operations.

        Parameters
        ----------
        model : nn.Module
            image is passed through this model
        block : nn.Module
            a block belonging to this model which activation is monitored
        current_pos : int
        parent_folder : str

        Returns
        -------
        boolean
            returns True if an extraction ran successfull. In this standard
            version, nothing is computed. Thus, False is returned.
        """
        return False

    def check(self, block):
        """Returns True if block fits the manager

        Parameters
        ----------
        block : nn.Module

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError()

    def _get_folder(self, parent_folder, current_pos):
        """Create a unique folder id to where data from the database are saved.
        The returned string looks like this:
            [position]-[name of manager]
        For example: 007-fp_block 

        Parameters
        ----------
        parent_folder : str
            Path to the specific model's database
        current_pos : int
            Position of the block in a model

        Returns
        -------
        string

        """
        assert type(self).name != "None", "Manager Class needs a unique name"
        folder = get_block_folder_str(current_pos, type(self).name)
        folder = join(parent_folder, folder)
        return folder

    def _save_activation_of_examples(self, images, model, block, folder, op_id):
        """Saves the activations of block to
        folder-[op_id]-activation_examples.npy

        Parameters
        ----------
        images: torch.Tensor
        model : nn.Module
        block : nn.Module
            For this block, an the activation is saved
        folder : str
            Where to save the npy file.
        op_id : str
            Identifier of operation, e.g., upper, lower, bn1, ...
        """

        act_getter = ActivationGetter(block)

        model(images)

        # prev. debug-check to determine if dying ReLU computation was right:
        # torch-where and np-where are similar
        # where_torch = list(
        #    torch.where(act_getter.out[0].sum(-1).sum(-1) == 0)[0]
        # )
        activation = cuda_to_numpy(act_getter.out)
        # np_where = list(np.where(activation[0, ...].sum(0).sum(0) == 0)[0])
        # print(np_where)

        np.save(
            join(folder, f'{op_id}-activation_examples.npy'),
            activation
        )

    def _extract_deg_of_es(self, model, operation, folder, op_id):
        """Usually this function is called for multiple operations:
        Determines the degree of end-stopping and saves these values to the 
        '[folder]/meta.json' file and saves the activations to an npy file:
        '[folder]/[op_id]-deg_of_es_activations.npy'


        Parameters
        ----------
        model : nn.Module
            [description]
        operation : nn.Module
            [description]
        folder : str
            where to save the meta.json file and the activations
        op_id : str
            Identifier of the operation
        """
        # horizontal
        image, weight_id0, weight_id1, weight_id2 = create_end_stopping_image_box(
            +1.)
        id0_pos, id1_pos, id2_pos, act_not_normed_pos = _degree_of_endstopping(
            model, operation, image, weight_id0, weight_id1, weight_id2
        )

        # vertical
        image, weight_id0, weight_id1, weight_id2 = create_end_stopping_image_box(
            -1.)
        id0_neg, id1_neg, id2_neg, act_not_normed_neg = _degree_of_endstopping(
            model, operation, image, weight_id0, weight_id1, weight_id2
        )
        # concat positive and negative activations
        # each activation has the shape (1, D, H, W)
        # output has shape: (2, D, H, W)
        activation = np.concatenate(
            [act_not_normed_pos, act_not_normed_neg], 0
        )

        # make sure pos and neg yield different results
        if np.abs(np.array(id0_pos) - np.array(id0_neg)).sum() == 0.:
            warnings.warn(
                ("ID0-pos and ID0-neg signal are equal! Check if this is an"
                 " input error!")
            )

        to_json = {
            f'{op_id}-id0_pos': id0_pos,
            f'{op_id}-id0_neg': id0_neg,

            f'{op_id}-id1_pos': id1_pos,
            f'{op_id}-id1_neg': id1_neg,

            f'{op_id}-id2_pos': id2_pos,
            f'{op_id}-id2_neg': id2_neg,
        }

        np.save(
            join(folder, f'{op_id}-deg_of_es_activations.npy'),
            activation
        )

        update_json(
            join(folder, META_NAME),
            to_json
        )

    def _get_relevant_operations(self, block, keys_only=False):
        raise NotImplementedError

    def extract_deg_of_es(self, model, block, current_pos, parent_folder):
        if not self.check(block):
            return False

        folder = self._get_folder(parent_folder, current_pos)

        rel_ops = self._get_relevant_operations(block)
        for key, operation in rel_ops.items():
            self._extract_deg_of_es(
                model, operation, folder, key
            )

        return True

    def save_activation_of_examples(self, model, block, current_pos, parent_folder, *, images):
        if not self.check(block):
            return False

        folder = self._get_folder(parent_folder, current_pos)

        rel_ops = self._get_relevant_operations(block)
        for key, operation in rel_ops.items():
            self._save_activation_of_examples(
                images, model, operation, folder, key
            )

        return True

    def _get_logger(self, cls_logger, current_pos, block):
        """ Create a logger for all kind of operations

        Parameters
        ----------
        cls_logger : class object of logger
            for an operation, instantiate a Logger object
        current_pos : int
        block : nn.Module
            this block is monitored by the logger

        Returns
        -------
        list of Logger
        """
        name = self._get_logger_prefix(cls_logger, current_pos)
        rel_ops = self._get_relevant_operations(block)
        out = []
        for key, operation in rel_ops.items():
            out.append(cls_logger(name + '/' + key, operation))

        return out


class FPBlockManager(IBlockManager):
    name = 'fp_block'
    num_conv_layers_in_sequence = 3

    def check(self, block):
        return isinstance(block, cb.FPBlockJOV)

    def _get_inner_block(self, block):
        """Return the block from which it is easy to grab the operations
        Parameters
        ----------
        block : nn.Module

        Returns
        -------
        nn.Module
        """
        return block.block_with_shortcut.block

    def _get_relevant_operations(self, block, keys_only=False):
        if keys_only:
            return [
                'upper', 'dw1', 'dw2',
                'mult', 'lower', 'res',
            ]
        inner_block = self._get_inner_block(block)
        return {
            'upper': inner_block.upper,
            'dw1': inner_block.left_dw,
            'dw2': inner_block.right_dw,
            'mult': inner_block.mult,
            'lower': inner_block.lower,
            'res': block,
        }

    def extract_meta(self, model, block, current_pos, parent_folder):
        if not self.check(block):
            return False

        # save both dws filter kernels (Dx3x3) as npy files
        # extract all angles, orientations, norms,
        folder = self._get_folder(parent_folder, current_pos)

        inner_block = self._get_inner_block(block)
        # save filters
        dw1 = inner_block.left_dw[0].weight.detach().cpu().numpy()[
            :, 0, ...]
        dw2 = inner_block.right_dw[0].weight.detach().cpu().numpy()[
            :, 0, ...]
        np.save(
            join(folder, 'filters.npy'),
            np.stack([dw1, dw2], 0)
        )

        # save linear combination values for upper
        upper = inner_block.upper[0].weight.detach().cpu().numpy()
        upper = upper[..., 0, 0]
        np.save(
            join(folder, 'lin_comb.npy'),
            upper
        )

        to_json = dict()
        # extract angles and norms
        to_json.update(type(self).extract_angles_and_norms(dw1, dw2))

        # do we really need this?
        # from the angles get the curvature
        # to_json.update({
        #    'taylor_curv': list(e_func.get_taylor_curvature(Z, gamma0=np.array(to_json['kd_angles'])))
        # })

        # extract orientations and norms
        to_json.update(type(self).extract_orientations_and_norms(dw1, dw2))

        update_json(
            join(folder, META_NAME),
            to_json
        )

        return True

    @staticmethod
    def extract_angles_and_norms(dw1, dw2):
        # TODO: write Test function
        dw1 = dw1.reshape(dw1.shape[0], dw1.shape[1] * dw1.shape[2])
        dw2 = dw2.reshape(dw2.shape[0], dw2.shape[1] * dw2.shape[2])

        angles = FPBlockManager._comp_angles(dw1, dw2)
        min_norm, max_norm = FPBlockManager._comp_norms(dw1, dw2)
        return {
            'kd_angles': arr_to_float_list(angles),
            'kd_min_norm': arr_to_float_list(min_norm),
            'kd_max_norm': arr_to_float_list(max_norm),
        }

    @staticmethod
    def extract_orientations_and_norms(dw1, dw2):
        # TODO: write test function
        # compute the direction vectors
        k = dw1.shape[-1]
        xx, yy = np.meshgrid(np.linspace(-1., 1., k), np.linspace(-1., 1., k))
        X = np.stack([xx, yy], 0)
        X = X / (np.linalg.norm(X, 2, 0) + 1e-6)
        dx = X[0, ...][np.newaxis, ...]
        dy = X[1, ...][np.newaxis, ...]

        # orientation vector = linear combination of weights and dir vectors
        x1 = (dw1 * dx).sum(-1).sum(-1)
        y1 = (dw1 * dy).sum(-1).sum(-1)
        or_vect1 = np.stack([x1, y1], -1)

        x2 = (dw2 * dx).sum(-1).sum(-1)
        y2 = (dw2 * dy).sum(-1).sum(-1)
        or_vect2 = np.stack([x2, y2], -1)

        angles = FPBlockManager._comp_angles(or_vect1, or_vect2)
        min_norm, max_norm = FPBlockManager._comp_norms(
            or_vect1, or_vect2
        )

        return {
            'or_angles': arr_to_float_list(angles),
            'or_min_norm': arr_to_float_list(min_norm),
            'or_max_norm': arr_to_float_list(max_norm),
        }

    def _comp_angles(vec1, vec2):
        # TODO: write test function
        norm1 = np.linalg.norm(vec1, axis=-1)
        norm2 = np.linalg.norm(vec2, axis=-1)

        vec1_normed = vec1 / (norm1.reshape(-1, 1) + 1e-6)
        vec2_normed = vec2 / (norm2.reshape(-1, 1) + 1e-6)

        scalar_products = (
            vec1_normed * vec2_normed
        ).sum(-1).clip(min=-1., max=1.)

        angles = np.arccos(scalar_products) / np.pi * 180.
        return angles

    def _comp_norms(vec1, vec2):
        # TODO: write test function
        norm1 = np.linalg.norm(vec1, axis=-1)
        norm2 = np.linalg.norm(vec2, axis=-1)
        min_norm = np.stack([norm1, norm2], -1).min(-1)
        max_norm = np.stack([norm1, norm2], -1).max(-1)
        return min_norm, max_norm


class FPBlockManagerLegacy(FPBlockManager):
    def check(self, block):
        is_res = isinstance(block, fp_blocks.ResidualAdapter)
        if is_res:
            inner_block = block.block
            return isinstance(inner_block, fp_blocks.FPL1DWR1U1)
        else:
            return False

    def _get_inner_block(self, block):
        return block.block

    def _get_relevant_operations(self, block, keys_only=False):
        if keys_only:
            return [
                'upper', 'dw1', 'dw2', 'mult', 'lower', 'res',
            ]
        inner_block = self._get_inner_block(block)
        return {
            'upper': inner_block.upper,
            'dw1': inner_block.left_dw,
            'dw2': inner_block.right_dw,
            'mult': inner_block.mult,
            'lower': inner_block.lower,
            'res': block,
        }


class BottleneckManager(IBlockManager):
    name = 'btlnck'
    num_conv_layers_in_sequence = 3

    def check(self, block):
        return isinstance(block, Bottleneck)

    def _get_relevant_operations(self, block, keys_only=False):
        if keys_only:
            return [
                'bn1', 'bn2', 'bn3', 'res',
            ]

        return {
            'bn1': block.bn1,
            'bn2': block.bn2,
            'bn3': block.bn3,
            'res': block,
        }


class PyrBlockManager(IBlockManager):
    name = 'pyrblock'
    num_conv_layers_in_sequence = 2

    """
    # Block contains a sequential called 'convolutions'
    0 : nn.BatchNorm2d(in_dim),
    1 : nn.Conv2d(in_dim, out_dim, kernel_size=k,
                       stride=stride, padding=k // 2, bias=False),
    2 : nn.BatchNorm2d(out_dim),
    3 : nn.ReLU(),
    4 : nn.Conv2d(out_dim, out_dim, kernel_size=k,
                       stride=1, padding=k // 2, bias=False),
    5 : nn.BatchNorm2d(out_dim),
    """

    def check(self, block):
        return isinstance(block, cb.PyramidBasicBlock)

    def _get_relevant_operations(self, block, keys_only=False):
        if keys_only:
            return [
                'bn1', 'bn2', 'relu', 'bn3', 'res',
            ]

        inner_block = block.block_with_shortcut.block
        return {
            'bn1': inner_block[0],
            'bn2': inner_block[2],
            'relu': inner_block[3],
            'bn3': inner_block[5],
            'res': block,
        }


# -----------------------------------------------------------
# ------------------ LOGGERS --------------------------------
# -----------------------------------------------------------
class IPerFeatureMapLogger(Logger):
    name = 'None'

    def __init__(self, name, block):
        """A Logger is a printer function that monitors activation values,
        the activations are transformed using _get_transformed_activation
        and stored in self.values. When calling the Logger, the list of values
        is typecasted to an numpy array. On this array, the reduce function is
        applied.


        Parameters
        ----------
        name : str
        block : nn.Module
            record the activation of this module
        """
        self.name = name
        # used in the restart function, do not remove
        self.init_name = name

        # a logger needs an activation getter and a reduce function
        # After each model feed-forward step, the activation getter saves
        # activation results at self.act_getter.out
        self.act_getter = ActivationGetter(block)

        # for better readability the reduction is written out here
        self.reduce = None
        # contains a list of lists of float numbers -> the batch averaged
        # entropy values
        self.values = []

    def __call__(self):
        # creates a (num_batches, num_f_maps) array with float64 values
        values = np.array(self.values)
        # average over batches and return as list of float64
        return list(values.mean(0))


class EntropyPerFeatureMapLogger(IPerFeatureMapLogger):
    name = 'ent_per_fmap'

    """ From the paper:
    We analyzed the entropy of all FP-neurons T_2 for the 
    FP-ResNet-50 (ImageNet) and the FP-ResNet-59 (Cifar-10). One hundred 
    randomly sampled images from the respective test set (in case of ImageNet, 
    the validation set) were passed to each network. For each input, we 
    computed the corresponding feature maps for every FP-block, one tensor 
    T_2 for every block. We normalized each feature map 
    T_2^m from real values to {0,1,...,255} and computed 
    the entropy of the pixel distribution over the 256 integer values. For the 
    100 input images, we obtained 100 entropy values for each feature map. 
    We averaged these 100 values resulting in the mean entropy for each feature 
    map, i.e., each FP-neuron.

    THIS FUNCTION RETURNS THE AVERAGE OVER THE BATCH SIZE

    """

    def _get_transformed_activation(self):
        # TODO: test this
        """For each image in the batch and for each feature map,
        the feature map is normalized to uint8 and the entropy of the pixel
        distribution is computed.
        The entropy values are averaged over all images in the batch.
        The function returns a list of float values. The list is as long as 
        the number of feature maps in the activation tensor.

        Returns
        -------
        list of float
            Batch averaged entropy for each feature map
        """
        # compute entropy of each feature-map
        # shape: (b, d, h, w)
        x = self.act_getter.out
        # placeholder for featre map entropy vals
        out = torch.zeros(x.shape[1])

        for b in range(x.shape[0]):
            # for each dimension, compute the entropy value
            for d in range(x.shape[1]):
                # normalize to uint8
                c = x[b, d, ...]
                c -= c.min()
                c /= (c.max() + 1e-3)
                c = (255. * c).round()

                # compute normalized distribution
                tmp = torch.histc(
                    c.view(-1),
                    bins=256, min=0, max=255
                )
                dist = tmp / tmp.sum()

                # with 1e-6 offset entropy becomes negative
                entropy = -1. * (
                    dist * torch.log2(dist + 1e-6)
                ).sum()
                entropy = entropy.clamp(min=0).item()

                # add the entropy, a whole batch is processed
                out[d] += entropy

        # normalize each value by the batch-size
        out = (out / float(x.shape[0])).detach().cpu().numpy()

        # there shouldn't be any NaN values
        assert not np.any(np.isnan(out))

        # cast to float
        out = [float(z) for z in list(out)]
        return out


class ActivationPerFeatureMapLogger(IPerFeatureMapLogger):
    name = 'act_per_fmap'

    def _get_transformed_activation(self):
        # compute entropy of each feature-map
        # normalize to uint8
        # shape: (b, d, h, w)
        x = self.act_getter.out

        # (b, d, h, w) -> d
        out = x.mean(0).mean(-1).mean(-1)

        # normalize each value by the batch-size
        out = (out).detach().cpu().numpy()

        # there shouldn't be any NaN values
        assert not np.any(np.isnan(out))

        # cast to float
        out = [float(z) for z in list(out)]
        return out


# ----------------------------------------------------------------------------
# ------------------------------ UTILS ---------------------------------------
# ----------------------------------------------------------------------------


def setup(in_options):
    if in_options.device is not None:
        pt_training.set_device(in_options.device)

    folder = in_options.model_folder
    model_folder = folder.split('/')[-1]

    options = load_json(join(folder, 'opt.json'))
    assert options is not None, f'no options at: {folder}'

    model = load_model(options, 'cpu', map_location=torch.device(
        'cpu'), new_model_path=join(folder, 'model.pt'),
        from_par_gpu=in_options.from_par_gpu
    ).to(get_device()).eval()

    # TODO: should we do this?
    #model.pre_transform = NORMALIZE

    model_type = options['model_type']
    print(model_folder, model_type)
    return model, model_folder, model_type


def create_end_stopping_image_box(sign=1., is_cifar=None):
    """Creates an image of rectangle with the image size depending on the 
    model (ImageNet or Cifar-10). All tuple combinations of (N +/- s) are the 
    corner points. Furthermore, three masks id0, id1, and id2 are created to 
    extract homogenous areas, edges, and corners. Note that the all outputs are
    subsample when an ImageNet network is used. The masks are supposed to have
    the same sum. 

    Parameters
    ----------
    sign : float, optional
        is it a negative or positive rectangle, by default 1.
    is_cifar : bool, optional
        input the image for a model trained on Cifar-10 or trained on imagenet,
        by default False

    Returns
    -------
    image : np.array in {-1., 0.} or {0., 1.}
    wid0 : np.array in {0., 1.}
    wid1 : np.array in {0., 1.}
    wid2 : np.array in {0., 1.}
        image and different intrinsic dimension masks.
    """
    if is_cifar is None:
        is_cifar = CIFAR_MODE

    if is_cifar:
        N = 32
        s = 16
        w = 4
        stride = 1
    else:
        N = 224
        s = 64
        w = 16
        stride = 4
    image = np.zeros((N, N))
    wid0 = np.zeros((N, N))
    wid1 = np.zeros((N, N))
    wid2 = np.zeros((N, N))

    assert sign == 1. or sign == -1.
    image[(N - s) // 2:(N + s) // 2, (N - s) // 2:(N + s) // 2] = sign

    # id0
    wid0[N // 2 - w:N // 2 + w, N // 2 - w:N // 2 + w] = 1
    wid0 = wid0[::stride, ::stride]
    # id1
    wid1[(N - w) // 2:(N + w) // 2,
         (N - s - w) // 2:(N - s + w) // 2] = 1
    wid1[(N - w) // 2:(N + w) // 2, -
         (N - s + w) // 2:-(N - s - w) // 2] = 1

    wid1[(N - s - w) // 2:(N - s + w) //
         2, (N - w) // 2:(N + w) // 2] = 1
    wid1[-(N - s + w) // 2:-(N - s - w) //
         2, (N - w) // 2:(N + w) // 2] = 1
    wid1 = wid1[::stride, ::stride]

    # id2
    wid2[(N - s - w) // 2:(N - s + w) // 2,
         (N - s - w) // 2:(N - s + w) // 2] = 1
    wid2[-(N - s + w) // 2:-(N - s - w) // 2,
         (N - s - w) // 2:(N - s + w) // 2] = 1

    wid2[(N - s - w) // 2:(N - s + w) // 2, -
         (N - s + w) // 2:-(N - s - w) // 2] = 1
    wid2[-(N - s + w) // 2:-(N - s - w) // 2, -
         (N - s + w) // 2:-(N - s - w) // 2] = 1
    wid2 = wid2[::stride, ::stride]

    return image, wid0, wid1, wid2


def _degree_of_endstopping(model, block, image, weight_id0, weight_id1, weight_id2):
    """Passes image to model and records the activations of block. The 
    activations are normalized to be in [0, 1] and then summed over using 
    different weighted masks.

    Parameters
    ----------
    model : nn.Module
        [description]
    block : nn.Module
        [description]
    image : np.array
        test image to compute the degree of endstopping
    weight_id0 : np.array
        mask for intrinsic dimension 0
    weight_id1 : np.array
        mask for intrinsic dimension 1
    weight_id2 : np.array
        mask for intrinsic dimension 2

    Returns
    -------
    id0 : list of float
        For each feature map: the intrinsic dimension 0 value
    id1 : list of float
        For each feature map: the intrinsic dimension 1 value
    id2 : list of float
        For each feature map: the intrinsic dimension 2 value
    activations : np.array
        actual activations of block when using image as input
    """
    act_getter = ActivationGetter(block)
    image = torch.Tensor(image[np.newaxis, :, :])
    image = torch.cat([image] * 3, 0).unsqueeze(0).to(get_device())

    # zero mean and standard deviation of one
    # this is the easiest way to have a proper normalization
    image = (image - image.mean()) / image.std()

    model(image)

    activations = act_getter.out
    activations = activations.detach().cpu().numpy()

    activations = normalize_act(activations)
    id0 = []
    id1 = []
    id2 = []
    for i in range(activations.shape[1]):
        tmp = activations[0, i, ...]
        id0.append((tmp.copy() * weight_id0).sum())
        id1.append((tmp.copy() * weight_id1).sum())
        id2.append((tmp.copy() * weight_id2).sum())

    return id0, id1, id2, activations


def normalize_act(activations):
    # from range [a, b] (a can be < 0) to range [0, 1]
    # compute energy and normalize it
    activations = activations**2.
    numerator = 3. * activations.std() + activations.mean()
    if numerator == 0:
        warnings.warn("Zero numerator")
        numerator = 1.

    activations = (activations / numerator)
    percentage_clipped_to_one = (activations > 1.).mean().round(3)
    if percentage_clipped_to_one > .1:
        assert percentage_clipped_to_one < .1
        warnings.warn(('High perc. of clipped values: ',
                       f'{percentage_clipped_to_one}'))
    activations = activations.clip(min=0., max=1.)
    return activations


def get_example_batch(is_cifar, as_numpy=False):
    if is_cifar:
        dataset = 'cifar_10_validation_subset'
    else:
        dataset = 'im_net_entropy'
    return _get_example_batch(dataset, as_numpy=as_numpy)


def _get_example_batch(dataset, as_numpy=False):

    data_loader = get_data_loaders(
        dataset, batch_size=5, num_workers=0
    )['val']
    for images, _ in data_loader:
        break

    if as_numpy:
        images = cuda_to_numpy(images)
    return images


def get_block_folder_str(position, name_of_manager):
    return f'{str(position).zfill(3)}-{name_of_manager}'


def update_json(path, new_dict, do_rewrite=False):
    """Update the dictionary of a json file. If do_rewrite is True,
    the dictionary currently stored in the file is overwritten.

    Parameters
    ----------
    path : str
        path to the json file
    new_dict : dict
        the dictionary in the file is updated with this dictionary:
        dict_to_save.update(new_dict)
    do_rewrite : bool, optional
        clear all that is currently in the json file, by default False
    """
    current_dict = load_json(path)
    if current_dict is None or do_rewrite:
        current_dict = dict()

    current_dict.update(new_dict)

    with open(path, 'w') as file:
        json.dump(current_dict, file)


def arr_to_float_list(arr):
    assert arr.ndim == 1
    return [float(arr[i]) for i in range(arr.shape[0])]


# ----------------------------------------------------------------------------
# ------------------------------DEBUGGING & TESTS-----------------------------
# ----------------------------------------------------------------------------


def _test_angles():
    N = 5000
    angles = np.random.randint(low=0, high=180, size=(N))

    V1 = []
    V2 = []
    for a1 in list(angles):
        a0 = np.random.randint(low=0, high=180)
        a0 = a0 / 180. * np.pi
        v = np.array([np.cos(a0), np.sin(a0)]) * (np.random.rand() + 1.)

        a1 = a1 / 180. * np.pi
        g = np.array([np.cos(a0 + a1), np.sin(a0 + a1)]) * \
            (np.random.rand() + 1.)

        V1.append(v)
        V2.append(g)

    V1 = np.stack(V1, 0)
    V2 = np.stack(V2, 0)

    predict = FPBlockManager._comp_angles(V1, V2)

    # near zero and 180 the method becomes a bit inaccurate
    assert np.abs(predict - angles).mean() < 1e-3
    assert np.abs(predict - angles).max() < .5
    print('Test successful')
    xxx = 0


def _compare_logger_to_activation():
    activation = np.load(
        '/nfshome/gruening/my_code/DLBio_repos/feature_products_cvpr/database_small/fp_resnet_50_layer_start_q1/000-fp_block/upper-activation_examples.npy'
    )

    big_log = load_json(
        '/nfshome/gruening/my_code/DLBio_repos/feature_products_cvpr/database_small/fp_resnet_50_layer_start_q1/loggers.json'
    )
    Z = np.array(big_log['val_fp_block;ent_per_fmap;00/upper'][0])

    log = load_json(
        '/nfshome/gruening/my_code/DLBio_repos/feature_products_cvpr/database_small/fp_resnet_50_layer_start_q1/000-fp_block/meta.json'
    )

    X = np.array(log['ent_per_fmap-upper'])
    Y = activation[0].sum(0).sum(0)

    y_w = np.where(Y == 0)[0]
    x_w = np.where(X == 0)[0]

    xxx = 0


def _debug():
    _plot_deg_of_es_images()
    # _compare_logger_to_activation()


if __name__ == '__main__':
    OPTIONS = get_options()
    main(OPTIONS)
