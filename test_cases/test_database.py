import json
import subprocess
import unittest
from os.path import isdir, isfile, join

import model_database_api as api
import numpy as np
import torch
import torch.nn as nn
from DLBio.helpers import check_mkdir, search_rgx
from DLBio.kwargs_translator import get_kwargs, to_kwargs_str
from DLBio.pt_training import _torch_save_model
from DLBio.pytorch_helpers import ActivationGetter, get_device
from helpers import load_model
from models.model_getter import get_model
from run_create_model_database import (DATABASE_PATH,
                                       create_end_stopping_image_box,
                                       get_block_managers, get_json_meta_key)

BASE_FOLDER = 'test_cases'
PATH_TO_EXE = 'run_create_model_database.py'


class DatabaseTest(unittest.TestCase):

    def test_dataframe_lengths(self):
        self._test_dataframe_lengths('CifarJOVFPNet_N9_s744', 'CifarJOVFPNet')

        self._test_dataframe_lengths(
            'allzero_CifarJOVFPNet', 'AllZero: CifarJOVFPNet')

    def _test_dataframe_lengths(self, model_folder_name, model_type):
        df = api.get_dataframe(model_folder_name)

        # first stack
        # the first block should be an FP-block with q=2 and out_dim 16
        # thus 32 filters
        self.check_fp_block(df, 0, 16, 2, 2, model_type)
        # the next 8 blocks are PyrBlock
        self.check_pyrblock(df, 1, 16, 5, model_type)
        self.check_pyrblock(df, 2, 16, 7, model_type)
        self.check_pyrblock(df, 3, 16, 9, model_type)
        self.check_pyrblock(df, 4, 16, 11, model_type)
        self.check_pyrblock(df, 5, 16, 13, model_type)
        self.check_pyrblock(df, 6, 16, 15, model_type)
        self.check_pyrblock(df, 7, 16, 17, model_type)
        self.check_pyrblock(df, 8, 16, 19, model_type)

        # second stack -> 32 filters
        # FP-block
        self.check_fp_block(df, 9, 32, 2, 21, model_type)
        # the next 8 blocks are PyrBlock
        self.check_pyrblock(df, 10, 32, 24, model_type)
        self.check_pyrblock(df, 11, 32, 26, model_type)
        self.check_pyrblock(df, 12, 32, 28, model_type)
        self.check_pyrblock(df, 13, 32, 30, model_type)
        self.check_pyrblock(df, 14, 32, 32, model_type)
        self.check_pyrblock(df, 15, 32, 34, model_type)
        self.check_pyrblock(df, 16, 32, 36, model_type)
        self.check_pyrblock(df, 17, 32, 38, model_type)

        # third stack -> 64 filters
        # FP-block
        self.check_fp_block(df, 18, 64, 2, 40, model_type)
        # the next 8 blocks are PyrBlocks
        self.check_pyrblock(df, 19, 64, 43, model_type)
        self.check_pyrblock(df, 20, 64, 45, model_type)
        self.check_pyrblock(df, 21, 64, 47, model_type)
        self.check_pyrblock(df, 22, 64, 49, model_type)
        self.check_pyrblock(df, 23, 64, 51, model_type)
        self.check_pyrblock(df, 24, 64, 53, model_type)
        self.check_pyrblock(df, 25, 64, 55, model_type)
        self.check_pyrblock(df, 26, 64, 57, model_type)

    def test_all_zero_model(self):
        df = api.get_dataframe('allzero_CifarJOVFPNet')
        self.check_logger_values(df)
        self.check_degofes_values(df, 16, 2)

    def check_logger_values(self, df):
        # check at each position if the entropy and degoes values are zero
        keys_ = DatabaseTest.get_meta_logger_json_keys()

        for _, row in df.iterrows():
            for key in keys_:
                self.assertTrue(is_nan_or_zero(row[key]), msg=(key, row[key]))

    def check_degofes_values(self, df, first_block_dim, q):
        keys_ = DatabaseTest.get_all_degofes_keys()
        # degofes keys are either for an fp-block or pyramid block
        keys_ = list(set(keys_).intersection(set(df.columns)))
        assert keys_

        for pos in range(df['position'].max()):
            tmp = DatabaseTest.get_subtable(df, {'position': pos})

            self.assertGreaterEqual(tmp.shape[0], 1)
            if pos == 0:
                # Degree of end-stopping was computed for position 0
                for _, row in tmp.iterrows():
                    for key in keys_:
                        op = key.split('-')[0]
                        x = row[key]
                        # since q=2 there are nan values for res and lower
                        if op in ['res', 'lower']:
                            if row['filter_idx'] < first_block_dim:
                                self.assertEqual(
                                    x, 0, msg=(pos, key, row[key]))
                            else:
                                self.assertTrue(
                                    x != x, msg=(pos, key, row[key]))
                        else:
                            self.assertEqual(x, 0, msg=(pos, key, row[key]))
            else:
                # All other degofes values should be nan
                for _, row in tmp.iterrows():
                    for key in keys_:
                        x = row[key]
                        self.assertTrue(x != x, msg=(pos, key, row[key]))

    def setUp(self):
        model_folder = 'allzero_CifarJOVFPNet'
        save_folder = join(BASE_FOLDER, 'models', model_folder)
        save_path = join(save_folder, 'model.pt')

        if not isfile(save_path):
            self.create_and_save_all_zero_model(save_folder, save_path)

        try:
            api.get_dataframe(model_folder)
        # no files found in model_folder
        except AssertionError:
            call_list = [
                'python', PATH_TO_EXE,
                '--model_folder', save_folder,
                '--bs', str(100),
                '--nw', str(0),
                '--depth_offset', str(2)
            ]
            print('Creating database:')
            print(call_list)
            subprocess.call(call_list)

    def create_and_save_all_zero_model(self, save_folder, save_path):
        # save AllZeroModel
        options = {
            'model_type': 'AllZero: CifarJOVFPNet',
            'in_dim': 3,
            'out_dim': 10,
            'model_kw': to_kwargs_str(
                {'N': [9], 'q': [2]}
            ),
        }

        check_mkdir(save_path)
        with open(join(save_folder, 'opt.json'), 'w') as file:
            json.dump(options, file)

        model_kwargs = get_kwargs(options['model_kw'])
        model = get_model(
            options['model_type'],
            options['in_dim'],
            options['out_dim'],
            get_device(),
            **model_kwargs
        )

        # quick check if the model really outputs all zero
        test_logger = ActivationGetter(model.stack1[0])
        test_input = torch.rand(3, 3, 32, 32).to(get_device())
        model(test_input)
        assert (test_logger.out**2.).sum() == 0

        _torch_save_model(model, save_path, True)
        print(f'saved model to: {save_path}')

        model = load_model(options, get_device(), new_model_path=save_path)

    def check_fp_block(self, df, position, out_dim, q, depth, model_type):
        # max dim = q* out_dim
        expected_dim = q * out_dim
        # if q > 1 there are many entries where the 'lower' layer has nan
        # entries
        fp_block_nans = expected_dim // q
        first_block = DatabaseTest.get_subtable(df, {'position': position})
        self.assertEqual(first_block.shape[0], expected_dim)
        self.check_equal(
            first_block,
            {
                'model_type': model_type,
                'block_id': 'fp_block',
                'depth': depth,
            })

        to_check = {
            'ent_per_fmap-bn1': expected_dim, 'ent_per_fmap-bn2': expected_dim,
            'ent_per_fmap-bn3': expected_dim,
            'ent_per_fmap-res': fp_block_nans,
            'act_per_fmap-bn1': expected_dim,
            'act_per_fmap-bn2': expected_dim,
            'act_per_fmap-bn3': expected_dim,
            'act_per_fmap-res': fp_block_nans,
            'position': 0, 'depth': 0,
            'filter_idx': 0, 'ent_per_fmap-upper': 0, 'ent_per_fmap-dw1': 0,
            'ent_per_fmap-dw2': 0, 'ent_per_fmap-mult': 0,
            'ent_per_fmap-lower': fp_block_nans,
            'act_per_fmap-upper': 0, 'act_per_fmap-dw1': 0,
            'act_per_fmap-dw2': 0,
            'act_per_fmap-mult': 0, 'act_per_fmap-lower': fp_block_nans,
            'kd_angles': 0, 'kd_min_norm': 0,
            'kd_max_norm': 0, 'or_angles': 0, 'or_min_norm': 0,
            'or_max_norm': 0,
        }

        # degree of end-stopping is only computed in first layer
        deg_of_es_check = {
            'upper-id0_pos': 0, 'upper-id0_neg': 0, 'upper-id1_pos': 0,
            'upper-id1_neg': 0,
            'upper-id2_pos': 0, 'upper-id2_neg': 0, 'dw1-id0_pos': 0,
            'dw1-id0_neg': 0,
            'dw1-id1_pos': 0, 'dw1-id1_neg': 0, 'dw1-id2_pos': 0,
            'dw1-id2_neg': 0,
            'dw2-id0_pos': 0, 'dw2-id0_neg': 0, 'dw2-id1_pos': 0,
            'dw2-id1_neg': 0,
            'dw2-id2_pos': 0, 'dw2-id2_neg': 0, 'mult-id0_pos': 0,
            'mult-id0_neg': 0,
            'mult-id1_pos': 0, 'mult-id1_neg': 0, 'mult-id2_pos': 0,
            'mult-id2_neg': 0,
            'lower-id0_pos': fp_block_nans, 'lower-id0_neg': fp_block_nans,
            'lower-id1_pos': fp_block_nans, 'lower-id1_neg': fp_block_nans,
            'lower-id2_pos': fp_block_nans, 'lower-id2_neg': fp_block_nans,
            'res-id0_pos': fp_block_nans, 'res-id0_neg': fp_block_nans,
            'res-id1_pos': fp_block_nans, 'res-id1_neg': fp_block_nans,
            'res-id2_pos': fp_block_nans, 'res-id2_neg': fp_block_nans
        }
        if position == 0:
            to_check.update(deg_of_es_check)
        else:
            # set each degree of end-stopping key to full number of nans
            # since it was not computed
            for key in deg_of_es_check.keys():
                to_check[key] = expected_dim

        self.check_num_nan_values(
            first_block, to_check
        )

    def check_pyrblock(self, df, position, expected_dim, depth, model_type):
        first_block = DatabaseTest.get_subtable(df, {'position': position})
        self.assertEqual(first_block.shape[0], expected_dim)
        self.check_equal(
            first_block,
            {
                'model_type': model_type,
                'block_id': 'pyrblock',
                'depth': depth,
            })

        to_check = {
            'ent_per_fmap-bn1': 0, 'ent_per_fmap-bn2': 0,
            'ent_per_fmap-bn3': 0,
            'ent_per_fmap-res': 0, 'act_per_fmap-bn1': 0,
            'act_per_fmap-bn2': 0,
            'act_per_fmap-bn3': 0, 'act_per_fmap-res': 0, 'position': 0,
            'depth': 0,
            'filter_idx': 0, 'ent_per_fmap-upper': expected_dim,
            'ent_per_fmap-dw1': expected_dim,
            'ent_per_fmap-dw2': expected_dim,
            'ent_per_fmap-mult': expected_dim,
            'ent_per_fmap-lower': expected_dim,
            'act_per_fmap-upper': expected_dim,
            'act_per_fmap-dw1': expected_dim, 'act_per_fmap-dw2': expected_dim,
            'act_per_fmap-mult': expected_dim,
            'act_per_fmap-lower': expected_dim, 'kd_angles': expected_dim,
            'kd_min_norm': expected_dim,
            'kd_max_norm': expected_dim,
            'or_angles': expected_dim, 'or_min_norm': expected_dim,
            'or_max_norm': expected_dim,
        }

        # degree of end-stopping is only computed in first layer
        deg_of_es_check = {
            'upper-id0_pos': expected_dim, 'upper-id0_neg': expected_dim,
            'upper-id1_pos': expected_dim, 'upper-id1_neg': expected_dim,
            'upper-id2_pos': expected_dim, 'upper-id2_neg': expected_dim,
            'dw1-id0_pos': expected_dim, 'dw1-id0_neg': expected_dim,
            'dw1-id1_pos': expected_dim, 'dw1-id1_neg': expected_dim,
            'dw1-id2_pos': expected_dim, 'dw1-id2_neg': expected_dim,
            'dw2-id0_pos': expected_dim, 'dw2-id0_neg': expected_dim,
            'dw2-id1_pos': expected_dim, 'dw2-id1_neg': expected_dim,
            'dw2-id2_pos': expected_dim, 'dw2-id2_neg': expected_dim,
            'mult-id0_pos': expected_dim, 'mult-id0_neg': expected_dim,
            'mult-id1_pos': expected_dim, 'mult-id1_neg': expected_dim,
            'mult-id2_pos': expected_dim, 'mult-id2_neg': expected_dim,
            'lower-id0_pos': expected_dim, 'lower-id0_neg': expected_dim,
            'lower-id1_pos': expected_dim, 'lower-id1_neg': expected_dim,
            'lower-id2_pos': expected_dim, 'lower-id2_neg': expected_dim,
            'res-id0_pos': 0, 'res-id0_neg': 0, 'res-id1_pos': 0,
            'res-id1_neg': 0, 'res-id2_pos': 0, 'res-id2_neg': 0
        }
        if position == 0:
            to_check.update(deg_of_es_check)
        else:
            # set each degree of end-stopping key to full number of nans
            # since it was not computed
            for key in deg_of_es_check.keys():
                to_check[key] = expected_dim

        self.check_num_nan_values(
            first_block, to_check
        )

    def check_equal(self, df, key_val_conditions):
        for _, row in df.iterrows():
            for key, val in key_val_conditions.items():
                assert key in df.columns
                self.assertEqual(row[key], val, msg=key)

    def check_num_nan_values(self, df, key_val_conditions):
        for key, val in key_val_conditions.items():
            assert key in df.columns
            num_nan = np.isnan(df[key]).sum()
            self.assertEqual(num_nan, val, msg=key)

    @ staticmethod
    def get_all_degofes_keys():
        block_managers_ = get_block_managers()

        out = []
        for block_manager in block_managers_:
            operations = block_manager._get_relevant_operations(
                None, keys_only=True
            )
            for op in operations:
                for id_num in [0, 1, 2]:
                    for pn in ['pos', 'neg']:
                        out.append(f'{op}-id{id_num}_{pn}')

        return out

    @ staticmethod
    def get_meta_logger_json_keys():
        out = []
        block_managers_ = get_block_managers()
        for block_manager in block_managers_:
            operations = block_manager._get_relevant_operations(
                None, keys_only=True
            )

            for logger in block_manager.cls_loggers_:
                for op in operations:
                    out.append(get_json_meta_key(logger.name, op))

        return out

    @staticmethod
    def get_subtable(df, key_val_conditions):
        where = np.ones(df.shape[0], dtype=bool)
        for key, val in key_val_conditions.items():
            tmp = df[key] == val
            where = np.logical_and(where, tmp)

        return df[where]


def is_nan_or_zero(x):
    return x != x or x == 0.


def plot_deg_of_es_images():
    _plot_def_of_es_images(True, 1, +1)
    _plot_def_of_es_images(True, 1, -1)
    _plot_def_of_es_images(False, 4, +1)
    _plot_def_of_es_images(False, 4, -1)


def _plot_def_of_es_images(is_cifar, stride, sign):
    import matplotlib.pyplot as plt
    image, weight_id0, weight_id1, weight_id2 = create_end_stopping_image_box(
        sign, is_cifar=is_cifar)

    kwargs = {'interpolation': 'nearest'}

    new_image = (0. * weight_id1.copy())
    new_image[image[::stride, ::stride] > 0] = 1.
    new_image[weight_id0 > 0] += 1.
    new_image[weight_id1 > 0] += 2.
    new_image[weight_id2 > 0] += 3.

    _, ax = plt.subplots(2, 3)
    ax[0][0].imshow(weight_id0, **kwargs)
    ax[0][1].imshow(weight_id1, **kwargs)
    ax[0][2].imshow(weight_id2, **kwargs)
    ax[1][0].imshow(image, **kwargs)
    ax[1][1].imshow(new_image, **kwargs)
    prefix = 'cifar_' if is_cifar else 'imnet_'
    prefix = prefix + 'pos_' if sign > 0 else prefix + 'neg_'
    plt.savefig(join(BASE_FOLDER, prefix + 'degree_of_endstopping_image.png'))


if __name__ == '__main__':
    plot_deg_of_es_images()
    db_folders = search_rgx(r'CifarJOVFPNet_N\d_s\d+', DATABASE_PATH)
    assert db_folders

    unittest.main()
