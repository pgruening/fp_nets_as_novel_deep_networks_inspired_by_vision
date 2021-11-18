from os.path import join

import numpy as np
import torch
from datasets.ds_cifar10_compression_test import Cifar10JpegCompression
from DLBio.helpers import check_mkdir, search_rgx
from DLBio.pytorch_helpers import get_device
from helpers import load_model
from log_tensorboard import log_tensorboard
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_FOLDER = 'experiments/exp_6'
MODEL_FOLDERS = [
    'experiments/exp_0/exp_data/trained_models',
    'experiments/exp_4/exp_data/trained_models'

]
RGXS = [
    r'(CifarJOVFPNet|CifarPyrResNet|CifarResNet)_N(\d)_s(\d+)',
    r'(CifarJOVFPNet-RNBasic)_N(\d)_s(\d+)'

]

PRED_SAVE_PATH = join(BASE_FOLDER, 'model_predictions')


def create_data():
    folders_ = get_folders()
    print(f'found {len(folders_)} folders.')

    for folder in folders_:
        compute_and_save_results(folder)


def get_folders():
    folders_ = []
    for rgx, path in zip(RGXS, MODEL_FOLDERS):
        tmp = search_rgx(rgx, path)
        assert tmp
        folders_ += [join(path, x) for x in tmp]

    return folders_


def compute_and_save_results(folder):
    """Loads a model from '[folder]/model.pt'. The model classifies ten
    different compression rates {0,...,90} of each image of the Cifar-10
    testset. 
    All predictions and the actual class are save as a numpy file. The file
    contains a matrix of shape (10000, 11). Each row corresponds to an image,
    the first column is the actual label, all subsequent columns are the model
    predictions at compression 0, 10, 20, ..., 90.

    Parameters
    ----------
    folder : string
        folder containing the model with filename model.pt
    """
    print(f'Running folder: {folder}')
    device = get_device()
    model = load_model(
        join(folder, 'opt.json'), device,
        new_model_path=join(folder, 'model.pt')
    )
    dataset = Cifar10JpegCompression()

    out = []
    for image, label in tqdm(dataset):
        with torch.no_grad():
            prediction = model(image.to(device))
        p_classes = prediction.max(1)[1]

        # save as (label, predictions)
        label = label.numpy()
        p_classes = p_classes.cpu().numpy()

        out.append(np.concatenate([label, p_classes]))

    out = np.stack(out, 0)

    name = folder.split('/')[-1]
    out_path = join(PRED_SAVE_PATH, name + '.npy')
    check_mkdir(out_path)
    np.save(out_path, out)


class FakeDataLoader():
    def __init__(self, dataset):
        self.dataset = dataset
        self.idx = 0

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.idx >= len(self.dataset):
            raise StopIteration()

        image, label = self.dataset[self.idx]
        self.idx += 1

        label = torch.cat([label] * image.shape[0], 0)

        return image, label

    def __iter__(self):
        return self


def check_data_on_tensorboard():
    dataset = Cifar10JpegCompression()
    dataloaders = {'val': FakeDataLoader(dataset)}
    log_tensorboard(
        BASE_FOLDER, join(BASE_FOLDER, 'tboard'),
        dataloaders=dataloaders, numBatches=100
    )


if __name__ == '__main__':
    # create_data()
    check_data_on_tensorboard()
