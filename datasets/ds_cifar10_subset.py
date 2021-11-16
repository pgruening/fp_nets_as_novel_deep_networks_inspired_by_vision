"""
This is only supposed for evaluation purposes. Hence, there is no training set.
"""
import json
import random
import warnings
from os.path import isfile

import torchvision.datasets as datasets
from config import DATA_FOLDER
from DLBio.helpers import load_json
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms

# this is the imagenet normalization:
# https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
# cifar-10 normalization: [0.4914, 0.4822, 0.4465]; [0.2470, 0.2435, 0.2616]
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

NUMBER_OF_SAMPLES = 500

PATH = 'datasets/ds_cifar10_subset_indices.json'


def get_data_loader(*, batch_size, num_workers=0, data_path=DATA_FOLDER, pin_memory=True):
    if not isfile(PATH):
        warnings.warn('Creating new index file.')
        create_random_indices()

    indices = load_json(PATH)
    dataset = get_dataset()

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=SubsetRandomSampler(indices)
    )
    return data_loader


def get_dataset():
    return datasets.CIFAR10(root=DATA_FOLDER, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        NORMALIZE,
    ]), download=True)


def create_random_indices():
    dataset = get_dataset()
    out = []
    C = len(dataset.classes)
    n = NUMBER_OF_SAMPLES // C
    for c in range(C):
        tmp = [
            i for i, t in enumerate(dataset.targets) if t == c
        ]
        random.shuffle(tmp)
        tmp = tmp[:n]

        out += tmp

    with open(PATH, 'w') as file:
        json.dump(out, file)


if __name__ == '__main__':
    create_random_indices()
