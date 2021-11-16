"""
Code copied from:
https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/trainer.py
"""
import torchvision.datasets as datasets
from config import CIFAR_10_ENTROPY_SPLIT_FILE, DATA_FOLDER
from DLBio.helpers import load_json
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms

from .helpers import HiddenPrints

# this is the imagenet normalization:
# https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
# cifar-10 normalization: [0.4914, 0.4822, 0.4465]; [0.2470, 0.2435, 0.2616]
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def get_data_loader(*, is_train, batch_size, num_workers=0, do_normalize=True, pin_memory=True):
    assert not is_train

    indices = load_json(CIFAR_10_ENTROPY_SPLIT_FILE)
    assert indices, f'No file found at {CIFAR_10_ENTROPY_SPLIT_FILE}'

    # Suppress annoying download print statement
    with HiddenPrints():
        dataset = get_dataset(is_train, do_normalize)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=SubsetSampler(indices),
        pin_memory=pin_memory,
    )
    return data_loader


def get_dataset(is_train, do_normalize=True):
    """
    Provide the Cifar-10 dataset with data augmentation from
    He et al.'s Deep Residual learning:

    "We follow the simple data augmentation in [24] for training: 4 pixels
    are padded on each side,and a 32×32 crop is randomly sampled from
    the padded image or its horizontal flip. For testing, we only
    evaluate the single view of the original 32×32 image."

    The data are downloaded to DATA_FOLDER.

    is_train: bool
        return the Training dataset or validation dataset with the right
        data augmentation
    do_normalize: bool
        add imagenet normalization to pre-processing chain

    returns Pytorch Dataset


    """
    if is_train:
        transform = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),  # second number is padding
            transforms.ToTensor(),
        ]
        if do_normalize:
            transform.append(NORMALIZE)

        transform = transforms.Compose(transform)
        return datasets.CIFAR10(
            root=DATA_FOLDER, train=True, transform=transform, download=True
        )

    else:
        transform = [
            transforms.ToTensor(),
        ]
        if do_normalize:
            transform.append(NORMALIZE)

        transform = transforms.Compose(transform)

        return datasets.CIFAR10(
            root=DATA_FOLDER, train=False, transform=transform, download=True
        )

# adopted from SubsetRandomSampler in
# https://github.com/pytorch/pytorch/blob/fe805794ac7a224c588a62955002de3aa270b75c/torch/utils/data/sampler.py#L57-L71
# no random permutation as in SubsetRandomSampler


class SubsetSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        # returns a generator
        # https://wiki.python.org/moin/Generators
        # In fact, we can turn a list comprehension into a generator expression
        # by replacing the square brackets ("[ ]") with parentheses
        return (idx for idx in self.indices)

    def __len__(self):
        return len(self.indices)
