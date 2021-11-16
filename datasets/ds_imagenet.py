import os
import random
from os.path import join, isdir

import cv2
import torchvision.transforms

import torch
from torch.utils.data import DataLoader, Dataset
import glob
from tensorpack import imgaug
import numpy as np
import config

# from pc48
TRAIN_IMAGES = config.IM_NET_TRAIN
VAL_IMAGES = config.IM_NET_VAL

IMAGENET_RESIZE = 224

# augmentation taken from
# https://github.com/fastai/imagenet-fast/blob/master/imagenet_nv/fastai_imagenet.py
# normalization seems to be the same as in pytorch-imagenet training
# https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198

MEAN = (
    [.5, .5, .5]
)
STD = (
    [.5, .5, .5]
)


def get_aug(is_train=None, alternate_aug=None):
    if alternate_aug == 'facebook':
        assert not is_train
        augmentors = [
            # torchvision.transforms.ToPILImage('RGB'),
            lambda x: imgaug.ResizeShortestEdge(
                256, cv2.INTER_CUBIC).augment(x),
            lambda x: imgaug.CenterCrop((224, 224)).augment(x),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=list(MEAN),
                std=list(STD)
            )
        ]
        return torchvision.transforms.Compose(augmentors)

    assert is_train is not None
    if is_train:
        aug = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage('RGB'),
            torchvision.transforms.RandomResizedCrop(IMAGENET_RESIZE),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(.4, .4, .4),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])

        ])
    else:
        aug = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage('RGB'),
            torchvision.transforms.Resize(int(IMAGENET_RESIZE * 1.14)),
            torchvision.transforms.CenterCrop(IMAGENET_RESIZE),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

    return aug


def get_dataloader(is_train=True, batch_size=32, num_workers=0, **kwargs):
    which_pc = kwargs.get('use_from', 'pc48')
    alternate_aug = kwargs.get('alternate_aug', None)
    bgr_to_rgb = kwargs.get('bgr_to_rgb', True)
    if alternate_aug == 'facebook':
        assert not bgr_to_rgb

    train_image = TRAIN_IMAGES[which_pc]
    val_images = VAL_IMAGES[which_pc]

    if is_train:
        dataset = ImageNetDataset(
            train_image, get_aug(is_train, alternate_aug), bgr_to_rgb
        )
    else:
        dataset = ImageNetDataset(
            val_images, get_aug(is_train, alternate_aug), bgr_to_rgb
        )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True
    )

    return data_loader


class ImageNetDataset(Dataset):
    im_idx = 0
    lab_idx = 1
    to_tens = torchvision.transforms.ToTensor()

    def __init__(self, folder, aug, bgr_to_rgb):
        """Dataset that assumes that all Imagenet images of the specific dataset
        (train or validation) are stored. In this folder there are 1000
        subfolders ("n0[7-digit-number]")for each class, containing JPEG images
        named ILSVR2021_[val|train]_[8-digit-number].JPEG

        Parameters
        ----------
        folder : str
            path to either train or validation dataset folder
        aug : torchvision.transforms.Compose
            Preprocessing and data augmentation applied to each image.
        """
        self.bgr_to_rgb = bgr_to_rgb
        self.images = dict()
        self.classes = []
        ctr_im = 0  # images
        ctr_cl = 0  # classes

        assert isdir(folder)
        directories = next(os.walk(folder))[1]
        directories = sorted(directories)

        for root in directories:
            # print(root)
            tmp = join(folder, root, '*.JPEG')
            files_ = glob.glob(tmp)
            assert files_, f'no files found for {tmp}'

            self.images.update(
                {ctr_im + i: (x, ctr_cl)
                 for i, x in enumerate(files_)}
            )
            ctr_im += len(files_)
            ctr_cl += 1

            self.classes.append(root)

        self.N = ctr_im - 1
        assert ctr_cl == 1000
        # print('done.')

        self.aug = aug

    def __getitem__(self, index):
        image = cv2.imread(self.images[index][self.im_idx])
        if self.bgr_to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.images[index][self.lab_idx]

        if self.aug is not None:
            image = self.aug(image)

        label = torch.tensor(label).long()

        return image, label

    def __len__(self):
        return self.N


def _test_same_classes():
    train_image = TRAIN_IMAGES['pc48']
    val_images = VAL_IMAGES['pc48']

    train = ImageNetDataset(train_image, get_aug(True)).classes
    val = ImageNetDataset(val_images, get_aug(False)).classes

    for x, y in zip(train, val):
        assert x == y

    print('Test successful')


def _check_augmentation():
    val_images = VAL_IMAGES['pc48']
    val = ImageNetDataset(val_images, get_aug(False, alternate_aug='facebook'))
    for x, y in val:
        pass


if __name__ == "__main__":
    # did pass test
    # _test_same_classes()
    _check_augmentation()
