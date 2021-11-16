"""
To start, you must set the environment variable `IMAGENET` to point to
wherever you have saved the ILSVRC2012 dataset. You must also set the
`TENSORPACK_DATASET` environment variable, because tensorpack may download
some things itself.

Kaggle download imagenet

to be consistent with the JOV-Paper use the forked repo:
https://github.com/pgruening/sequential-imagenet-dataloader

How to download ImageNet:
* Create Kaggle useraccount
* Create json file in /root/.kaggle/kaggle.json:
    {"username":"??","key":"??"} (look at notes)

* run in terminal:
pip install kaggle
kaggle competitions download -c imagenet-object-localization-challenge

... this will download 150gb of data

* run in terminal:
unzip imagent archive

* install the following packages with pip
tensorpack 0.8.7
tensorflow 1.4

* run in terminal:
python preprocess_sequential.py

"""
from imagenet_seq.data import Loader as SeqINLoader
import os
from config import IMAGENET_LMDB

# set environment variable to imagenet lmbdb file
os.environ["IMAGENET"] = IMAGENET_LMDB


def get_dataloader(is_train, batch_size, num_workers, use_cuda=False):
    if use_cuda:
        print('data_loader: using cuda')

    if is_train:
        return SeqINLoader(
            'train', batch_size=batch_size, shuffle=True,
            num_workers=num_workers, cuda=use_cuda
        )
    else:
        return SeqINLoader(
            'val', batch_size=batch_size, shuffle=False,
            num_workers=num_workers, cuda=use_cuda
        )


def _test_dataloader():
    dataloader = SeqINLoader(
        'val', batch_size=128, shuffle=False,
        num_workers=1, cuda=False
    )

    for x, y in dataloader:
        print(x.max(), x.mean(), x.std())


if __name__ == '__main__':
    _test_dataloader()
