import os
from os.path import join

import cv2
import numpy as np
from datasets import data_getter
from DLBio.helpers import check_mkdir, to_uint8_image
from DLBio.pytorch_helpers import cuda_to_numpy, get_device
import config


def save_some_images(n=100):
    os.environ["IMAGENET"] = config.IMAGENET_LMDB
    dataloader = data_getter.get_data_loaders(
        'sequential_imagenet', 1, num_workers=2, use_cuda=False
    )['val']

    check_mkdir('im_net_entropy')
    check_mkdir(join('im_net_entropy', 'png'))
    check_mkdir(join('im_net_entropy', 'npy'))

    for i, sample in enumerate(dataloader):
        arr = sample[0][0, ...].detach().cpu().numpy()
        np.save(join('im_net_entropy', 'npy', f'test_{i}.npy'), arr)

        image = cuda_to_numpy(sample[0][0, ...])
        image = to_uint8_image(image)

        cv2.imwrite(join('im_net_entropy', 'png', f'test_{i}.png'), image)

        if i > n:
            return
