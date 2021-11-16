import glob
from os.path import join

import numpy as np
import torch
from config import IM_NET_ENTROPY_SUBSET
from torch.utils.data import DataLoader, Dataset

# No worries about lacking normalization etc.:
# all augmentation steps are already save in a numpy file
# see exe_create_imagenet_subset_for_entropy_eval.py


def get_data_loader(batch_size, num_workers=0):

    dataset = ImageList()

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )


class ImageList(Dataset):
    def __init__(self):
        self.images = [
            np.load(x) for x in glob.glob(
                join(IM_NET_ENTROPY_SUBSET, '*.npy')
            )
        ]

    def __getitem__(self, index):
        x = torch.tensor(self.images[index])
        y = torch.tensor([0]).long()

        return x, y

    def __len__(self):
        return len(self.images)
