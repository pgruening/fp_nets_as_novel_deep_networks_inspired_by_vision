import numpy as np
import torch
from datasets import data_getter
from DLBio.pytorch_helpers import cuda_to_numpy
from DLBio.helpers import to_uint8_image, check_mkdir
from scipy.stats import pearsonr
from tqdm import tqdm
from os.path import join
import cv2

import matplotlib.pyplot as plt

BATCH_SIZE = 1

OUT_FOLDER = 'imagenet_examples'
NUM_IMAGES = 500


def run():
    def get_score(a, b):
        a = a.cpu().numpy()
        b = b.cpu().numpy()

        return pearsonr(a.flatten(), b.flatten())[0]

    def make_plot(a, b, ctr, score):
        a = cuda_to_numpy(a[0, ...])
        a = to_uint8_image(a)

        b = cuda_to_numpy(b[0, ...])
        b = to_uint8_image(b)

        _, ax = plt.subplots(1, 2)
        ax[0].imshow(a)
        ax[1].imshow(b)
        ax[0].set_title(f'PLCC: {round(score,3)}')
        path = join(OUT_FOLDER, 'plcc', str(ctr).zfill(4) + '.png')
        check_mkdir(path)
        plt.savefig(path)
        plt.close()

    dataloader_seq = data_getter.get_data_loaders(
        'sequential_imagenet', batch_size=1, num_workers=1,
        use_cuda=[False]
    )['val']

    dataloader_legacy = data_getter.get_data_loaders(
        'legacy_imagenet', batch_size=1, num_workers=0,
        alternate_aug=['facebook'],
        use_from=['pc14'], bgr_to_rgb=[False],
    )['val']

    ctr = 0
    print('starting ...')
    for x, y in dataloader_seq:
        if y != 0:
            continue

        for x_leg, y_leg in dataloader_legacy:
            score = get_score(x_leg, x)
            #make_plot(x, x_leg, ctr, score)

            if score > 0.99:
                mean_ = []
                std_ = []
                for n in range(3):
                    x0 = x[0, n, ...].min()
                    x1 = x[0, n, ...].max()

                    y0 = x_leg[0, n, ...].min()
                    y1 = x_leg[0, n, ...].max()

                    std = (y1 - y0) / (x1 - x0)
                    mean = y0 - std * x0
                    mean_.append(mean)
                    std_.append(std)

                print(f'mean: {mean_}')
                print(f'std: {std_}')
                xxx = 0

            ctr += 1
            if y_leg != 0:
                return


def run_3():
    def save_some_files(dataloader, folder):
        ctr = 0
        for x, y in dataloader:
            x = cuda_to_numpy(x[0, ...])
            x = to_uint8_image(x)
            if int(y) > 20:
                continue
            y = str(int(y[0])).zfill(4)
            path = join(folder, y, str(ctr).zfill(4) + '.png')
            check_mkdir(path)
            cv2.imwrite(path, x)
            ctr += 1
            if ctr > NUM_IMAGES:
                return

            if ctr % 10 == 0:
                print(ctr)

    if True:
        dataloader_seq = data_getter.get_data_loaders(
            'sequential_imagenet', batch_size=1, num_workers=1,
            use_cuda=[False]
        )['val']

        save_some_files(dataloader_seq, join(OUT_FOLDER, 'sequential'))

    dataloader_legacy = data_getter.get_data_loaders(
        'legacy_imagenet', batch_size=1, num_workers=0,
        alternate_aug=['facebook'],
        use_from=['pc14']
    )['val']
    save_some_files(dataloader_legacy, join(OUT_FOLDER, 'legacy'))


def run_1():
    def get_vals_(dataloader):

        mean = torch.zeros(3)
        std = torch.zeros(3)
        ctr = 0
        for x, _ in tqdm(dataloader):
            mean += x.mean([0, 2, 3])
            std += x.std([0, 2, 3])
            ctr += 1

        print(
            'mean: ',
            [round(float(x), 3) for x in list(mean / ctr)]

        )

        print(
            'std: ',
            [round(float(x), 3) for x in list(std / ctr)]
        )

    if False:
        dataloader_seq = data_getter.get_data_loaders(
            'sequential_imagenet', batch_size=500, num_workers=1,
            use_cuda=[False]
        )['val']

        get_vals_(dataloader_seq)

    dataloader_legacy = data_getter.get_data_loaders(
        'legacy_imagenet', batch_size=500, num_workers=0,
        alternate_aug=['facebook'],
        use_from=['pc14']
    )['val']
    get_vals_(dataloader_legacy)

    # sequential
    # mean: [-0.21, -0.109, -0.051]
    # std: [0.558, 0.535, 0.551]

    # legacy
    # mean:  [0.476, 0.447, 0.396]
    # std:  [0.275, 0.267, 0.277]
    xxx = 0


def run_2():
    dataloader_seq = data_getter.get_data_loaders(
        'sequential_imagenet', batch_size=BATCH_SIZE, num_workers=1,
        use_cuda=[False]
    )['val']
    dataloader_legacy = data_getter.get_data_loaders(
        'legacy_imagenet', batch_size=BATCH_SIZE, num_workers=0,
        alternate_aug=['facebook'],
        use_from=['pc14']
    )['val']

    for x_seq, _ in dataloader_seq:
        max_plcc = 0.
        ctr = 0
        best_fit = None
        for x_leg, _ in dataloader_legacy:
            x_seq = np.array(x_seq)
            x_leg = np.array(x_leg)
            plcc = pearsonr(x_leg.flatten(), x_seq.flatten())[0]

            if plcc > 1. - 1e-3:
                xxx = 0

            if ctr % 100 == 0:
                print(ctr, max_plcc, plcc)
            if plcc > max_plcc:
                max_plcc = plcc
                best_fit = [x_seq, x_leg]

            ctr += 1

        xxx = 0


if __name__ == '__main__':
    run()
