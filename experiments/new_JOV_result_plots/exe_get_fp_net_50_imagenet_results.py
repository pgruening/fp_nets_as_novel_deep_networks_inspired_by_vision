"""
Code to reproduce JOV-Paper results for the FP-net-50, based on the ResNet-50 
trained on the sequential-dataloader ImageNet dataset. This dataloader has some 
augmentations and pre-processing steps that differ from the 'legacy'-ImageNet 
dataset where the MobileNet was trained on.

Numbers in the paper:

(from https://github.com/tensorpack/tensorpack/tree/master/examples/ResNet):
ResNet-50 (baseline)25.5 23.61 

Our Results:
FP-net-50 (q=0.8)24.3 23.80
FP-net-50 (q=1)25.9 23.24

"""
import json
import os
from os.path import isfile, join

import cv2
import numpy as np
import torch
from datasets import data_getter
from DLBio import pt_training
from DLBio.helpers import check_mkdir, load_json, to_uint8_image
from DLBio.pytorch_helpers import cuda_to_numpy, get_device
from helpers import load_model
from tqdm import tqdm
import config

MODEL_FOLDERS_ = [
    'experiments/imagenet_resnet/resnet50_imagenet_layer_start_q08',
    # 'experiments/imagenet_resnet/resnet50_imagenet_layer_start_q1'
]
DEVICE = get_device()

BATCH_SIZE = 50

USE_SEQUENTIAL = True


def run():
    os.environ["IMAGENET"] = config.IMNET_LMDB_LINK
    if USE_SEQUENTIAL:
        dataloader = data_getter.get_data_loaders(
            'sequential_imagenet', batch_size=BATCH_SIZE, num_workers=1,
            use_cuda=[False]
        )['val']
    else:
        dataloader = data_getter.get_data_loaders(
            'legacy_imagenet', batch_size=BATCH_SIZE, num_workers=0,
            alternate_aug=['facebook'],
            use_from=['link'],
            bgr_to_rgb=[False]
        )['val']

    for folder in MODEL_FOLDERS_:
        model = load_model(
            join(folder, 'opt.json'),
            'cpu',
            strict=True,
            new_model_path=join(folder, 'model.pt'),
            map_location=torch.device(DEVICE),
            from_par_gpu=True
        ).eval()
        print(model.layer1[0].block.upper[0].weight[0, 0, 0])
        model = model.to(DEVICE)

        out = _eval(model, dataloader)
        accuracy = _acc(out)

        # TODO: the counter iteration does not seem to work
        ctr = 0
        file_name = f'validation_acc_{str(ctr).zfill(2)}.json'
        while isfile(join(folder, file_name)):
            ctr += 1
            file_name = f'validation_acc_{str(ctr).zfill(2)}.json'

        with open(join(folder, file_name), 'w') as file:
            json.dump({
                'acc': float(accuracy),
                'error': float((1. - accuracy) * 100.),
            }, file)

        file_name = f'validation_results_{str(ctr).zfill(2)}.json'
        with open(join(folder, file_name), 'w') as file:
            json.dump(out, file)

        print(folder)
        print(f'Accuracy: {accuracy}')
        print(f'Error: {(1. - accuracy) * 100.}')
        print('-' * 80)


def _eval(model, dataloader):
    X = []
    Y = []
    ctr = 0
    for sample in tqdm(dataloader):
        images, targets = sample[0].to(DEVICE), sample[1].to(DEVICE)
        #print(images.mean(), images.std(), images.min(), images.max())
        #assert images.shape[0] == 1
        pred = model(images)
        y_pred = np.array(pred.detach().cpu())
        y_pred = np.argmax(y_pred, 1)

        targets = np.array(targets.detach().cpu())

        X += list(y_pred)
        Y += list(targets)

        ctr += 1
        if ctr % 10 == 0:
            accuracy = _acc({'pred': X, 'tar': Y})
            print('Error ', (1. - accuracy) * 100.)

    print(len(X))
    print(len(Y))

    X = [int(x) for x in X]
    Y = [int(y) for y in Y]

    out = {'pred': X, 'tar': Y}
    return out


def _acc(result_dict):
    pred = np.array(result_dict['pred'])
    tar = np.array(result_dict['tar'])
    return (pred == tar).mean()


def save_some_images(n=100):
    os.environ["IMAGENET"] = '/nfshome/gruening/my_code'
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


if __name__ == "__main__":
    # save_some_images()
    with torch.no_grad():
        run()
