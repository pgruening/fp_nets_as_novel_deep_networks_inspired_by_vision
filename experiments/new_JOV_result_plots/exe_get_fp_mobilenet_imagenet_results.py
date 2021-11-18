"""
Code to reproduce JOV-Paper results for the FP-net-50, based on the ResNet-50 
trained on the sequential-dataloader ImageNet dataset. This dataloader has some 
augmentations and pre-processing steps that differ from the 'legacy'-ImageNet 
dataset where the MobileNet was trained on.
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
from DLBio.pytorch_helpers import cuda_to_numpy, get_device, get_num_params
from helpers import load_model
from tqdm import tqdm

MODEL_FOLDERS_ = [
    'experiments/imagenet_mobilenet/vanilla_mobilenet_imagenet',
    'experiments/imagenet_mobilenet/mobilenet_layer_start_q_3',
    'experiments/imagenet_mobilenet/mobilenet_layer_start_q_2',

]
DEVICE = get_device()

BATCH_SIZE = 50


def run():
    dataloader = data_getter.get_data_loaders(
        'legacy_imagenet', batch_size=BATCH_SIZE, num_workers=0,
        use_from=['link']
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
        model = model.to(DEVICE)
        print(folder.split('/')[-1])
        print(f'num params: {get_num_params(model)}')

        out = _eval(model, dataloader)
        accuracy = _acc(out)

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
    for sample in tqdm(dataloader):
        images, targets = sample[0].to(DEVICE), sample[1].to(DEVICE)
        #assert images.shape[0] == 1
        pred = model(images)
        y_pred = np.array(pred.detach().cpu())
        y_pred = np.argmax(y_pred, 1)

        targets = np.array(targets.detach().cpu())

        X += list(y_pred)
        Y += list(targets)

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


def setup(in_options):
    if in_options.device is not None:
        pt_training.set_device(in_options.device)

    folder = in_options.model_folder
    model_folder = folder.split('/')[-1]

    options = load_json(join(folder, 'opt.json'))
    assert options is not None, f'no options at: {folder}'

    model = load_model(options, 'cpu', map_location=torch.device(
        'cpu'), new_model_path=join(folder, 'model.pt'),
        from_par_gpu=in_options.from_par_gpu
    ).to(get_device()).eval()

    # TODO: should we do this?
    #model.pre_transform = NORMALIZE

    model_type = options['model_type']
    print(model_folder, model_type)
    return model, model_folder, model_type


if __name__ == "__main__":
    # save_some_images()
    with torch.no_grad():
        run()
