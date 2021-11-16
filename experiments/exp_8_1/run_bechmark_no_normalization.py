"""
Using the repositroy: https://github.com/RobustBench/robustbench

Note that for our standardized evaluation of Linf-robustness we use the full 
version of AutoAttack which is slower but more accurate (for that just use 
adversary = AutoAttack(model, norm='Linf', eps=8/255)).

torchattacks
https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Performance%20Comparison%20(CIFAR10).ipynb

"""
import argparse
import json
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchattacks
from datasets.ds_cifar10 import NORMALIZE, get_data_loader
from DLBio import pt_training
from DLBio.helpers import check_mkdir, load_json, to_uint8_image
from DLBio.pytorch_helpers import get_device, get_num_params
from helpers import load_model
import random

BASE_FOLDER = 'fp_net_after_jov/experiments/exp_8_1'


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str)
    parser.add_argument('--device', type=int)

    return parser.parse_args()


def run_fgsm(in_options):
    model, model_name = setup(in_options)
    # Usually, the data are normalized during batch-creation in the dataloader.
    # However, the FGSM method expects images in range [0, 1]. Thus, the
    # normalization step is done during model inference. By setting the model's
    # 'pre_transform' attribute to sth. different than None (default value).
    model.pre_transform = NORMALIZE
    # test dataset
    dataloader = get_data_loader(
        is_train=False, batch_size=512, num_workers=0, do_normalize=False
    )
    classes = dataloader.dataset.classes

    out_dict = {
        'folder': in_options.folder,
        'model_name': model_name,
        'num_params': float(get_num_params(model, False)),
    }

    # from: https://pypi.org/project/torchattacks/
    # All images should be scaled to [0, 1] with transform[to.Tensor()]
    # before used in attacks.
    device = get_device()
    # run for different epsilon values
    for i in [0, 1, 2, 4, 8, 16]:
        ctr = 0
        atk = torchattacks.FGSM(model, eps=float(i) / 255.)
        predictions = []
        orig_predictions = []
        labels_ = []
        # batch-wise load images and distort them (atk)
        for images, labels in dataloader:
            assert images.min() >= 0. and images.max() <= 1.

            orig_pred = model(images.to(device))
            orig_pred = orig_pred.max(-1)[1].cpu().numpy()

            adv_images = atk(images, labels) if i > 0 else images.to(device)
            with torch.no_grad():
                pred = model(adv_images)
            confidences_ = torch.softmax(pred, -1)
            pred = pred.max(-1)[1].cpu().numpy()
            labels = labels.cpu().numpy()

            # create an example image for a successful adversarial attack
            is_wrong = list(np.nonzero(pred != labels)[0])
            idx = random.choice(is_wrong)
            confidence = confidences_[idx, pred[idx]].item()
            pred_class = classes[pred[idx]]
            true_class = classes[labels[idx]]

            if i >= 0:
                plot_example(
                    images[idx, ...],
                    adv_images[idx, ...],
                    model_name,
                    i,
                    ctr,
                    pred_class,
                    true_class,
                    confidence
                )
                ctr += 1

            predictions.append(pred)
            labels_.append(labels)
            orig_predictions.append(orig_pred)

        # aggregate all data and write them to a json file

        predictions = np.concatenate(predictions)
        orig_predictions = np.concatenate(orig_predictions)
        labels_ = np.concatenate(labels_)

        assert predictions.shape[0] == 10000
        assert orig_predictions.shape[0] == 10000
        assert labels_.shape[0] == 10000

        acc = (predictions == labels_).mean()
        out_dict[f'robust_acc_{i}'] = float(acc)

        orig_acc = (orig_predictions == labels_).mean()
        out_dict[f'orig_acc_{i}'] = float(orig_acc)

        number_of_changes = (predictions != orig_predictions).mean()
        out_dict[f'num_changes_{i}'] = float(number_of_changes)

        print(
            f'{model_name} (eps = {i}): orig: {orig_acc}, acc: {acc}, nc: {number_of_changes}'
        )

    log_path = join(BASE_FOLDER, f'logs/{model_name}.json')
    check_mkdir(log_path)

    with open(log_path, 'w') as file:
        json.dump(out_dict, file)


def plot_example(image, adv_ex, model_name, eps, ctr, pred_class, true_class, confidence):
    # (C, H, W) -> (H, W, C)
    image = image.permute([1, 2, 0]).detach().cpu().numpy()
    adv_ex = adv_ex.permute([1, 2, 0]).detach().cpu().numpy()

    delta = (
        127. + 255. * (adv_ex - image)
    ).astype('uint8')

    image = (255. * image).astype('uint8')
    adv_ex = (255. * adv_ex).astype('uint8')

    _, ax = plt.subplots(1, 3)
    ax[0].imshow(image)
    ax[0].set_title(
        f'GT: {true_class}'
    )

    ax[1].imshow(delta)
    ax[1].set_title(
        f'Eps: {int(eps)}/255'
    )

    ax[2].imshow(adv_ex)
    ax[2].set_title(
        f'P: {pred_class}, Conf.: {round(confidence,3)}'
    )

    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    plt.tight_layout()

    out_path = join(
        BASE_FOLDER, 'examples', model_name,
        str(eps).zfill(2), str(ctr).zfill(4) + '.png'
    )
    check_mkdir(out_path)

    plt.savefig(out_path)
    plt.close()


def setup(in_options):
    if in_options.device is not None:
        pt_training.set_device(in_options.device)

    folder = in_options.folder
    model_name = folder.split('/')[-1]

    options = load_json(join(folder, 'opt.json'))
    assert options is not None, f'no options at: {folder}'

    model = load_model(options, 'cpu', map_location=torch.device(
        'cpu'), new_model_path=join(folder, 'model.pt')
    ).to(get_device()).eval()
    model.pre_transform = NORMALIZE

    print(model_name)
    return model, model_name


if __name__ == '__main__':
    OPTIONS = get_options()
    run_fgsm(OPTIONS)
