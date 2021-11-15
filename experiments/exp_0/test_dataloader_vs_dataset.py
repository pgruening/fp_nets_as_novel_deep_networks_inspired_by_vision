from os.path import join

import numpy as np
import torch
from DLBio.helpers import load_json, search_rgx
from DLBio.pytorch_helpers import get_device
from tqdm import tqdm

BASE_FOLDER = '/nfshome/gruening/my_code/DLBio_repos/fp_net_after_jov/experiments/exp_0/exp_data/trained_models'
RGX = r'(CifarJOVFPNet|CifarPyrResNet|CifarResNet)_N(\d)_s(\d+)'


def run():
    import sys
    sys.path.append('/nfshome/gruening/my_code/DLBio_repos/fp_net_after_jov')
    from datasets.ds_cifar10 import get_dataset
    from helpers import load_model

    dataset = get_dataset(False)
    folders_ = search_rgx(RGX, BASE_FOLDER)
    for folder in folders_:
        print(folder)
        with torch.no_grad():
            test_model_out_same_as_in_log(
                join(BASE_FOLDER, folder), dataset, load_model)


def test_model_out_same_as_in_log(folder, dataset, load_model):
    device = get_device()
    opt = load_json(join(folder, 'opt.json'))
    assert opt is not None

    model = load_model(
        opt, device,
        new_model_path=join(folder, 'model.pt')
    ).eval()

    X = []
    Y = []

    for x, y in tqdm(dataset):
        x = x.unsqueeze(0).to(device)
        pred = model(x)

        pred = pred.max(-1)[1]
        pred = pred.cpu().item()

        X.append(pred)
        Y.append(y)

    log = load_json(join(folder, 'log.json'))
    assert log is not None

    log_val = max(log['val_acc'])

    new_val = (np.array(X) == np.array(Y)).mean()

    assert np.abs(log_val - new_val) < 1e-9


if __name__ == '__main__':
    run()
