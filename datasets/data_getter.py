from . import ds_cifar10
from . import ds_cifar10_subset
from . import ds_mnist
from . import ds_cifar10_entropy_val_subset
from . import ds_imnet_entropy
from . import ds_imagenet
import warnings


def get_data_loaders(dataset, *, batch_size, num_workers, **kwargs):
    if dataset == 'cifar_10':
        use_val = kwargs.get('use_val', [True])[0]
        if use_val:
            val_data_loader = ds_cifar10.get_data_loader(
                is_train=False, batch_size=batch_size,
                num_workers=num_workers
            )
        else:
            warnings.warn('No validation data-loader is used')
            val_data_loader = None

        return {
            'train': ds_cifar10.get_data_loader(
                is_train=True, batch_size=batch_size,
                num_workers=num_workers
            ),
            'val': val_data_loader,
            'test': None
        }
    elif dataset == 'cifar_10_subset':
        return {
            'train': None,
            'val': ds_cifar10_subset.get_data_loader(
                batch_size=batch_size,
                num_workers=num_workers
            ),
            'test': None
        }
    elif dataset == 'mnist':
        return{
            'train': ds_mnist.get_dataloader(
                True, batch_size=batch_size),
            'val': ds_mnist.get_dataloader(
                False, batch_size=batch_size),
            'test': None
        }

    elif dataset == 'cifar_10_validation_subset':
        return {
            'train': None,
            'val': ds_cifar10_entropy_val_subset.get_data_loader(
                is_train=False, batch_size=batch_size,
                num_workers=num_workers
            ),
            'test': None
        }

    elif dataset == 'im_net_entropy':
        return {
            'train': None,
            'val': ds_imnet_entropy.get_data_loader(
                batch_size, num_workers=num_workers
            ),
            'test': None
        }

    elif dataset == 'sequential_imagenet':
        from . import ds_imagenet_sequential
        use_cuda = kwargs.get('use_cuda', [False])[0]

        return {
            'train': ds_imagenet_sequential.get_dataloader(
                True, batch_size, num_workers, use_cuda=use_cuda
            ),
            'val': ds_imagenet_sequential.get_dataloader(
                False, batch_size, num_workers, use_cuda=use_cuda
            ),
            'test': None,
        }

    elif dataset == 'legacy_imagenet':
        use_from = kwargs.get('use_from', ['pc48'])[0]
        alternate_aug = kwargs.get('alternate_aug', [None])[0]
        bgr_to_rgb = kwargs.get('bgr_to_rgb', [True])[0]

        return {
            'train': ds_imagenet.get_dataloader(
                True, batch_size, num_workers,
                use_from=use_from,
                bgr_to_rgb=bgr_to_rgb
            ),
            'val': ds_imagenet.get_dataloader(
                False, batch_size, num_workers,
                use_from=use_from,
                alternate_aug=alternate_aug,
                bgr_to_rgb=bgr_to_rgb
            ),
            'test': None
        }

    else:
        raise ValueError(f'Unkown dataset: {dataset}')
