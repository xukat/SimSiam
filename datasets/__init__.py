import torch
import torchvision
from .random_dataset import RandomDataset
import os

import sys
sys.path.append("../")
from rnn_basic.data_utils import TensileSampleDatasetImagesOnly

def get_dataset(dataset, data_dir, transform, train=True, download=False, debug_subset_size=None):
    split = 'train' if train else 'val'

    if dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'stl10':
        dataset = torchvision.datasets.STL10(data_dir, split='train+unlabeled' if train else 'test', transform=transform, download=download)
    elif dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'imagenet':
        dataset = torchvision.datasets.ImageNet(data_dir, split='train' if train == True else 'val', transform=transform, download=download)
    elif dataset == 'random':
        dataset = RandomDataset()
    elif dataset == 'tensile':
        dataset = TensileSampleDatasetImagesOnly(
                os.path.join(data_dir, "experiment_info_cleaned.csv"),
                os.path.join(data_dir, "all_images"),
                mask_dir=None,
                dset=split,
                train_split=0.8,
                image_extractor='vanilla',
                mask_background=False,
                augmentation=transform)
    elif dataset == 'tensile_masked':
        dataset = TensileSampleDatasetImagesOnly(
                os.path.join(data_dir, "experiment_info_cleaned.csv"),
                os.path.join(data_dir, "all_images"),
                mask_dir=os.path.joing(data_dir, "all_masks"),
                dset=split,
                train_split=0.8,
                image_extractor='vanilla',
                mask_background=True,
                augmentation=transform)

    else:
        raise NotImplementedError

    if debug_subset_size is not None:
        if isinstance(dataset, TensileSampleDatasetImagesOnly):
            dataset = torch.utils.data.Subset(dataset, range(0, debug_subset_size)) # take only one batch
        else:
            dataset = torch.utils.data.Subset(dataset, range(0, debug_subset_size)) # take only one batch
            dataset.classes = dataset.dataset.classes
            dataset.targets = dataset.dataset.targets

    return dataset
