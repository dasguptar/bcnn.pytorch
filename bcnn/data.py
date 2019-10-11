"""Handle datasets, transforms, and data loaders."""

import argparse
from pathlib import Path
from typing import List

from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

Transform = object

common_transforms: List[Transform] = [
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
]


def get_data_loader(split: str, args: argparse.Namespace) -> DataLoader:
    """Return dataloader for specified split."""
    split_transforms: List[Transform]
    if split == 'train':
        split_transforms = [
            transforms.Resize(448),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.RandomCrop(448),
        ]
    elif split == 'test':
        split_transforms = [
            transforms.Resize(448),
            transforms.CenterCrop(448),
        ]

    dataset: ImageFolder = ImageFolder(
        root=Path(args.datadir) / split,
        transform=transforms.Compose(split_transforms + common_transforms)
    )
    dataloader: DataLoader = DataLoader(
        dataset=dataset,
        batch_size=args.batchsize,
        shuffle=True if split == 'train' else False,
        num_workers=args.workers,
        pin_memory=True,
    )
    return dataloader
