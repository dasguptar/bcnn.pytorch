"""Argument parser for all programs."""

import argparse
import pathlib


def parse_args_main() -> argparse.Namespace:
    """Parse arguments for main training script."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Parse arguments for main.py')
    parser.add_argument('--datadir', type=pathlib.Path, default='./data/CUB_200_2011/original/',
                        help='Path to folder containing data')
    parser.add_argument('--savedir', type=pathlib.Path, default='./ckpt/test',
                        help='Directory for checkpointing to disk')
    parser.add_argument('--load', type=str, default='',
                        help='Checkpoint filename to load state dicts')
    parser.add_argument('--batchsize', type=int, default=64,
                        help='Batchsize for each GPU')
    parser.add_argument('--epochs', type=int, nargs='+', default=[55, 25],
                        help='Number of epochs for partial and full finetuning')
    parser.add_argument('--lr', type=float, nargs='+', default=[1.0, 1e-2],
                        help='Learning rate (multiplied by factor for new layers)')
    parser.add_argument('--wd', type=float, nargs='+', default=[1e-8, 1e-5],
                        help='Weight decay (multiplied by factor for new layers)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--stepfactor', type=float, default=0.1,
                        help='Step size for reducing learning rate')
    parser.add_argument('--patience', type=int, default=3,
                        help='How long to wait before dropping LR')
    parser.add_argument('--gpus', type=int, default=[], nargs='+',
                        help='Space separated list of gpus to use')
    parser.add_argument('--seed', type=int, default=12345,
                        help='Random seed for reproducibility')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel data loader threads')
    args: argparse.Namespace = parser.parse_args()
    return args
