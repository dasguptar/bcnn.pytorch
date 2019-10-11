"""Bilinear CNN training script."""

import argparse
import logging
import pathlib
import random
from typing import Any, Dict, Tuple

from bcnn import BilinearModel, Trainer, get_data_loader

from config import parse_args_main

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler('expt.log', mode='w')
    ])
logger = logging.getLogger()


def checkpoint(
        trainer: Trainer,
        epoch: int,
        accuracy: float,
        savedir: pathlib.Path,
        config: argparse.Namespace) -> None:
    """Save a model checkpoint at specified location."""
    checkpoint: Dict[str, Any] = {
        "model": trainer.model.state_dict(),
        "optim": trainer.optimizer.state_dict(),
        "epoch": epoch,
        "accuracy": accuracy,
        "config": config,
    }
    logger.debug("==> Checkpointing Model")
    torch.save(checkpoint, savedir / 'checkpoint.pt')


def run_epochs_for_loop(
        trainer: Trainer,
        epochs: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        savedir: pathlib.Path,
        config: argparse.Namespace,
        scheduler: ReduceLROnPlateau = None):
    """Run train + evaluation loop for specified epochs.

    Save checkpoint to specified save folder when better optimum is found.
    If LR scheduler is specified, change LR accordingly.
    """
    best_acc: float = 0.0
    for epoch in range(epochs):
        (train_loss, train_acc) = trainer.train(train_loader)  # type: Tuple[float, float]
        (test_loss, test_acc) = trainer.test(test_loader)  # type: Tuple[float, float]
        logger.info("Epoch %d: TrainLoss %f \t TrainAcc %f" % (epoch, train_loss, train_acc))
        logger.info("Epoch %d: TestLoss %f \t TestAcc %f" % (epoch, test_loss, test_acc))
        if scheduler is not None:
            scheduler.step(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint(trainer, epoch, test_acc, savedir, config)


def main():
    """Train bilinear CNN."""
    args: argparse.Namespace = parse_args_main()
    logger.debug(args)

    # random seeding
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if len(args.gpus) > 0:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
        device: torch.device = torch.device('cuda:0')

    args.savedir.mkdir(parents=True, exist_ok=True)

    train_loader: DataLoader = get_data_loader('train', args)
    test_loader: DataLoader = get_data_loader('test', args)

    model: nn.Module = BilinearModel(num_classes=200)
    model = torch.nn.DataParallel(model)
    criterion: nn.Module = nn.CrossEntropyLoss()
    model.to(device)
    criterion.to(device)

    logger.debug("==> PRETRAINING NEW BILINEAR LAYER ONLY")
    for param in model.module.features.parameters():
        param.requires_grad = False
    optimizer: optim.optimizer.Optimizer = optim.SGD(
        model.module.classifier.parameters(),
        lr=args.lr[0],
        weight_decay=args.wd[0],
        momentum=args.momentum,
        nesterov=True,
    )
    pretrainer: Trainer = Trainer(
        model,
        criterion,
        optimizer,
        device,
    )
    scheduler: ReduceLROnPlateau = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=args.stepfactor,
        patience=args.patience,
        verbose=True,
        threshold=1e-4,
    )
    run_epochs_for_loop(
        trainer=pretrainer,
        epochs=args.epochs[0],
        train_loader=train_loader,
        test_loader=test_loader,
        savedir=args.savedir,
        config=args,
        scheduler=scheduler,
    )

    logger.debug("==> FINE-TUNING OLDER LAYERS AS WELL")
    for param in model.module.features.parameters():
        param.requires_grad = True
    optimizer: optim.optimizer.Optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr[1],
        weight_decay=args.wd[1],
        momentum=args.momentum,
        nesterov=True,
    )
    finetuner: Trainer = Trainer(
        model,
        criterion,
        optimizer,
        device,
    )
    scheduler: ReduceLROnPlateau = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=args.stepfactor,
        patience=args.patience,
        verbose=True,
        threshold=1e-4,
    )
    run_epochs_for_loop(
        trainer=finetuner,
        epochs=args.epochs[1],
        train_loader=train_loader,
        test_loader=test_loader,
        savedir=args.savedir,
        config=args,
        scheduler=scheduler,
    )


if __name__ == "__main__":
    main()
