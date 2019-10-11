"""Trainer class to abstract rudimentary training loop."""

from typing import Tuple

import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from tqdm import tqdm


class Trainer(object):
    """Trainer class to abstract rudimentary training loop."""

    def __init__(
            self,
            model: Module,
            criterion: Module,
            optimizer: Optimizer,
            device: torch.device) -> None:
        """Set trainer class with model, criterion, optimizer. (Data is passed to train/eval)."""
        super(Trainer, self).__init__()
        self.model: Module = model
        self.criterion: Module = criterion
        self.optimizer: Optimizer = optimizer
        self.device: torch.device = device

    def train(self, loader: DataLoader) -> Tuple[float, float]:
        """Train model using batches from loader and return accuracy and loss."""
        total_loss, total_acc = 0.0, 0.0
        self.model.train()
        for _, (inputs, targets) in tqdm(enumerate(loader), total=len(loader), desc='Training'):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            _, predicted = torch.max(outputs, 1)
            total_loss += loss.item()
            total_acc += (predicted == targets).float().sum().item() / targets.numel()
        return total_loss / len(loader), 100.0 * total_acc / len(loader)

    def test(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model using batches from loader and return accuracy and loss."""
        with torch.no_grad():
            total_loss, total_acc = 0.0, 0.0
            self.model.eval()
            for _, (inputs, targets) in tqdm(enumerate(loader), total=len(loader), desc='Testing '):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                _, predicted = torch.max(outputs, 1)
                total_loss += loss.item()
                total_acc += (predicted == targets).float().sum().item() / targets.numel()
        return total_loss / len(loader), 100.0 * total_acc / len(loader)
