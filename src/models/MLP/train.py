import logging
from typing import Tuple, List
from math import sqrt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error

from utils import relative_absolute_error

logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    config,
) -> Tuple[List[float], List[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Training model")
    model.train()
    rmse_list = []
    rae_list = []
    loss_list = []

    for epoch in range(config.num_epochs):
        epoch_rmse = 0
        epoch_rae = 0
        epoch_loss = 0
        num_batches = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            targets_np = targets.cpu().detach().numpy()
            outputs_np = outputs.cpu().detach().numpy()

            epoch_rmse += sqrt(mean_squared_error(targets_np, outputs_np))
            epoch_rae += relative_absolute_error(targets_np, outputs_np)
            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= num_batches
        epoch_rmse /= num_batches
        epoch_rae /= num_batches
        rmse_list.append(epoch_rmse)
        rae_list.append(epoch_rae)
        loss_list.append(epoch_loss)

        logger.info(
            f"Epoch [{epoch+1}/{config.num_epochs}], Loss: {loss.item():.4f}, RMSE: {epoch_rmse:.4f}, RAE: {epoch_rae:.4f}"
        )

    # Calculate the average RMSE and RAE
    average_rmse = np.mean(rmse_list)
    average_rae = np.mean(rae_list)
    average_loss = np.mean(loss_list)
    logger.info(
        f"Average Train Loss: {average_loss:.4f}, Average Train RMSE: {average_rmse:.4f}, Average Train RAE: {average_rae:.4f}"
    )

    return rmse_list, rae_list
