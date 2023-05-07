import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from math import sqrt
from typing import Tuple
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from utils import relative_absolute_error, mean_absolute_percentage_error


logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    X_test_tensor: torch.Tensor,
    y_test_tensor: torch.Tensor,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_test_tensor, y_test_tensor = X_test_tensor.to(device), y_test_tensor.to(device)
    logger.info("Evaluating model")
    model.eval()

    with torch.no_grad():
        test_predictions = model(X_test_tensor)
    test_predictions_np = (
        test_predictions.cpu().numpy()
    )  # Move tensor to CPU before converting to NumPy array
    y_test_np = (
        y_test_tensor.cpu().numpy()
    )  # Move tensor to CPU before converting to NumPy array

    test_rmse = 0.0
    test_rae = 0.0
    test_loss = 0.0
    test_mae = 0.0
    test_mape = 0.0

    for i, (X_batch, y_batch) in enumerate(test_loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        with torch.no_grad():
            batch_predictions = model(X_batch)
            batch_loss = criterion(batch_predictions, y_batch)
        test_loss += batch_loss.item()
        batch_predictions_np = batch_predictions.cpu().numpy()
        y_batch_np = y_batch.cpu().numpy()
        batch_rmse = sqrt(mean_squared_error(y_batch_np, batch_predictions_np))
        test_rmse += batch_rmse
        batch_rae = relative_absolute_error(y_batch_np, batch_predictions_np)
        test_rae += np.sum(batch_rae)
        batch_mae = mean_absolute_error(y_batch_np, batch_predictions_np)
        test_mae += batch_mae
        batch_mape = mean_absolute_percentage_error(y_batch_np, batch_predictions_np)
        test_mape += batch_mape

    avg_test_rmse = test_rmse / (i + 1)
    avg_test_rae = test_rae / (i + 1)
    avg_test_loss = test_loss / (i + 1)
    avg_test_mae = test_mae / (i + 1)
    avg_test_mape = test_mape / (i + 1)

    logger.info(
        f"Average Test Loss: {avg_test_loss:.4f}, Average Test RMSE: {avg_test_rmse:.4f}, Average Test RAE: {avg_test_rae:.4f}, Average Test MAE: {avg_test_mae:.4f}, Average Test MAPE: {avg_test_mape:.4f}"
    )

    return test_predictions_np, y_test_np
