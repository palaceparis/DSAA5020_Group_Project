import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from omegaconf import DictConfig, OmegaConf
import hydra
from pathlib import Path
import hydra.utils
import os

from mlp import MLP
from utils import relative_absolute_error
from data import load_data, preprocess_data, create_dataloaders
from train import train_model
from eval import evaluate_model
from vis import visualize_predictions


@hydra.main(
    version_base=None, config_path="../../configuration/modules", config_name="MLP"
)
def main(cfg: DictConfig) -> None:
    # Load your emissions data
    emissions = load_data(cfg.file_path)

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(emissions, cfg)

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        X_train, y_train, X_test, y_test, cfg
    )

    # Define model, loss function, and optimizer
    hidden_sizes = cfg.model.hidden_sizes  # Adjust the number and size of hidden layers
    dropout_rate = cfg.model.dropout_rate  # Adjust the dropout rate
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    model = MLP(input_size, hidden_sizes, output_size, dropout_rate)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.optimizer.learning_rate)

    # Train the model
    rmse_list, rae_list = train_model(model, train_loader, criterion, optimizer, cfg)

    # Save the RMSE and RAE lists to an Excel file
    metrics_df = pd.DataFrame({"RMSE": rmse_list, "RAE": rae_list})
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    metrics_df.to_excel(output_dir / "metrics_per_epoch.xlsx", index=False)

    # Evaluate the model
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()
    test_predictions_np, y_test_np = evaluate_model(
        model, test_loader, criterion, X_test_tensor, y_test_tensor
    )

    # Visualize the predictions for each province
    visualize_predictions(model, test_predictions_np, y_test_np, cfg)


if __name__ == "__main__":
    main()
