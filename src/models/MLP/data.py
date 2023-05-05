import pandas as pd
import torch
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def load_data(file_path: str) -> pd.DataFrame:
    emissions = pd.read_csv(file_path, header=None)
    return emissions


def create_dataset(emissions, cfg):
    look_back = cfg.look_back
    target_day = cfg.target_day
    X, y = [], []
    for i in range(len(emissions) - look_back - target_day + 1):
        X.append(emissions[i : (i + look_back), :])
        y.append(emissions[i + look_back + target_day - 1, :])
    return np.array(X), np.array(y)


def preprocess_data(data: pd.DataFrame, cfg) -> Tuple[np.ndarray, np.ndarray]:
    X, y = create_dataset(data.values, cfg)
    X = X.reshape(X.shape[0], -1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_seed
    )
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config,
) -> Tuple[DataLoader, DataLoader]:
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    # Create TensorDatasets and DataLoaders for train and test sets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    batch_size = config.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
