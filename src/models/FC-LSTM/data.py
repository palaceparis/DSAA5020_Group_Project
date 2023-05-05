import random
import numpy as np
from mxnet import nd
from mxnet.gluon.data.dataloader import DataLoader
from mxnet.gluon.data.dataset import ArrayDataset


def load_data(data_sample_file):
    """
    Load data from the data_sample_file.
    """
    Xr_sample, Xd_sample, Xw_sample, Yp_sample, regions = nd.load(data_sample_file)
    return Xr_sample, Xd_sample, Xw_sample, Yp_sample, regions


def split_data(X, Y, train_ratio=0.8, validate_ratio=0.1, seed=None):
    """
    Split data into train, validate, and test sets.
    """
    n_sample = X.shape[0]
    idx = list(range(n_sample))
    idx_train = random.sample(idx, int(train_ratio * len(idx)))
    idx_remaining = list(set(idx) - set(idx_train))  # exclude idx_train
    idx_validate = random.sample(idx_remaining, int(validate_ratio * len(idx)))
    idx_test = list(
        set(idx_remaining) - set(idx_validate)
    )  # exclude idx_train and idx_validate

    return idx_train, idx_validate, idx_test


def get_dataloaders(X, Y, idx_train, idx_validate, idx_test, batch_size):
    """
    Create DataLoader objects for train, validate, and test sets.
    """

    # caculate number of train, validate, test samples
    n_train = len(idx_train)
    n_test = len(idx_test)

    data_train = ArrayDataset(X[idx_train], Y[idx_train])
    data_validate = ArrayDataset(X[idx_validate], Y[idx_validate])
    data_test = ArrayDataset(X[idx_test], Y[idx_test])

    loader_train = DataLoader(data_train, batch_size=batch_size)
    loader_validate = DataLoader(data_validate, batch_size=batch_size)
    loader_test = DataLoader(data_test, batch_size=batch_size)

    return loader_train, loader_validate, loader_test, n_train, n_test
