import time
from mxnet import autograd
from mxnet.gluon import Trainer
from mxnet.metric import MAE, RMSE
import logging
import os

import mxnet as mx
import numpy as np
import pandas as pd
from mxnet import autograd, nd
from mxnet.gluon import Trainer, nn
from mxnet.gluon.data.dataloader import DataLoader
from mxnet.gluon.data.dataset import ArrayDataset
from mxnet.metric import MAE, RMSE

from sklearn.metrics import mean_absolute_error
from utils import mean_absolute_percentage_error

logging.basicConfig(level=logging.INFO)


def get_devices():
    if mx.context.num_gpus() > 0:
        devices = [mx.gpu(i) for i in range(mx.context.num_gpus())]  # mxnet
        os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
        os.environ["MXNET_CUDNN_LIB_CHECKING"] = "0"
    else:
        devices = [mx.cpu()]  # mxnet
    return devices


def train_one_epoch(net, loader_train, devices, loss_fun, trainer, batch_size, n_train):
    """
    Train the model for one epoch.
    """
    start = time.time()
    train_loss_acc = 0
    for x, y in loader_train:
        x_list = mx.gluon.utils.split_and_load(x, devices, even_split=False)
        y_list = mx.gluon.utils.split_and_load(y, devices, even_split=False)
        with autograd.record():
            y_hat_list = [net(x1) for x1 in x_list]
            losses = [loss_fun(y_hat, y1) for y_hat, y1 in zip(y_hat_list, y_list)]
        for loss in losses:
            loss.backward()
        trainer.step(batch_size)
        train_loss_acc += sum([loss.sum().asscalar() for loss in losses])

    train_loss_mean = train_loss_acc / n_train  # mse
    train_rmse = np.sqrt(train_loss_mean)
    duration = time.time() - start
    return train_rmse, duration


def validate(net, loader_validate, devices, threshold):
    """
    Validate the model.
    """
    mae = MAE()
    rmse = RMSE()
    for x, y in loader_validate:
        x = x.copyto(devices[0])
        y = y.copyto(devices[0])
        y_hat = net(x)
        pred = nd.ceil(y_hat - y - threshold)
        mae.update(y, pred)
        rmse.update(y, pred)
    return mae.get()[1], rmse.get()[1]


def test(net, loader_test, devices, loss_fun, n_test):
    """
    Test the model.
    """
    start = time.time()
    test_loss_acc = 0
    y_test_list = []
    y_hat_test_list = []

    for x, y in loader_test:
        x = x.copyto(devices[0])
        y = y.copyto(devices[0])
        y_hat = net(x)
        loss = loss_fun(y_hat, y)
        test_loss_acc += nd.sum(loss).asscalar()
        y_test_list.append(y.asnumpy())
        y_hat_test_list.append(y_hat.asnumpy())

    test_loss_mean = test_loss_acc / n_test  # mse
    test_rmse = np.sqrt(test_loss_mean)
    y_test_concat = np.concatenate([np.squeeze(y) for y in y_test_list])
    y_hat_test_concat = np.concatenate([np.squeeze(y_hat) for y_hat in y_hat_test_list])
    test_mae = mean_absolute_error(y_test_concat, y_hat_test_concat)
    test_mape = mean_absolute_percentage_error(y_test_concat, y_hat_test_concat)
    duration = time.time() - start

    return test_rmse, test_mae, test_mape, duration, y_test_list, y_hat_test_list


def train_and_evaluate(
    net,
    loader_train,
    loader_validate,
    loader_test,
    devices,
    loss_fun,
    trainer,
    batch_size,
    epochs,
    n_train,
    n_test,
    threshold,
    run,
    runs,
):
    """
    Train, validate, and test the model for multiple runs.
    """
    train_records = []
    validate_records = []

    for e in range(epochs):
        train_rmse, train_duration = train_one_epoch(
            net, loader_train, devices, loss_fun, trainer, batch_size, n_train
        )
        train_records.append([e, train_rmse])
        logging.info(
            f"FS-LSTM, run {run+1}/{runs}, train, Epoch {e + 1}/{epochs}, train RMSE: {train_rmse}, duration: {train_duration}s"
        )

        val_mae, val_rmse = validate(net, loader_validate, devices, threshold)
        validate_records.append([e, val_mae, val_rmse])

    test_rmse, test_mae, test_mape, test_duration, y_test_list, y_hat_test_list = test(
        net, loader_test, devices, loss_fun, n_test
    )
    logging.info(
        f"Test RMSE: {test_rmse}, Test MAE: {test_mae}, Test MAPE: {test_mape}, duration: {test_duration}s"
    )

    train_records = np.array(train_records)
    validate_records = np.array(validate_records)
    test_records = np.array([[test_rmse]])

    return train_records, validate_records, test_records, y_test_list, y_hat_test_list
