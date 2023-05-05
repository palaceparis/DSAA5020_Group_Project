import hydra
import hydra.utils
from omegaconf import DictConfig, OmegaConf
import random
import logging
import mxnet as mx
from mxnet import autograd, nd
from mxnet.gluon import Trainer, nn
from mxnet.gluon.data.dataloader import DataLoader
from mxnet.gluon.data.dataset import ArrayDataset
from mxnet.metric import MAE, RMSE

from pytictoc import TicToc
import matplotlib.pyplot as plt
import os

from data import load_data, split_data, get_dataloaders
from fc_lstm import LSTM_m
from train import get_devices, train_and_evaluate
from results_pre import write_records

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../../configuration/modules", config_name="FC-LSTM"
)
def main(cfg: DictConfig) -> None:
    # Configurations
    devices = get_devices()
    epochs = cfg.epochs
    data_sample_file = cfg.data_sample_file
    batch_size = cfg.batch_size
    Tp = cfg.Tp
    runs = cfg.runs
    threshold = cfg.threshold

    # Load data
    Xr_sample, Xd_sample, Xw_sample, Yp_sample, regions = load_data(data_sample_file)
    X = Xr_sample
    Y = Yp_sample
    n_sample, N, F, T = X.shape

    # Split data
    idx_train, idx_validate, idx_test = split_data(X, Y)

    # Get DataLoader objects
    loader_train, loader_validate, loader_test, n_train, n_test = get_dataloaders(
        X, Y, idx_train, idx_validate, idx_test, batch_size
    )

    # Initialize the model
    net = LSTM_m(Tp)
    net.initialize(ctx=devices, force_reinit=True)
    loss_fun = mx.gluon.loss.L2Loss()
    trainer = Trainer(net.collect_params(), "adam")

    # Train and evaluate the model
    for run in range(runs):
        (
            train_r,
            val_r,
            test_r,
            y_test_list,
            y_hat_test_list,
        ) = train_and_evaluate(
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
        )
        if run == 0:
            train_records, validate_records, test_records = train_r, val_r, test_r
        else:
            train_records += train_r
            validate_records += val_r
            test_records += test_r
        train_r_sum = sum([record[1] for record in train_r])
        train_r_count = len(train_r)
        train_r_average = train_r_sum / train_r_count

        Train_RMSE_message = (
            f"Average train RMSE for run {run+1}/{runs}, {train_r_average}"
        )
        Test_RMSE_message = f"Average test RMSE for run {run+1}/{runs},{test_r}"
        separator = "-" * len(Train_RMSE_message)
        logging.info(separator)
        logging.info(Train_RMSE_message)
        logging.info(Test_RMSE_message)
        logging.info(separator)

    train_records /= runs
    validate_records /= runs
    test_records /= runs

    # Average train RMSE over all epochs
    train_records_sum = sum([record[1] for record in train_records])
    train_records_count = len(train_records)
    train_records_average = train_records_sum / train_records_count
    logging.info(f"For all runs, average train RMSE {train_records_average}")

    # Test RMSE
    logging.info(f"For all runs, Average test RMSE {test_records}")

    # Write out records
    write_records(cfg, train_records, validate_records, test_records)


if __name__ == "__main__":
    main()
