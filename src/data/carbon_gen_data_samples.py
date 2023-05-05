from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from glob import glob
import os

import mxnet as mx
import numpy as np
import pandas as pd
from mxnet import ndarray as nd
from pytictoc import TicToc
import sys

sys.path.append("code/scripts")
from share import Td, Tp, Tr, Tw

# use stop watch
t = TicToc()  # create instance of class
t.tic()  # Start timer

t.toc("flow aggregation, calculate hourly traffic flows")

emissions = pd.read_csv(
    "/Users/tonygong/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/Projects-Data/Emission-Prediction-Main/emissionsWithoutHeader.csv",
    header=None,
)
emissions_np = np.array(emissions)
emissions_nd = nd.array(emissions_np)


tmp = pd.read_csv(
    "/Users/tonygong/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/Projects-Data/Emission-Prediction-Main/carbon-MISTAGCN-outputs/ordered_province_daily.csv"
)
tmp = np.array(tmp)
constant = abs(tmp.min().min()) + 1e-6
tmp_shifted = tmp + constant
tmp_np = np.array(tmp_shifted)
tmp_nd = nd.array(tmp_np)

aqi = pd.read_csv(
    "/Users/tonygong/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/Projects-Data/Emission-Prediction-Main/carbon-MISTAGCN-outputs/aqi.csv"
)
aqi_np = np.array(aqi)
aqi_nd = nd.array(aqi_np)
aqi_nd = aqi_nd.expand_dims(axis=2)

data = nd.stack(emissions_nd, tmp_nd, axis=2)
data = nd.concat(data, aqi_nd, dim=2)
data = nd.array(data)

n_time_slices, N, F = data.shape
data = nd.transpose(data, axes=(1, 2, 0))  # N, F, n_time_slices

Tr = 10  # 7
Td = 2  # 3
Tw = 2  # 2
Tp = 1  # 3
n_sample = n_time_slices - Tp - Tw * 30  # the last Tp time slices are reserved for Yp

Yp_sample = nd.zeros((n_sample, N, Tp))
Xr_sample = nd.zeros(
    (n_sample, N, F, Tr)
)  # the primary Tw*7*24 time slices are reserved for Xw
Xd_sample = nd.zeros((n_sample, N, F, Td))
Xw_sample = nd.zeros((n_sample, N, F, Tw))
for k in range(n_sample):
    Yp_sample[k] = data[:, 0, k + Tw * 30 : k + Tw * 30 + Tp]
    Xr_sample[k] = data[:, :, k + Tw * 30 - Tr : k + Tw * 30]
    for k1 in range(Td):
        Xd_sample[k, :, :, k1] = data[:, :, k + Tw * 30 - (Td - k1) * 7]

    for k1 in range(Tw):
        Xw_sample[k, :, :, k1] = data[:, :, k + Tw * 30 - (Tw - k1) * 30]


regions = nd.arange(1, 32)  # Start from 1 and end at 32 (exclusive)
nd.save(
    "Data/MISTAGCN/data-samples", [Xr_sample, Xd_sample, Xw_sample, Yp_sample, regions]
)
t.toc("gen data samples, saving data samples sucessfully")
