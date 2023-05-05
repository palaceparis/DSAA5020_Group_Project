import sqlite3
from mxnet import ndarray as nd
import pandas as pd
from pytictoc import TicToc
import numpy as np

from share import (
    latitude_min,
    latitude_span,
    M_lat,
    longitude_min,
    longitude_span,
    N_lng,
    R,
)

# use stop watch
t = TicToc()  # create instance of class
t.tic()  # Start timer

t.toc("gen distance graph, calculate ADG")

provinces_distance = pd.read_csv(
    "/Users/tonygong/Projects/Emission_Prediction_Data/distance.csv", header=None
)

# Assuming provinces_distance is a NumPy array of shape (n_regions, n_regions)
provinces_distance = np.array(provinces_distance)
n_regions = 31
# Calculate the adjacency values using the inverse distance method
provinces_adjacency = 1.0 / (provinces_distance + np.eye(n_regions))

# Apply softmax to the adjacency values to normalize them
provinces_adjacency = nd.array(provinces_adjacency)
provinces_ADG = nd.softmax(provinces_adjacency, axis=1)

# Save the adjacency matrix
nd.save("data/Carbon/provinces_adg", provinces_ADG)

t.toc("finish ADG aggregation using provinces_distance data")
