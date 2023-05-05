from calendar import month
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import pandas as pd
from mxnet import ndarray as nd
import os
from pytictoc import TicToc
import multiprocessing as mp

from share import (
    latitude_min,
    latitude_span,
    M_lat,
    longitude_min,
    longitude_span,
    N_lng,
    dates,
    months,
)


#######

# use stop watch
t = TicToc()  # create instance of class
t.tic()  # Start timer

t.toc("gen distance graph, calculate ADG")

grp_difference = pd.read_csv(
    "/Users/tonygong/Projects/Emission_Prediction_Data/grp_diff_matrix.csv"
)
grp_difference = abs(grp_difference)
# Assuming provinces_distance is a NumPy array of shape (n_regions, n_regions)
grp_difference = np.array(grp_difference)
n_regions = 31
# Calculate the adjacency values using the inverse distance method
grp_difference_adjacency = 1.0 / (grp_difference + np.eye(n_regions))

# Apply softmax to the adjacency values to normalize them
grp_difference_adjacency = nd.array(grp_difference_adjacency)
carbon_AIG = nd.softmax(grp_difference_adjacency, axis=1)

# Save the adjacency matrix
nd.save("data/Carbon/carbon_aig", carbon_AIG)

t.toc("finish ADG aggregation using provinces_distance data")
