from datetime import date
from itertools import count
from share import *
from pytictoc import TicToc
import pandas as pd
from mxnet import ndarray as nd
import numpy as np
import sqlite3

from share_y_gen_graphs import dates, months


emissions = pd.read_csv(
    "/Users/tonygong/Projects/Emission_Prediction_Data/emissionsWithoutHeader.csv",
    header=None,
)

# use stop watch
t = TicToc()  # create instance of class
t.tic()  # Start timer

t.toc("gen correlation graph, calculate ACGs")


# Assuming emissions DataFrame has columns representing regions
df_counts = emissions

# Calculate the Pearson correlation of df_counts and replace NaN values with -1
carbon_ACG = df_counts.corr(method="pearson")
carbon_ACG = carbon_ACG.fillna(value=-1)  # substitute 'nan' to '-1'
carbon_ACG = nd.array(carbon_ACG)
carbon_ACG = nd.softmax(carbon_ACG, axis=1)

# Store the adjacency matrix (correlation graph)
nd.save("data/Carbon/carbon_ACG", carbon_ACG)

t.toc("gen correlation graph, finish ACG aggregation")
