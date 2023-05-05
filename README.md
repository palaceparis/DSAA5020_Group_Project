# DSAA5020_Group_Project

DSAA5020 Group Project: Carbon Emission Prediction by GNNs

# Run

Run a certain modle (AGCRN, MLP or FC-LSTM)):

`python src/models/choose_a_model/main.py`

# Description

This project aims to predict the daily carbon emissions of 31 Chinese provinces using historical data from January 1st, 2019 to December 31st, 2022. The input is a 1461x31 matrix, where each row represents the daily carbon emission predictions for each province, and the output is a 1x31 matrix providing the forecast for the next day based on the past 10 days' data.

# Dependencies
