# DSAA5020 Group Project: Carbon Emission Prediction by GNNs


# Run

Run a certain modle (AGCRN, MLP or FC-LSTM)):

`python src/models/choose_a_model/main.py`

# Description

This course project aims to predict the daily carbon emissions of 31 Chinese provinces using historical data from January 1st, 2019 to December 31st, 2022. The input is a 1461x31 matrix, where each row represents the daily carbon emission predictions for each province, and the output is a 1x31 matrix providing the forecast for the next day based on the past 10 days' data.

# Methods Reference

AGCRN -> [1] L. Bai, L. Yao, C. Li, X. Wang, and C. Wang, “Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting.” arXiv, Oct. 21, 2020. Accessed: Feb. 16, 2023. [Online]. Available: http://arxiv.org/abs/2007.02842

FC-LSTM -> [2] S. Tao, H. Zhang, F. Yang, Y. Wu, and C. Li, “Multiple Information Spatial–Temporal Attention based Graph Convolution Network for traffic prediction,” Applied Soft Computing, vol. 136, p. 110052, Mar. 2023, doi: 10.1016/j.asoc.2023.110052.
