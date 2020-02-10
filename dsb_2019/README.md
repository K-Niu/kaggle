# 2019 Data Science Bowl
Final ranking: Top 17% (593/3479)

Blog post: https://kelvinniu.com/posts/2019-data-science-bowl/

The competition data is assumed to be unzipped to `data/...`

## Feature Engineering
*`preprocessing/flat.py`: perform feature engineering to be used as input in nonsequential models such as gradient boosted decision trees and DNNs
*`preprocessing/rnn.py`: perform feature engineering to be used as input in sequential models such as RNNs

## Model
* `models/lightgbm.py`: gradient boosted decision tree model using LightGBM
* `models/rnn.py`: RNN model using Keras/Tensorflow