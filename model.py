import preprocessing as prep
import numpy as np
import pandas as pd
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

rmse = lambda a, p: np.sqrt(mean_squared_error(a, p))

train_df = pd.read_csv("train.csv", low_memory=False)
test_df = pd.read_csv("test.csv", low_memory=False)
y = train_df.pop("price")

X_train = prep.feature_engineering(train_df)
X_test = prep.feature_engineering(test_df)
X_combined = pd.concat([X_train, X_test], axis=0)
X_processed = prep.feature_engineering_combined(X_combined)
X_train = X_processed[:33538, :]
X_test = X_processed[33538:, :]

Xtr, Xval, ytr, yval = train_test_split(X_train, y, test_size=0.3)

trainset = lgbm.Dataset(Xtr, ytr)
valset = lgbm.Dataset(Xval, yval, reference=trainset)
params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 35,
    "learning_rate": 0.005,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_split_gain": 0.5,
    "min_child_weight": 1,
    "min_child_samples": 5,
    "n_estimators": 5000,
    "verbose": 1,
}

model = lgbm.train(
    params, trainset, num_boost_round=1500, valid_sets=valset, early_stopping_rounds=50
)

output_df = pd.DataFrame()
output_df["Id"] = test_df.id
output_df["Predicted"] = model.predict(X_test)
output_df.to_csv("lgbm_baseline.csv", index=False)
