import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pathlib

import pandas as pd
from data.pre_process import PATH_FEATURES, PATH_TIME_WINDOWS


features_train = pd.read_csv(pathlib.Path(PATH_FEATURES, "train.csv"))
y_train = np.load(str(pathlib.Path(PATH_TIME_WINDOWS, "data_train_y.npy")))
features_test = pd.read_csv(pathlib.Path(PATH_FEATURES, "test.csv"))
y_test = np.load(str(pathlib.Path(PATH_TIME_WINDOWS, "data_test_y.npy")))

columns_order = features_train.columns.values
X_train = features_train[columns_order]
X_test = features_test[columns_order]

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

print("score on test set", clf.score(X_test, y_test))
