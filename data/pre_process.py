import pathlib
import pandas as pd
import numpy as np

import tsfresh
from tsfresh import extract_relevant_features, extract_features


ACTIVITY_CLASS_ENCODING = {"Running": 0, "Standing_still": 1, "Walking": 2}

PATH_DATA = pathlib.Path(__file__).parent
PATH_RAW = PATH_DATA / "raw"
PATH_RAW_ACC = PATH_DATA / "raw_acc"
PATH_TIME_WINDOWS = PATH_DATA / "time_windows"
PATH_FEATURES = PATH_DATA / "features"


def create_directories():
    for p in (PATH_RAW_ACC, PATH_TIME_WINDOWS, PATH_FEATURES):
        p.mkdir(exist_ok=True)


def extract_acc_data_from_raw():
    for path_f in PATH_RAW.iterdir():
        path_f_acc = PATH_RAW_ACC / path_f.name
        with path_f.open() as fp:
            acc_data = ""
            for line in fp.readlines():
                if line.split("\t")[1] == "ACC":
                    acc_data += line
            if acc_data:
                with path_f_acc.open("w") as fp:
                    fp.write(acc_data)


def extract_time_windows_from_acc(file_nrs, data_type):
    all_data = []
    for path_f_acc in PATH_RAW_ACC.iterdir():
        i = int(path_f_acc.stem.split("_")[-1])
        if i in file_nrs:
            all_data.append(extract_time_windows_from_acc_file(path_f_acc))
    all_data = np.vstack(all_data)
    nr_data_points = all_data.shape[0]
    indx = np.arange(nr_data_points)
    np.random.shuffle(indx)
    np.save(str(PATH_TIME_WINDOWS / f"data_{data_type}_X.npy"), all_data[indx, 1:])
    np.save(str(PATH_TIME_WINDOWS / f"data_{data_type}_y.npy"), all_data[indx, 0])


def extract_time_windows_from_acc_file(path_f_acc):
    path_f = pathlib.Path(path_f_acc)
    df = pd.read_csv(path_f, sep="\t", names=["time", "type", "acc_x", "acc_y", "acc_z"])
    activity = path_f.stem.rsplit("_", maxsplit=1)[0]
    activity_encoded = ACTIVITY_CLASS_ENCODING[activity]
    df["acc"] = np.linalg.norm(df[["acc_x", "acc_y", "acc_z"]], axis=1)
    time_window_size = 100
    stride = 10
    i = 0
    data = df["acc"]
    N = data.size
    all_data = []
    while (i + time_window_size) < N:
        all_data.append(data[i:i + time_window_size])
        i += stride
    all_data = np.vstack(all_data)
    nr_time_windows, _ = all_data.shape
    activity_encoded_vec = np.ones((nr_time_windows, 1)) * activity_encoded
    all_data = np.hstack([activity_encoded_vec, all_data])
    return all_data


def unpack_time_windows_data(X_raw):
    nr_data_points, time_window_size = X_raw.shape
    X = []
    for i, x in enumerate(X_raw):
        X.append(
            np.hstack(
                [np.ones((time_window_size, 1)) * i,
                 np.arange(0, time_window_size).reshape(-1, 1),
                 x.reshape(-1, 1)
                 ]
            )
        )
    return np.vstack(X)


def create_features_from_training_data():
    path_X = pathlib.Path(PATH_TIME_WINDOWS, "data_train_X.npy")
    path_y = pathlib.Path(PATH_TIME_WINDOWS, "data_train_y.npy")
    X_raw = np.load(str(path_X))
    y = np.load(str(path_y))
    X = unpack_time_windows_data(X_raw)
    y = pd.Series(y)
    timeseries = pd.DataFrame(X, columns=("id", "time", "x"))
    timeseries = timeseries[["id", "time", "x"]]
    features_filtered_direct = extract_relevant_features(timeseries, y, column_id='id', column_sort='time')
    features_filtered_direct.to_csv(pathlib.Path("data", "features", "train.csv"), index=False)


def create_features_test():
    path_X = pathlib.Path(PATH_TIME_WINDOWS, "data_test_X.npy")
    X_raw = np.load(str(path_X))
    X = unpack_time_windows_data(X_raw)
    features_train = pd.read_csv(pathlib.Path(PATH_FEATURES, "train.csv"))
    kind_to_fc_parameters = tsfresh.feature_extraction.settings.from_columns(features_train)
    df_test = pd.DataFrame(X, columns=("id", "time", "x"))
    df_test = df_test[["id", "time", "x"]]
    features_test = extract_features(
        df_test, column_id='id', column_sort='time', kind_to_fc_parameters=kind_to_fc_parameters
    )
    features_test.to_csv(pathlib.Path(PATH_FEATURES, "test.csv"), index=False)


if __name__ == "__main__":
    create_directories()
    extract_acc_data_from_raw()
    extract_time_windows_from_acc((1, 2, 3, 4), data_type="train")
    extract_time_windows_from_acc((5, ), data_type="test")
    create_features_from_training_data()
    create_features_test()
