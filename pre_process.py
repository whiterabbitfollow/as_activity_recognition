import zipfile
from pathlib import Path

import dvc.api
import numpy as np
import pandas as pd
import tsfresh
from tsfresh import extract_features, extract_relevant_features

ENCODING = {"Running": 0, "Standing_still": 1, "Walking": 2}

def create_directories(dirs:list):
    directories = []
    for arg in dirs:
        directories.append(Path(arg))
        arg.mkdir(parents=True, exist_ok=True)
    return directories


def extract_acc_data_from_raw(path_raw:Path, path_acc:Path):
    for path_f in path_raw.iterdir():
        path_f_acc = path_acc / path_f.name
        with path_f.open() as fp:
            acc_data = ""
            for line in fp.readlines():
                if line.split("\t")[1] == "ACC":
                    acc_data += line
            if acc_data:
                with path_f_acc.open("w") as fp:
                    fp.write(acc_data)


def extract_time_windows_from_acc(file_nrs:list, data_type:str, path_acc:Path, path_time_windows:Path, encoding:dict):
    all_data = []
    for path_f_acc in path_acc.iterdir():
        i = int(path_f_acc.stem.split("_")[-1])
        if i in file_nrs:
            all_data.append(extract_time_windows_from_acc_file(path_f_acc, encoding))
    all_data = np.vstack(all_data)
    nr_data_points = all_data.shape[0]
    indx = np.arange(nr_data_points)
    np.random.shuffle(indx)
    np.save(str(path_time_windows / f"data_{data_type}_X.npy"), all_data[indx, 1:])
    np.save(str(path_time_windows / f"data_{data_type}_y.npy"), all_data[indx, 0])


def extract_time_windows_from_acc_file(path_f_acc:Path, encoding:dict):
    path_f = Path(path_f_acc)
    df = pd.read_csv(path_f, sep="\t", names=["time", "type", "acc_x", "acc_y", "acc_z"])
    activity = path_f.stem.rsplit("_", maxsplit=1)[0]
    activity_encoded = encoding[activity]
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


def unpack_time_windows_data(X_raw:np.ndarray):
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


def create_features_from_training_data(time_windows:Path):
    path_X = Path(time_windows, "data_train_X.npy")
    path_y = Path(time_windows, "data_train_y.npy")
    X_raw = np.load(str(path_X))
    y = np.load(str(path_y))
    X = unpack_time_windows_data(X_raw)
    y = pd.Series(y)
    timeseries = pd.DataFrame(X, columns=("id", "time", "x"))
    timeseries = timeseries[["id", "time", "x"]]
    features_filtered_direct = extract_relevant_features(timeseries, y, column_id='id', column_sort='time')
    features_filtered_direct.to_csv(Path("data", "features", "train.csv"), index=False)


def create_features_test(time_windows:Path, features:Path):
    path_X = Path(time_windows, "data_test_X.npy")
    X_raw = np.load(str(path_X))
    X = unpack_time_windows_data(X_raw)
    features_train = pd.read_csv(Path(features, "train.csv"))
    kind_to_fc_parameters = tsfresh.feature_extraction.settings.from_columns(features_train)
    df_test = pd.DataFrame(X, columns=("id", "time", "x"))
    df_test = df_test[["id", "time", "x"]]
    features_test = extract_features(
        df_test, column_id='id', column_sort='time', kind_to_fc_parameters=kind_to_fc_parameters
    )
    features_test.to_csv(Path(features, "test.csv"), index=False)


if __name__ == "__main__":
    
    
    config = dvc.api.params_show()
    data_path = Path(config['data_paths']["data"])
    raw_path = data_path / config['data_paths']["raw"]
    acc_path = data_path / config['data_paths']["acc"]
    windows = data_path / config['data_paths']["time"]
    features = data_path / config['data_paths']["feat"]
    input_data = config['data_paths']["zip"]
    print("Creating directories")
    paths = [data_path, raw_path, acc_path, windows, features]
    dirs = create_directories(paths)
    with zipfile.ZipFile(input_data, 'r') as zip_ref:
        zip_ref.extractall(raw_path)
    print("Extracting acc data")
    extract_acc_data_from_raw(raw_path, acc_path)
    print("Extracting time windows")
    extract_time_windows_from_acc(file_nrs=(1, 2, 3, 4), data_type="train", path_acc=acc_path, path_time_windows=windows, encoding=ENCODING)
    extract_time_windows_from_acc(file_nrs = (5, ), data_type="test", path_acc=acc_path, path_time_windows=windows, encoding=ENCODING)
    print("Creating features")
    create_features_from_training_data(time_windows=windows)
    create_features_test(time_windows=windows, features=features)
