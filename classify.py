import argparse
import configparser
import importlib
from hashlib import md5
from pathlib import Path


import numpy as np
import ruamel.yaml as yaml
from pandas import Series
from sklearn.metrics import (accuracy_score, explained_variance_score,
                             f1_score, mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
import dvc.api
# Default scorers
REGRESSOR_SCORERS = {
    "MAPE": mean_absolute_percentage_error,
    "MSE": mean_squared_error,
    "MAE": mean_absolute_error,
    "R2": r2_score,
    "EXVAR": explained_variance_score,
}
CLASSIFIER_SCORERS = {
    "F1": f1_score,
    "ACC": accuracy_score,
    "PREC": precision_score,
    "REC": recall_score,
    "AUC": roc_auc_score,
}

def gen_from_tuple(obj_tuple: list, *args) -> list:
    """
    Imports and initializes objects from yml file. Returns a list of instantiated objects.
    :param obj_tuple: (full_object_name, params)
    """
    library_name = ".".join(obj_tuple[0].split(".")[:-1])
    class_name = obj_tuple[0].split(".")[-1]
    global tmp_library
    tmp_library = None
    tmp_library = importlib.import_module(library_name)
    global temp_object
    temp_object = None
    global params
    params = obj_tuple[1]
    if len(args) > 0:
        global positional_arg
        positional_arg = args[0]
        exec(f"temp_object = tmp_library.{class_name}(positional_arg, **{params})", globals())
        del positional_arg
    elif len(args) == 0:
        exec(f"temp_object = tmp_library.{class_name}(**params)", globals())
    else:
        raise ValueError("Too many positional arguments")
    del params
    del tmp_library
    return temp_object


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    
    print("=========================================")
    print("Reading config file...")
    config.read(args.config)
    data_path = Path(config["data_paths"]["data"])
    windows = data_path / config["data_paths"]["time"]
    print(f"Loading data from time windows folder: {windows}")
    # Extract features
    X_train = np.load(str(windows / "data_train_X.npy"))
    y_train = np.load(str(windows / "data_train_y.npy"))
    X_test = np.load(str(windows / "data_test_X.npy"))
    y_test = np.load(str(windows / "data_test_y.npy"))
    print("Data loaded")
    ##################################
    # Set model name and params here #
    ##################################
    params = dvc.api.params_show()
    model = params.pop("model")
    params = params['params']
    param_dict = {}
    for param in params:
        key = list(param.keys())[0]
        value = list(param.values())[0]
        param_dict[key] = value
    params = param_dict
    input("Press enter to continue")
    ###################################
    # Saves configuration to configs/ #
    ###################################
    obj_tuple = (model, params)
    my_hash = md5(str(obj_tuple).encode("utf-8")).hexdigest()
    dict_ = {"name" : model, "params" : params}    
    print(f"Generated model with hash: {my_hash}")
    config_path = Path(config["result_paths"]["config"], config["result_paths"]["model_file"])
    result_path = Path(config["result_paths"]["results"], config["result_paths"]["scores"])
    config_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        yaml.dump(dict_, f)
    ####################################
    #     Creates object from tuple    #
    ####################################
    clf = gen_from_tuple(obj_tuple)
    ####################################
    #             Science              #
    ####################################
    clf.fit(X_train, y_train)
    score_dict = {}
    for key, value in CLASSIFIER_SCORERS.items():
        try:
            score_dict.update({key :value(y_test, clf.predict(X_test))})
        except ValueError as e:
            if "average=" in str(e):
                score_dict.update({key : value(y_test, clf.predict(X_test), average='weighted')})
    
    del temp_object
    ####################################
    #             Saving               #
    ####################################
    Series(score_dict).to_json(result_path)
    ####################################
    #           Visualising            #
    ####################################
    # Balance
    from sklearn.model_selection import StratifiedKFold
    from yellowbrick.classifier import (ROCAUC, ClassificationReport,
                                        ClassPredictionError, ConfusionMatrix,
                                        DiscriminationThreshold,
                                        PrecisionRecallCurve)
    from yellowbrick.model_selection import (CVScores, DroppingCurve,
                                             FeatureImportances, LearningCurve)
    from yellowbrick.target import ClassBalance, FeatureCorrelation

    from pre_process import ENCODING
    # For Visualizing the Data
    tar_viz = (ClassBalance, FeatureCorrelation)
    # For Visualizing the Feature Selection
    mod_viz = (DroppingCurve, CVScores, LearningCurve, FeatureImportances)
    # For Visualizing the Model
    cls_viz = (ConfusionMatrix, ROCAUC, PrecisionRecallCurve, ClassPredictionError, ClassificationReport, ClassPredictionError)
    features = np.array(range(X_train.shape[1]))
    parent = Path(result_path).parent
    # Balance
    visualizer = ClassBalance(labels=Series(y_train).unique())
    visualizer.fit(y_train)
    visualizer.show(outpath = str(parent / "balance.pdf"))
    # Pearson
    visualizer = FeatureCorrelation(labels=features)
    visualizer.fit(X_train, y_train)        
    visualizer.show(outpath = str(parent / config['plots']['correlation'])) 
    # Fischer
    visualizer = FeatureCorrelation(labels=features, method='mutual_info-classification', sort=True)
    visualizer.fit(X_train, y_train)       
    visualizer.show(outpath = str(parent / config['plots']['information']))
    # Feature Selection
    visualizer = DroppingCurve(clf, scoring='f1_weighted')
    visualizer.fit(X_train, y_train)
    visualizer.show(outpath = str(parent / config['plots']['dropping']))
    # Learning Curve
    cv = StratifiedKFold(n_splits=12)
    sizes = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]
    visualizer = LearningCurve(clf, cv=cv, scoring='f1_weighted', train_sizes=sizes, n_jobs=4)
    visualizer.fit(X_train, y_train)
    visualizer.show(outpath = str(parent / config['plots']['learning']))
    # Cross Validation
    visualizer = CVScores(clf, cv=cv, scoring='f1_weighted')
    visualizer.fit(X_train, y_train)
    visualizer.show(outpath = str(parent / config['plots']['cross_validation']))
    # Feature Importance
    visualizer = FeatureImportances(clf)
    visualizer.fit(X_train, y_train)
    visualizer.show(outpath = str(parent / config['plots']['feature_importance']))
    # ROCAUC
    visualizer = ROCAUC(clf, classes=list(ENCODING.items())[:])
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show(outpath = str(parent / config['plots']['rocauc']))
    
    

    