import collections
from sklearn.datasets import make_blobs, make_moons, make_classification, load_boston, load_iris, load_diabetes, load_wine
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import collections

generated = {
    "blobs": make_blobs,
    "moons": make_moons,
    "classification": make_classification,
}
real = {
    "boston": load_boston,
    "iris": load_iris,
    "diabetes": load_diabetes,
    "wine": load_wine,
}

class Data(collections.namedtuple('Data', 'name, shuffle, random_state, train_size, stratify')):
    def __new__(cls, loader, node):
        return super().__new__(cls, **loader.construct_mapping(node))
        
    def __call__(self, **kwargs):
        for kwarg in kwargs:
            if hasattr(self, kwarg):
                setattr(self, kwarg, kwargs[kwarg])
                kwargs.pop(kwarg)
        if self.name in real:
            big_X, big_y = real[self.name](return_X_y=True, **kwargs)
        elif self.name in generated:
            big_X, big_y = generated[self.name]( **kwargs)
        elif isinstance(Path(self.name), Path) and Path(self.name).exists() and str(self.name).endswith(".npz"):
            data = np.load(self.name)
            big_X = data['X']
            big_y = data['y']
        elif isinstance(Path(self.name), Path) and Path(self.name).exists() and str(self.name).endswith(".csv"):
            assert "target" in self.params, "target column must be specified"
            df = pd.read_csv(self.name)
            big_X = df.drop(self.target, axis = 1)
            big_y = df[self.target]
        elif isinstance(Path(self.name), Path) and Path(self.name).exists() and str(self.name).endswith(".json"):
            assert "target" in self.params, "target column must be specified"
            df = pd.read_json(self.name)
            big_X = df.drop(self.target, axis = 1)
            big_y = df[self.target]
        else:
            raise ValueError(f'Unknown dataset: {self.name}')
        if self.stratify == True:
            stratify = big_y
        else:
            stratify = None
        X_train, X_test, y_train, y_test =  train_test_split(big_X, big_y, shuffle = self.shuffle, random_state = self.random_state, train_size = self.train_size, stratify = stratify)
        return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
yaml.add_constructor('!Data', Data)
if __name__ == '__main__':
    document = """
    !Data
    name: data/features/data.npz
    shuffle : True
    random_state : 42
    train_size : 2500
    stratify : True
    """
    data = yaml.load(document, Loader = yaml.Loader)
    data = data()
    assert "X_train" in data
