from data import Data
from model import Model
import yaml



def load_experiment(model_file, data_file):
    with open(data_file, "r") as f:
        document = yaml.load(f, Loader = yaml.Loader)
    document =  str(document['data'])
    document =  str(document)
    data = yaml.load("!Data\n" + document, Loader = yaml.Loader)
    assert isinstance(data, Data)
    data = data()
    
    with open(model_file, "r") as f:
        document = yaml.load(f, Loader = yaml.Loader)
    document =  str(document['model'])
    config = yaml.load("!Model\n" +document, Loader = yaml.Loader)
    assert isinstance(config, Model)
    model = config.load()
    
    return data, model



if __name__ == '__main__':
    model_file = "model.yaml"
    data_file = "data.yaml"
    data, model = load_experiment(model_file, data_file)
    assert "X_train" in data, "X_train not found"
    assert "X_test" in data, "X_test not found"
    assert "y_train" in data, "y_train not found"
    assert "y_test" in data, "y_test not found"
    assert hasattr(model, 'fit'), "Model must have a fit method"
    assert hasattr(model, 'predict'), "Model must have a predict method"
    