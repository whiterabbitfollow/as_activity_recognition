import yaml
import collections
import importlib
from sklearn.pipeline import Pipeline
from dataclasses import dataclass

class Model(collections.namedtuple('Model', ('model', 'pipeline', 'defence'), defaults=(None, None))):
    def __new__(cls, loader, node):
        return super().__new__(cls, **loader.construct_mapping(node))

# defaults=(None,)
# @dataclass
# class Model:
#     model: dict
#     pipeline: dict = None
#     classifier : bool = True
#     library : str =  "sklearn"
#     time_series : bool = False
    

    def gen_from_dict(self, obj_dict: dict, *args) -> list:
        """
        Imports and initializes objects from yml file. Returns a list of instantiated objects.
        :param obj_tuple: (full_object_name, params)
        """
        key = list(obj_dict.keys())[0]
        value = obj_dict[key]
        obj_tuple = (key, value)
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
    
    
    def load(self):
        # Initialize model
        model = self.gen_from_dict(self.model)
        pipe_list = []
        i = 0
        # Initialize pipeline
        if  self.pipeline is not None:
            if "cache" in self.pipeline:
                cache = self.pipeline.pop("cache")
            for step in self.pipeline:
                name = list(step.keys())[i]
                component = step[name]
                obj_ = self.gen_from_dict(component)
                pipe_list.append((name, obj_))
                i += 0
            pipe_list.append(("model", model))
            model = Pipeline(pipe_list)
        if self.defence is not None:
            raise NotImplementedError("Defences are not implemented yet")
        return model
    
    
yaml.add_constructor('!Model', Model)
if __name__ == '__main__':
    document = """\
    model:
        sklearn.ensemble.RandomForestClassifier:
            n_estimators : 10,
            max_depth : 1,
    pipeline:
        - preprocessor:
            sklearn.preprocessing.StandardScaler:
                with_mean : True
                with_std : True
        - feature_selector:
            sklearn.feature_selection.SelectKBest:
                k: 30
    """
    document = "!Model\n" + document
    print(document)
    input("Press Enter to continue...")
    config = yaml.unsafe_load(document)
    model = config.load()
    assert hasattr(model, 'fit'), "Model must have a fit method"
    assert hasattr(model, 'predict'), "Model must have a predict method"
    