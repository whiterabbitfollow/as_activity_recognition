1. Install packages. `python -m pip install -r requirements.txt`
3. Create the directory ./data/raw
2. Move IMU data to ./data/raw.
3. You should now have the following structure:
```
├── data
│   ├── pre_process.py
│   └── raw
│       ├── Running_1.txt
...
│       ├── Running_5.txt
│       ├── Standing_still_1.txt
...
│       ├── Standing_still_5.txt
│       ├── Walking_1.txt
...
│       └── Walking_5.txt
├── classify.py
├── preprocess.py
├── README.md
└── requirements.txt
```
4. Run `python data/pre_process.py`
5. You should now have the following structure. 
```
.
├── data
│   ├── features
│   │   ├── test.csv
│   │   └── train.csv
│   ├── pre_process.py
│   ├── raw
│   │   ├── Running_1.txt
...
│   │   ├── Running_5.txt
│   │   ├── Standing_still_1.txt
...
│   │   ├── Standing_still_5.txt
│   │   ├── Walking_1.txt
...
│   │   └── Walking_5.txt
│   ├── raw_acc
│   │   ├── Running_1.txt
...
│   │   ├── Running_5.txt
│   │   ├── Standing_still_1.txt
...
│   │   ├── Standing_still_5.txt
│   │   ├── Walking_1.txt
...
│   │   └── Walking_5.txt
│   └── time_windows
│       ├── data_test_X.npy
│       ├── data_test_y.npy
│       ├── data_train_X.npy
│       └── data_train_y.npy
├── main.py
├── README.md
└── requirements.txt
```
6. Run `dvc repro` to train the default model according to the pipeline outlined in `dvc.yaml`. Folders and filenames are specified in `config.toml`. Model input parameters are in `params.yaml`. All tested parameters are in `configs`. 
