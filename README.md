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
├── main.py
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
6. Run `main.py` to train a simple model.
