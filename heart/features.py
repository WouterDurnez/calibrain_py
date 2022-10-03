"""
ECG data feature calculation functions
"""

import numpy as np
import pandas as pd
from joblib import Memory
import toml

from pyhrv.hrv import hrv

from utils.helper import load_config, import_data_frame, log
from heart.preprocessing import HeartPreprocessor

mem = Memory(location='../cache/heart', verbose=0)

class HeartFeatures:

    def __init__(self, rr_data: pd.DataFrame = None, **params):

        self.data = None
        self.rr_col = None

        # Load params
        self.load_params(**params)

        # Load data
        if rr_data is not None:
            self.load_data(data=rr_data)

    def load_data(self, rr_data):
        self.rr_data = rr_data

    def load_params(self, **params):

        params = params if params else {}

        # Set basic attributes
        self.rr_col = (
            params['rr_col'] if 'rr_col' in params else 'rr_int'
        )

    ############
    # FEATURES #
    ############

    # Time-domain features
    def get_mean_rri_diff(self):

        self.mean_rri_diff = rr_data['rr_int'].diff().mean()

    # Frequency domain features
    def get_rmssd(self):
        self.rmssd = hrv(rr_data[self.rr_col])['rmssd']

    # Non-linear domain features



if __name__ == '__main__':

    print("Test area!")

    # Load config
    with open("../configs/test.toml") as config_file:
        config = toml.load(config_file)
    heart_preprocessing_params = config["clt"]["heart"]["preprocessing"]

    # Get data
    data = import_data_frame(path="../data/klaas_202209130909/clt/ecg.csv")

    # Try pipeline
    rr_data = HeartPreprocessor(
        data=data, **heart_preprocessing_params
    ).pipeline()

    # Feature calculation
    feature_object = HeartFeatures(data=rr_data)

