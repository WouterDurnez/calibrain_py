"""
ECG data feature calculation functions
"""

import numpy as np
import pandas as pd
from joblib import Memory
import toml

from pyhrv.hrv import hrv
from pyhrv import frequency_domain, nonlinear

from utils.helper import load_config, import_data_frame, log
from heart.preprocessing import HeartPreprocessor

import warnings

mem = Memory(location='../cache/heart', verbose=0)

class HeartFeatures:

    def __init__(self, rr_data: pd.DataFrame = None, **params):

        self.rr_data = None
        self.time_col = None
        self.rr_col = None

        # Initialize attributes
        self.mean_rri = None
        self.rri_diff_mean = None
        self.rri_diff_min = None
        self.rri_diff_max = None
        self.rmssd = None
        self.sdsd = None
        self.sdnn =None

        self.rel_power_vlf = None
        self.rel_power_lf = None
        self.rel_power_hf = None
        self.lf_hf_ratio = None

        self.sd1 = None
        self.sd2 = None
        self.sd_ratio = None

        self.features = None

        # Load params
        if params:
            self.load_params(**params)

        # Load data
        if rr_data is not None:
            self.load_data(rr_data=rr_data)

    def load_data(self, rr_data):
        self.rr_data = rr_data

    def load_params(self, **params):

        if hasattr(self, 'params'):
            log('⚠️ Overwriting previous parameters!')

        params = params if params else {}

        # Set basic attributes
        self.rr_col = (
            params['rr_col'] if 'rr_col' in params else 'rr_int'
        )
        self.time_col = (
            params['time_col'] if 'time_col' in params else 'timestamp'
        )

        # If no step arguments are given: run everything with default parameters
        params.setdefault('time_domain', True)
        params.setdefault('frequency_domain', True)
        params.setdefault('nonlinear_domain', True)
        params.setdefault('detrend', False)

        self.params = params

    ############
    # FEATURES #
    ############

    # Time-domain features
    def get_time_domain_hrv_features(self):
        # TODO: look into UserWarning TINN computation
        """
        Calculates all time-domain hrv features
        """
        self.mean_rri = hrv(rpeaks=self.rr_data[self.time_col])['nni_mean']

        self.rri_diff_mean = hrv(rpeaks=self.rr_data[self.time_col])['nni_diff_mean']
        self.rri_diff_min = hrv(rpeaks=self.rr_data[self.time_col])['nni_diff_min']
        self.rri_diff_max = hrv(rpeaks=self.rr_data[self.time_col])['nni_diff_max']

        self.rmssd = hrv(rpeaks=self.rr_data[self.time_col])['rmssd']
        self.sdsd = hrv(rpeaks=self.rr_data[self.time_col])['sdsd']
        self.sdnn = hrv(rpeaks=self.rr_data[self.time_col])['sdnn']

    # Frequency domain features
    def get_frequency_domain_hrv_features(self, detrend: bool = False):
        warnings.warn("Frequency domain features can be confounded by duration of recording.")

        self.rel_power_vlf = frequency_domain.welch_psd(
            rpeaks=self.rr_data[self.time_col],
            show=False,
            detrend = detrend,
        )['fft_rel'][0]
        self.rel_power_lf = frequency_domain.welch_psd(
            rpeaks=self.rr_data[self.time_col],
            show=False,
            detrend = detrend,
        )['fft_rel'][1]
        self.rel_power_hf = frequency_domain.welch_psd(
            rpeaks=self.rr_data[self.time_col],
            show=False,
            detrend = detrend,
        )['fft_rel'][1]

        self.lf_hf_ratio = frequency_domain.welch_psd(
            rpeaks=self.rr_data[self.time_col],
            show=False,
            detrend = detrend
        )['fft_ratio']

    # Non-linear domain features
    def get_nonlinear_domain_hrv_features(self):

        self.sd1 = nonlinear.nonlinear(rpeaks=self.rr_data[self.time_col])['sd1']
        self.sd2 = nonlinear.nonlinear(rpeaks=self.rr_data[self.time_col])['sd2']
        self.sd_ratio = nonlinear.nonlinear(rpeaks=self.rr_data[self.time_col])['sd_ratio']

    def pipeline(self, data: pd.DataFrame = None, **params):

        # Load new parameters if provided
        self.load_params(**params)

        # Load data if provided
        if data is not None:
            self.load_data(rr_data=self.rr_data)

        # Calculate requested features
        if self.params['time_domain']:
            self.get_time_domain_hrv_features()
        if self.params['frequency_domain']:
            self.get_frequency_domain_hrv_features()
        if self.params['nonlinear_domain']:
            self.get_nonlinear_domain_hrv_features()

        # Combine features dict
        self.features = {
            feature: value
            for feature, value in self.__dict__.items()
            if feature
               in ['mean_rri', 'rri_diff_mean', 'rri_diff_min', 'rri_diff_max', 'rmssd',
                   'sdsd', 'sdnn', 'rel_power_vlf', 'rel_power_lf', 'rel_power_hf',
                   'lf_hf_ratio', 'sd1', 'sd2', 'sd_ratio']
        }

        #self.features = pd.DataFrame(self.features, index = [0])


if __name__ == '__main__':

    print("Test area!")

    # Load config
    with open("../configs/test.toml") as config_file:
        config = toml.load(config_file)
    heart_preprocessing_params = config["clt"]["heart"]["preprocessing"]
    heart_features_params = config["clt"]["heart"]["features"]

    # Get data
    data = import_data_frame(path="../data/klaas_202209130909/clt/ecg.csv")

    # Try pipeline
    rr_data = HeartPreprocessor().pipeline(data=data, **heart_preprocessing_params)

    # Feature calculation
    feature_object =  HeartFeatures(rr_data=rr_data)
    feature_object.pipeline(data=rr_data, **heart_features_params)

