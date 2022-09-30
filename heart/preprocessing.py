"""
Heart data preprocessing functions
"""
import numpy as np
import pandas as pd
import toml
import utils.helper as hlp
from utils.helper import log, import_data_frame

from ecgdetectors import Detectors



#from biosppy.signals import ecg
#pd.options.plotting.backend = 'plotly'
#import plotly.io as pio
#pio.renderers.default = "browser"
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import plotly.express as px
#import plotly.graph_objects as go
#from scipy.signal import welch

class HeartPreprocessor:
    def __init__(self, data: pd.DataFrame = None, **params):

        # Load params
        self.data = None
        self.ecg_col = None
        self.time_col = None
        self.load_params(**params)

        # Load data
        if data is not None:
            self.load_data(data=data)

    def load_data(self, data: pd.DataFrame):
        """
        Load eye-tracking data for processing
        """
        self.data = data

        # Run some checks
        assert (
            self.ecg_col in self.data
        ), f"Could not find ecg column '{self.ecg_col}' in data frame."
        assert (
            self.time_col in self.data
        ), f"Could not find time column '{self.time_col}' in data frame."
        assert (
            self.data[self.time_col].dtype == "float"
        ), f"Need unix epoch timestamps in time column '{self.time_col}'"

    def load_params(self, **params):

        params = params if params else {}

        # Set basic attributes
        self.ecg_col = (
            params["ecg_col"] if "ecg_col" in params else "ecg"
        )
        self.time_col = (
            params["time_col"] if "time_col" in params else "timestamp"
        )

        # If no step arguments are given: run everything with default parameters
        params.setdefault("rr_peak_detection_params", True)
        self.params = params

    def check_sfreq(self):

        """
        Calculates sampling frequency
        """

        self.duration = (self.data[self.time_col].iloc[-1] - self.data[self.time_col].iloc[0]) / 1000
        self.sample_rate = len(self.data) / self.duration

    def rr_peak_detection(self, detector: str = "engzee_detector"):

        """
        Detects rr-peaks and outputs indexes of peaks
        """

        # Initialize class using sampling rate of recording
        detectors = Detectors(self.sample_rate)

        detectors = {
            'hamilton_detector': detectors.hamilton_detector,
            'christov_detector': detectors.christov_detector,
            'engzee_detector': detectors.engzee_detector,
        }

        # get timestamps of r-peaks
        r_peak_indexes = detectors[detector](self.data[self.ecg_col])
        r_peak_timestamps = pd.DataFrame(self.data[self.time_col].iloc[r_peak_indexes],
                                         columns=[self.time_col])

        # create new df with timestamps of r-peaks and rr-intervals
        r_peak_timestamps['rr_int'] = r_peak_timestamps.diff()

        return r_peak_timestamps

    def pipeline(self, data: pd.DataFrame = None, **params):

        # Load new parameters if provided
        self.load_params(**params)

        # Load data if provided
        if data is not None:
            self.load_data(data=data)

        assert (
                hasattr(self, "data") and self.data is not None
        ), "Need to load data first! Either load data with the `load` method, or pass a `data` argument to the pipeline function."

        # Line up arguments
        rr_peak_detection_params = self.params["rr_peak_detection_params"]

        # get sfreq
        self.check_sfreq()

        # detect r-peaks
        if rr_peak_detection_params:
            log("⚙️ Detecting r-peaks.", color="green")

            # Must be dict or bool
            assert isinstance(
                rr_peak_detection_params, (bool, dict)
            ), "Please pass a boolean or a dictionary with method parameters to the `remove_outliers_params` method!"

            # If bool, make dict
            if rr_peak_detection_params is True:
                rr_peak_detection_params = {}

            # Set default parameters
            rr_peak_detection_params.setdefault("detector", "engzee_detector")

            # Execute method
            r_peak_timestamps = self.rr_peak_detection(
                **rr_peak_detection_params,
            )

        # Return preprocessed data frame
        return r_peak_timestamps


if __name__ == '__main__':

    print("Test area!")

    # Load config
    with open("../configs/test.toml") as config_file:
        config = toml.load(config_file)
    heart_preprocessing_params = config["mrt"]["heart"]["preprocessing"]

    # get data
    data = import_data_frame(path="../data/klaas_202209130909/clt/ecg.csv")

    # Try pipeline
    rr_data = HeartPreprocessor(
        data=data, **heart_preprocessing_params
    ).pipeline()

    #test = HeartPreprocessor(data)

    '''
    data = import_data_frame('../data/klaas_202209130909/clt/ecg.csv')
    data.reset_index(drop=False, inplace=True)

    duration = (data.timestamp.iloc[-1] - data.timestamp.iloc[0])/1000
    sample_rate = len(data)/duration

    results = ecg.ecg(data.ecg, sampling_rate=130, show=False, interactive=False)
    results = dict(**results)

    rtimestamps = data.iloc[results['rpeaks']].timestamp - data.iloc[results['rpeaks']].timestamp.shift(1)

    f, Px = welch(data.ecg, fs=sample_rate, nperseg=2048, scaling='spectrum')
    plt.semilogy(f, Px)
    plt.show()

    data['datetime'] = pd.to_datetime(data.timestamp, unit='ms')

    total_duration = (data.datetime.iloc[-1] - data.datetime.iloc[0]).total_seconds()
    total_samples = len(data)
    sample_rate_global = total_samples/total_duration
    print(f'Total duration: {total_duration} seconds - total samples: {total_samples} --> sample rate of {sample_rate_global} Hz')

    data['datetime_shift'] = data.datetime.shift(-1)
    data['difference'] = (data.datetime_shift - data.datetime).dt.total_seconds()
    data_new = data.loc[(data.difference > 0.0)]
    test = data.difference.value_counts()
    plt.plot(test.values)
    plt.show()

    def sr(datetime_series):
        total_duration = (datetime_series.iloc[-1] - datetime_series.iloc[0])/1000
        total_samples = len(datetime_series)
        sample_rate_local = total_samples / total_duration
        return sample_rate_local

    test = data.timestamp.rolling(1000).apply(sr).rolling(1000).mean()
    test.plot(title='rolling_sample_rate')
    fig = go.Figure()
    fig.add_trace(
        trace={'x': data.index,
               'y': data.ecg}
                  )
    #fig.show()
    '''


