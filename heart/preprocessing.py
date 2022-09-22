"""
Heart data preprocessing functions
"""
import numpy as np
import pandas as pd
from biosppy.signals import ecg
pd.options.plotting.backend = 'plotly'
import plotly.io as pio
pio.renderers.default = "browser"
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from utils.helper import import_data_frame
from scipy.signal import welch

if __name__ == '__main__':

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

    '''data['datetime'] = pd.to_datetime(data.timestamp, unit='ms')

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
    test.plot(title='rolling_sample_rate')'''
    fig = go.Figure()
    fig.add_trace(
        trace={'x': data.index,
               'y': data.ecg}
                  )
    fig.show()

