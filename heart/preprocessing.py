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
mpl.use('MacOSX')
import plotly.express as px

def fix_timestamps(row):

    part1 = str(row.timestamp).split('.')[0]
    part2 = str(row.ts).split('.')[0]
    return float(part1 + '.' + part2)

if __name__ == '__main__':

    data = pd.read_csv(filepath_or_buffer='../data/dummy/ECG.csv')
    data.reset_index(drop=False, inplace=True)
    data.columns = ['timestamp','ts','ecg']
    data['timestamp_full'] = data.apply(fix_timestamps, axis=1)

    start = data.timestamp_full.iloc[0]
    stop = data.timestamp_full.iloc[-1]

    new_timestamps = [start + i*1000/130 for i in range(len(data))]
    data['timestamp_new'] = new_timestamps
    stop_new = data.timestamp_new.iloc[-1]

    fig = px.line(data, y=['timestamp_full','timestamp_new'])
    fig.show()

    results = dict(**ecg.ecg(data.ecg[:1000], sampling_rate=130, show=False, interactive=False))
    rtimestamps = data.iloc[results['rpeaks']].timestamp - data.iloc[results['rpeaks']].timestamp.shift(1)

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
    '''fig = go.Figure()
    fig.add_trace(
        trace={'x': data.index,
               'y': data.ECG}
                  )
    fig.show()'''

