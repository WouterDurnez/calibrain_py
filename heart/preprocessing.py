"""
Heart data preprocessing functions
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.express as px
import toml
from tqdm import tqdm
from plotly_resampler import FigureResampler, register_plotly_resampler

import utils.helper as hlp
from utils.helper import log, import_data_frame

pd.options.plotting.backend = 'plotly'
import plotly.io as pio
pio.renderers.default = "browser"
import matplotlib.pyplot as plt
import plotly.graph_objects as go


if __name__ == '__main__':

    data = pd.read_csv(filepath_or_buffer='../data/dummy/ECG.csv')
    data.reset_index(drop=True, inplace=True)

    register_plotly_resampler()

    fig = go.Figure()
    fig.add_trace(
        trace={'x': data.index,
               'y': data.ECG}
                  )
    fig.show()

    duration_first_last = data.time.iloc[-1] - data.time.iloc[0]