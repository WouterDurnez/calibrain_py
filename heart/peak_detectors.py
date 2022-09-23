#
# Visualize different r-peak detectors' performance
#

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from utils.helper import import_data_frame

from ecgdetectors import Detectors

from bokeh.plotting import figure, show, save, output_file
from bokeh.models import ColumnDataSource, Span, HoverTool
from bokeh.layouts import gridplot

if __name__ == '__main__':

    # import data
    data = import_data_frame('../data/klaas_202209130909/clt/ecg.csv')
    data.reset_index(drop=False, inplace=True)

    # extract sf
    duration = (data.timestamp.iloc[-1] - data.timestamp.iloc[0])/1000
    sample_rate = len(data)/duration

    # plot raw data
    fig = go.Figure()
    fig.add_trace(
        trace={
            'x': data.index,
            'y': data.ecg
                }
                  )
    fig.show()

    # TODO: do we need denoising? filtering? baseline fitting?

    ####################
    # R-peak Detection #
    ####################

    # take smaller sample of data for plotting purposes
    data = data[:10000]

    # Initialize class using sampling rate of recording
    detectors = Detectors(sample_rate)

    # specify detectors
    detectors = {
        'hamilton_detector': detectors.hamilton_detector,
        'christov_detector': detectors.christov_detector,
        'engzee_detector': detectors.engzee_detector,
        'pan_tompkins_detector': detectors.pan_tompkins_detector,
        'swt_detector': detectors.swt_detector,
        'two_average_detector': detectors.two_average_detector,
        #'wqrs_detectpr': detectors.wqrs_detector, # unable to proceed with this one, TODO!
    }

    r_peaks = {}
    for detector in detectors.keys():
        r_peaks[detector] = detectors[detector](data.ecg)

    #############################################
    # Visualize detections by for each detector #
    #############################################

    p1 = figure(
               plot_width=1400, plot_height=100,
               tools="hover,pan,zoom_in,xwheel_zoom, reset,save",
               active_drag='pan',
               active_scroll='xwheel_zoom',
               #x_axis_type='datetime',
               )
    p1.title.text = 'hamilton_detector'

    source = ColumnDataSource(data={
        "index": data.index,
        "ecg": data.ecg,
        "time": data.time,
    })

    # add a line renderer
    p1.line(x="index", y="ecg", line_width=2, source=source)

    hover = p1.select(dict(type=HoverTool))
    hover.tooltips = [("index", "@index"), ("time", "@time"), ("ecg", "@ecg")]
    hover.formatters = {'@time': 'datetime'}
    hover.mode = 'vline'

    for ix in r_peaks['hamilton_detector']:
        sp_detection = Span(location=ix,
                        dimension='height', line_color='green', line_dash='solid', line_width=2)
        p1.add_layout(sp_detection)

    p2 = figure(
               plot_width=1400, plot_height=100,
               tools="hover,pan,zoom_in,xwheel_zoom, reset,save",
               active_drag='pan',
               active_scroll='xwheel_zoom',
               x_range=p1.x_range,
                y_range=p1.y_range
               )
    p2.title.text = 'christov_detector'

    # add a line renderer
    p2.line(x="index", y="ecg", line_width=2, source=source)

    for ix in r_peaks['christov_detector']:
        sp_detection = Span(location=ix,
                        dimension='height', line_color='red', line_dash='solid', line_width=2)
        p2.add_layout(sp_detection)

    p3 = figure(
        plot_width=1400, plot_height=100,
        tools="hover,pan,zoom_in,xwheel_zoom, reset,save",
        active_drag='pan',
        active_scroll='xwheel_zoom',
        x_range=p1.x_range,
        y_range=p1.y_range
    )
    p3.title.text = 'engzee_detector'

    # add a line renderer
    p3.line(x="index", y="ecg", line_width=2, source=source)

    for ix in r_peaks['engzee_detector']:
        sp_detection = Span(location=ix,
                            dimension='height', line_color='yellow', line_dash='solid', line_width=2)
        p3.add_layout(sp_detection)

    p4 = figure(
        plot_width=1400, plot_height=100,
        tools="hover,pan,zoom_in,xwheel_zoom, reset,save",
        active_drag='pan',
        active_scroll='xwheel_zoom',
        x_range=p1.x_range,
        y_range=p1.y_range
    )
    p4.title.text = 'pan_tompkins_detector'

    # add a line renderer
    p4.line(x="index", y="ecg", line_width=2, source=source)

    for ix in r_peaks['pan_tompkins_detector']:
        sp_detection = Span(location=ix,
                            dimension='height', line_color='black', line_dash='solid', line_width=2)
        p4.add_layout(sp_detection)

    p5 = figure(
        plot_width=1400, plot_height=100,
        tools="hover,pan,zoom_in,xwheel_zoom, reset,save",
        active_drag='pan',
        active_scroll='xwheel_zoom',
        x_range=p1.x_range,
        y_range=p1.y_range
    )
    p5.title.text = 'swt_detector'

    # add a line renderer
    p5.line(x="index", y="ecg", line_width=2, source=source)

    for ix in r_peaks['swt_detector']:
        sp_detection = Span(location=ix,
                            dimension='height', line_color='blue', line_dash='solid', line_width=2)
        p5.add_layout(sp_detection)

    p6 = figure(
        plot_width=1400, plot_height=100,
        tools="hover,pan,zoom_in,xwheel_zoom, reset,save",
        active_drag='pan',
        active_scroll='xwheel_zoom',
        x_range=p1.x_range,
        y_range=p1.y_range
    )
    p6.title.text = 'two_average_detector'

    # add a line renderer
    p6.line(x="index", y="ecg", line_width=2, source=source)

    for ix in r_peaks['two_average_detector']:
        sp_detection = Span(location=ix,
                            dimension='height', line_color='orange', line_dash='solid', line_width=2)
        p6.add_layout(sp_detection)

    # Inspect differences in RR-ints
    hamilton_detector_diff = pd.Series(r_peaks['hamilton_detector']).diff()
    christov_detector_diff = pd.Series(r_peaks['christov_detector']).diff()
    engzee_detector_diff = pd.Series(r_peaks['engzee_detector']).diff()
    pan_tompkins_detector_diff = pd.Series(r_peaks['pan_tompkins_detector']).diff()
    swt_detector_diff = pd.Series(r_peaks['swt_detector']).diff()
    two_average_detector_diff = pd.Series(r_peaks['two_average_detector']).diff()

    p_diff = figure(
        plot_width=1400, plot_height=400,
        tools="hover,pan,zoom_in,xwheel_zoom, reset,save",
        active_drag='pan',
        active_scroll='xwheel_zoom',
        # x_axis_type='datetime',
    )
    p_diff.title.text = 'rr_ints'

    source1 = ColumnDataSource(data={
        "index": hamilton_detector_diff.index,
        "diff": hamilton_detector_diff,
    })
    source2 = ColumnDataSource(data={
        "index": christov_detector_diff.index,
        "diff": christov_detector_diff,
    })
    source3 = ColumnDataSource(data={
        "index": engzee_detector_diff.index,
        "diff": engzee_detector_diff,
    })
    source4 = ColumnDataSource(data={
        "index": pan_tompkins_detector_diff.index,
        "diff": pan_tompkins_detector_diff,
    })
    source5 = ColumnDataSource(data={
        "index": swt_detector_diff.index,
        "diff": swt_detector_diff,
    })
    source6 = ColumnDataSource(data={
        "index": two_average_detector_diff.index,
        "diff": two_average_detector_diff,
    })

    # add a line renderer
    p_diff.line(x="index", y="diff", line_width=2, source=source1, legend_label = 'hamilton', line_color = 'green')
    p_diff.line(x="index", y="diff", line_width=2, source=source2, legend_label = 'christov', line_color = 'red')
    p_diff.line(x="index", y="diff", line_width=2, source=source3, legend_label = 'engzee', line_color = 'yellow')
    p_diff.line(x="index", y="diff", line_width=2, source=source4, legend_label = 'pan_tompkins', line_color = 'black')
    p_diff.line(x="index", y="diff", line_width=2, source=source5, legend_label = 'swt', line_color = 'blue')
    p_diff.line(x="index", y="diff", line_width=2, source=source6, legend_label = 'two_average', line_color = 'orange')

    hover = p_diff.select(dict(type=HoverTool))
    hover.tooltips = [("index", "@index"), ("diff", "@diff")]
    hover.mode = 'vline'

    p = gridplot([[p1], [p2], [p3], [p4], [p5], [p6], [p_diff]])

    show(p)
    output_file('detectors.html')
    save(p)