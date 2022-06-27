"""
Pupil size preprocessing functions
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.helper import log, import_data_frame

pd.options.plotting.backend = 'plotly'
import plotly.express as px
import plotly.graph_objects as go


def clean_missing_data(
    data: pd.Series, to_replace: list | float = -1, value=np.nan
):
    """
    Set missing values as NaN rather than some integer
    :param data: data Series to clean
    :param to_replace: value or list of values to replace
    :param value: what value to replace the unwanted elements with
    :return: None (all inplace)
    """

    to_replace = (
        [to_replace] if not isinstance(to_replace, list) else to_replace
    )

    for element in to_replace:
        data.replace(to_replace=element, value=value, inplace=True)


def calculate_gaze_velocity(
    time_series: pd.Series, pupil_series: pd.Series, shift=1
) -> pd.Series:
    """
    Calculate a vector (pd.Series) containing velocity
    :param time_series: array of timestamps
    :param pupil_series: array of pupil dilation values
    :param shift: how many samples to look back, or to look forward, in calculating velocity
    :return: array containing momentaneous velocity values
    """

    """Velocity is displacement over delta time: Kret, M. E., & Sjak-Shie, E. E. (2019).
    Preprocessing pupil size data: Guidelines and code. Behavior Research Methods,
    51(3), 1336–1342. https://doi.org/10.3758/s13428-018-1075-y"""

    # Speed between here and the next point
    velocity_to_next = np.abs(
        (pupil_series - pupil_series.shift(periods=shift))
        / (time_series - time_series.shift(periods=shift))
    )

    # Speed between here and the previous point
    velocity_to_prev = np.abs(
        (pupil_series.shift(periods=-shift) - pupil_series)
        / (time_series.shift(periods=-shift) - time_series)
    )

    return pd.concat([velocity_to_prev, velocity_to_next], axis=1).max(axis=1)


def remove_outliers_mad(
    data: pd.DataFrame,
    on: str = 'velocity',
    pupil_col: str = 'left_pupil_size',
    n: int = 3,
    show_plot: bool = False,
):
    """
    Remove outliers based on mean absolute deviation
    :param data: data frame to filter
    :param on: column on which to apply the criterion
    :param n: threshold factor
    :param show_plot: visualize cut-off
    :return: None (inplace)
    """

    # Original number of data points
    n_before = len(data)
    n_missing_before = data.left_pupil_size.isna().sum()

    # Median absolute deviation
    median = data[on].median()
    mad = (np.abs(data[on] - median)).median()

    # Visualize if requested
    if show_plot:

        fig = px.line(
            data,
            x='timestamp',
            y='velocity',
            title=f'MAD outlier removal (n = {n})',
        )
        fig.add_hline(y=median + n * mad, line_color='red')
        fig.update_layout(
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            plot_bgcolor='white',
        )
        fig.update_traces(connectgaps=False)
        fig.show()

    # Filter
    bool_series = data[on] > median + n * mad
    data.loc[bool_series, pupil_col] = np.nan

    # New number of data points
    n_missing_after = data.left_pupil_size.isna().sum()
    new_missing = n_missing_after - n_missing_before
    percentage_reduction = round(
        new_missing / (n_before - n_missing_before), 2
    )

    log(
        f"Removed outliers based on '{on}' column: {n_before - n_missing_before} -> {n_before - n_missing_after} data points ({percentage_reduction}% less)."
    )


def remove_edge_artifacts(
    data: pd.DataFrame,
    time_col: str = 'timestamp',
    pupil_col: str = 'left_pupil_size',
    min_ms: int = 75,
    buffer_ms: int = 50,
    show_plot: bool = False,
) -> list:
    """
    Find gaps in `pupil_col` based on `missing_val`, and return those
     that meet a minimum criterion (i.e., must be at least `min_ms` long)
    :param data: data frame containing our samples
    :param time_col: column containing timestamps (unix epoch!)
    :param pupil_col: column containing pupil dilation values
    :param min_ms: threshold for gap detection
    :param buffer_ms: extra time buffer to remove around gaps
    :return: data frame with extra column of 'remove' flags, list of tuples (gap start, gap stop, gap duration)
    """

    # Get timestamps and missingness series
    time = np.array(data[time_col])
    missing = np.array((data[pupil_col].isna()))

    # Initalize variables
    start = None
    gaps = []

    # Visualize if requested
    if show_plot:
        fig = px.line(
            data,
            x=time_col,
            y=pupil_col,
            title=f'Edge artifact removal (min gap = {min_ms} ms, buffer = {buffer_ms} ms)',
        )
        fig.update_layout(
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            plot_bgcolor='white',
        )

    # Go over timestamps and missings
    for t, m in tqdm(zip(time, missing), total=len(time)):

        # If not missing...
        if m == 0:

            # ... and we have a start flag ...
            if start is not None:
                # ... then create a new gap and reset the start flag
                gaps.append((start, t, t - start))

                start = None

        # ... otherwise, if missing, ...
        else:

            # ... and we did not set a start flag yet ...
            if start is None:
                # ... set one!
                start = t

    # If we set a minimum duration (ms) for gaps, enforce it here
    if min_ms:
        gaps = [(x, y, z) for (x, y, z) in gaps if z >= min_ms]

    if len(gaps) > 100 and show_plot:
        log(
            f'{len(gaps)} gaps identified in edge artifact removal! Only drawing the first 100.',
            color='magenta',
        )

    # How much will we remove, beyond the edges of the gap?
    for idx, (start, stop, _) in enumerate(gaps):

        # Broaden the window
        start -= buffer_ms
        stop += buffer_ms

        # Flag our 'to remove' parts
        data.loc[
            data[time_col].between(start, stop, inclusive='left'), pupil_col
        ] = np.nan

        if show_plot and idx <= 100:
            fig.add_vrect(
                start, stop, fillcolor='orange', line_width=0, opacity=0.5
            )

    if show_plot:
        fig.show()

    return gaps


def pipeline(
    data: pd.DataFrame,
    time_col: str = 'timestamp',
    pupil_col: str = 'left_pupil_size',
    outliers: bool = True,
    edge_artifacts: bool = True,
    show_plots: bool = False,
):
    assert (
        data[time_col].dtype == 'float'
    ), f"Need unix epoch timestamps in time column '{time_col}'"
    assert (
        pupil_col in data.columns
    ), f"Could not find pupil size column '{pupil_col}' in data frame."

    # Get rid of integer values representing missingness
    log('Cleaning missing data.')
    clean_missing_data(data.left_pupil_size)

    # Add velocity column
    log('Adding velocity column.')
    data['velocity'] = calculate_gaze_velocity(
        time_series=data[time_col], pupil_series=data[pupil_col]
    )

    # Clean data based on outlier criterion
    if outliers:
        log('Removing outliers based on MAD.')
        remove_outliers_mad(data=data, show_plot=show_plots)

    # Get rid of edge artifacts
    gaps = None
    if edge_artifacts:
        log('Removing edge artifacts.')
        gaps = remove_edge_artifacts(
            data=data,
            time_col=time_col,
            pupil_col=pupil_col,
            show_plot=show_plots,
        )

    # Set timestamp as index
    #data.set_index('timestamp', inplace=True)

    return gaps


if __name__ == '__main__':

    print('Test area!')

    # Prep data
    data = import_data_frame(path='../data/8_202205091310/MRT/eye.csv')
    data = data.filter(
        items=[
            'timestamp',
            'time',
            'verbose.left.eyeopenness',
            'verbose.left.gazedirectionnormalized.x',
            'verbose.left.gazedirectionnormalized.y',
            'verbose.left.gazedirectionnormalized.z',
            'verbose.left.pupildiametermm',
            'verbose.right.eyeopenness',
            'verbose.right.gazedirectionnormalized.x',
            'verbose.right.gazedirectionnormalized.y',
            'verbose.right.gazedirectionnormalized.z',
            'verbose.right.pupildiametermm',
            'gaze_object',
        ]
    )
    data.columns = (
        'timestamp',
        'time',
        'left_openness',
        'left_gaze_direction_x',
        'left_gaze_direction_y',
        'left_gaze_direction_z',
        'left_pupil_size',
        'right_openness',
        'right_gaze_direction_x',
        'right_gaze_direction_y',
        'right_gaze_direction_z',
        'right_pupil_size',
        'gaze_object',
    )

    # Try pipeline
    gaps = pipeline(
        data=data, outliers=True, edge_artifacts=False, show_plots=True
    )

    fig = data.left_pupil_size.plot()
    data.set_index('timestamp', inplace=True)
    for args in (
        {'method': 'cubic'},
        {'method': 'spline', 'order': 3},
        {'method': 'from_derivatives'},
    ):
        data['left_pupil_size_smoothed'] = data.left_pupil_size.interpolate(
            **args
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.left_pupil_size_smoothed,
                mode='lines',
                name=f'left_pupil_size_smoothed_{args["method"]}',
                opacity=0.3,
            )
        )
    fig.show()
