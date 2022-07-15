"""
Pupil size preprocessing functions
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.express as px
import toml
from tqdm import tqdm

import utils.helper as hlp
from utils.helper import log, import_data_frame

pd.options.plotting.backend = 'plotly'


class EyePreprocessor:
    def __init__(self, data: pd.DataFrame = None, **params):

        # Load params
        self.data = None
        self.pupil_col = 'left_pupil_size'
        self.time_col = 'timestamp'
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
            self.pupil_col in self.data
        ), f"Could not find pupil size column '{self.pupil_col}' in data frame."
        assert (
            self.time_col in self.data
        ), f"Could not find time column '{self.time_col}' in data frame."
        assert (
            self.data[self.time_col].dtype == 'float'
        ), f"Need unix epoch timestamps in time column '{self.time_col}'"

    def load_params(self, **params):

        # Set basic attributes
        self.pupil_col = (
            params['pupil_col'] if 'pupil_col' in params else 'left_pupil_size'
        )
        self.time_col = (
            params['time_col'] if 'time_col' in params else 'timestamp'
        )

        params.setdefault('clean_missing_data_params', {})
        params.setdefault('add_velocity_params', {})
        params.setdefault('remove_outliers_params', {})
        params.setdefault('remove_edge_artifacts_params', {})
        self.params = SimpleNamespace(**params)

    def clean_missing_data(self, to_replace: list | float = -1, value=np.nan):
        """
        Set missing values as NaN rather than some integer
        :param to_replace: value or list of values to replace
        :param value: what value to replace the unwanted elements with
        """

        if value == 'NaN':
            value = np.nan

        to_replace = (
            [to_replace] if not isinstance(to_replace, list) else to_replace
        )

        for element in to_replace:
            self.data[self.pupil_col].replace(
                to_replace=element, value=value, inplace=True
            )

    def calculate_gaze_velocity(self, shift: int = 1):
        """
        Calculate a vector (pd.Series) containing velocity
        :param shift: how many samples to look back, or to look forward, in calculating velocity
        :return: array containing momentaneous velocity values
        """

        """Velocity is displacement over delta time: Kret, M. E., & Sjak-Shie, E. E. (2019).
        Preprocessing pupil size data: Guidelines and code. Behavior Research Methods,
        51(3), 1336–1342. https://doi.org/10.3758/s13428-018-1075-y"""

        # Speed between here and the next point
        velocity_to_next = np.abs(
            (
                self.data[self.pupil_col]
                - self.data[self.pupil_col].shift(periods=shift)
            )
            / (
                self.data[self.time_col]
                - self.data[self.time_col].shift(periods=shift)
            )
        )

        # Speed between here and the previous point
        velocity_to_prev = np.abs(
            (
                self.data[self.pupil_col].shift(periods=-shift)
                - self.data[self.pupil_col]
            )
            / (
                self.data[self.time_col].shift(periods=-shift)
                - self.data[self.time_col]
            )
        )

        return pd.concat([velocity_to_prev, velocity_to_next], axis=1).max(
            axis=1
        )

    def remove_outliers_mad(
        self,
        on_col: str = 'velocity',
        n: int = 3,
        show_plot: bool = False,
    ):
        """
        Remove outliers based on mean absolute deviation
        :param on_col: column on which to apply the criterion
        :param n: threshold factor
        :param show_plot: visualize cut-off
        :return: None (inplace)
        """

        # Original number of data points
        n_before = len(self.data)
        n_missing_before = self.data.left_pupil_size.isna().sum()

        # Median absolute deviation
        median = self.data[on_col].median()
        mad = self.data[on_col].mad() #(np.abs(self.data[on_col] - median)).median()

        # Visualize if requested
        if show_plot:
            fig = px.line(
                self.data,
                x=self.time_col,
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
        bool_series = self.data[on_col] > median + n * mad
        self.data.loc[bool_series, self.pupil_col] = np.nan

        # New number of data points
        n_missing_after = self.data.left_pupil_size.isna().sum()
        new_missing = n_missing_after - n_missing_before
        percentage_reduction = round(
            new_missing / (n_before - n_missing_before), 2
        )

        log(
            f'Outliers removed: {n_before - n_missing_before} -> {n_before - n_missing_after}'
            f' data points ({percentage_reduction}% less).'
        )

    def remove_edge_artifacts(
        self,
        min_ms: int = 75,
        buffer_ms: int = 50,
        show_plot: bool = False,
    ) -> list:
        """
        Find gaps in `pupil_col` based on_col `missing_val`, and return those
         that meet a minimum criterion (i.e., must be at least `min_ms` long)
        :param min_ms: threshold for gap detection
        :param buffer_ms: extra time buffer to remove around gaps
        :param show_plot: visualize the results
        :return: data frame with extra column of 'remove' flags, list of tuples (gap start, gap stop, gap duration)
        """

        # Get timestamps and missingness series
        time = np.array(self.data[self.time_col])
        missing = np.array((self.data[self.pupil_col].isna()))

        # Initalize variables
        start = None
        gaps = []

        # Visualize if requested
        if show_plot:
            fig = px.line(
                self.data,
                x=self.time_col,
                y=self.pupil_col,
                title=f'Edge artifact removal (min gap = {min_ms} ms, buffer = {buffer_ms} ms)',
            )
            fig.update_layout(
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                plot_bgcolor='white',
            )

        # Go over timestamps and missings
        iterator = (
            tqdm(zip(time, missing), desc='Processing gaps', total=len(time))
            if hlp.VERBOSITY > 1
            else zip(time, missing)
        )
        for t, m in iterator:

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
            self.data.loc[
                self.data[self.time_col].between(
                    start, stop, inclusive='left'
                ),
                self.pupil_col,
            ] = np.nan

            if show_plot and idx <= 100:
                fig.add_vrect(
                    start, stop, fillcolor='orange', line_width=0, opacity=0.5
                )

        if show_plot:
            fig.show()

        # Save gap info
        self.gaps = gaps

    def pipeline(self, data: pd.DataFrame = None, **params):

        # Load new parameters if provided
        self.load_params(**params)

        # Load data if provided
        if data is not None:
            self.load_data(data=data)

        assert (
            hasattr(self, 'data') and self.data is not None
        ), 'Need to load data first! Either load data with the `load` method, or pass a `data` argument to the pipeline function.'

        # Line up arguments
        clean_missing_data_params = self.params.clean_missing_data_params
        add_velocity_params = self.params.add_velocity_params
        remove_outliers_params = self.params.remove_outliers_params
        remove_edge_artifacts_params = self.params.remove_edge_artifacts_params

        # Get rid of integer values representing missingness
        if clean_missing_data_params:

            # Must be dict or bool
            assert isinstance(
                clean_missing_data_params, (bool, dict)
            ), 'Please pass a boolean or a dictionary with method parameters to the `remove_outliers_params` method!'

            # If bool, make dict
            if clean_missing_data_params is True:
                clean_missing_data_params = {}

            log('Cleaning missing data.')
            self.clean_missing_data(**clean_missing_data_params)

        # Add velocity column
        if 'velocity' not in self.data or add_velocity_params:
            log('Adding velocity column.')

            # Must be dict or bool
            assert isinstance(
                add_velocity_params, (bool, dict)
            ), 'Please pass a boolean or a dictionary with method parameters to the `remove_outliers_params` method!'

            # If bool, make dict
            if add_velocity_params is True:
                add_velocity_params = {}

            self.data['velocity'] = self.calculate_gaze_velocity(
                **add_velocity_params
            )

        # Clean data based on_col outlier criterion
        if remove_outliers_params:
            log('Removing outliers based on MAD.')

            # Must be dict or bool
            assert isinstance(
                remove_outliers_params, (bool, dict)
            ), 'Please pass a boolean or a dictionary with method parameters to the `remove_outliers_params` method!'

            # If bool, make dict
            if remove_outliers_params is True:
                remove_outliers_params = {}

            # Only velocity is implemented
            if 'on_col' in remove_outliers_params.keys():
                assert (
                    remove_outliers_params['on_col'] == 'velocity'
                ), 'Currently, outlier removal only works with velocity. Set `on_col` column to `velocity`.'

            # Set default parameters
            remove_outliers_params.setdefault('on_col', 'velocity')
            remove_outliers_params.setdefault('n', 3)
            remove_outliers_params.setdefault('show_plot', False)

            # Execute method
            self.remove_outliers_mad(**remove_outliers_params)

        # Get rid of edge artifacts
        if remove_edge_artifacts_params:
            log('Removing edge artifacts.')

            # Must be dict or bool
            assert isinstance(
                remove_edge_artifacts_params, (bool, dict)
            ), 'Please pass a boolean or a dictionary with method parameters to the `remove_outliers_params` method!'

            # If bool, make dict
            if remove_edge_artifacts_params is True:
                remove_edge_artifacts_params = {}

            # Set default parameters
            remove_edge_artifacts_params.setdefault('min_ms', 75)
            remove_edge_artifacts_params.setdefault('buffer_ms', 50)
            remove_edge_artifacts_params.setdefault('show_plot', False)

            # Execute method
            self.remove_edge_artifacts(
                **remove_edge_artifacts_params,
            )

        # Return preprocessed data frame
        return self.data


if __name__ == '__main__':
    print('Test area!')

    # Prep data
    data = import_data_frame(path='../data/7_202205091017/MRT/eye.csv')
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

    # Load config
    with open('../configs/test.toml') as config_file:
        config = toml.load(config_file)
    eye_preprocessing_params = config['mrt']['eye']['preprocessing']

    # Try pipeline
    preprocessed_data = EyePreprocessor(
        data=data, **eye_preprocessing_params
    ).pipeline()

    """fig = data.left_pupil_size.plot()
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
    fig.show()"""
