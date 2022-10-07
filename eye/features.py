"""
Eye tracking feature calculation functions
"""

import numpy as np
import pandas as pd
import plotly.express as px

from eye.preprocessing import EyePreprocessor
from utils.base import Processor
from utils.helper import load_config, import_data_frame, log
from joblib import Memory
from sklearn.preprocessing import normalize

mem = Memory(location='../cache/eye', verbose=0)


class EyeFeatures(Processor):
    def __init__(self, data: pd.DataFrame = None, **params):

        # Initialize attributes
        self.gaze_object_col = None
        self.aoi_mapping = None

        # Initialize features
        self.transition_matrix = None
        self.entropy = None
        self.absolute_gaze_switches = None
        self.relative_gaze_switches = None

        super().__init__()

        # Load params and data if given
        if params is not None:
            self.load_params(**params)
        if data is not None:
            self.load_data(data=data)

    def load_params(self, **params):
        """
        Load feature calculation parameter set
        """
        super().load_params(**params)

        # Set basic attributes
        self.gaze_object_col = (
            params['gaze_object_col']
            if 'gaze_object_col' in params
            else 'gaze_object'
        )
        self.aoi_mapping = (
            params['aoi_mapping'] if 'aoi_mapping' in params else None
        )

        # If no step arguments are given: run everything with default parameters
        self.params.setdefault('entropy', True)
        self.params.setdefault('absolute_gaze_switches', True)
        self.params.setdefault('relative_gaze_switches', True)
        self.params.setdefault('absolute_gaze_switches_to', False)
        self.params.setdefault('relative_gaze_switches_to', False)

    def load_data(self, data:pd.DataFrame):
        """
        Load eye-tracking data for feature calculation
        """
        super().load_data(data=data)

    def map_aois(self):

        log(
            '🏷️ Mapping areas of interest.',
            color='green',
        )

        assert hasattr(self, 'data'), '⚠️ Feed me data first!'

        if not self.aoi_mapping:
            log('⚠️ No AOI mapping provided. Moving on...', color='red')

        data = self.data.copy()
        data['AOI'] = data[self.gaze_object_col]
        data.AOI.replace(to_replace=self.aoi_mapping, inplace=True)

        self.data = data

    ############
    # FEATURES #
    ############

    def create_transition_matrix(self):

        # Mapping step (fill in here when required)
        if self.aoi_mapping:
            self.map_aois()

        # Convert to data frame, selecting AOI column if present (otherwise, use gaze object column)
        df = pd.DataFrame(
            self.data[self.gaze_object_col]
            if 'AOI' not in self.data
            else self.data.AOI
        )
        df.columns = ['AOI']

        log(
            '⚙️ Creating transition matrix.',
            color='green',
        )

        # Create data frame with shifted column, use it to calculate transitions
        df['AOI_shift'] = df['AOI'].shift(-1)
        df['count'] = 1
        transition_matrix = (
            df.groupby(['AOI', 'AOI_shift']).count().unstack().fillna(0)
        )
        transition_matrix.columns = transition_matrix.columns.get_level_values(
            1
        )

        # Get column and index labels, and look for missing ones
        col_labels = list(transition_matrix.columns)
        row_labels = list(transition_matrix.index)
        all_labels = set(col_labels + row_labels)
        missing_cols = all_labels.difference(col_labels)
        missing_rows = all_labels.difference(row_labels)

        # If any *are* missing, we need to insert them
        if not len(missing_cols) == 0:
            for col in missing_cols:
                transition_matrix[col] = 0
        if not len(missing_rows) == 0:
            for row in missing_rows:
                transition_matrix.loc[row] = 0

        # Sort columns and rows
        transition_matrix.columns = [
            str(lvl) for lvl in transition_matrix.columns
        ]
        transition_matrix.index = [str(lvl) for lvl in transition_matrix.index]
        transition_matrix = transition_matrix.reindex(
            sorted(transition_matrix.columns), axis=1
        )
        transition_matrix = transition_matrix.reindex(
            sorted(transition_matrix.index), axis=0
        )

        assert (
            transition_matrix.shape[0] == transition_matrix.shape[1]
        ), 'Something went wrong! Transition matrix is not square.'

        # Done!
        self.transition_matrix = transition_matrix

    def get_entropy(self, quiet: bool = True) -> float:
        """
        Calculate gaze entropy from time series
        :param quiet: set to True to silence logging
        :return: entropy value
        """

        log('⚙️ Calculating gaze entropy.', color='green')

        # Get transition matrix
        if not hasattr(self, 'transition_matrix'):
            self.create_transition_matrix()
        dim = self.transition_matrix.shape[0]

        # Normalize dummy
        M_norm = (
            self.transition_matrix.values
            / np.array(np.sum(self.transition_matrix, axis=1))[:, np.newaxis]
        )

        # Fill nan values by zeros (python returned nan if divided by zero because there were
        # no gaze collisions in that AOI)
        M_norm[np.isnan(M_norm)] = 0

        # Get stationary distribution by solving equations
        A = np.append(
            np.transpose(M_norm) - np.identity(dim), [[1] * dim], axis=0
        )
        b = np.transpose(np.array([0] * dim + [1]))

        # Try solving, else return np.nan
        try:
            stationary = np.linalg.solve(
                np.transpose(A).dot(A), np.transpose(A).dot(b)
            )
        except np.linalg.LinAlgError as e:
            if not quiet:
                log(
                    f'⚠️ Failed to calculate gaze entropy, likely due to singular matrix. -- {e}'
                )
            return np.nan

        # Calculate entropy
        entropy = 0
        for i in range(M_norm.shape[0]):
            for j in range(M_norm.shape[1]):
                val = (
                    0
                    if M_norm[i, j] == 0
                    else M_norm[i, j] * np.log(M_norm[i, j])
                )
                entropy -= stationary[i] * val

        self.entropy = entropy

    def get_gaze_switches(self, mode: str = 'absolute', to: str | list = None):

        # Check arguments
        assert mode in (
            'absolute',
            'relative',
        ), '⚠️ When calculating gaze switches, only mode must be "absolute" or "relative".'

        # If there is a list of objects, we'll recursively pass them on (and get out)
        if isinstance(to, list):
            for object in to:
                self.get_gaze_switches(mode=mode, to=object)
            return

        log(
            f'⚙️ Calculating {mode} gaze switches{f" to <{to}>" if to else ""}.',
            color='green',
        )

        # Get transition matrix
        if not hasattr(self, 'transition_matrix'):
            self.create_transition_matrix()

        # Get absolute number of gaze switches
        if not to:
            self.absolute_gaze_switches = (
                self.transition_matrix.values.sum()
                - np.trace(self.transition_matrix)
            )

            # If we wanted relative, then calculate proportion
            # (we need absolute switches in either case)
            if mode == 'relative':
                self.relative_gaze_switches = (
                    self.absolute_gaze_switches
                    / self.transition_matrix.values.sum()
                )

        # If an object of interest is supplied, calculate switches to that object specifically
        # (Make sure it's not a list, but this shouldn't happen anymore.)
        elif not isinstance(to, list):
            data = self.data.copy()
            data = data[[self.gaze_object_col]]

            # Calculate instruction screen fixations
            data['check'] = None
            data['gaze_object_prev'] = data.gaze_object.shift(1)
            data.loc[
                (data['gaze_object'] == to) & (data['gaze_object_prev'] != to),
                'check',
            ] = True

            setattr(
                self, f'absolute_gaze_switches_to_{to}', data.check.count()
            )

            # If we wanted relative, then calculate proportion
            # (we need absolute switches in either case)
            # TODO: store this in a dict (self.features[feature_name])?
            if mode == 'relative':
                setattr(
                    self,
                    f'relative_gaze_switches_to_{to}',
                    getattr(self, f'absolute_gaze_switches_to_{to}')
                    / self.transition_matrix.values.sum(),
                )

    def pipeline(self, data: pd.DataFrame = None, **params):

        # Load new parameters if provided
        if params is not None:
            self.load_params(**params)

        # Load data if provided
        if data is not None:
            self.load_data(data=data)

        # Calculate requested features
        self.create_transition_matrix()
        if self.params['entropy']:
            self.get_entropy()
        if self.params['absolute_gaze_switches']:
            self.get_gaze_switches(mode='absolute')
        if self.params['relative_gaze_switches']:
            self.get_gaze_switches(mode='relative')
        if self.params['absolute_gaze_switches_to']:
            self.get_gaze_switches(
                mode='absolute', to=self.params['absolute_gaze_switches_to']
            )
        if self.params['relative_gaze_switches_to']:
            self.get_gaze_switches(
                mode='relative', to=self.params['relative_gaze_switches_to']
            )

        # Combine features in dict # TODO: Make this a data frame
        self.features = {
            feature: value
            for feature, value in self.__dict__.items()
            if feature
            in [
                'entropy',
            ]
            or feature.__contains__('gaze_switches')
        }

    #########################
    # Visualization methods #
    #########################

    def visualize_transition_matrix(self, normalized: bool = False):

        if not hasattr(self, 'transition_matrix'):
            self.create_transition_matrix()

        if normalized:
            levels = self.transition_matrix.columns
            tm = pd.DataFrame(
                normalize(self.transition_matrix), columns=levels, index=levels
            )
        else:
            tm = self.transition_matrix.copy()

        fig = px.imshow(
            tm,
            color_continuous_scale='Blues',
            title=f'<b>Transition matrix</b>{" (normalized)" if normalized else ""}',
        )
        fig.update_layout(title_x=0.5)
        fig.show()


if __name__ == '__main__':
    # Load config
    config = load_config(path='../configs/test.toml')

    # Load and select data
    data = import_data_frame(path='../data/klaas_202209130909/MRT/eye.csv')
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

    # Preprocess data
    prepped_data = EyePreprocessor(
        data=data, **config['mrt']['eye']['preprocessing']
    ).pipeline()

    # Feature calculation
    feature_object = EyeFeatures(data=prepped_data)
    feature_object.pipeline(**config['mrt']['eye']['features'])
    feature_object.visualize_transition_matrix(normalized=True)
