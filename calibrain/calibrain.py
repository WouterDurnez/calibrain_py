"""
  ___      _ _ _             _
 / __|__ _| (_) |__ _ _ __ _(_)_ _
| (__/ _` | | | '_ \ '_/ _` | | ' \
 \___\__,_|_|_|_.__/_| \__,_|_|_||_|

- Coded by Wouter Durnez & Jonas De Bruyne
"""

import os
from datetime import datetime as dt
from pathlib import Path
from time import strptime, mktime, time

import numpy as np
import pandas as pd
import toml
from tqdm import tqdm

import utils.helper as hlp
from eye.features import EyeFeatures
from eye.preprocessing import EyePreprocessor
from performance.preproc_and_features import build_performance_data_frame
from utils.helper import log, import_data_frame

tqdm.pandas()

################
# Data classes #
################

DATA_TYPES = ['heart', 'eye', 'subjective', 'event']


class CalibrainTask:
    """
    Boilerplate class for Calibrain measurements tasks
    """

    def __init__(self, dir: str | Path, **task_config):

        # Accepted columns and their new names
        self.EYE_COLUMNS = dict(
            zip(
                [
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
                ],  # Old names
                [
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
                ],  # New names
            )
        )

        # Set data type defaults: process unless specifically turned off
        for type in DATA_TYPES:
            task_config.setdefault(type, True)

        # Which data types will we tackle?
        for type in task_config.keys():
            if not type == False:
                setattr(self, type, True)

        # Store config
        self.config = task_config

        # Set directory
        self.dir = Path(dir) if not isinstance(dir, Path) else dir

        # Get task name and log
        self.task_name = dir.stem.upper()
        as_title = True if hlp.VERBOSITY > 1 else False
        log(
            f'Initializing {self.task_name}.',
            color='blue',
            verbosity=1,
            title=as_title,
        )

        # TODO: Extract below from init, it messes with the pipeline to have these steps hardcoded on initialization (e.g., adding trial labels comes in between)
        # Load data
        self._import_data()

        # Preprocess data
        self._preprocess_data()

        # Calculate features
        self._calculate_features()

        # Done!
        log(
            f'\U0001f3c1 Done with generic data components in {self.task_name}, moving on to custom data...',
            verbosity=1,
        )

    ##################
    # IMPORT METHODS #
    ##################

    def _import_data(self):

        if self.heart:
            log('📋 Importing RR data.')
            self._import_heart()

        if self.events:
            log('📋 Importing event data.')
            self._import_events()

        if self.subjective:
            log('📋 Importing subjective data.')
            self._import_subjective()

        if self.eye:
            log('📋 Importing eye tracking data.')
            self._import_eye()

        if self.eye or self.heart:
            self._add_condition_labels()

    def _import_heart(self):
        self.heart_data = import_data_frame(path=self.dir / 'raw-heart.csv')

    def _import_eye(self):

        self.eye_data = import_data_frame(path=self.dir / 'eye.csv')

        # If gaze object is available, keep it, otherwise proceed without it
        if 'gaze_object' not in self.eye_data:
            self.EYE_COLUMNS.pop('gaze_object')
        self.eye_data = self.eye_data.filter(items=self.EYE_COLUMNS.keys())
        self.eye_data.rename(columns=self.EYE_COLUMNS, inplace=True)

    def _import_events(self):

        # Read and format data
        self.events_data = import_data_frame(path=self.dir / 'events.csv')
        self.events_data.replace(
            to_replace={
                'New subfolder: MRT': 'task_start',
                'New subfolder: CLT': 'task_start',
                'Marker: measuring baseline': 'baseline_start',
                'Marker: finished measuring baseline': 'baseline_end',
                'Condition: 0': 'practice',
                'Condition: 1': 'easy',
                'Condition: Q1': 'easy_q',
                'Condition: 2': 'medium',
                'Condition: Q2': 'medium_q',
                'Condition: 3': 'hard',
                'Condition: Q3': 'hard_q',
            },
            inplace=True,
        )

        # Drop some rows
        allowed = [
            'task_start',
            'baseline_start',
            'baseline_end',
            'practice',
            'easy',
            'easy_q',
            'medium',
            'medium_q',
            'hard',
            'hard_q',
        ]
        self.events_data = self.events_data.loc[
            self.events_data.event.isin(allowed)
        ]

    def _import_subjective(self):

        self.subjective_data = import_data_frame(
            path=self.dir / 'questionnaire.csv'
        )
        self.subjective_data['pe'] = 10 - self.subjective_data['pe']
        self.subjective_data['nasa_score'] = self.subjective_data[
            ['pd', 'md', 'td', 'pe', 'ef', 'fl']
        ].mean(axis=1)

    ######################
    # PROCESSING METHODS #
    ######################

    def _add_condition_labels(self):

        if not (self.eye or self.heart):
            log('⚠️ There is no eye-tracking or heart data to label!')
            return

        # Get timestamps to make bins
        bins = self.events_data.timestamp
        labels = [
            np.nan,
            'baseline',
            np.nan,
            'practice',
            'easy',
            np.nan,
            'medium',
            np.nan,
            'hard',
        ]

        if self.eye:
            log('🏷️ Labeling eye data.')

            # Add labels
            self.eye_data['event'] = pd.cut(
                self.eye_data.timestamp,
                bins=bins,
                right=False,
                labels=labels,
                ordered=False,
            )

        if self.heart:
            log('🏷️ Labeling RR data.')
            # Add labels
            self.heart_data['event'] = pd.cut(
                self.heart_data.timestamp,
                bins=bins,
                right=False,
                labels=labels,
                ordered=False,
            )

            # Lose some weight
            self.heart_data.drop(labels=['time'], axis=1, inplace=True)

    def _preprocess_data(self):

        if self.heart:
            pass  # TODO

        if self.eye:
            log('🚀 Preprocessing eye tracking data.')
            self.config['eye'].setdefault('preprocessing', {})
            self._preprocess_eye()

    def _preprocess_eye(self):

        # Set default parameters
        eye_prep_config = self.config['eye']['preprocessing']
        for step in (
            'add_velocity_params',
            'clean_missing_data_params',
            'remove_edge_artifacts_params',
            'remove_outliers_params',
        ):
            eye_prep_config.setdefault(step, True)

        # Create EyePreprocessor object, load data and parameters, and run through pipeline
        ep = EyePreprocessor()
        self.eye_data = ep.pipeline(
            data=self.eye_data, **self.config['eye']['preprocessing']
        )

    ###############################
    # FEATURE CALCULATION METHODS #
    ###############################

    def _calculate_features(self):

        if self.heart:
            pass  # TODO

        if self.eye:
            log('🚀 Calculating eye tracking features.')
            self.config['eye'].setdefault('features', {})
            self._calculate_eye_features()

    def _calculate_eye_features(self):

        # TODO: Calculate per condition
        # TODO: Merge all features in feature data frame

        # Set default parameters
        eye_feat_config = self.config['eye']['features']

        # Create EyeFeatures object, load data and parameters, and run through pipeline
        ef = EyeFeatures()
        ef.pipeline(
            data=self.eye_data, **eye_feat_config
        )
        self.eye_features = ef.features

    ###################
    # Generic methods #
    ###################

    def __str__(self):
        return (
            f'Calibrain {self.task_name.upper()} object containing:\n'
            f'\t-Eye data:\t\t\t{self.eye}\n'
            f'\t-Heart data:\t\t{self.heart}\n'
            f'\t-Event data:\t\t{self.event}\n'
            f'\t-Subjective data:\t{self.subjective}'
        )

    def __repr__(self):
        return (
            f'Calibrain {self.task_name.upper()} object containing:\n'
            f'\t-Eye data:\t\t\t{self.eye}\n'
            f'\t-Heart data:\t\t{self.heart}\n'
            f'\t-Event data:\t\t{self.event}\n'
            f'\t-Subjective data:\t{self.subjective}'
        )


class CalibrainCLT(CalibrainTask):
    """
    Cognitive Load Task
    """

    def __init__(self, dir: str | Path, **task_config):
        # Initialize and import requested data
        super().__init__(dir=dir, **task_config)
        self._import_performance()
        self._preprocess_performance()

        log(f'\U0001f3c1 Done with {self.task_name}!', verbosity=1)

    # Import performance data
    def _import_performance(self):
        log('📋 Importing performance data.')
        self.performance_data = import_data_frame(
            path=self.dir / 'performance-clt.csv'
        )

    def _preprocess_performance(self):
        self.performance_features = build_performance_data_frame(
            data=self.performance_data, task=self.task_name
        )


class CalibrainMRT(CalibrainTask):
    """
    Mental Rotation Task
    """

    def __init__(self, dir: str | Path, **task_config):
        # Initialize and import requested data
        super().__init__(dir=dir, **task_config)
        self._import_performance()
        self._preprocess_performance()
        #self._add_trial_info_performance()
        #self._get_trial_epochs()
        #self._add_trial_labels()

        log(f'\U0001f3c1 Done with {self.task_name}!', verbosity=1)

    # Import performance data
    def _import_performance(self):
        log('📋 Importing performance data.')
        self.performance_data = import_data_frame(
            path=self.dir / 'performance-mrt.csv'
        )

    def _preprocess_performance(self):
        self.performance_features = build_performance_data_frame(
            data=self.performance_data, task=self.task_name
        )

    def _add_trial_info_performance(self):
        """
        For the MRT, there are analysis on trial-level. Therefore, we have to label the data first.
        We do this using the performance data.
        """
        self.performance_data['trial'] = (
            self.performance_data.groupby(['condition']).cumcount() + 1
        )

    def _get_trial_epochs(self):
        self.trial_bounds = self.performance_data.filter(
            ['condition', 'trial', 'timestamp', 'reaction_time']
        )
        self.trial_bounds.rename(
            columns={'timestamp': 'timestamp_end', 'trial': 'trial_id'},
            inplace=True,
        )
        # Convert reaction time from seconds to milliseconds
        self.trial_bounds['reaction_time'] *= 1000

        self.trial_bounds['timestamp_start'] = (
            self.trial_bounds['timestamp_end']
            - self.trial_bounds['reaction_time']
        )
        # Reorder before converting from wide to long and drop reaction_time
        self.trial_bounds = self.trial_bounds.filter(
            ['condition', 'trial_id', 'timestamp_start', 'timestamp_end']
        )
        # Convert from wide to long to create bins later on
        self.trial_bounds = pd.wide_to_long(
            self.trial_bounds,
            stubnames='timestamp',
            sep='_',
            i=['condition', 'trial_id'],
            j='event_type',
            suffix=r'\w+',
        )
        self.trial_bounds.reset_index(inplace=True)

    def _add_trial_labels(self):
        """
        Note: only to be executed AFTER _add_trial_info_performance and _get_trial_epochs
        """

        # Get timestamps to make bins
        bins = self.trial_bounds.timestamp

        # Create list with labels
        # Get every second label (each label is in list column twice now)
        labels = list(self.trial_bounds.trial_id)[::2]

        # Insert value between each element; this labels the time between trials
        def insert_between_elements(lst, item):
            result = [item] * (len(lst) * 2 - 1)
            result[0::2] = lst
            return result

        labels = insert_between_elements(labels, np.nan)

        if self.eye:
            log('🏷️ Labeling eye data (trials).')
            # Add labels
            self.eye_data['trial'] = pd.cut(
                self.eye_data.timestamp,
                bins=bins,
                right=False,
                labels=labels,
                ordered=False,
            )


class CalibrainData:
    def __init__(
        self,
        dir: str | Path,
        **task_params,
    ):

        super().__init__()

        # Check and set directory
        self.dir = Path(dir) if not isinstance(dir, Path) else dir
        self._check_valid_dir()
        self.id = '_'.join(self.dir.stem.split('_')[:-1])
        self.time_created = mktime(
            strptime(self.dir.stem.split('_')[-1], '%Y%m%d%H%M')
        )
        self.time_processed = time()
        log(
            f'🧠 Processing Calibrain data: user {self.id}, recorded on {dt.fromtimestamp(self.time_created)}.',
            verbosity=1,
            color='blue'
        )

        # Define valid tasks
        self.__valid_tasks = ('clt', 'mrt')

        # Set some default behavior
        task_params = {} if task_params is None else task_params
        task_params.setdefault('mrt', True)
        task_params.setdefault('clt', True)

        # Import data
        self._import_data(**task_params)

    def _check_valid_dir(self):
        """
        Check whether directory contains appropriate folders and files
        """
        files_and_folders = os.listdir(self.dir)
        assert 'CLT' in files_and_folders, f'⚠️ Expected CLT folder in directory <{self.dir}>!'
        assert 'MRT' in files_and_folders, f'⚠️ Expected MRT folder in directory <{self.dir}>!'
        assert (
            'demographics.csv' in files_and_folders
        ), '⚠️ Expected demographics file in directory!'

    def _import_data(self, **task_params):

        # Demographics
        self.demo = (
            import_data_frame(path=self.dir / 'demographics.csv')
            .iloc[0, :]
            .rename('demographics')
        )

        # Import data for specified measures
        tasks_to_process = (
            task for task in task_params.keys() if task_params[task]
        )
        for task in tasks_to_process:

            # Check if task is valid
            assert (
                task in self.__valid_tasks
            ), f'"{task} is not a valid task! Try {self.__valid_tasks}.'

            # Unpack measure parameters and set defaults
            measure_params = (
                task_params[task]
                if isinstance(task_params[task], dict)
                else {}
            )
            measure_params.setdefault('heart', {})
            measure_params.setdefault('eye', {})
            measure_params.setdefault('events', {})
            measure_params.setdefault('subjective', {})

            # Create appropriate task
            match task:

                # Cognitive load task
                case 'clt':
                    self.clt = CalibrainCLT(
                        dir=self.dir / 'CLT', **measure_params
                    )

                case 'mrt':
                    self.mrt = CalibrainMRT(
                        dir=self.dir / 'MRT', **measure_params
                    )

    def __repr__(self):
        return f'Calibrain data object <id {self.demo.id}; timestamp {self.demo.timestamp}>'

    def __str__(self):
        return f'Calibrain data object <id {self.demo.id}; timestamp {self.demo.timestamp}>'


if __name__ == '__main__':
    # Let's go
    hlp.hi(verbosity=3)

    # Load config
    with open('../configs/test.toml') as config_file:
        config = toml.load(config_file)

    # Temp
    # config.pop('mrt')

    dir = Path('../data/7_202205091017')
    #data_folders = [f for f in dir.iterdir() if f.is_dir()]
    #data = []

    #for df in data_folders:
    #    try:
    #        data.append(CalibrainData(dir=df, **config))
    #    except Exception as e:
    #        log(f'Failed for <{df}>...', color='red',verbosity=1)
    #        log(e)

    data = CalibrainData(dir=dir, **config)