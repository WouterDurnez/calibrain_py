"""
  ___      _ _ _             _
 / __|__ _| (_) |__ _ _ __ _(_)_ _
| (__/ _` | | | '_ \ '_/ _` | | ' \
 \___\__,_|_|_|_.__/_| \__,_|_|_||_|

- Coded by Wouter Durnez & Jonas De Bruyne
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.helper import log, hi, import_dataframe

tqdm.pandas()


################
# Data classes #
################


class CalibrainTask:
    """
    Boilerplate class for Calibrain measurements tasks
    """

    def __init__(self, dir: str | Path, **import_args):

        # Set directory
        self.dir = Path(dir) if not isinstance(dir, Path) else dir

        # Load data
        self._import_data(**import_args)

    def _import_data(
        self,
        heart: bool = True,
        bounds: bool = True,
        subjective: bool = True,
        eye: bool = True,
        time_fixer: bool = True,
    ):

        if heart:
            log('Importing RR data.')
            self._import_heart()

        if bounds:
            log('Importing event data.')
            self._import_events()
        if subjective:
            log('Importing subjective data.')
            self._import_subjective()
        if eye:
            log('Importing eye tracking data.')
            self._import_eye()

        if eye or heart:
            self._add_condition_labels(eye=eye, heart=heart)

    def _import_heart(self):
        self.heart = import_dataframe(path=self.dir / 'raw-heart.csv')

    def _import_eye(self):
        self.eye = import_dataframe(path=self.dir / 'eye.csv')

    def _import_events(self):

        # Read and format data
        self.events = import_dataframe(path=self.dir / 'events.csv')
        self.events.replace(
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
        self.events = self.events.loc[self.events.event.isin(allowed)]

    def _import_subjective(self):
        self.subjective = import_dataframe(path=self.dir / 'questionnaire.csv')
        self.subjective['nasa_score'] = self.subjective[
            ['pd', 'md', 'td', 'pe', 'ef', 'fl']
        ].mean(axis=1)

    def _add_condition_labels(self, eye: bool = False, heart: bool = False):

        if not (eye or bool):
            pass

        # Get timestamps to make bins
        bins = self.events.timestamp
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

        if eye:
            log('Labeling eye data.')
            # Add labels
            self.eye['event'] = pd.cut(
                self.eye.timestamp,
                bins=bins,
                right=False,
                labels=labels,
                ordered=False,
            )

        if heart:
            log('Labeling RR data.')
            # Add labels
            self.heart['event'] = pd.cut(
                self.heart.timestamp,
                bins=bins,
                right=False,
                labels=labels,
                ordered=False,
            )
            # Lose some weight
            self.heart.drop(labels=['timestamp', 'time'], axis=1, inplace=True)


class CalibrainCLT(CalibrainTask):
    """
    Cognitive Load Task
    """

    def __init__(self, dir: str | Path, **import_args):
        # Initialize and import requested data
        log('Initializing CLT.', color='red')
        super().__init__(dir=dir)
        self._import_performance()

    # Import performance data
    def _import_performance(self):
        log('Importing performance data.')
        self.performance = import_dataframe(
            path=self.dir / 'performance-clt.csv'
        )


class CalibrainMRT(CalibrainTask):
    """
    Mental Rotation Task
    """

    def __init__(self, dir: str | Path, **import_args):
        # Initialize and import requested data
        log('Initializing MRT.', color='red')
        super().__init__(dir=dir, **import_args)
        self._import_performance()
        self._add_trial_info_performance()
        self._get_trial_epochs()
        self._add_trial_labels(eye=True)

    # Import performance data
    def _import_performance(self):
        log('Importing performance data.')
        self.performance = import_dataframe(
            path=self.dir / 'performance-mrt.csv'
        )

    def _add_trial_info_performance(self):
        '''
        For the MRT, there are analysis on trial-level. Therefore, we have to label the data first.
        We do this using the performance data.
        '''
        self.performance['trial'] = (
            self.performance.groupby(['condition']).cumcount() + 1
        )

    def _get_trial_epochs(self):
        self.trial_bounds = self.performance.filter(['condition', 'trial', 'timestamp', 'reaction_time'])
        self.trial_bounds.rename(columns={'timestamp': 'timestamp_end', 'trial': 'trial_id'}, inplace=True)
        # reaction time s --> ms
        self.trial_bounds['reaction_time'] = self.trial_bounds['reaction_time']*1000
        self.trial_bounds['timestamp_start'] = self.trial_bounds['timestamp_end'] - self.trial_bounds['reaction_time']
        # reorder before converting from wide to long and drop reaction_time
        self.trial_bounds = self.trial_bounds.filter(['condition', 'trial_id', 'timestamp_start', 'timestamp_end'])
        # convert from wide to long to create bins later on
        self.trial_bounds = pd.wide_to_long(
            self.trial_bounds,
            stubnames = 'timestamp',
            sep='_',
            i = ['condition', 'trial_id'],
            j = 'event_type',
            suffix=r'\w+'
        )
        self.trial_bounds.reset_index(
            inplace=True
        )

    def _add_trial_labels(self, eye: bool = False):
        '''
        Note: only to be executed AFTER _add_trial_info_performance and _get_trial_epochs
        '''

        # Get timestamps to make bins
        bins = self.trial_bounds.timestamp
        # create list with labels
        ## get every second label (each label is in list column twice now)
        labels = list(self.trial_bounds.trial_id)[::2]
        # insert value between each element; this labels the time between trials
        def insert_between_elements(lst, item):
            result = [item] * (len(lst) * 2 - 1)
            result[0::2] = lst
            return result
        labels = insert_between_elements(labels, np.nan)

        if eye:
            log('Labeling eye data (trials).')
            # Add labels
            self.eye['trial'] = pd.cut(
                self.eye.timestamp,
                bins=bins,
                right=False,
                labels=labels,
                ordered=False,
            )


class CalibrainData:
    def __init__(
        self,
        dir: str | Path,
    ):
        super().__init__()

        # Check and set directory
        self.dir = Path(dir)
        self._check_valid_dir()

        # Import data
        self._import_data()
        self.pp = self.demo.id

    def _check_valid_dir(self):
        """
        Check whether directory contains appropriate folders and files
        """
        files_and_folders = os.listdir(self.dir)
        assert 'CLT' in files_and_folders, 'Expected CLT folder in directory!'
        assert 'MRT' in files_and_folders, 'Expected MRT folder in directory!'
        assert (
            'demographics.csv' in files_and_folders
        ), 'Expected demographics file in directory!'

    def _import_data(self):

        # Demographics
        self.demo = (
            import_dataframe(path=self.dir / 'demographics.csv')
            .iloc[0, :]
            .rename('demographics')
        )

        # CLT
        self.clt = CalibrainCLT(
            dir=self.dir / 'CLT',
            heart=True,
            eye=True,
            bounds=True,
            subjective=True,
        )

        # MRT
        self.mrt = CalibrainMRT(
            dir=self.dir / 'MRT',
            heart=True,
            eye=True,
            bounds=True,
            subjective=True,
        )

    def __repr__(self):
        return f'Calibrain data object <id {self.demo.id}; timestamp {self.demo.timestamp}>'

    def __str__(self):
        return f'Calibrain data object <id {self.demo.id}; timestamp {self.demo.timestamp}>'


if __name__ == '__main__':
    hi('Test!')

    path_to_data = '../data/7_202205091017'
    data = CalibrainData(dir=path_to_data)