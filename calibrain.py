"""
  ___      _ _ _             _
 / __|__ _| (_) |__ _ _ __ _(_)_ _
| (__/ _` | | | '_ \ '_/ _` | | ' \
 \___\__,_|_|_|_.__/_| \__,_|_|_||_|

- Coded by Wouter Durnez & Jonas De Bruyne
"""


from pathlib import Path
import pandas as pd
import os
from utils.helper import log, clean_col_name, hi, import_dataframe


################
# Data classes #
################


class CalibrainTask:
    """
    Boilerplate class for Calibrain measurements tasks
    """

    def __init__(self, dir: str | Path, **import_args):

        super().__init__()

        # Set directory
        self.dir = Path(dir) if not isinstance(dir, Path) else dir

        # Load data
        self.import_data(**import_args)

    def import_data(
        self,
        heart: bool = True,
        bounds: bool = True,
        subjective: bool = True,
        eye: bool = True,
    ):

        if heart:
            log('Importing RR data.')
            self.import_heart()
        if bounds:
            log('Importing event data.')
            self.import_bounds()
        if subjective:
            log('Importing subjective data.')
            self.import_subjective()
        if eye:
            log('Importing eye tracking data.')
            self.import_eye()

    def import_heart(self):
        self.heart = import_dataframe(path=self.dir / 'raw-heart.csv')

    def import_eye(self):
        self.eye = import_dataframe(path=self.dir / 'eye.csv')

    def import_bounds(self):
        self.bounds = import_dataframe(path=self.dir / 'events.csv')

    def import_subjective(self):
        self.subjective = import_dataframe(path=self.dir / 'questionnaire.csv')
        self.subjective['nasa_score'] = self.subjective[
            ['pd', 'md', 'td', 'pe', 'ef', 'fl']
        ].mean(axis=1)


class CalibrainCLT(CalibrainTask):
    """
    Cognitive Load Task
    """

    def __init__(self, dir: str | Path, **import_args):
        log('Initializing CLT.', color='red')
        super().__init__(dir=dir)
        self.import_performance()

    # Import performance data
    def import_performance(self):
        log('Importing performance data.')
        self.performance = import_dataframe(
            path=self.dir / 'performance-clt.csv'
        )


class CalibrainMRT(CalibrainTask):
    """
    Mental Rotation Task
    """

    def __init__(self, dir: str | Path, **import_args):
        log('Initializing MRT.', color='red')
        super().__init__(dir=dir, **import_args)
        self.import_performance()

    # Import performance data
    def import_performance(self):
        log('Importing performance data.')
        self.performance = import_dataframe(
            path=self.dir / 'performance-mrt.csv'
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
        self.import_data()
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

    def import_data(self):

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
            events=True,
            subjective=True,
        )

        # MRT
        self.mrt = CalibrainMRT(
            dir=self.dir / 'MRT',
            heart=True,
            eye=True,
            events=True,
            subjective=True,
        )

    def __repr__(self):

        return f'Calibrain data object <id {self.demo.id}; timestamp {self.demo.timestamp}>'

    def __str__(self):

        return f'Calibrain data object <id {self.demo.id}; timestamp {self.demo.timestamp}>'


if __name__ == '__main__':

    hi('Test!')

    path_to_data = 'data/7_202205091017'
    data = CalibrainData(dir=path_to_data)

    events = data.clt.bounds