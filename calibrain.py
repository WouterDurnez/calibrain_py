from pathlib import Path
import pandas as pd
import os

"""
Main class: CalibrainData

Contains: CalibrainTask
    * CalibrainMRT
    * CalibrainCLT
"""

class CalibrainTask():
    def __init__(self,
                 dir: str|Path):
        super().__init__()

        # Set directory
        self.dir = Path(dir) if not isinstance(dir,Path) else dir

    def import_heart(self):
        self.heart = pd.read_csv(self.dir / 'raw-heart.csv')
        # Create column with formatted timestamps
        self.heart['timestamp_datetime'] = pd.to_datetime(self.heart['Timestamp'], unit='ms',
                                                                origin='unix')

    def import_bounds(self):
        self.bounds = pd.read_csv(self.dir / 'events.csv')
        # Create column with formatted timestamps
        self.heart['timestamp_datetime'] = pd.to_datetime(self.heart['Timestamp'], unit='ms',
                                                          origin='unix')

    def import_subjective(self):
        self.subjective = pd.read_csv(self.dir / 'questionnaire.csv')
        self.subjective['nasa_score'] = self.subjective[
            ['PD', 'MD', 'TD', 'PE', 'EF', 'FL']
        ].mean(axis=1)

    # Fix the incorrect timestamps in the heart data
    def fix_timestamps_rr(self):
        # check if there are any NaN values in the dataframe
        nans = sum(self.heart['RRI'].isnull())
        # raise error if nan != 0
        assert nans == 0, f"There are {nans} NaN values in the RRI data, expected zero"
        # fix timestamps based on rr data
        self.heart['cumsum_RRI'] = self.heart['RRI'].cumsum(axis=0)
        self.heart['cumsum_RRI_td'] = pd.to_timedelta(
            self.heart['cumsum_RRI'], 'ms')
        self.heart['timestamp_datetime_new'] = self.heart['timestamp_datetime'][
                                                                      0] + \
                                                                  self.heart['cumsum_RRI_td']
        # Delete columns used for calculation
        self.heart.drop('cumsum_RRI', axis=1, inplace=True)
        self.heart.drop('cumsum_RRI_td', axis=1, inplace=True)

class CalibrainCLT(CalibrainTask):

    def __init__(self, dir: str|Path):

        super().__init__(dir=dir)

    # Import performance data
    def import_performance(self):
        self.performance_clt = pd.read_csv(self.dir / 'performance-clt.csv')


class CalibrainMRT(CalibrainTask):

    def __init__(self, dir: str|Path):

        super().__init__(dir=dir)

    # Import performance data
    def import_performance(self):
        self.performance_mrt = pd.read_csv(self.dir / 'performance-mrt.csv')


class CalibrainData():
    def __init__(
        self,
        dir: str|Path,
    ):
        super().__init__()

        # Check and set directory
        self.dir = Path(dir)
        self._check_valid_dir()

        # Check and set participant number
        self.pp_number = int(self.dir.stem.split('_')[0])

        # Import data
        self.import_data()


    def _check_valid_dir(self):
        """
        Check whether directory contains appropriate folders and files
        """
        files_and_folders = os.listdir(self.dir)
        assert 'CLT' in files_and_folders, "Expected CLT folder in directory!"
        assert 'MRT' in files_and_folders, "Expected MRT folder in directory!"
        assert 'demographics.csv' in files_and_folders, "Expected demographics file in directory!"

    def import_data(self):

        # Demographics
        self.demo = pd.read_csv(filepath_or_buffer=self.dir / 'demographics.csv')

        # CLT
        self.clt = CalibrainCLT(dir=self.dir / 'CLT')
        self.clt.import_heart()
        self.clt.import_bounds()
        self.clt.import_subjective()
        self.clt.import_performance()
        self.clt.fix_timestamps_rr()

        # MRT
        self.mrt = CalibrainMRT(dir=self.dir / 'MRT')
        self.mrt.import_heart()
        self.mrt.import_bounds()
        self.mrt.import_subjective()
        self.mrt.import_performance()




if __name__ == '__main__':

    path_to_data = Path("data/7_202205091017")

    data = CalibrainData(dir=path_to_data)
