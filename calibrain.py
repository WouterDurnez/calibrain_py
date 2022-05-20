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


class CalibrainCLT(CalibrainTask):

    def __init__(self, dir: str|Path):

        super().__init__(dir=dir)

class CalibrainData():
    def __init__(
        self,
        dir: str|Path,
    ):
        super().__init__()

        # Check and set directory
        self.dir = Path(dir)
        self._check_valid_dir()

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




if __name__ == '__main__':

    path_to_data = Path("data/7_202205091017")

    data = CalibrainData(dir=path_to_data)
