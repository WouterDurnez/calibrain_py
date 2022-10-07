"""
  ___      _ _ _             _
 / __|__ _| (_) |__ _ _ __ _(_)_ _
| (__/ _` | | | '_ \ '_/ _` | | ' \
\___\__,_|_|_|_.__/_| \__,_|_|_||_|

Base classes

- Coded by Wouter Durnez & Jonas De Bruyne
"""
import pandas as pd
from utils.helper import log


class Processor:
    def __init__(self):

        # Initialize core attributes
        self.data = None
        self.params = None

    def load_data(self, data: pd.DataFrame|None):

        assert isinstance(data, pd.DataFrame), '⚠️ Data argument needs to be a `pandas.DataFrame`!'
        if data is None:
            return
        self.data = data

    def load_params(self, **params):

        # If no parameters are provided, create an empty param dict
        params = params if params is not None else {}

        # Check if parameters have been set yet, they will be overwritten
        if hasattr(self, 'params') and self.params is not None:

            log('⚠️ Overwriting previous parameters!')
            self.params.update(params)

        # ...otherwise, set supplied parameters
        else:
            self.params = params

            # Additional parameters should be set here, e.g.:
            # `self.params.setdefault('my_new_parameter', .5)`