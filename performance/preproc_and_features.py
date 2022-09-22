"""
  ___      _ _ _             _
 / __|__ _| (_) |__ _ _ __ _(_)_ _
| (__/ _` | | | '_ \ '_/ _` | | ' \
 \___\__,_|_|_|_.__/_| \__,_|_|_||_|

- Coded by Wouter Durnez & Jonas De Bruyne
"""

import numpy as np
import pandas as pd

from utils.helper import log, import_data_frame


def drop_first_trials(data_chunk: pd.DataFrame, n: int = None) -> pd.DataFrame:

    # Number of rows to drop
    n_rows = (
        n if n else int(data_chunk.iloc[0].condition + 1)
    )   # N (in N-back) + 1

    # Set accuracy to 'drop' (0)
    data_chunk.loc[data_chunk.index[:n_rows], 'accuracy'] = np.nan
    return data_chunk


def filter_by_mad(
    data: pd.DataFrame, on_col: str = 'reaction_time', n: int = 3
):

    # Original number of data points
    n_before = len(data)
    n_missing_before = data[on_col].isna().sum()

    # Median absolute deviation
    median = data[on_col].median()
    mad = data[on_col].mad()  # (np.abs(self.data[on_col] - median)).median()

    # Filter
    bool_series = data[on_col] > median + n * mad
    data.loc[bool_series, on_col] = np.nan

    # New number of data points
    n_missing_after = data[on_col].isna().sum()
    new_missing = n_missing_after - n_missing_before
    percentage_reduction = round(
        new_missing / (n_before - n_missing_before), 2
    )

    log(
        f'ℹ️ Outliers removed: {n_before - n_missing_before} -> {n_before - n_missing_after}'
        f' data points ({percentage_reduction}% less).',
        color='green',
    )


def build_performance_data_frame(
    data: pd.DataFrame, task: str
) -> pd.DataFrame:
    """
    Build performance matrix for a task
    :param data: performance data frame from Calibrain task
    :return: performance matrix (data frame)
    """

    task = task.lower()
    assert task in ('clt', 'mrt'), 'Only CLT and MRT are supported!'

    log(f'⚙️ Calculating performance stats for {task.upper()}.')

    # Drop first trials
    data = data.groupby('condition').apply(
        lambda df: drop_first_trials(df, n=(None if task == 'clt' else 1))
    )

    # Get some counts
    matrix = data.groupby('condition').accuracy.value_counts()
    matrix.rename('count', inplace=True)
    matrix = matrix.reset_index()

    # Label trial accuracy
    match task:
        case 'clt':
            to_replace = {-1: 'incorrect', 1: 'correct', 0: 'dropped'}
        case 'mrt':
            to_replace = {1: 'correct', 0: 'incorrect'}

    matrix.accuracy.replace(to_replace, inplace=True)
    matrix = matrix.pivot(index='condition', columns='accuracy')
    matrix.columns = matrix.columns.get_level_values(1)
    matrix.columns.rename('value')

    # Add missing columns (if any) and calculate proportions
    matrix['total'] = data.groupby('condition').accuracy.count()
    iterate_over = ['correct', 'incorrect']
    if task == 'clt':
        iterate_over.append('dropped')
    for col in iterate_over:
        if col not in matrix:
            matrix[col] = 0
        matrix[f'{col}_proportion'] = matrix[col] / matrix.total

    # MRT also contains reaction time
    if task == 'mrt':

        # First filter reaction time for outliers
        filter_by_mad(data=data, on_col='reaction_time')

        # Helper percentile calculator
        def percentile(n):
            def percentile_(x):
                return np.nanpercentile(x, n)

            percentile_.__name__ = f'percentile_{n}'
            return percentile_

        # Calculate RT statistics
        rt_stats = data.groupby('condition').reaction_time.agg(
            [
                'mean',
                'std',
                'median',
                'min',
                'max',
                percentile(5),
                percentile(50),
                percentile(95),
            ]
        )
        rt_stats.columns = [f'rt_{col}' for col in rt_stats.columns]

        # Append new columns
        matrix = matrix.join(rt_stats)

    # Fill NaN
    matrix.fillna(0, inplace=True)

    # Cleanup
    matrix = matrix.reset_index()
    matrix.drop(columns=['accuracy'], inplace=True, errors='ignore')

    # Rename condition levels
    matrix.condition.replace(to_replace={
        0: 'practice',
        1: 'easy',
        2: 'medium',
        3: 'hard',
    }, inplace=True)
    matrix.set_index('condition', inplace=True)
    matrix.columns.name = None

    return matrix


if __name__ == '__main__':

    clt_data = import_data_frame(
        path='../data/7_202205091017/CLT/performance-clt.csv'
    )
    mrt_data = import_data_frame(
        path='../data/7_202205091017/MRT/performance-mrt.csv'
    )

    clt_performance = build_performance_data_frame(data=clt_data, task='clt')
    mrt_performance = build_performance_data_frame(data=mrt_data, task='mrt')
