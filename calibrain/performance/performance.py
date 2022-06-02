"""
  ___      _ _ _             _
 / __|__ _| (_) |__ _ _ __ _(_)_ _
| (__/ _` | | | '_ \ '_/ _` | | ' \
 \___\__,_|_|_|_.__/_| \__,_|_|_||_|

- Coded by Wouter Durnez & Jonas De Bruyne
"""

import numpy as np
import pandas as pd

from calibrain import calibrain

from utils.helper import log

def calculate_performance_CLT(data: pd.DataFrame):


    '''
    Calculates performance features and returns dataframe containing the features
    :param data: pd.DataFrame containing performance data of CLT; CalibrainData.clt.performance
    :return: pd.DataFrame containing features of subject's performance
    '''

    log("Calculating performance measures.")

    # copy so we don't change original data
    data_new = data.copy()

    def delete_firsts(data_new: pd.DataFrame):

        '''
        Delete first trials of each block
        Block 1: 2 trials;
        Block 2: 3 trials;
        Block 3: 4 trials
        '''

        def mask_first_2trials(df):
         result = np.ones_like(df)
         result[0] = 0
         result[1] = 0
         return result

        def mask_first_3trials(df):
         result = np.ones_like(df)
         result[0] = 0
         result[1] = 0
         result[2] = 0
         return result

        def mask_first_4trials(df):
         result = np.ones_like(df)
         result[0] = 0
         result[1] = 0
         result[2] = 0
         result[3] = 0
         return result

        mask = (
         data_new.loc[data_new.condition == 1]
         .groupby('condition')['condition']
         .transform(mask_first_2trials)
         .astype(bool)
        )
        data_new.loc[data_new.condition == 1] = data_new.loc[data_new.condition == 1].loc[mask]

        mask = (
         data_new.loc[data_new.condition == 2]
         .groupby('condition')['condition']
         .transform(mask_first_3trials)
         .astype(bool)
        )
        data_new.loc[data_new.condition == 2] = data_new.loc[data_new.condition == 2].loc[mask]

        mask = (
         data_new.loc[data_new.condition == 3]
         .groupby('condition')['condition']
         .transform(mask_first_4trials)
         .astype(bool)
        )
        data_new.loc[data_new.condition == 3] = data_new.loc[data_new.condition == 3].loc[mask]

    def calculate_accuracy(data_new: pd.DataFrame):

        wrong_count = (
            data_new.loc[data_new.accuracy == -1]
                .groupby('condition', as_index=False)
                .accuracy.count()
        )

        correct_count = (
            data_new.loc[data_new.accuracy == 1]
                .groupby('condition', as_index=False)
                .accuracy.count()
        )

        dropped_count = (
            data_new.loc[data_new.accuracy == 0]
                .groupby('condition', as_index=False)
                .accuracy.count()
        )

        total_count = (
            data_new.groupby('condition', as_index=False)
                .accuracy.count()
        )

        wrong_count.rename(columns={'accuracy': 'wrong_count'}, inplace=True)
        correct_count.rename(columns={'accuracy': 'correct_count'}, inplace=True)
        dropped_count.rename(columns={'accuracy': 'dropped_count'}, inplace=True)
        total_count.rename(columns={'accuracy': 'total_count'}, inplace=True)

        return wrong_count, correct_count, dropped_count, total_count

    def get_performance_matrix(wrong_count, correct_count, dropped_count, total_count):

        performance_matrix = pd.merge(correct_count, wrong_count, how='outer')
        performance_matrix = pd.merge(performance_matrix, dropped_count, how='outer')
        performance_matrix = pd.merge(performance_matrix, total_count, how='outer')

        # Replace nan values with zeros
        performance_matrix.fillna(0, inplace=True)

        # Calculate percentage correct, wrong and dropped
        performance_matrix['correct_prop'] = (
                performance_matrix['correct_count'] / performance_matrix['total_count']
        )
        performance_matrix['wrong_prop'] = (
                performance_matrix['wrong_count'] / performance_matrix['total_count']
        )
        performance_matrix['dropped_prop'] = (
                performance_matrix['dropped_count'] / performance_matrix['total_count']
        )

        # delete condition 0 and NaN
        to_be_deleted = performance_matrix.loc[
            (performance_matrix['condition'] == 0)
            | (performance_matrix['condition'] == np.nan)
            ]
        to_be_deleted_rows_indeces = list(to_be_deleted.index)
        performance_matrix.drop(to_be_deleted_rows_indeces, axis=0, inplace=True)

        return performance_matrix

    delete_firsts(data_new)
    wrong_count, correct_count, dropped_count, total_count = calculate_accuracy(data_new)
    performance_matrix = get_performance_matrix(wrong_count, correct_count, dropped_count, total_count)

    return performance_matrix

def calculate_performance_MRT(data: pd.DataFrame):

    # copy so we don't change original data
    data_new = data.copy()

    def mask_first_trial(df):
        result = np.ones_like(df)
        result[0] = 0
        return result

    def calculate_accuracy(data_new: pd.DataFrame):

        total_count = df_perf_MRT.groupby('condition', as_index=False).accuracy.count()
        wrong_count = (
            data_new.loc[data_new.accuracy == 0]
                .groupby('condition', as_index=False)
                .accuracy.count()
        )
        correct_count = (
            data_new.loc[data_new.accuracy == 1]
                .groupby('condition', as_index=False)
                .accuracy.count()
        )

        wrong_count.rename(columns={'accuracy': 'wrong_count'}, inplace=True)
        correct_count.rename(columns={'accuracy': 'correct_count'}, inplace=True)
        total_count.rename(columns={'accuracy': 'total_count'}, inplace=True)

        return wrong_count, correct_count, total_count

    def clean_rt_data(data_new: pd.DataFrame):
        # delete unusually fast responses (< 500 ms)
        threshold = 0.5
        data_new['rt_clean'] = data_new['reaction_time'].copy()
        data_new.loc[data_new['reaction_time'] < threshold, 'rt_clean'] = None

        # delete outliers (> median + 2.5 x SD)
        Ms = data_new.groupby('condition').rt_clean.median()
        SDs = data_new.groupby('condition').rt_clean.std()
        for condition in list(data_new.condition.unique()):
            data_new.loc[
                (data_new.rt_clean > Ms[condition] + 2.5 * SDs[condition]), 'rt_clean'
            ] = None

    def calculate_rt_measures(data_new):

        median_rt = data_new.groupby('condition', as_index=False)['rt_clean'].median()
        median_rt.rename(columns={"rt_clean": "median_rt"}, inplace=True)

        sd_rt = data_new.groupby('condition', as_index=False)['rt_clean'].std()
        sd_rt.rename(columns={"rt_clean": "sd_rt"}, inplace=True)

        return median_rt, sd_rt

    def get_performance_matrix(wrong_count, correct_count, total_count, median_rt, sd_rt):
        performance_matrix = pd.merge(correct_count, wrong_count, how='outer')
        performance_matrix = pd.merge(performance_matrix, total_count, how='outer')

        # Replace nan values with zeros
        performance_matrix.fillna(0, inplace=True)

        # Calculate percentage correct, wrong and dropped
        performance_matrix['correct_prop'] = (
                performance_matrix['correct_count'] / performance_matrix['total_count']
        )
        performance_matrix['wrong_prop'] = (
                performance_matrix['wrong_count'] / performance_matrix['total_count']
        )

        performance_matrix = pd.merge(performance_matrix, median_rt, how='outer')
        performance_matrix = pd.merge(performance_matrix, sd_rt, how='outer')

        # delete condition 0 and NaN
        to_be_deleted = performance_matrix.loc[
            (performance_matrix['condition'] == 0)
            | (performance_matrix['condition'] == np.nan)
            ]
        to_be_deleted_rows_indeces = list(to_be_deleted.index)
        performance_matrix.drop(to_be_deleted_rows_indeces, axis=0, inplace=True)

        return performance_matrix

    # delete first trial of each block
    mask = (
        data_new.groupby('condition')['condition'].transform(mask_first_trial).astype(bool)
    )
    df_perf_MRT = data_new.loc[mask]

    # get accuracy measures
    wrong_count, correct_count, total_count = calculate_accuracy(data_new)

    # clean rt data
    clean_rt_data(data_new)

    # get rt measures
    median_rt, sd_rt = calculate_rt_measures(data_new)

    # get performance matrix
    performance_matrix = get_performance_matrix(
        wrong_count, correct_count, total_count, median_rt, sd_rt
    )

    return performance_matrix

if __name__ == '__main__':

    path_to_data = '../../data/7_202205091017'
    data = calibrain.CalibrainData(dir=path_to_data)

    performance_data = data.mrt.performance
    test = calculate_performance_MRT(performance_data)