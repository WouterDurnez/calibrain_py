#
# Analysis of performance on MRT
#

from pathlib import Path

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import toml

import utils.helper as hlp
from calibrain.calibrain import CalibrainData

if __name__ == '__main__':

    hlp.hi(verbosity=3)

    # Load config
    with open('../configs/test.toml') as config_file:
        config = toml.load(config_file)

    # import data (manual for now)
    dirs = [
        Path('../data/Dennis_202210030001'),
        Path('../data/Carl_202210030000'),
        Path('../data/Stephanie_202210041526'),
        Path('../data/Arian_202210051014'),
    ]

    # MRT
    mrt_perf = {}
    for dir in dirs:
        data = CalibrainData(dir=dir, **config)
        mrt_perf[data.id] = data.mrt.performance_features

    data_mrt = pd.concat(mrt_perf).reset_index()
    data_mrt.rename(columns={"level_0": "id"}, inplace=True)

    # delete practice data
    data_mrt = data_mrt.loc[data_mrt.condition != 'practice']

    # plot performance
    sns.boxplot(
        data=data_mrt,
        x='condition',
        y='correct_proportion',
    )
    plt.show()

    sns.barplot(
        data=data_mrt,
        x='condition',
        y='correct_proportion',
        hue='id'
    )
    plt.show()

    # CLT
    clt_perf = {}
    for dir in dirs:
        data = CalibrainData(dir=dir, **config)
        mrt_perf[data.id] = data.clt.performance_features

    data_clt = pd.concat(mrt_perf).reset_index()
    data_clt.rename(columns={"level_0": "id"}, inplace=True)

    # delete practice data
    data_clt = data_clt.loc[data_clt.condition != 'practice']

    sns.boxplot(
        data=data_clt,
        x='condition',
        y='correct_proportion',
        #hue='id'
    )
    plt.show()