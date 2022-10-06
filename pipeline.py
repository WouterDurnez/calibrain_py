"""
  ___      _ _ _             _
 / __|__ _| (_) |__ _ _ __ _(_)_ _
| (__/ _` | | | '_ \ '_/ _` | | ' \
 \___\__,_|_|_|_.__/_| \__,_|_|_||_|

- Coded on_col Wouter Durnez & Jonas De Bruyne

This file should contain the entire pipeline, from raw data to results.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from calibrain import calibrain
from utils.helper import log


# from tqdm import tqdm


def get_aggregated_subjective_data(database: list, task: str):
    """
    :param database: list of all CalibrainData objects (one for each pp)
    :param task: 'clt' or 'mrt'
    :return: aggregated dataframe
    """
    # TODO: also add pp that have NaN data
    all_subjective_data = []
    for data in database:
        df = getattr(data, task).subjective
        df["pp"] = data.pp
        all_subjective_data.append(df)

    df = pd.concat(all_subjective_data)

    return df


def get_aggregated_performance_data(database: list, task: str):
    all_perf_data = []
    for data in database:
        df = getattr(data, task).performance_features
        df["pp"] = data.pp
        all_perf_data.append(df)

    df = pd.concat(all_perf_data)

    return df


if __name__ == "__main__":

    data_dir = "./data"
    folder_names = [
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
    ]

    # manual exclusion of participants
    folder_names.remove("./data/6_202205090923")
    folder_names.remove("./data/24_202206011607")
    folder_names.remove("./data/Alex_202204251533")
    folder_names.remove("./data/11_202205131006")
    folder_names.remove("./data/15_202205201103")

    # import all data
    database = []  # this method is only temporary
    for pp_folder in folder_names:
        log(pp_folder)
        data = calibrain.CalibrainData(dir=pp_folder)
        database.append(data)

    # Some pps have trials that are logged multiple times, we fix this here
    # get n_trials for each pp
    n_trials_dict = {}
    for pp_folder in database:
        n_trials = len(pp_folder.clt.performance)
        n_trials_dict[database.index(pp_folder)] = n_trials

    # get indices of calibrain data objects in which there are too many CLT trials
    to_fix = [k for k, v in n_trials_dict.items() if v > 300]

    # remove duplicates in those objects
    for index in to_fix:
        database[index].clt.performance.drop_duplicates(
            subset="timestamp", keep="first", inplace=True
        )  # TODO: find out which one to keep, 'first' for now

    ############
    # Analysis #
    ############

    clt_agg_subj = get_aggregated_subjective_data(database, "clt").reset_index(
        drop=True
    )
    mrt_agg_subj = get_aggregated_subjective_data(database, "mrt").reset_index(
        drop=True
    )
    # without physical demand in mrt
    mrt_agg_subj["nasa_without_pd"] = mrt_agg_subj[["md", "td", "pe", "ef", "fl"]].mean(
        axis=1
    )
    # save this data for analysis in R
    clt_agg_subj.to_csv(
        os.path.join("/Users/jonas/Desktop/CaliBrain_preprocdata", "clt_subj.csv")
    )
    mrt_agg_subj.to_csv(
        os.path.join("/Users/jonas/Desktop/CaliBrain_preprocdata", "mrt_subj.csv")
    )

    # plot the data
    # clt - subjective - overall
    sns.set_theme(style="whitegrid")
    ax = sns.boxplot(x="condition", y="nasa_score", data=clt_agg_subj)
    ax.set_title("CLT")
    ax.set_ylim(1, 10)
    plt.show()

    # clt - subjective - on_col factor
    sns.set(rc={"figure.figsize": (12, 12)})
    fig, axs = plt.subplots(ncols=3, nrows=2)
    sns.boxplot(x="condition", y="pd", data=clt_agg_subj, ax=axs[0, 0])
    sns.boxplot(x="condition", y="md", data=clt_agg_subj, ax=axs[0, 1])
    sns.boxplot(x="condition", y="td", data=clt_agg_subj, ax=axs[0, 2])
    sns.boxplot(x="condition", y="pe", data=clt_agg_subj, ax=axs[1, 0])
    sns.boxplot(x="condition", y="ef", data=clt_agg_subj, ax=axs[1, 1])
    sns.boxplot(x="condition", y="fl", data=clt_agg_subj, ax=axs[1, 2])
    for m, subplot in np.ndenumerate(axs):
        subplot.set_ylim(0, 10)
        subplot.set_ylabel("score")
    axs[0, 0].set_title("physical demand")
    axs[0, 1].set_title("mental demand")
    axs[0, 2].set_title("temporal demand")
    axs[1, 0].set_title("performance")
    axs[1, 1].set_title("effort")
    axs[1, 2].set_title("frustration")
    plt.show()

    # mrt - subjective - overall
    sns.set(rc={"figure.figsize": (12, 8)})
    fig, axs = plt.subplots(ncols=1, nrows=1)
    ax = sns.boxplot(x="condition", y="nasa_score", data=mrt_agg_subj)
    ax.set_title("MRT - NASA-TLX overall")
    ax.set_ylim(0, 10)
    plt.show()
    # without physical demand
    sns.set(rc={"figure.figsize": (12, 8)})
    fig, axs = plt.subplots(ncols=1, nrows=1)
    ax = sns.boxplot(x="condition", y="nasa_without_pd", data=mrt_agg_subj)
    ax.set_title("MRT - NASA-TLX overall")
    ax.set_ylim(0, 10)
    plt.show()

    # mrt - subjective - on_col factor
    sns.set(rc={"figure.figsize": (12, 12)})
    fig, axs = plt.subplots(ncols=3, nrows=2)
    sns.boxplot(x="condition", y="pd", data=mrt_agg_subj, ax=axs[0, 0])
    sns.boxplot(x="condition", y="md", data=mrt_agg_subj, ax=axs[0, 1])
    sns.boxplot(x="condition", y="td", data=mrt_agg_subj, ax=axs[0, 2])
    sns.boxplot(x="condition", y="pe", data=mrt_agg_subj, ax=axs[1, 0])
    sns.boxplot(x="condition", y="ef", data=mrt_agg_subj, ax=axs[1, 1])
    sns.boxplot(x="condition", y="fl", data=mrt_agg_subj, ax=axs[1, 2])
    for m, subplot in np.ndenumerate(axs):
        subplot.set_ylim(0, 10)
        subplot.set_ylabel("score")
    axs[0, 0].set_title("physical demand")
    axs[0, 1].set_title("mental demand")
    axs[0, 2].set_title("temporal demand")
    axs[1, 0].set_title("performance")
    axs[1, 1].set_title("effort")
    axs[1, 2].set_title("frustration")
    plt.show()

    # # CLT_agg_perf = get_aggregated_performance_data(database, 'clt')
    #
    # # ax = sns.barplot(x='condition', y='correct_prop', data=CLT_agg_perf)
    # # ax.set_title("MRT")
    # # plt.show()

    # MRT - performance
    mrt_agg_perf = get_aggregated_performance_data(database, "mrt")
    mrt_agg_perf.to_csv(
        os.path.join("/Users/jonas/Desktop/CaliBrain_preprocdata", "mrt_perf.csv")
    )

    sns.set(rc={"figure.figsize": (12, 8)})
    ax = sns.boxplot(x="condition", y="correct_prop", data=mrt_agg_perf)
    ax.set_title("MRT - performance")
    plt.show()

    ax = sns.boxplot(x="condition", y="median_rt", data=mrt_agg_perf)
    ax.set_title("MRT - reaction times")
    ax.set_ylabel("reaction time (s)")
    plt.show()
