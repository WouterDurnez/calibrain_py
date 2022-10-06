"""
Helper functions
"""

import os
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import pyfiglet
import toml
from colorama import Fore, Style

VERBOSITY = 3
TIMESTAMPED = True


# Set parameters
def set_params(
    verbosity: int = None,
    timestamped: bool = None,
):
    global VERBOSITY
    global TIMESTAMPED

    VERBOSITY = verbosity if verbosity else VERBOSITY
    TIMESTAMPED = timestamped if timestamped is not None else TIMESTAMPED


def hi(title=None, **params):
    """
    Say hello. (It's stupid, I know.)
    If there's anything to initialize, do so here.
    """

    print("\n")
    print(Fore.BLUE, end="")
    print(pyfiglet.figlet_format("Calibrain", font="small")[:-2])
    print(Style.RESET_ALL)

    if title:
        log(title, title=True, color="blue")

    # Set params on_col request
    if params:
        set_params(**params)


# Fancy print
def log(*message, verbosity=3, sep="", timestamped=None, title=False, color=None):
    """
    Print wrapper that adds timestamp, and can be used to toggle levels of logging info.

    :param message: message to print
    :param verbosity: importance of message: level 1 = top importance, level 3 = lowest importance
    :param timestamped: include timestamp at start of log
    :param sep: separator
    :param title: toggle whether this is a title or not
    :param color: text color
    :return: /
    """

    # Set colors
    color_dict = {
        "red": Fore.RED,
        "blue": Fore.BLUE,
        "green": Fore.GREEN,
        "yellow": Fore.YELLOW,
        "magenta": Fore.MAGENTA,
        "cyan": Fore.CYAN,
    }
    assert (
        color in color_dict or color is None
    ), "Please pick a valid color for logging (red, green, blue, yellow, magenta or cyan)."
    if color and color in color_dict:
        color = color_dict[color]

    # Title always get shown
    verbosity = 1 if title else verbosity

    # Print if log level is sufficient
    if verbosity <= VERBOSITY:

        # Print title
        if title:
            n = len(*message)
            if color:
                print(color, end="")
            print("\n" + (n + 4) * "#")
            print("# ", *message, " #", sep="")
            print((n + 4) * "#" + "\n" + Style.RESET_ALL)

        # Print regular
        else:
            ts = timestamped if timestamped is not None else TIMESTAMPED
            t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            if color:
                print(color, end="")
            print(
                (str(t) + (" - " if sep == "" else "-")) if ts else "",
                *message,
                Style.RESET_ALL,
                sep=sep,
            )

    return


def time_it(f: Callable):
    """
    Timer decorator: shows how long execution of function took.
    :param f: function to measure
    :return: /
    """

    def timed(*args, **kwargs):
        t1 = time.time()
        res = f(*args, **kwargs)
        t2 = time.time()

        log(f"{f.__name__} took {round(t2 - t1, 3)} seconds to complete.")

        return res

    return timed


def set_dir(*dirs):
    """
    If folder doesn't exist, make it.

    :param dir: directory to check/create
    :return: path to dir
    """

    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
            log(
                "WARNING: Data directory <{dir}> did not exist yet, and was created.".format(
                    dir=dir
                ),
                verbosity=1,
            )
        else:
            log(f"<{dir}> folder accounted for.", verbosity=3)


def clean_col_name(col: str):
    """
    Clean up column name strings
    :param col: string containing name
    :return: new string without whitespaces, parentheses, etc.
    """
    col = col.lower()
    col = col.replace(" ", "_")
    col = col.replace(":", "")
    col = col.replace("(", "")
    col = col.replace(")", "")
    col = col[1:] if col.startswith("_") else col
    col = col[-1] if col.endswith("_") else col

    return col


def import_data_frame(path: str | Path, **kwargs):
    df = pd.read_csv(filepath_or_buffer=path, **kwargs)
    df.columns = [clean_col_name(col) for col in df.columns]
    if "timestamp" in df.columns:
        df["time"] = pd.to_datetime(df.timestamp, unit="ms", origin="unix")

    return df


def load_config(path: str | Path):

    with open(path) as config_file:
        config = toml.load(config_file)

    return config


if __name__ == "__main__":
    hi()
