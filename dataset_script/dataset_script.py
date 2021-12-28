import pandas as pd
import numpy as np
import yaml
import math
import warnings
import argparse

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")


from eventsFactory import getEvents
from reco import getRecoResults

"""USAGE:

    python dataset_script.py -i <input_data_directory> -o <output_data_directory> -c <config_directory> -run <last_4_digits_of_run
    
    The I/O directories should be ../data/ (as default)
    
    The configuration directories should be ../config/ (as default)
    
    NOTE that data and config files should be named accordingly
    
    EXAMPLE: 
    
    python dataset_script.py -run 1231 
    
"""

# CONSTANTS
USE_TRIGGER = False
RUN_TIME_SHIFT = 0
KEEP = ["FPGA", "TDC_CHANNEL", "HIT_DRIFT_TIME", "m"]


def argParser():
    """manages command line arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, default="../data/", help="input data directory"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="../data/", help="output data directory"
    )
    parser.add_argument(
        "-c", "--config", type=str, default="../config/", help="config directory"
    )
    parser.add_argument("-run", "--run", type=int, default=1252, help="run number")

    return parser.parse_args()


def buildDataframe(stream_df, cfg):

    df = pd.DataFrame()

    # reco (da sistemare getEvents e getRecoResults)
    print("Getting events...")
    events = getEvents(stream_df, cfg, RUN_TIME_SHIFT, USE_TRIGGER)
    print("Reconstructing tracks...")
    resultsList, resultsData, resultsHits, resultsDf = getRecoResults(
        events, USE_TRIGGER
    )
    print("Building dataframe...")
    # out df
    for df_ in resultsDf:
        df_ = df_[KEEP]
        df = pd.concat([df, df_], axis=0, ignore_index=True)

    # add a sequential channel tag
    df.loc[(df["FPGA"] == 0), "CH"] = df["TDC_CHANNEL"]
    df.loc[(df["FPGA"] == 1), "CH"] = df["TDC_CHANNEL"] + 128
    df_ = df.drop(["FPGA", "TDC_CHANNEL"], axis=1)
    df_["CH"] = df_["CH"].astype(np.uint32)

    # clean dataset
    df = df_[["CH", "HIT_DRIFT_TIME", "m"]]
    df = df[(df["HIT_DRIFT_TIME"] > -200) & (df["HIT_DRIFT_TIME"] < 600)]

    # rad to deg conversion
    df["THETA"] = np.arctan(df["m"]) * 180.0 / math.pi

    print("Dataframe ready!")

    return df


def saveChannels(df, OUTPUT_PATH, RUNNUMBER):

    FILE_NAME = f"RUN00{RUNNUMBER}_channels.h5"
    save_to = OUTPUT_PATH + FILE_NAME

    print("Saving data...")
    channels = []
    for channel in np.unique(df["CH"]):
        channels.append(df[df["CH"] == channel])
        df[df["CH"] == channel].to_hdf(save_to, key=f"ch{channel}", mode="a")

    return channels


def main(args):

    # store command line arguments
    DATA_PATH = args.input
    CONFIG_PATH = args.config
    RUNNUMBER = args.run
    OUTPUT_PATH = args.output

    # link data and config files
    data_file = DATA_PATH + f"RUN00{RUNNUMBER}_data.txt"
    config_file = CONFIG_PATH + f"RUN00{RUNNUMBER}.yml"

    # read data from file
    print("Reading data from file...")
    stream_df = pd.read_csv(data_file)
    # read config from file
    print("Reading config from file...")
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    df = buildDataframe(stream_df, cfg)
    channels = saveChannels(df, OUTPUT_PATH, RUNNUMBER)

    return


if __name__ == "__main__":
    args = argParser()
    main(args)
