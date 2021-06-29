import warnings

import numpy as np
import pandas as pd
from pyod.models.hbos import HBOS
from scipy.stats import zscore

from utils import write_parquet_from_pandas
from utils import read_parquet

warnings.filterwarnings("ignore")


def _clean_dataset(df, verbose=False):
    return df


def clean_dataset(file="Taxi_Trips.parquet", verbose=False):
    """
    This function reads in all the data, cleans it and saves the cleaned dataframe as parquet file in the data
    directory.
    ----------------------------------------------
    :param
        file(String): Name of the input data file. Default is Taxi_Trips.parquet.
        verbose(boolean): Set 'True' to get detailed logging information.
    """
    print(f"Started cleaning the data set")
    cleaned_df = _clean_dataset(
        read_parquet(file), verbose=verbose
    )
    write_parquet_from_pandas(
        cleaned_df, filename="Taxi_Trips_cleaned.parquet"
    )
    print(f"Finished cleaning the data set")