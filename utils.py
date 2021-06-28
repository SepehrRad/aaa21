import os
import warnings
from datetime import datetime

import pandas as pd
import pyarrow
from pyarrow import csv, parquet

warnings.filterwarnings("ignore")


def get_data_path():
    """
    This function finds the path of data folder in the current project.
    ----------------------------------------------
    :return:
        String: the data folder path
    :raises
        FileNotFoundError: if the data directory can't be located.
    """
    if os.path.isdir(os.path.join(os.getcwd(), "data")):
        path = os.path.join(os.getcwd(), "data")
    elif os.path.isdir(os.path.join(os.getcwd(), "..", "data")):
        path = os.path.join(os.getcwd(), "..", "data")
    else:
        raise FileNotFoundError("The data folder could not be found")

    return path


def write_parquet(local_csv_file, filename):
    """
    This function reads a csv file and saves it as a parquet.
    ----------------------------------------------
    :param
        local_csv_file(String): Given csv file path.
        filename(String): Name of the parquet file.
    """
    table = csv.read_csv(local_csv_file)
    parquet.write_table(table, filename)


def read_parquet(file, path=get_data_path(), columns=None):
    """
    This function reads a parquet file & returns it as a pd.DataFrame.
    ----------------------------------------------
    :param
        file(String): Name of file.
        path(String): Path to parquet file.Data directory is chosen by default.
        columns(list[String]): Only reads specified columns from file. Default is None(reads all columns).
        For more information use pyarrow.parquet.read_table documentation.
    """
    parquet_path = os.path.join(path, file)
    table = parquet.read_table(parquet_path, columns=columns)
    df = table.to_pandas()
    df = df.reset_index(drop=True)
    return df


def add_weather_data(df):
    """
    This function reads weather data sets and
    merges them to a data frame.
    ----------------------------------------------
    :param
        df(pandas.DataFrame): Given data frame
    :return
        pandas.DataFrame: The weather data fram for chicago
    """

    # Since there is no vectorized way of converting str to datetime, the map function is used
    df["Trip Start Timestamp"] = df["Trip Start Timestamp"].map(
        lambda date_time_str: datetime.strptime(date_time_str, "%m/%d/%Y %I:%M:%S %p")
    )
    # Temp column used for merging
    df["Trip Start Timestamp_temp"] = df["Trip Start Timestamp"].map(
        lambda x: x.replace(minute=0)
    )

    weather_features = {
        "humidity": "Humidity(%)",
        "pressure": "Pressure(hPa)",
        "temperature": "Temperature(C)",
        "wind_direction": "Wind Direction(Meteoro. Degree)",
        "wind_speed": "Wind Speed(M/S)",
    }

    for feature in weather_features:
        feature_df = pd.read_csv(f"data/{feature}.csv", parse_dates=["datetime"])
        year_mask = feature_df["datetime"].dt.year == 2015
        _ = feature_df.loc[year_mask]
        chicago_information = _[["datetime", "Chicago"]]

        if feature == "temperature":
            # Converting temperature from Kelvin to Celsius
            chicago_information["Chicago"] = chicago_information["Chicago"] - 273.15

        chicago_information.columns = [
            "Hourly Timestamp",
            weather_features.get(feature),
        ]

        df = df.merge(
            chicago_information,
            how="left",
            left_on="Trip Start Timestamp_temp",
            right_on="Hourly Timestamp",
            validate="m:1",
        )
        df.drop(["Hourly Timestamp"], axis=1, inplace=True)

    df.drop(["Trip Start Timestamp_temp"], axis=1, inplace=True)

    return df


def write_parquet_from_pandas(df, filename, path=get_data_path()):
    """
    This function reads a pandas data frame and converts it to
    a parquet file.
    ----------------------------------------------
    :param
        df(pandas.DataFrame): Given data frame
        path(String): Path to parquet file.Data directory is chosen by default.
        filename(String): Name of file.
    """

    parquet_path = os.path.join(path, filename)

    # The number of threads is explicitly set to one to avoid string conversion problems
    table = pyarrow.Table.from_pandas(df, nthreads=1)
    pyarrow.parquet.write_table(table, parquet_path)
