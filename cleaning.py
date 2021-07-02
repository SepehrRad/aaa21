import warnings

import numpy as np
import pandas as pd
from scipy.stats import zscore

from utils import write_parquet_from_pandas
from utils import read_parquet

warnings.filterwarnings("ignore")

column_description = {
    "categorical_features": ["Payment Type", "Company"],
    "temporal_features": ["Trip Start Timestamp", "Trip End Timestamp"],
    "spatial_features": ["Pickup Census Tract", "Dropoff Census Tract",
                         "Pickup Community Area", "Dropoff Community Area"
                         ]
}


def _remove_invalid_spatial_entries(df):
    """
    Removes invalid spatial entries by checking if the Pickup/Dropoff Census Tract as well as the Community Area columns
    are both null at the same time.
    ----------------------------------------------
    :param
        df(pd.DataFrame): DataFrame to be processed.
    :returns
        pd.DataFrame: Processed DataFrame.
    """
    num_entries = df.shape[0]
    df = df[~(df['Pickup Census Tract'].isna() & df['Pickup Community Area'].isna() |
              df['Dropoff Census Tract'].isna() & df['Dropoff Community Area'].isna())]
    print(f"{num_entries - df.shape[0]} invalid spatial entries have been successfully removed!")
    return df


def _set_column_types(df):
    """
    Sets the column types for the given DataFrame.
    ----------------------------------------------
    :param
        df(pd.DataFrame): DataFrame to be processed.
    :returns
        pd.DataFrame: Processed DataFrame.
    """
    spatial_cols = column_description.get("spatial_features")
    df[spatial_cols] = df[spatial_cols].astype("str")


def _remove_invalid_numeric_data(df, verbose=False):
    """
    This functions removes negative (faulty) numeric values from the given DataFrame & returns the processed DataFrame.
    ----------------------------------------------
    :param
        df(pd.DataFrame): DataFrame to be processed.
        verbose(boolean): Set 'True' to get detailed logging information.
    :returns
        pd.DataFrame: Processed DataFrame.
    """
    df_numeric_view = df.select_dtypes(include="number")
    sum_negative_entries = 0
    for col in df_numeric_view.columns:
        if col in ["Trip Seconds", "Trip Miles"]:
            negative_entries = df[(df[col] <= 0) | (df[col].isna())]
        else:
            negative_entries = df[(df[col] < 0) | (df[col].isna())]
        if verbose:
            print(f"--> {negative_entries.shape[0]} invalid entries found in {col}")
        sum_negative_entries += negative_entries.shape[0]
        df.drop(negative_entries.index, inplace=True)
    print(f"{sum_negative_entries} invalid entries have been successfully dropped!")
    return df.reset_index(drop=True)


def _remove_outliers(
        df,
        excluded_cols=None,
        zscore_threshold=1.5,
        verbose=False,
):
    """
    This functions removes outliers by applying the 'zscore'-algorithm on all numeric columns.
    ----------------------------------------------
    :param
        df(pd.DataFrame): DataFrame to be processed.
        excluded_cols(list): Columns without outlier detection.
        zscore_threshold(float): Hyperparameter for 'zscore'-algorithm.
        verbose(boolean): Set 'True' to get detailed logging information.
    :returns
        pd.DataFrame: Processed DataFrame.
    """
    outlier_count = 0
    df_numeric_view = df.select_dtypes(include="number")

    for col in df_numeric_view.columns:
        if excluded_cols and col in excluded_cols:
            continue
        df[f"{col}_zscore"] = np.around(np.abs(zscore(df[col])), decimals=1)
        outlier = df[df[f"{col}_zscore"] > zscore_threshold]
        outlier_count += outlier.shape[0]
        df.drop(outlier.index, inplace=True)
        if verbose:
            print(
                f"--> {outlier.shape[0]} outlier detected and removed from {col} column using zscore"
            )
    df = df.loc[:, ~df.columns.str.contains("zscore")]

    print(f"Outlier detection completed. Number of removed outlier: {outlier_count}")

    return df.reset_index(drop=True)


def _merge_additional_columns(df, file="Taxi_Trips.parquet"):
    """
    This function merges the given Dataframe with columns so far missing from the whole Dataframe.
    ----------------------------------------------
    :param
        df(pd.DataFrame): DataFrame to be processed.
        file(String): Name of the input data file. Default is Taxi_Trips.parquet.
    :returns:
        pd.DataFrame: Merged DataFrame.
    """
    add_df = read_parquet(file, columns=['Trip ID',
                                         'Taxi ID',
                                         'Trip Start Timestamp',
                                         'Trip End Timestamp',
                                         'Payment Type',
                                         'Company',
                                         'Pickup Centroid Latitude',
                                         'Pickup Centroid Longitude',
                                         # 'Pickup Centroid Location', # redundant
                                         'Dropoff Centroid Latitude',
                                         'Dropoff Centroid Longitude',
                                         # 'Dropoff Centroid  Location' # redundant
                                         ])
    add_df.rename(columns={"Dropoff Centroid  Location": "Dropoff Centroid Location"}, inplace=True)
    df = df.merge(
        add_df,
        how="left",
        left_on="Trip ID",
        right_on="Trip ID"
    )
    categorical_cols = column_description.get("categorical_features")
    df[categorical_cols] = df[categorical_cols].astype("category")
    df.drop(columns=['Trip ID'], inplace=True)
    invalid_entries = df.shape[0]
    df.replace("", float("NaN"), inplace=True)
    df.dropna(subset=["Pickup Centroid Latitude", "Dropoff Centroid Latitude"], inplace=True)
    invalid_entries -= df.shape[0]
    print(f"{invalid_entries} invalid entries from Pickup/Dropoff Centroid locations have been successfully dropped!")
    return df.reset_index(drop=True)


def _add_cyclical_features(df):
    """
    This function turns the Trip Start/End Timestamp columns into 5 columns each, divided by Month, Day, Hour, Weekday
    and Is Weekend. Afterwards the Timestamp columns are removed and the processed DataFrame is returned.
    ----------------------------------------------
    :param
        df(pd.DataFrame): DataFrame to be processed.
    """

    for col in column_description.get("temporal_features"):
        df[col] = pd.to_datetime(df[col], format='%m/%d/%Y %I:%M:%S %p')
        name = col[:-10]
        df[f"{name} Month"] = df[col].dt.month
        df[f"{name} Day"] = df[col].dt.day
        df[f"{name} Hour"] = df[col].dt.hour
        df[f"{name} Weekday"] = df[col].dt.dayofweek
        df[f"{name} Is Weekend"] = (df[f"{name} Weekday"] > 4).astype(int)
        df.drop(columns=[col], inplace=True)


def clean_dataset(file="Taxi_Trips.parquet", verbose=False):
    """
    This function reads in all the data, cleans it and saves the cleaned dataframe as parquet file in the data
    directory.
    ----------------------------------------------
    :param
        file(String): Name of the input data file. Default is Taxi_Trips.parquet.
        verbose(boolean): Set 'True' to get detailed logging information.
    """
    print("Read the data set")
    df = read_parquet(file, columns=['Trip ID',
                                     'Trip Seconds',
                                     'Trip Miles',
                                     'Pickup Census Tract',
                                     'Dropoff Census Tract',
                                     'Pickup Community Area',
                                     'Dropoff Community Area',
                                     'Fare',
                                     'Tips',
                                     'Tolls',
                                     'Extras',
                                     'Trip Total'
                                     ])
    print("Start cleaning the data set")
    # Remove all entries without a Census Tract and a Community Area
    df = _remove_invalid_spatial_entries(df)
    # Set the column types
    _set_column_types(df)
    # Remove invalid numeric entries
    df = _remove_invalid_numeric_data(df, verbose=verbose)
    # Outlier detection
    df = _remove_outliers(
        df,
        verbose=verbose
    )
    # Merge further columns
    df = _merge_additional_columns(df, file=file)
    print("Finished cleaning the data set")
    print("Add cyclical features")
    # Add cyclical features
    _add_cyclical_features(df)
    # Sort table by start date
    df.sort_values(by=['Trip Start Month', 'Trip Start Day', 'Trip Start Hour'])
    # Save data
    print("Saving the data set")
    write_parquet_from_pandas(
        df, filename="Taxi_Trips_cleaned.parquet"
    )
