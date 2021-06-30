import warnings

import numpy as np
from scipy.stats import zscore

from utils import write_parquet_from_pandas
from utils import read_parquet

warnings.filterwarnings("ignore")

column_description = {
    "cyclical_features": [
        "start_month",
        "start_day",
        "start_hour",
        "end_hour",
        "end_day",
        "end_month",
    ],
    "categorical_features": ["Payment Type", "Company"],
    "temporal_features": ["Trip Start Timestamp", "Trip End Timestamp"],
    "spatial_features": ["Pickup Census Tract", "Dropoff Census Tract",
                         "Pickup Community Area", "Dropoff Community Area"],
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
    df = read_parquet(file, columns=['Trip Seconds', 'Trip Miles', 'Pickup Census Tract', 'Dropoff Census Tract',
                                     'Pickup Community Area', 'Dropoff Community Area', 'Fare', 'Tips', 'Tolls',
                                     'Extras', 'Trip Total'
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
    df.reset_index(drop=True)
    print("Finished cleaning the data set")
    print("Saving the data set")
    write_parquet_from_pandas(
        df, filename="Taxi_Trips_cleaned.parquet"
    )

# TODO: Remove double spacing in column name
# TODO: Remove empty strings
# TODO: Merge further columns with cleaned data set
# TODO: Add cyclical features
