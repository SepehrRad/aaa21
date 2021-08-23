import geopandas
import h3.api.numpy_int as h3
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

import geo_engineering


def add_weekday(df):
    """
    This function adds weekdays (Sunday, Monday,..) to the data frame.
    ----------------------------------------------
    :param
        df(pd.DataFrame): DataFrame to which weekday names should be added.
    :return
        pandas.DataFrame: The data frame with additional weekday column
    """
    # Creating Weekday columns
    df["Pickup Day"] = df["Trip Start Timestamp"].dt.day_name()
    df["Drop-off Day"] = df["Trip End Timestamp"].dt.day_name()
    return df


def add_time_interval(df):
    """
    This function adds time zones (morning, evening, etc.) based on the hour of the day.
    ----------------------------------------------
    :param
        df(pd.DataFrame): DataFrame to which time zone information should be added.
    """
    # Pickup time interval
    df.loc[
        df["Trip Start Hour"].between(5, 11, inclusive="left"), "Pickup Time_Interval"
    ] = "morning"
    df.loc[
        df["Trip Start Hour"].between(11, 17, inclusive="left"), "Pickup Time_Interval"
    ] = "midday"
    df.loc[
        df["Trip Start Hour"].between(17, 23, inclusive="left"), "Pickup Time_Interval"
    ] = "evening"
    df.loc[
        ((df["Trip Start Hour"] >= 23) | (df["Trip Start Hour"] < 5)),
        "Pickup Time_Interval",
    ] = "night"

    # Drop-off time interval
    df.loc[
        df["Trip End Hour"].between(5, 11, inclusive="left"), "Drop-off Time_Interval"
    ] = "morning"
    df.loc[
        df["Trip End Hour"].between(11, 17, inclusive="left"), "Drop-off Time_Interval"
    ] = "midday"
    df.loc[
        df["Trip End Hour"].between(17, 23, inclusive="left"), "Drop-off Time_Interval"
    ] = "evening"
    df.loc[
        ((df["Trip End Hour"] >= 23) | (df["Trip End Hour"] < 5)),
        "Drop-off Time_Interval",
    ] = "night"


def add_holidays(df):
    """
    This function adds 2015 holidays to the data frame.
    ----------------------------------------------
    :param
        df (pandas.DataFrame): Given data frame.
    :returns
        pandas.DataFrame: The given data frame with the holiday information.
    """
    cal = calendar()
    holidays = cal.holidays(
        df["Trip Start Timestamp"].min(), df["Trip Start Timestamp"].max()
    )
    df["Holiday"] = (
        df["Trip Start Timestamp"].dt.date.astype("datetime64").isin(holidays)
    )
    return df


def add_spatial_features(df, with_hex=False, hex_res=None, dropoff=False):
    """
    This function adds the relevant spatial features for prediction.
    ----------------------------------------------
    :param
        df(pandas.DataFrame): Given data frame
    :return
        geoPandas.GeoDataFrame: The merged geo data frame
    """
    gdf = geopandas.read_file("data/community_areas.geojson")
    gdf["area_numbe"] = gdf["area_numbe"].astype(float).astype(int)
    gdf["Community Area Center"] = gdf.geometry.centroid
    gdf["Community Area Center Lat"] = gdf["Community Area Center"].y
    gdf["Community Area Center Long"] = gdf["Community Area Center"].x

    if with_hex:
        gdf = geo_engineering._get_hexes(gdf)
        gdf["hex_6_center"] = gdf.hex_6.apply(lambda hex: h3.h3_to_geo(hex))
        gdf["hex_7_center"] = gdf.hex_7.apply(lambda hex: h3.h3_to_geo(hex))
        gdf["hex_6_center_lat"] = [
            center_point[0] for center_point in gdf["hex_6_center"]
        ]
        gdf["hex_6_center_lon"] = [
            center_point[1] for center_point in gdf["hex_6_center"]
        ]
        gdf["hex_7_center_lat"] = [
            center_point[0] for center_point in gdf["hex_7_center"]
        ]
        gdf["hex_7_center_lon"] = [
            center_point[1] for center_point in gdf["hex_7_center"]
        ]
        _get_dist_features(
            df=gdf,
            lon=gdf[f"hex_{hex_res}_center_lon"],
            lat=gdf[f"hex_{hex_res}_center_lat"],
        )
        gdf = gdf[[f"hex_{hex_res}", "City Center Distance", "Airport Distance"]]
        merged_df = df.merge(
            gdf,
            how="left",
            left_on=f"hex_{hex_res}_pickup",
            right_on=f"hex_{hex_res}",
        )
        merged_df.drop(columns=[f"hex_{hex_res}"], inplace=True)

    else:
        _get_dist_features(
            df=gdf,
            lon=gdf["Community Area Center Long"],
            lat=gdf["Community Area Center Lat"],
        )
        gdf = gdf[["area_numbe", "City Center Distance", "Airport Distance"]]
        df["Pickup Community Area"] = (
            df["Pickup Community Area"].astype(float).astype(int)
        )
        merged_df = df.merge(
            gdf,
            how="left",
            validate="m:1",
            left_on="Pickup Community Area",
            right_on="area_numbe",
        )
        if dropoff is True:
            merged_df["Dropoff Community Area"] = (merged_df["Dropoff Community Area"].astype(float).astype(int))
            gdf = gdf.rename(columns={"area_numbe": "area_numbe_dropoff", "City Center Distance": "City Center Distance Dropoff", "Airport Distance": "Airport Distance Dropoff"})
            merged_df = merged_df.merge(
                gdf,
                how="left",
                validate="m:1",
                left_on="Dropoff Community Area",
                right_on="area_numbe_dropoff",
            )
            merged_df.drop(columns=["area_numbe_dropoff"], inplace=True)
            merged_df["Dropoff Community Area"] = merged_df["Dropoff Community Area"].astype(
                str
            )
        merged_df.drop(columns=["area_numbe"], inplace=True)
        merged_df["Pickup Community Area"] = merged_df["Pickup Community Area"].astype(
            str
        )
    return merged_df


def _get_dist_features(df, lon, lat):
    """
    This function calculates the haversine distance of a point to Chicago city center and the air port.
    ----------------------------------------------
    :param
       df ((geo)Pandas.DataFrame): The given data frame.
       lon (float/np.array): The longitude.
       lat (float/np.array): The latitude.
    :return: (geo)Pandas.DataFrame: Data frame with the added features
    """
    center_lon = -87.623177
    center_lat = 41.881832
    airport_lon = -87.904724
    airport_lat = 41.978611
    center_lon, center_lat, lon_c, lat_c = map(
        np.radians, [center_lon, center_lat, lon, lat]
    )
    _ = np.sin((lat_c - center_lat) / 2.0) ** 2 + (
        np.cos(center_lat) * np.cos(lat_c) * np.sin((lon_c - center_lon) / 2.0) ** 2
    )
    df["City Center Distance"] = 6371 * 2 * np.arcsin(np.sqrt(_))
    airport_lon, airport_lat, lon_a, lat_a = map(
        np.radians, [airport_lon, airport_lat, lon, lat]
    )
    __ = np.sin((lat_a - airport_lat) / 2.0) ** 2 + (
        np.cos(airport_lat) * np.cos(lat_a) * np.sin((lon_a - airport_lon) / 2.0) ** 2
    )
    df["Airport Distance"] = 6371 * 2 * np.arcsin(np.sqrt(__))
    return df
