import geo_engineering
import pandas as pd
import numpy as np


def _add_weather_data(df, weather_df, temporal_resolution):
    """
    Doc String!
    """
    _ = weather_df.groupby([pd.Grouper(key='Trip Start Timestamp',
                                       freq=temporal_resolution,
                                       offset=f"{'23h00min' if temporal_resolution == '6H' else '00h00min'}")]).agg(
        {"Humidity(%)": np.mean,
         "Pressure(hPa)": np.mean,
         "Temperature(C)": np.mean,
         "Wind Direction(Meteoro. Degree)": np.mean,
         "Wind Speed(M/S)": np.mean})
    df = df.merge(_, how="left", on="Trip Start Timestamp")
    return df


def create_aggregated_data(df, weather_df, temporal_resolution, use_hexes=False, only_pickup=True, hex_resolution=None):
    """
    Doc String!
    """
    if use_hexes:
        df = geo_engineering.add_community_areas_with_hexagons(df, return_geojson=False)
        if only_pickup:
            df.drop(list(df.filter(regex='Dropoff')), axis=1, inplace=True)
        geo_col = f"hex_{hex_resolution}_pickup"
    else:
        geo_col = 'Pickup Community Area'
    df = df.groupby([pd.Grouper(key='Trip Start Timestamp',
                                freq=temporal_resolution,
                                offset=f"{'23h00min' if temporal_resolution == '6H' else '00h00min'}"), geo_col]).agg(
        {"Trip ID": 'count'})
    df.rename(columns={"Trip ID": f"Demand ({temporal_resolution})"}, inplace=True)
    df.reset_index(inplace=True)
    df = _add_weather_data(df=df, weather_df=weather_df, temporal_resolution=temporal_resolution)
    return df
