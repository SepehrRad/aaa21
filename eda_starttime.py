import folium

import hexagon
import utils
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def add_time_buckets(
        df
):
    conditions = [
        (df['Trip Start Hour'] >= 22) & (df['Trip Start Hour'] <= 3),
        (df['Trip Start Hour'] > 3) & (df['Trip Start Hour'] <= 9),
        (df['Trip Start Hour'] > 9) & (df['Trip Start Hour'] <= 15),
        (df['Trip Start Hour'] > 15) & (df['Trip Start Hour'] <=21)
    ]
    time_buckets = ['22-4h', '4-10h', '10-16h', '16-22h']
    df['Time Bucket'] = np.select(conditions, time_buckets)
    return df


def add_time_interval(
        df
):
    """
    This function adds time zones (morning, evening, etc.) based on the hour of the day.
    ----------------------------------------------
    :param
        df(pd.DataFrame): DataFrame to which time zone information should be added.
    """
    # Pickup time interval
    df.loc[df['Trip Start Hour'].between(5, 11, inclusive='left'), 'Pickup Time_Interval'] = 'morning'
    df.loc[df['Trip Start Hour'].between(11, 17, inclusive='left'), 'Pickup Time_Interval'] = 'midday'
    df.loc[df['Trip Start Hour'].between(17, 23, inclusive='left'), 'Pickup Time_Interval'] = 'evening'
    df.loc[((df['Trip Start Hour'] >= 23) | (df['Trip Start Hour'] < 5)), 'Pickup Time_Interval'] = 'night'

    return df


def total_countplot(
        df,
        time_interval = False
):
    sns.color_palette("dark")
    if time_interval is False:
        col = 'Trip Start Hour'
    else:
        col = 'Pickup Time_Interval'
    plot = sns.countplot(data=df, x=col)
    plt.ticklabel_format(style='plain', axis='y', useOffset=False)
    return plot


def weekend_weekday_countplot(
        df,
        time_interval = False,
        weekdays = False
):
    sns.color_palette("dark")
    if time_interval is False:
        col = 'Trip Start Hour'
    else:
        col = 'Pickup Time_Interval'
    if weekdays is False:
        hue = None
    else:
        hue = 'Trip Start Weekday'
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.countplot(data=df.loc[df['Trip Start Is Weekend'] == 0], x=col, ax=axes[0], hue=hue)
    axes[0].set_title('Weekdays')
    sns.countplot(data=df.loc[df['Trip Start Is Weekend'] == 1], x=col, ax=axes[1], hue=hue)
    axes[1].set_title('Weekend')

    axes[0].ticklabel_format(useOffset=False, style="plain", axis="y")
    axes[1].ticklabel_format(useOffset=False, style="plain", axis="y")


def area_peak_hours(
        df
):
    # Returns a list of areas with their peak hour
    df_grp = df.groupby(['Pickup Community Area'])['Trip Start Hour'].apply(pd.Series.value_counts)
    return df_grp


def area_peak_hours_map(
        df,
        gdf
):
    base_map = folium.Map(location=[41.91364, -87.72645])
    folium.Choropleth(
        geo_data=gdf,
        name="choropleth",
        key_on="feature.properties.hex",
        fill_color="YlGn",
        fill_opacity=0.7,
        line_opacity=.1,
        legend_name="Total Trips",
    ).add_to(base_map)

    geo_j = gdf.to_json()
    geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {'fillColor': 'orange'})
    geo_j.add_to(base_map)

    return base_map


def hex_peak_hours_map():

    hex_gdf = hexagon.census_tract_to_hexagon(save=True)
    base_map = folium.Map(location=[41.91364, -87.72645], zoom_start=11)
    geo_j = folium.GeoJson(data=hex_gdf, style_function=lambda x: {'fillColor': 'orange'})
    geo_j.add_to(base_map)
    return base_map
    """base_map = folium.Map(location=[41.91364, -87.72645])
    folium.Choropleth(
        geo_data=hex_gdf,
        name="choropleth",
        data=gdf,
        columns=["hex", "Count"],
        key_on="feature.properties.hex",
        fill_color="YlGn",
        fill_opacity=0.7,
        line_opacity=.1,
        legend_name="Total Trips",
    ).add_to(base_map)

    folium.LayerControl().add_to(base_map)
    base_map"""