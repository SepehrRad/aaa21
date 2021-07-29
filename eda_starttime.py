import folium
import branca.colormap as cm
import matplotlib.pyplot
import numpy as np

import hexagon
import seaborn as sns
from matplotlib import pyplot as plt


def total_countplot(
        df,
        time_interval=False
):
    col = 'Pickup Time_Interval'
    title = "Total annual number of trips per time interval"
    if time_interval is False:
        col = 'Trip Start Hour'
        title = "Total annual number of trips per hour"

    sns.countplot(data=df, x=col)
    plt.ticklabel_format(style='plain', axis='y', useOffset=False)
    plt.title(title)
    plt.show()


def weekend_weekday_countplot(
        df,
        time_interval=False,
        weekdays=False,
        sharey=False
):
    if time_interval is False:
        col = 'Trip Start Hour'
        title = "Annual number of trips per hour"
    else:
        col = 'Pickup Time_Interval'
        title = "Annual number of trips per time interval"
    if weekdays is False:
        hue = None
    else:
        hue = 'Trip Start Weekday'
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=sharey)
    sns.countplot(data=df.loc[df['Trip Start Is Weekend'] == 0], x=col, ax=axes[0], hue=hue)
    axes[0].set_title(title + ' - Weekdays')
    sns.countplot(data=df.loc[df['Trip Start Is Weekend'] == 1], x=col, ax=axes[1], hue=hue)
    axes[1].set_title(title + ' - Weekend')

    axes[0].ticklabel_format(useOffset=False, style="plain", axis="y")
    axes[1].ticklabel_format(useOffset=False, style="plain", axis="y")


def days_of_week_countplot(
        df,
        time_interval=False
):
    hue_order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    if time_interval is False:
        col = 'Trip Start Hour'
        title = "Day of the weeks' trips - Hourly"
    else:
        col = 'Pickup Time_Interval'
        title = "Day of the week' trips - Per time interval"
    sns.countplot(data=df, x=col, hue='Pickup Day', hue_order=hue_order)
    plt.ticklabel_format(style='plain', axis='y', useOffset=False)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.show()


def area_peak_hours(
        df,
        time_zone=None
):
    if time_zone is not None:
        df = df.loc[df['Pickup Time_Interval'] == time_zone]
    df_grp = df.groupby(['Pickup Community Area', 'Trip Start Hour']).size().reset_index(name="Total Trips")
    df_grp = df_grp.loc[df_grp.groupby('Pickup Community Area')['Total Trips'].idxmax()]
    return df_grp

CHICAGO_COORD = [41.91364, -87.72645]


def area_peak_hours_map(
        df,
        gdf,
        time_zone=None
):
    legend_name = "Peak Hour per Pickup Community Area"
    threshold = [0, 4, 8, 12, 16, 20, 24]
    base_map = folium.Map(location=CHICAGO_COORD, tiles="cartodbpositron")
    if time_zone is not None:
        df = df.loc[df['Pickup Time_Interval'] == time_zone]
        legend_name = legend_name + " during " + time_zone
        if time_zone is not 'night':
            threshold = df['Trip Start Hour'].unique()
            _ = threshold[-1]
            _ = _+1
            threshold = np.append(threshold, _)
        #else:
            #threshold = df['Trip Start Hour'].unique()
            #_ = threshold[-2]
            #_ = _ + 1
            #threshold = np.append(threshold, _)
    df = df.groupby(['Pickup Community Area', 'Trip Start Hour']).size().reset_index(name="Total Trips")
    df = df.loc[df.groupby('Pickup Community Area')['Total Trips'].idxmax()]

    df['Pickup Community Area'] = df['Pickup Community Area'].astype('float').astype('int').astype('str')
    #linear = cm.linear.YlOrRd_09.scale(0,23).to_step(n=24, index=range(0, 23))
    #colormap = cm.LinearColormap(vmin=df['Trip Start Hour'].min(), vmax=df['Trip Start Hour'].max(),colors=['red','lightblue'])
    folium.Choropleth(
        data=df,
        geo_data=gdf,
        name="choropleth",
        columns=['Pickup Community Area', 'Trip Start Hour'],
        key_on="feature.properties.area_num_1",
        fill_color='YlOrRd',
        bins=6,
        fill_opacity=0.6,
        line_opacity=0.3,
        legend_name=legend_name,
        threshold_scale=threshold
    ).add_to(base_map)
    """
    geo_j = gdf.to_json()
    geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {'fillColor': 'orange'})
    geo_j.add_to(base_map)
    """
    folium.TileLayer("cartodbdark_matter", name="dark mode", control=True).add_to(
        base_map
    )
    folium.TileLayer("openstreetmap", name="open street map", control=True).add_to(
        base_map
    )
    folium.LayerControl().add_to(base_map)
    return base_map
