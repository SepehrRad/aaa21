import folium
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import utils

CHICAGO_COORD = [41.86364, -87.72645]


def create_choropleth(
    df,
    target_col,
    agg_col,
    target_name,
    cmap="YlOrRd",
    agg_strategy="median",
    log_scale=False,
    use_hexes=False,
    geo_json=None,
):
    """
    This function creates a folium choropleth based on different targets of the given data.

    ----------------------------------------------

    :param
        df(pd.DataFrame): Data that is used to make the choropleth.
        agg_col(String): Aggregates data based on this column.
        target_name(String): This column is used to calculate an aggregation target.
        agg_strategy(String): The data will be aggregated based on this strategy.
        log_scale(bool): Shows data on log scale.
        geo_json(geojson): The given geojson to use in choropleth.
        use_hexes(bool): Whether it is a choropleth with hexes or not.
        cmap(String): The chosen colormap.
    :return:
        folium.Choropleth: The created choropleth.
    """
    key = "feature.properties.area_num_1"
    opacity = 0.7
    highlight = True
    if geo_json is None:
        geo_json = utils.read_geo_dataset("community_areas.geojson")
    if use_hexes:
        key = "feature.id"
        highlight = False
    data = (
        df.groupby([agg_col])[target_col]
        .agg(agg_strategy)
        .reset_index(name=target_name)
    )
    data[agg_col] = data[agg_col].astype("str")
    base_map = folium.Map(location=CHICAGO_COORD, tiles="cartodbpositron")
    legend_name = f"{target_name} per {agg_col}"
    if log_scale:
        data[target_name] = np.log(data[target_name])
        legend_name = legend_name + " (Log_Scale)"
    data[target_name] = data[target_name].round(2)
    if not use_hexes:
        geo_json = geo_json.merge(
            data, how="left", left_on="area_num_1", right_on=agg_col
        )
    choropleth = folium.Choropleth(
        geo_data=geo_json,
        name="choropleth",
        data=data,
        columns=[agg_col, target_name],
        key_on=key,
        fill_color=cmap,
        fill_opacity=opacity,
        line_opacity=0.3,
        legend_name=legend_name,
        highlight=highlight,
    ).add_to(base_map)

    folium.TileLayer("cartodbdark_matter", name="dark mode", control=True).add_to(
        base_map
    )
    folium.TileLayer("openstreetmap", name="open street map", control=True).add_to(
        base_map
    )
    folium.LayerControl().add_to(base_map)
    if not use_hexes:
        choropleth.geojson.add_child(
            folium.features.GeoJsonTooltip(
                fields=["community", target_name],
                aliases=[f"{agg_col}: ", f"{target_name}: "],
                style=(
                    "background-color: white; color: #333333; "
                    "font-family: arial; "
                    "font-size: 12px; padding: 10px;"
                ),
            )
        )
    return base_map


def area_peak_hours_map(df, gdf, time_zone=None, cmap="YlOrRd", use_hexes=False):
    """
    This function creates a folium choropleth based on peak demand hours.

    ----------------------------------------------

    :param
        df(pd.DataFrame): Data that is used to make the choropleth.
        gdf(geojson): The given geojson to use in choropleth.
        time_zone(String): If set, plot the map for the specific time interval.
        use_hexes(bool): Whether it is a choropleth with hexes or not.
        cmap(String): The chosen colormap.
    :return:
        folium.Choropleth: The created choropleth.
    """
    key = "feature.properties.area_num_1"
    opacity = 0.7
    legend_name = "Peak Hour per Pickup Community Area"
    threshold = [0, 4, 8, 12, 16, 20, 24]
    base_map = folium.Map(location=CHICAGO_COORD, tiles="cartodbpositron")
    if use_hexes:
        key = "feature.id"
        opacity = 0.6
    if time_zone is not None:
        df = df.loc[df["Pickup Time_Interval"] == time_zone]
        legend_name = legend_name + " during " + time_zone
        threshold = df["Trip Start Hour"].unique()
        _ = threshold[-1]
        _ = _ + 1
        threshold = np.append(threshold, _)
    df = (
        df.groupby(["Pickup Community Area", "Trip Start Hour"])
        .size()
        .reset_index(name="Total Trips")
    )
    df = df.loc[df.groupby("Pickup Community Area")["Total Trips"].idxmax()]

    df["Pickup Community Area"] = (
        df["Pickup Community Area"].astype("float").astype("int").astype("str")
    )

    folium.Choropleth(
        data=df,
        geo_data=gdf,
        name="choropleth",
        columns=["Pickup Community Area", "Trip Start Hour"],
        key_on=key,
        fill_color=cmap,
        fill_opacity=opacity,
        line_opacity=0.3,
        legend_name=legend_name,
        threshold_scale=threshold,
    ).add_to(base_map)

    folium.TileLayer("cartodbdark_matter", name="dark mode", control=True).add_to(
        base_map
    )
    folium.TileLayer("openstreetmap", name="open street map", control=True).add_to(
        base_map
    )
    folium.LayerControl().add_to(base_map)
    return base_map


def starttime_total_countplot(df, time_interval=False):
    """
    This function creates a countplot to visualize starting times distribution throughout the day.

    ----------------------------------------------

    :param
        df(pd.DataFrame): Data that is used to make the choropleth.
        time_zone(String): If set, creates the plot for the specific time interval.
    :return:
        pyplot.show: The created seaborn countplot.
    """
    col = "Pickup Time_Interval"
    title = "Annual number of trips per time interval"
    if time_interval is False:
        col = "Trip Start Hour"
        title = "Annual number of trips per hour"

    sns.countplot(data=df, x=col)
    plt.ticklabel_format(style="plain", axis="y", useOffset=False)
    plt.title(title)
    plt.show()


def starttime_weekend_weekday_countplot(df, time_interval=False, weekdays=False):
    """
    This function creates two countplots to visualize difference in temporal demand difference between weekdays and
    weekends.

    ----------------------------------------------

    :param
        df(pd.DataFrame): Data that is used to make the choropleth.
        time_zone(String): If set, creates the plot for the specific time interval.
        weekdays(bool): Whether to plot weekdays hue on plots or not.
    :return:
        pyplot.show: The created seaborn countplot.
    """
    if time_interval is False:
        col = "Trip Start Hour"
        title = "Annual number of trips per hour"
    else:
        col = "Pickup Time_Interval"
        title = "Annual number of trips per time interval"
    if weekdays is False:
        hue = None
    else:
        hue = "Trip Start Weekday"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.countplot(
        data=df.loc[df["Trip Start Is Weekend"] == 0], x=col, ax=axes[0], hue=hue
    )
    axes[0].set_title(title + " - Weekdays")
    sns.countplot(
        data=df.loc[df["Trip Start Is Weekend"] == 1], x=col, ax=axes[1], hue=hue
    )
    axes[1].set_title(title + " - Weekend")
    axes[0].ticklabel_format(useOffset=False, style="plain", axis="y")
    axes[1].ticklabel_format(useOffset=False, style="plain", axis="y")
    if time_interval is False:
        axes[0].set_ylim(40000, 1120000)
        axes[1].set_ylim(40000, 1120000)
    else:
        axes[0].set_ylim(70000, 6000000)
        axes[1].set_ylim(70000, 6000000)


def starttime_days_of_week_countplot(df, time_interval=False):
    """
    This function creates a countplots to visualize difference in temporal demand difference between each day of
    the week.

    ----------------------------------------------

    :param
        df(pd.DataFrame): Data that is used to make the choropleth.
        time_zone(String): If set, creates the plot for the specific time interval.
    :return:
        pyplot.show: The created seaborn countplot.
    """
    hue_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    if time_interval is False:
        col = "Trip Start Hour"
        title = "Day of the weeks' trips - Hourly"
    else:
        col = "Pickup Time_Interval"
        title = "Day of the week' trips - Per time interval"
    sns.countplot(data=df, x=col, hue="Pickup Day", hue_order=hue_order)
    plt.ticklabel_format(style="plain", axis="y", useOffset=False)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.show()


def starttime_area_peak_hours(df, time_zone=None):
    """
    This function returns a grouped series of community area IDs and their peak demand hour.

    ----------------------------------------------

    :param
        df(pd.DataFrame): Data that is used to make the choropleth.
        time_zone(String): If set, creates the Seifor the specific time interval.
    :return:
        pandas.Series: A series of community area IDs and their peak demand hour.
    """
    if time_zone is not None:
        df = df.loc[df["Pickup Time_Interval"] == time_zone]
    df_grp = (
        df.groupby(["Pickup Community Area", "Trip Start Hour"])
        .size()
        .reset_index(name="Total Trips")
    )
    df_grp = df_grp.loc[df_grp.groupby("Pickup Community Area")["Total Trips"].idxmax()]
    return df_grp
