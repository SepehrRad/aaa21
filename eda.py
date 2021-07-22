import folium
import numpy as np

import utils

CHICAGO_COORD = [41.91364, -87.72645]


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
        agg_col(String): Aggregates data based on this column
        target_name(String): This column is used to calculate an aggregation target
        agg_strategy(String): The data will be aggregated based on this strategy
        log_scale(bool): Shows data on log scale.
        geo_json(geojson): The given geojson to use in choropleth
        use_hexes(bool): Whether it is a choropleth with hexes or not.
        cmap(String): The chosen colormap.
    :return:
        folium.Choropleth: The created choropleth
    """
    key = "feature.properties.area_num_1"
    opacity = 0.9
    if geo_json is None:
        geo_json = utils.read_geo_dataset("community_areas.geojson")
    if use_hexes:
        key = "feature.id"
        opacity = 0.6
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
    folium.Choropleth(
        geo_data=geo_json,
        name="choropleth",
        data=data,
        columns=[agg_col, target_name],
        key_on=key,
        fill_color=cmap,
        fill_opacity=opacity,
        line_opacity=0.3,
        legend_name=legend_name,
    ).add_to(base_map)

    folium.TileLayer("cartodbdark_matter", name="dark mode", control=True).add_to(
        base_map
    )
    folium.TileLayer("openstreetmap", name="open street map", control=True).add_to(
        base_map
    )
    folium.LayerControl().add_to(base_map)
    return base_map
