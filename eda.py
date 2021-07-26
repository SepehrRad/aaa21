import folium
import numpy as np

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
    opacity = 0.7
    highlight = True
    if geo_json is None:
        geo_json = utils.read_geo_dataset("community_areas.geojson")
    if use_hexes:
        key = "feature.id"
        highlight = False
    data = (
        df.groupby([agg_col])[target_col].agg(agg_strategy).reset_index(name=target_name)
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
            data,
            how="left",
            left_on="area_num_1",
            right_on=agg_col
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
        choropleth.geojson.add_child(folium.features.GeoJsonTooltip(fields=['community', target_name],
                                                                    aliases=[f'{agg_col}: ', f'{target_name}: '],
                                                                    style=(
                                                                        "background-color: white; color: #333333; "
                                                                        "font-family: arial; "
                                                                        "font-size: 12px; padding: 10px;"),
                                                                    ))
    return base_map
