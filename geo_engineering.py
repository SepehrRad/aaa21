import json

import geopandas
import h3.api.numpy_int as h3
import numpy as np
from geojson.feature import *

import hexagon


def hexagons_to_geojson(df_with_hex, res):
    """
    This function produces GeoJSON representation for the hex columns in the given data set.
    ----------------------------------------------
    :param
        df_with_hex(pandas.DataFrame | geoPandas.GeoDataFrame): The Given data frame which contains the hex columns
        res(int): The hex resolution
    :return
        geojson: The GeoJSON representation of the hex column
    """
    list_features = []

    for i, row in df_with_hex.iterrows():
        feature = Feature(geometry=row[f"hex_{res}_geometry"], id=row[f"hex_{res}"])
        list_features.append(feature)

    feat_collection = FeatureCollection(list_features)

    geojson_result = json.dumps(feat_collection)

    return geojson_result


def _get_hexes(df):
    """
    This function get hexagons with resolution 6 and 7 and their corresponding
    geometries and adds them to the data frame
    ----------------------------------------------
    :param
        df(pandas.DataFrame | geoPandas.GeoDataFrame): Given data frame
    :return
        pandas.DataFrame: The data frame with additional hex columns
    """
    df["hex_7"] = hexagon.get_hexagon_vect(
        lat=df["Community Area Center Lat"],
        lng=df["Community Area Center Long"],
        resolution=7,
    )
    df["hex_6"] = hexagon.get_hexagon_vect(
        lat=df["Community Area Center Lat"],
        lng=df["Community Area Center Long"],
        resolution=6,
    )

    df["hex_6_geometry"] = df.hex_6.apply(
        lambda x: {
            "type": "Polygon",
            "coordinates": [h3.h3_to_geo_boundary(h=x, geo_json=True)],
        }
    )

    df["hex_7_geometry"] = df.hex_7.apply(
        lambda x: {
            "type": "Polygon",
            "coordinates": [h3.h3_to_geo_boundary(h=x, geo_json=True)],
        }
    )
    return df


def _prepare_for_merge(df, gdf):
    """
    This function makes the necessary preparations, which should be done before merging
    the community area geo information to the original dataset.
    ----------------------------------------------
    :param
        df(pandas.DataFrame): Given data frame
        gdf(geoPandas.GeoDataFrame): given geo data frame
    :return
        pandas.DataFrame: The processed data frame
        geoPandas.GeoDataFrame: The processed geo data frame
    """
    df.dropna(inplace=True)
    df["Pickup Community Area"] = df["Pickup Community Area"].astype(float).astype(int)
    df["Dropoff Community Area"] = (
        df["Dropoff Community Area"].astype(float).astype(int)
    )
    gdf["area_numbe"] = gdf["area_numbe"].astype(float).astype(int)
    gdf["Community Area Center"] = gdf.geometry.centroid
    gdf["Community Area Center Lat"] = gdf["Community Area Center"].y
    gdf["Community Area Center Long"] = gdf["Community Area Center"].x
    gdf = _get_hexes(gdf)

    gdf = gdf[
        [
            "area_numbe",
            "geometry",
            "Community Area Center",
            "Community Area Center Lat",
            "Community Area Center Long",
            "hex_6",
            "hex_7",
            "hex_6_geometry",
            "hex_7_geometry",
        ]
    ]
    return df, gdf


def _merge_geo_information(df, gdf):
    """
    This function merges the geo information from the community areas into the original data frame
    ----------------------------------------------
    :param
        df(pandas.DataFrame): Given data frame
        gdf(geoPandas.GeoDataFrame): given geo data frame
    :return
        geoPandas.GeoDataFrame: The merged geo data frame
    """
    merged_df = df.merge(
        gdf,
        how="left",
        validate="m:1",
        left_on="Pickup Community Area",
        right_on="area_numbe",
    )
    merged_df = merged_df.merge(
        gdf,
        how="left",
        validate="m:1",
        left_on="Dropoff Community Area",
        right_on="area_numbe",
        suffixes=("_pickup", "_dropoff"),
    )
    # Casting pandas Dataframe to GeoPandas Dataframe
    merged_gdf = geopandas.GeoDataFrame(merged_df, geometry="geometry_pickup")
    merged_gdf.drop(columns=["area_numbe_dropoff", "area_numbe_pickup"], inplace=True)
    return merged_gdf


def add_community_areas_with_hexagons(df, return_geojson=False):
    """
    This is a wrapper function that merges the geo information from the community areas
    into the original data frame.
    ----------------------------------------------
    :param
        df(pandas.DataFrame): Given data frame
        return_geojson(bool): Whether to return the corresponding GeoJson data or not
    :return
        geoPandas.GeoDataFrame: The merged geo data frame
        geojson (Optional): The GeoJson representation of the hex column with resolution 6
        geojson (Optional): The GeoJson representation of the hex column with resolution 7
    """
    gdf = geopandas.read_file("data/community_areas.geojson")
    df, gdf = _prepare_for_merge(df, gdf)
    merged_gdf = _merge_geo_information(df, gdf)
    if return_geojson:
        geojson_hex6 = hexagons_to_geojson(gdf, res=6)
        geojson_hex7 = hexagons_to_geojson(gdf, res=7)
        return merged_gdf, geojson_hex6, geojson_hex7
    else:
        return merged_gdf


def add_community_names(df):
    """
    This function adds the names of Community Area names to a pd.DataFrame.
    ----------------------------------------------
    :param
        df(pandas.DataFrame): Given data frame
    :return
        merged_df(pandas.DataFrame): DataFrame with Community Area names
    """
    gdf = geopandas.read_file("data/community_areas.geojson")
    df.dropna(inplace=True)
    df["Pickup Community Area"] = df["Pickup Community Area"].astype(float).astype(int)
    df["Dropoff Community Area"] = (
        df["Dropoff Community Area"].astype(float).astype(int)
    )

    gdf["area_numbe"] = gdf["area_numbe"].astype(float).astype(int)
    gdf = gdf[["area_numbe", "community"]]
    gdf.rename(columns={"community": "Community Area Name"}, inplace=True)
    merged_df = df.merge(
        gdf,
        how="left",
        validate="m:1",
        left_on="Pickup Community Area",
        right_on="area_numbe",
    )
    merged_df = merged_df.merge(
        gdf,
        how="left",
        validate="m:1",
        left_on="Dropoff Community Area",
        right_on="area_numbe",
        suffixes=("_pickup", "_dropoff"),
    )
    return merged_df
