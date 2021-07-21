import geopandas as gpd
import h3.api.numpy_int as h3
from geopandas import GeoDataFrame
from h3.unstable import vect
from shapely.geometry import Polygon

import utils


def get_hexagon(lat, lng, res=6):
    """
        Returns the h3 index for a latitude and longitude at given resolution  .
        ----------------------------------------------
        :param
            lat(int): Latitude.
            lng(int): Longitude.
            res(int): Resolution of h3 index.
        :returns
            hexagon(int): H3 index.
    """
    hexagon = h3.geo_to_h3(lat, lng, res)
    return hexagon


def get_hexagon_vect(lat, lng, res=6):
    """
        Returns the h3 indices for arrays of latitudes and longitudes in given resolution.
        ----------------------------------------------
        :param
            lat(array): Array of latitudes.
            lng(array): Array of longitudes.
            res(int): Resolution of h3 index.
        :returns
            hexagon(array): Array of h3 indices.
    """
    column = vect.geo_to_h3(lat, lng, res)
    return column


def polygons_to_hexagon(
        gdf=None,
        res=6,
        filename="Boundaries - Community Areas (current).geojson",
        path=utils.get_data_path(),
        save=False,
        save_name=None,
):
    """
        This function generates the h3 indices and their boundaries based on given polygons
        & returns them in a GeoDataFrame.
        ----------------------------------------------
        :param
            gdf(gdp.GeoDataFrame): GeoDataFrame containing polygons as geometry. If None the function
            reads a GeoJson  from path.
            res(int): Resolution of h3 index.
            filename(str): File to read, if no gdf is given.
            path(str): Path of file to read, if no gdf is given.
            save(boolean): If True saves the created GeoDataFrame as GeoJson in path.
            save_name(str): Name of GeoJson.
        :returns
            hex_gdf(gdp.GeoDataFrame): GeoDataFrame containing h3 indices and their boundaries.
    """
    hexagon_index = []
    hexagon_boundaries = []
    if gdf is None:
        gdf = utils.read_geo_dataset(filename=filename, path=path)
    gdf = gdf.explode(column="geometry").reset_index()
    for polygon in gdf.geometry:
        hexagons = h3.polyfill(polygon.__geo_interface__, res=res, geo_json_conformant=True)
        for hexagon in hexagons:
            hexagon_index.append(hexagon)
            hexagon_boundaries.append(Polygon(h3.h3_to_geo_boundary(hexagon, True)))

    # check if all centroids are in a hexagon
    hex_centroids = get_hexagon_vect(gdf.geometry.centroid.y, gdf.geometry.centroid.x, res=res)
    for hex_centroid in hex_centroids:
        if hex_centroid not in hexagon_index:
            hexagon_index.append(hex_centroid)
            hexagon_boundaries.append(Polygon(h3.h3_to_geo_boundary(hex_centroid, True)))

    hex_gdf = gpd.GeoDataFrame(
        {"hex": hexagon_index, "geometry": hexagon_boundaries}, crs="EPSG:4326"
    )
    hex_gdf = hex_gdf.drop_duplicates("hex")


    if save is True:
        if save_name is None:
            save_name = f"Hex{res}.geosjon"
        utils.write_geo_dataset(hex_gdf, filename=save_name)
    return hex_gdf
