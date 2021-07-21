import geopandas as gpd
import h3.api.numpy_int as h3
from geopandas import GeoDataFrame
from h3.unstable import vect
from shapely.geometry import Polygon

import utils


def get_hexagon(lat, lng, resolution=9):
    hexagon = h3.geo_to_h3(lat, lng, resolution)
    return hexagon


def get_hexagon_vect(lat, lng, resolution=9):
    column = vect.geo_to_h3(lat, lng, resolution)
    return column


def census_tract_to_hexagon(
        gdf=None,
        res=9,
        filename="Boundaries - City.geojson",
        path=utils.get_data_path(),
        save=False,
        save_name=None,
):
    hexagon_index = []
    hexagon_boundaries = []
    if gdf is None:
        gdf = utils.read_geo_dataset(filename=filename, path=path)
        geometry = gdf["geometry"]

    gdf = gdf.explode(column="geometry").reset_index()
    for polygon in gdf.geometry:
        hexagons = h3.polyfill(polygon.__geo_interface__, res=res, geo_json_conformant=True)
        for hexagon in hexagons:
            hexagon_index.append(hexagon)
            hexagon_boundaries.append(Polygon(h3.h3_to_geo_boundary(hexagon, True)))

    hex_gdf = gpd.GeoDataFrame(
        {"hex": hexagon_index, "geometry": hexagon_boundaries}, crs="EPSG:4326"
    )
    hex_gdf = hex_gdf.drop_duplicates("hex")
    if save is True:
        if save_name is None:
            save_name = f"Hex{res}.geosjon"
        utils.write_geo_dataset(hex_gdf, filename=save_name)
    return hex_gdf
