from h3.unstable import vect
import h3.api.numpy_int as h3
from shapely.geometry import Polygon
import geopandas as gpd
import utils
from geopandas import GeoDataFrame


def get_hexagon(lat, lng, resolution=9):
    hexagon = h3.geo_to_h3(lat, lng, resolution)
    return hexagon


def get_hexagon_vect(lat, lng, resolution=9):
    column = vect.geo_to_h3(lat, lng, resolution)
    return column


def census_tract_to_hexagon(
        geometry=None,
        res=9,
        filename="Boundaries - Census Tracts - 2010.geojson",
        path=utils.get_data_path(),
        save=False,
        save_name = None
):
    hexagon_index = []
    hexagon_boundaries = []
    if geometry is None:
        gdf = utils.read_geo_dataset(filename=filename, path=path)
        geometry = gdf["geometry"]
    for polygon in geometry:
        polygon = list(polygon)[0]
        hexagons = h3.polyfill(polygon.__geo_interface__, res=res, geo_json_conformant=True)
        for hexagon in hexagons:
            hexagon_index.append(hexagon)
            hexagon_boundaries.append(Polygon(h3.h3_to_geo_boundary(hexagon, True)))

    hex_gdf = gpd.GeoDataFrame({"hex": hexagon_index, "geometry": hexagon_boundaries}, crs="EPSG:4326")
    hex_gdf = hex_gdf.drop_duplicates("hex")
    if save is True:
        if save_name is None:
            save_name = f"Hex{res}.geosjon"
        utils.write_geo_dataset(hex_gdf, filename=save_name)
    return hex_gdf


# Slower than census_tract_to_hexagon
def polygon_to_hexagon(geometry, res=9):
    hex_gdf = gpd.GeoDataFrame(columns=["hex", "geometry"], crs="EPSG:4326")
    hex_gdf["geometry"] = geometry
    hex_gdf["hex"] = hex_gdf.apply(lambda x: h3.polyfill(x["geometry"].__geo_interface__,
                                                         res=res,
                                                         geo_json_conformant=True), axis=1)
    hex_gdf.drop(columns=["geometry"], inplace=True)
    hex_gdf = hex_gdf.explode("hex", ignore_index=True)
    hex_gdf.dropna(inplace=True)
    hex_gdf = hex_gdf.drop_duplicates("hex")
    hex_gdf["geometry"] = hex_gdf.apply(lambda x: Polygon(h3.h3_to_geo_boundary(x["hex"], True)), axis=1)
    return hex_gdf
