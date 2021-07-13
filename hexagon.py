from h3.unstable import vect
import h3.api.numpy_int as h3
from shapely.geometry import Polygon
import geopandas as gpd
from geopandas import GeoDataFrame


def get_hexagon(lat, lng, resolution=8):
    hexagon = h3.geo_to_h3(lat, lng, resolution)
    return hexagon


def get_hexagon_vect(lat, lng, res=8):
    column = vect.geo_to_h3(lat, lng, res)
    return column


def get_poly(multi_poly):
    poly = list(multi_poly)
    return (poly[0])


def census_tract_to_hexagon(geometry, res=8):
    hex_code = []
    polygon_list = []
    for polygon in geometry:
        hex_array = h3.polyfill(polygon.__geo_interface__, res=res, geo_json_conformant=True)
        for hex in hex_array:
            hex_code.append(hex)
            polygon_list.append(Polygon(h3.h3_to_geo_boundary(hex, True)))

    hex_gdf = gpd.GeoDataFrame({"hex": hex_code, "geometry": polygon_list}, crs="EPSG:3857")
    hex_gdf = hex_gdf.drop_duplicates("hex")
    return hex_gdf
