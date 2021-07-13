from h3.unstable import vect
import h3.api.numpy_int as h3


def get_hexagon(lat, lng, resolution=8):
    hexagon = h3.geo_to_h3(lat, lng, resolution)
    return hexagon


def get_hexagon_vect(lat, lng, res=8):
    column = vect.geo_to_h3(lat, lng, res)
    return column

