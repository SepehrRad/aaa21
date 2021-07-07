import h3


def get_hexagon(lat, lng, resolution=8):
    hexagon = h3.geo_to_h3(lat, lng, resolution)
    return hexagon
