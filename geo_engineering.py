import random
from shapely.geometry import Point


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
    df['Pickup Community Area'] = df['Pickup Community Area'].astype(int)
    df['Dropoff Community Area'] = df['Dropoff Community Area'].astype(int)
    gdf['area_numbe'] = gdf['area_numbe'].astype(int)
    gdf = gdf[['area_numbe', 'geometry']]
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
    merged_df = df.merge(gdf, how="left", validate='m:1', left_on='Pickup Community Area', right_on='area_numbe')
    merged_df = merged_df.merge(gdf, how="left", validate='m:1', left_on='Dropoff Community Area',
                                right_on='area_numbe', suffixes=("_pickup", "_dropoff"))
    # Casting pandas Dataframe to GeoPandas Dataframe
    merged_gdf = geopandas.GeoDataFrame(merged_df, geometry='geometry_pickup')
    merged_gdf.drop(columns=['area_numbe_dropoff', 'area_numbe_pickup'], inplace=True)
    return merged_gdf


def _generate_random_points_in_polygon(polygon):
    """
    This function generates a random point in a given polygon
    ----------------------------------------------
    :param
        polygon(shapely.geometry.polygon): the given polygon within which a random point should be generated
    :return
        shapely.Point: The created random point inside the given polygon
    """
    min_x, min_y, max_x, max_y = polygon.bounds
    while True:
        random_point = Point(random.uniform(float(min_x), float(max_x)), random.uniform(float(min_y), float(max_y)))
        # Check if the generated point is inside the polygon
        if polygon.contains(random_point):
            return random_point


def generate_random_pickup_dropoff_loc(df, gdf):
    """
    This is a wrapper function that merges the geo information from the community areas into the original data frame
    and creates random pickup and drop-off points for each trip in the given data set
    ----------------------------------------------
    :param
        df(pandas.DataFrame): Given data frame
        gdf(geoPandas.GeoDataFrame): given geo data frame
    :return
        geoPandas.GeoDataFrame: The merged geo data frame with random pickup/drop-off locations
    """
    df, gdf = _prepare_for_merge(df, gdf)
    merged_gdf = _merge_geo_information(df, gdf)

    merged_gdf['pickup location'] = merged_gdf.apply(
        lambda row: _generate_random_points_in_polygon(row['geometry_pickup']), axis=1)
    merged_gdf['dropoff location'] = merged_gdf.apply(
        lambda row: _generate_random_points_in_polygon(row['geometry_dropoff']), axis=1)
    return merged_gdf

