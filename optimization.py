import random
from shapely.geometry import Point
from shapely import geometry
import folium


def add_optimization_results_to_data_set(df, geometry_map, hexagon_map, optimization_problem):
    """
    This function adds the optimization results to a data set.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that is used to aggregate the results
        geometry_map(pd.DataFrame): Hexagon map
        hexagon_map(pd.DataFrame): Hexagon geometry map
        optimization_problem(pulp.pulp.LpProblem): The solved optimization problem
    :return
        pd.DataFrame: Optimization results
    """
    df['Level 2 Station'] = 0
    df['Level 3 Station'] = 0
    for v in optimization_problem.variables():
        if v.varValue > 0:
            if v.name.startswith('Gamma'):
                index = int(v.name[6:])
                df.at[index, "Level 2 Station"] = v.varValue
            else:
                index = int(v.name[6:])
                df.at[index, "Level 3 Station"] = v.varValue

    df.reset_index(inplace=True)
    df = df.merge(hexagon_map, how='left', on='hexagon_number')
    geometry_map['hex_6_pickup'] = geometry_map['hex_6_pickup'].astype(str)
    df = df.merge(geometry_map, how='left', on='hex_6_pickup')
    return df


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


def show_stations_on_map(df, geo_json):
    """
    This function creates a folium choropleth for charging stations.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that is used to make the choropleth.
        geo_json(geojson): The given geojson to use in choropleth.
    :return:
        folium.Choropleth: The created choropleth.
    """
    CHICAGO_COORD = [41.86364, -87.72645]
    base_map = folium.Map(location=CHICAGO_COORD, tiles="cartodbpositron")
    folium.Choropleth(
        geo_data=geo_json,
        name="choropleth",
        fill_color='gray',
        fill_opacity=0.1
    ).add_to(base_map)

    for index, row in df.iterrows():
        if row['Level 2 Station'] > 0:
            number_of_stations = row['Level 2 Station']
            for i in range(number_of_stations):
                p1, p2, p3, p4, p5, p6, p7 = row['hex_6_geometry_pickup'].get('coordinates')[0]
                vertices = [p1, p2, p3, p4, p5, p6, p7]
                poly = geometry.Polygon([[p[1], p[0]] for p in vertices])
                station_loc = _generate_random_points_in_polygon(poly)
                folium.Marker(
                    (station_loc.x, station_loc.y),
                    popup=f"Expected Daily Taxi Demand:{row['Expected Daily Taxi Demand']}",
                    tooltip="Level 2 Station", icon=folium.Icon(prefix='fa', color='green', icon='car')
                ).add_to(base_map)

        if row['Level 3 Station'] > 0:
            number_of_stations = row['Level 3 Station']
            for i in range(number_of_stations):
                p1, p2, p3, p4, p5, p6, p7 = row['hex_6_geometry_pickup'].get('coordinates')[0]
                vertices = [p1, p2, p3, p4, p5, p6, p7]
                poly = geometry.Polygon([[p[1], p[0]] for p in vertices])
                station_loc = _generate_random_points_in_polygon(poly)
                folium.Marker(
                    (station_loc.x, station_loc.y),
                    popup=f"Expected Daily Taxi Demand:{row['Expected Daily Taxi Demand']}",
                    tooltip="Level 3 Station", icon=folium.Icon(prefix='fa', color='red', icon='car')
                ).add_to(base_map)

    folium.TileLayer("cartodbdark_matter", name="dark mode", control=True).add_to(
        base_map
    )
    folium.TileLayer("openstreetmap", name="open street map", control=True).add_to(
        base_map
    )
    legend_2 = folium.FeatureGroup(name=f"<span style=color:{'green'}>{'Level 2 Station'}</span>")
    base_map.add_child(legend_2)
    legend_3 = folium.FeatureGroup(name=f"<span style=color:{'red'}>{'Level 3 Station'}</span>")
    base_map.add_child(legend_3)
    folium.map.LayerControl('topleft', collapsed=False).add_to(base_map)
    return base_map
