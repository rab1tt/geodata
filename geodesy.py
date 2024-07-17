import geopandas
import contextily
import geopandas
import math
import numpy
from shapely import geometry
from functools import lru_cache
from geopy.distance import distance


class Grid(geopandas.geodataframe.GeoDataFrame):

    geodataframe = None
    cell_width_degrees = None
    cell_height_degrees = None
    buffer_indices = None
    origin = None

    def __init__(
        self,
        min_latitude,
        min_longitude,
        max_latitude,
        max_longitude,
        max_side_length_meter,
        buffer_size_cells,
    ):

        self.origin = (min_latitude, min_longitude)

        distance_lon_max = distance(
            (min_latitude, min_longitude), (min_latitude, max_longitude)
        ).m
        n_x = math.ceil(distance_lon_max / max_side_length_meter) + 1
        distance_lat_max = distance(
        (min_latitude, min_longitude), (max_latitude, min_longitude)
        ).m
        n_y = math.ceil(distance_lat_max / max_side_length_meter) + 1
        X = numpy.linspace(min_longitude, max_longitude, n_x, endpoint=True)
        Y = numpy.linspace(min_latitude, max_latitude, n_y, endpoint=True)

        self.cell_width_degrees = X[1] - X[0]
        self.cell_height_degrees = Y[1] - Y[0]

        self.buffer_indices = [
            (x, y)
            for x in range(-buffer_size_cells, buffer_size_cells)
            for y in range(-buffer_size_cells, buffer_size_cells)
        ]

        self.geodataframe = (
            geopandas.GeoDataFrame(Grid.generate_grid(X, Y))
            .set_index(["i", "j"])
            .set_crs(epsg=4326)
        )

    @staticmethod
    def generate_grid(X, Y):
        for i, x in enumerate(X[:-1]):
            for j, y in enumerate(Y[:-1]):
                yield (
                    {
                        "geometry": geometry.Polygon(
                            [(x, y), (X[i + 1], y), (X[i + 1], Y[j + 1]), (x, Y[j + 1])]
                        ),
                        "i": i,
                        "j": j,
                    }
                )

    def index_of(self, latitude, longitude):
        j = int((latitude - self.origin[0]) / self.cell_height_degrees)
        i = int((longitude - self.origin[1]) / self.cell_width_degrees)
        return (i, j)

    def valid_indices_for_neighbourhood(self, center_index):
        index = self.geodataframe.index
        (I, J) = index.max()
        for candidate in self.buffer_indices:
            i = candidate[0] + center_index[0]
            j = candidate[1] + center_index[1]
            if i >= 0 and i <= I and j >= 0 and j <= J:
                yield (i, j)

    def window(self, latitude, longitude, max_distance_meters):
        center_index = self.index_of(latitude, longitude)
        indices = list(self.valid_indices_for_neighbourhood(center_index))
        window = self.geodataframe.loc[indices]
        centroids = window.to_crs(window.estimate_utm_crs()).centroid.to_crs(epsg=4326)
        for (i, j), distance in centroids.apply(
            lambda _: distance((latitude, longitude),( _.y, _.x)).m
        ).items():
            if distance < max_distance_meters:
                yield (i, j, distance)


class DynamicGrid(geopandas.geodataframe.GeoDataFrame):

    geodataframe = None
    cell_width_degrees = None
    cell_height_degrees = None
    I = None
    J = None
    origin = None
    cell_at = None

    def __init__(
        self,
        min_latitude,
        min_longitude,
        max_latitude,
        max_longitude,
        max_side_length_meter,
    ):

        self.origin = (min_latitude, min_longitude)

        distance_lon_max = distance(
            (min_latitude, min_longitude), (min_latitude, max_longitude)
        ).m
        n_x = math.ceil(distance_lon_max / max_side_length_meter)
        distance_lat_max = distance(
            (min_latitude, min_longitude), (max_latitude, min_longitude)
        ).m
        n_y = math.ceil(distance_lat_max / max_side_length_meter)

        self.cell_width_degrees = (max_longitude - min_longitude) / (n_x)
        self.cell_height_degrees = (max_latitude - min_latitude) / (n_y)
        (self.I, self.J) = self.index_of(max_latitude, max_longitude)
        
        self.cell_at = lru_cache(maxsize=1000)(self.__cell_at)

    def index_of(self, latitude, longitude):
        j = int((latitude - self.origin[0]) / self.cell_height_degrees)
        i = int((longitude - self.origin[1]) / self.cell_width_degrees)
        return (i, j)

    def __cell_at(self, i, j):
        # workaround to ensure proper garbage collection cf. https://stackoverflow.com/a/68550238
        x = self.origin[1] + i * self.cell_width_degrees
        y = self.origin[0] + j * self.cell_height_degrees
        return {
            "geometry": geometry.Polygon(
                [
                    (x, y),
                    (x + self.cell_width_degrees, y),
                    (x + self.cell_width_degrees, y + self.cell_height_degrees),
                    (x, y + self.cell_height_degrees),
                ]
            ),
            "i": i,
            "j": j,
        }

    def window(self, latitude, longitude, max_distance_meters):
        (i, j) = self.index_of(latitude, longitude)
        center_cell = self.cell_at(i, j)
        cell_above = self.cell_at(i, j + 1)
        cell_right = self.cell_at(i + 1, j)
        boudaries_center = center_cell["geometry"].exterior.coords.xy
        boundaries_above = cell_above["geometry"].exterior.coords.xy
        boundaries_right = cell_right["geometry"].exterior.coords.xy
        d_y = distance(
            (boudaries_center[1][0],
            boudaries_center[0][0]),
            (boundaries_above[1][0],
            boundaries_above[0][0]),
        ).m
        d_x = distance(
            (boudaries_center[1][0],
            boudaries_center[0][0]),
            (boundaries_right[1][0],
            boundaries_right[0][0]),
        ).m
        d_i = math.ceil(max_distance_meters / d_x)
        d_j = math.ceil(max_distance_meters / d_y)
        window = (
            geopandas.GeoDataFrame(
                self.cell_at(x, y)
                for x in range(max(0, i - d_i), min(self.I, i + d_i))
                for y in range(max(0, j - d_j), min(self.J, j + d_j))
            )
            .set_index(["i", "j"])
            .set_crs(epsg=4326)
        )
        centroids = window.to_crs(window.estimate_utm_crs()).centroid.to_crs(epsg=4326)
        selection = geopandas.GeoDataFrame(
            {"i": i, "j": j, "distance": distance}
            for ((i, j), distance) in centroids.apply(
                lambda _: distance((latitude, longitude), (_.y, _.x)).m
            ).items()
            if distance < max_distance_meters
        ).set_index((["i", "j"]))
        return window.join(selection, how="inner")
