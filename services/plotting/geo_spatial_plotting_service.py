from typing import Any, Sequence
from numpy import unique
from pandas import DataFrame
from folium import Map as FoliumMap, Marker as FoliumMarker

from geopy.geocoders.base import Geocoder
from folium.plugins import HeatMap, MarkerCluster
from IPython.display import display as ipython_display


class GeoSpatialPlottingService:
    def __init__(
        self,
        geocoder: Geocoder,
        locations: Sequence[str] | None = None,
        custom_mappings: dict[str, str] | None = None,
    ) -> None:
        self.raw_locations = locations
        self.geocoder = geocoder
        self.world_map: FoliumMap
        self.map_dataframe: DataFrame
        self.custom_mappings = custom_mappings

    def _get_coordinates(
        self,
        location: str,
        address_chunk_index: int = -1,
    ) -> tuple[float, float] | tuple[None, None]:
        try:
            if address_chunk_index == -1:
                precise_location = location
            else:
                precise_location = location.split(",")[address_chunk_index].strip()

            if self.custom_mappings and precise_location in self.custom_mappings.keys():
                precise_location = self.custom_mappings[precise_location]

            coordinates: Any = self.geocoder.geocode(precise_location)
            return coordinates.latitude, coordinates.longitude

        except:
            address_chunk_index += 1
            if address_chunk_index == len(location.split(",")):
                return None, None
            return self._get_coordinates(location, address_chunk_index)

    def _initialize_map_dataframe(self):
        unique_locations: list[str] = list(unique(self.raw_locations))
        counts = [self.raw_locations.count(location) for location in unique_locations]
        latitudes, longitudes = zip(
            *(self._get_coordinates(location) for location in unique_locations)
        )
        self.map_dataframe = DataFrame(
            zip(unique_locations, counts, latitudes, longitudes),
            columns=["location", "count", "latitude", "longitude"],
        )
        self.map_dataframe.dropna(
            subset=["latitude", "longitude"],
            inplace=True,
            ignore_index=True,
        )

    def _initialize_map(self):
        highest_count_index: int = self.map_dataframe["count"].idxmax()
        self.world_map = FoliumMap(
            zoom_start=4,
            location=[
                self.map_dataframe["latitude"].iloc[highest_count_index],
                self.map_dataframe["longitude"].iloc[highest_count_index],
            ],
        )

    def _apply_heatmap(self):
        heat_data: list[list[Any]] = [
            [point[0], point[1], value]
            for point, value in zip(
                zip(self.map_dataframe["latitude"], self.map_dataframe["longitude"]),
                self.map_dataframe["count"],
            )
        ]
        HeatMap(heat_data).add_to(self.world_map)

    def _apply_markers(self):
        marker_cluster = MarkerCluster()
        for _, row in self.map_dataframe.iterrows():
            FoliumMarker(
                location=[row["latitude"], row["longitude"]],
                popup=f"{row['location']}: {row['count']}",
            ).add_to(marker_cluster)
        marker_cluster.add_to(self.world_map)

    def _configure_world_map(
        self,
        heatmap_applied: bool = True,
        markers_applied: bool = True,
    ):
        self._initialize_map_dataframe()
        self._initialize_map()
        if heatmap_applied:
            self._apply_heatmap()
        if markers_applied:
            self._apply_markers()
        return self.world_map

    def display_in_notebook(self, locations: Sequence[str] | None = None):
        if locations is not None:
            self.raw_locations = locations
        self._configure_world_map()
        ipython_display(self.world_map)
