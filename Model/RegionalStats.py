import geopandas as gpd
import xarray as xr


class RegionalStats:
    def _get_geodataframe(self, url_opendap):
        ds = xr.open_dataset(url_opendap)
        df_full = ds.to_dataframe()
        df_full = df_full.reset_index()
        try:
            df_full_gdf = gpd.GeoDataFrame(
                df_full,
                geometry=gpd.points_from_xy(df_full.lon, df_full.lat),
                crs=4326,
            )
        except:
            df_full_gdf = gpd.GeoDataFrame(
                df_full,
                geometry=gpd.points_from_xy(df_full.longitude, df_full.latitude),
                crs=4326,
            )
        return df_full_gdf

    def _get_zone_to_clip(self, geojson):
        gdf_input = gpd.read_file(geojson)
        return gdf_input

    def get_stats(self, inputs):
        geodataframe = self._get_geodataframe(inputs["url_opendap_data"])
        clip_zone = self._get_zone_to_clip(inputs["geojson_to_clip"])
        mask = gpd.clip(geodataframe, clip_zone)
        stats = {
            "max": mask[inputs["variable_name"]].values.max(),
            "min": mask[inputs["variable_name"]].values.min(),
            "mean": mask[inputs["variable_name"]].values.mean(),
        }
        return stats

    def get_categorized_values(self, inputs):
        geodataframe = self._get_geodataframe(inputs["url_opendap_data"])
        clip_zone = self._get_zone_to_clip(inputs["geojson_to_clip"])
        mask = gpd.clip(geodataframe, clip_zone)
        values_length = len(mask[inputs["variable_name"]].values)
        values = mask[inputs["variable_name"]].groupby(mask[inputs["variable_name"]].values)
        results = []
        percent = 0
        for group in values.groups:
            values_group = len(values.indices[group])
            results.append(
                {
                    "category": int(group),
                    "percent": (values_group * 100) / values_length,
                }
            )
            percent += values_group * 100 / values_length

        results.append({"category": "No data", "percent": 100 - percent})
        return results
