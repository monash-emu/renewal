import rioxarray as rx
import geopandas as gp
import pyproj
import pandas as pd
import polars as pl
import numpy as np
import pycountry
import json
import urllib.request
from pathlib import Path
from xarray import Dataset

from emu_renewal.inputs import DATA_PATH, RAW_MOB_PATH, raster_to_polydf

# Stop GeoPandas warning about area intersection calculations
import warnings

warnings.filterwarnings("ignore", "Geometry is in a geographic CRS")

import logging

logging.basicConfig()
logger = logging.getLogger("mobility")
logger.setLevel(logging.INFO)

# Gridded Population of the World 30sec 2020 dataset
# https://www.earthdata.nasa.gov/data/projects/gpw
DEFAULT_POP_RASTER_DS_PATH = DATA_PATH / "population/gpw_v4_population_count_rev11_2020_30_sec.tif"
# Alternative datasets exist, for example;
# https://github.com/lulingliu/GlobPOP


def population_from_gadm(
    iso3: str, gadm_level: int, pop_ds: Dataset, write_json=True
) -> dict[str, float]:

    # Use cached json if available;
    json_pop_path = DATA_PATH / f"population/gadm_est/{iso3}_{gadm_level}.json"
    if json_pop_path.exists():
        logger.info(f"Loading existing population from {json_pop_path}")
        return json.load(open(json_pop_path, "r"))

    # GADM administrative boundaries as obtained from:
    # https://gadm.org/download_country.html

    # These files are directly downloadable as of 21/01/2025 via
    # https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{iso3}_{gadm_level}.json.zip
    # Download the appropriate GADM boundaries json (or use cached if it exists)
    source = f"https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{iso3}_{gadm_level}.json.zip"
    dest = DATA_PATH / f"population/gadm_input_json/gadm41_{iso3}_{gadm_level}.json.zip"
    if not dest.exists():
        urllib.request.urlretrieve(source, dest)

    poly_df = gp.read_file(dest)

    # Set up the dimensions of a population cell
    pix_dim = float((pop_ds.coords["x"][1] - pop_ds.coords["x"][0]).data)

    pop_dict = {}

    # Loop over the polygons relevant to the country
    for i_poly, poly_id in enumerate(poly_df[f"GID_{gadm_level}"]):

        # Get the relevant gridded population data, ensuring correct projection
        poly_bounds = np.array(
            poly_df.loc[i_poly, "geometry"].bounds
        )  # Furthest reach of polygon in each direction
        expanded_bounds = poly_bounds + np.array(
            ([-pix_dim, -pix_dim, pix_dim, pix_dim])
        )  # Extend bounds to span every population data coordinate that could be relevant
        pop_clipped = pop_ds.rio.clip_box(*expanded_bounds)
        pop_df = raster_to_polydf(pop_clipped, "population")
        pop_df = pop_df.set_crs(
            pyproj.CRS.from_wkt(pop_ds.rio.crs.to_wkt())
        )  # Reconcile projection of polygon and population data

        # Find population based on intersections
        isect = gp.overlay(pop_df, poly_df.iloc[i_poly : i_poly + 1], keep_geom_type=False)
        pop_val = float((isect.area / pix_dim**2 * isect.population).sum())
        pop_dict[poly_id] = pop_val
        logger.info(f"{poly_id} has population of {round(pop_val / 1e3)} thousand")

    if write_json:
        json.dump(pop_dict, open(DATA_PATH / f"population/gadm_est/{iso3}_{gadm_level}.json", "w"))

    return pop_dict


def mobility_from_population(
    country_mobility: pl.DataFrame,
    iso3: str,
    gadm_level: int,
    country_pop_data: dict[str, float],
    data_col: str = "all_day_bing_tiles_visited_relative_change",
) -> pd.Series:
    # Calculate weighted average over patches
    country_mob_series = pd.Series(0.0, index=country_mobility["ds"].unique(), dtype=float)
    total_pop = 0.0
    for pid in country_pop_data:
        if pid in country_mobility["polygon_id"]:
            cur_data = country_mobility.filter(pl.col("polygon_id") == pid)
            mob_series = pd.Series(index=cur_data["ds"].unique(), data=cur_data[data_col]).dropna()
            region_pop = country_pop_data[pid]
            country_mob_series += (
                mob_series.reindex(country_mob_series.index, method="nearest") * region_pop
            )
            total_pop += region_pop
    weighted_country_mob = country_mob_series / total_pop
    return weighted_country_mob


class FacebookMobilityBuilder:
    """Builds weighted mobility Series from GADM and gridded population datasets, and handles
    standard I/O for the emu_renewal use case
    """

    pop_ds: Dataset
    fb_data: pl.DataFrame

    def __init__(self, raster_pop_path: Path = DEFAULT_POP_RASTER_DS_PATH):
        """Initialize using the specified raster population data

        Args:
            raster_pop_path: Path to gridded raster population data. Defaults to DEFAULT_POP_RASTER_DS_PATH.
        """
        self.pop_ds = rx.open_rasterio(raster_pop_path)
        # Compile Facebook data
        data20 = pl.read_csv(
            RAW_MOB_PATH / "movement-range-data-2020-03-01--2020-12-31.txt",
            separator="\t",
            schema_overrides={"ds": pl.datatypes.Date},
        )
        data21_22 = pl.read_csv(
            RAW_MOB_PATH / "movement-range-2022-05-22.txt",
            separator="\t",
            schema_overrides={"ds": pl.datatypes.Date},
        )
        self.fb_data = pl.concat([data20, data21_22])

    def build_mobility(self, iso3, gadm_level, write_csv=True) -> pd.DataFrame:
        mobility_csv_path = DATA_PATH / f"mobility/{iso3}_fbmob_data.csv"

        pop_dict = population_from_gadm(iso3, gadm_level, self.pop_ds)
        pop_info = (
            f"Total population for {iso3} is {round(sum(pop_dict.values()) / 1e6, 3)} million"
        )
        logger.info(pop_info)

        country_mobility = self.fb_data.filter(pl.col("country") == iso3)

        weighted_country_mob = mobility_from_population(
            country_mobility, iso3, gadm_level, pop_dict
        )

        if write_csv:
            weighted_country_mob.to_csv(mobility_csv_path)

        return weighted_country_mob
