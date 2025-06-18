import rioxarray as rx
import geopandas as gp
import pyproj
import pandas as pd
import polars as pl
import numpy as np
import pycountry
import shapely as shp
import json
import urllib.request
from pathlib import Path
from xarray import Dataset, DataArray
from typing import Optional

from emu_renewal.inputs import DATA_PATH, RAW_MOB_PATH

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


def raster_to_polydf(
    raster_ds: DataArray, data_name: str, out_type: shp.GeometryType = shp.GeometryType.POINT
) -> gp.GeoDataFrame:
    """
    Convert a raster dataset of regularly spaced
    coordinates into polygons representing the polygons
    that would have each coordinate as their centroid.

    Args:
        raster_ds: The rasterised dataset
        data_name: Name for the data column

    Returns:
        The square polygons within the geopandas format
    """
    square_size = (raster_ds.coords["x"][1] - raster_ds.coords["x"][0]).data
    buffer = square_size * 0.5
    geoms = []
    nodata_mask = raster_ds.rio.nodata

    data = raster_ds.data[0]
    n_valid = (data != nodata_mask).sum()
    out_data = np.empty(n_valid)
    valid_idx = 0

    for ix, x in enumerate(raster_ds.coords["x"].data):
        for iy, y in enumerate(raster_ds.coords["y"].data):
            cell_data = data[iy, ix]
            if cell_data != nodata_mask:
                if out_type == shp.GeometryType.POLYGON:
                    geoms.append(
                        shp.Polygon(
                            [
                                (x - buffer, y - buffer),
                                (x + buffer, y - buffer),
                                (x + buffer, y + buffer),
                                (x - buffer, y + buffer),
                            ]
                        )
                    )
                elif out_type == shp.GeometryType.POINT:
                    geoms.append(shp.Point((x, y)))
                else:
                    raise Exception(f"Invalid output geometry type {out_type}")
                out_data[valid_idx] = cell_data
                valid_idx += 1

    data = data.flatten()
    return gp.GeoDataFrame({data_name: out_data}, geometry=geoms)


def retag_poly_revision(poly_id, new_rev=1):
    parts = poly_id.split("_")
    return f"{parts[0]}_{new_rev}"


def retag_gidcol(poly_df, gadm_level, revision=1) -> gp.GeoDataFrame:
    pdf = poly_df.copy()
    gid_col = f"GID_{gadm_level}"
    pdf[gid_col] = [retag_poly_revision(pid) for pid in pdf[gid_col]]
    return pdf


def polyids_from_gadm(iso3: str, gadm_level: int, force_rev: int = 1) -> list[str]:
    if iso3 == "USA":
        # Facebook uses FIPS county level classifications for the USA, rather the GADM sets
        # Obtained from
        # https://community.esri.com/t5/arcgis-enterprise-portal-questions/where-can-i-find-a-shapefile-with-all-us-counties/td-p/307592
        dest = DATA_PATH / "population/gadm_input_json/UScounties.zip"
        poly_df = gp.read_file(dest)
        return list(poly_df["FIPS"])

    source = f"https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{iso3}_{gadm_level}.json.zip"
    dest = DATA_PATH / f"population/gadm_input_json/gadm41_{iso3}_{gadm_level}.json.zip"
    if not dest.exists():
        urllib.request.urlretrieve(source, dest)

    poly_df = gp.read_file(dest)
    pid_col = f"GID_{gadm_level}"

    if force_rev is not None:
        poly_df = retag_gidcol(poly_df, gadm_level)

    return list(poly_df[pid_col])


def polydf_from_gadm(iso3: str, gadm_level: int, force_rev: int = 1):
    if iso3 == "USA":
        # Facebook uses FIPS county level classifications for the USA, rather the GADM sets
        dest = DATA_PATH / "population/gadm_input_json/UScounties.zip"
        poly_df = gp.read_file(dest)
        return poly_df

    source = f"https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{iso3}_{gadm_level}.json.zip"
    dest = DATA_PATH / f"population/gadm_input_json/gadm41_{iso3}_{gadm_level}.json.zip"
    if not dest.exists():
        urllib.request.urlretrieve(source, dest)

    poly_df = gp.read_file(dest)

    if force_rev is not None:
        poly_df = retag_gidcol(poly_df, gadm_level)

    return poly_df


def population_from_gadm(
    iso3: str,
    gadm_level: int,
    pop_ds: Dataset,
    force_rebuild=False,
    write_json=True,
    poly_ids=None,
    geom_method=shp.GeometryType.POINT,
    process_gadm_func=None,
) -> dict[str, float]:

    if poly_ids is None:
        poly_ids = []

    # Use cached json if available;
    if iso3 == "USA":
        json_pop_path = DATA_PATH / f"population/gadm_est/{iso3}.json"
    else:
        json_pop_path = DATA_PATH / f"population/gadm_est/{iso3}_{gadm_level}.json"
    if json_pop_path.exists() and not force_rebuild:
        logger.info(f"Loading existing population from {json_pop_path}")
        return json.load(open(json_pop_path, "r"))

    # GADM administrative boundaries as obtained from:
    # https://gadm.org/download_country.html

    # These files are directly downloadable as of 21/01/2025 via
    # https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{iso3}_{gadm_level}.json.zip
    # Download the appropriate GADM boundaries json (or use cached if it exists)
    poly_df = polydf_from_gadm(iso3, gadm_level)

    # source = f"https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{iso3}_{gadm_level}.json.zip"
    # dest = DATA_PATH / f"population/gadm_input_json/gadm41_{iso3}_{gadm_level}.json.zip"
    # if not dest.exists():
    #     urllib.request.urlretrieve(source, dest)

    # poly_df = gp.read_file(dest)

    if process_gadm_func is not None:
        poly_df = process_gadm_func(poly_df)

    # Set up the dimensions of a population cell
    pix_dim = float((pop_ds.coords["x"][1] - pop_ds.coords["x"][0]).data)

    pop_dict = {}

    if iso3 == "USA":
        poly_col = "FIPS"
    else:
        poly_col = f"GID_{gadm_level}"

    # Loop over the polygons relevant to the country
    for i_poly, poly_id in enumerate(poly_df[poly_col]):

        if poly_id in poly_ids:

            # Get the relevant gridded population data, ensuring correct projection
            poly_bounds = np.array(
                poly_df.loc[i_poly, "geometry"].bounds
            )  # Furthest reach of polygon in each direction
            expanded_bounds = poly_bounds + np.array(
                ([-pix_dim, -pix_dim, pix_dim, pix_dim])
            )  # Extend bounds to span every population data coordinate that could be relevant
            pop_clipped = pop_ds.rio.clip_box(*expanded_bounds)
            pop_df = raster_to_polydf(pop_clipped, "population", geom_method)
            pop_df = pop_df.set_crs(
                pyproj.CRS.from_wkt(pop_ds.rio.crs.to_wkt())
            )  # Reconcile projection of polygon and population data

            # Find population based on query
            # This is _dramatically_ faster than previous methods, but doesn't do the fancy
            # intersection supported by overlay etc
            # Really only makes sense for GeometryType.POINT
            if geom_method != shp.GeometryType.POINT:
                raise Exception(f"Invalid geom_method {geom_method}")

            qres = pop_df.sindex.query(poly_df.iloc[i_poly : i_poly + 1].geometry, "contains")

            pop_val = pop_df.loc[qres[1]].population.sum()

            # +++ Old code for intersection methods;
            # isect = gp.overlay(pop_df, poly_df.iloc[i_poly : i_poly + 1], keep_geom_type=False)
            # We may want to reinstate this at some point
            # if geom_method == shp.GeometryType.POLYGON:
            #    pop_val = float((isect.area / pix_dim**2 * isect.population).sum())
            # elif geom_method == shp.GeometryType.POINT:
            #    pop_val = float(isect.population.sum())
            # else:
            #    raise Exception(f"Invalid geom_method {geom_method}")
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
) -> pd.DataFrame:
    # Calculate weighted average over patches
    country_mob_series = pd.Series(0.0, index=country_mobility["ds"].unique(), dtype=float)
    total_pop = 0.0
    regional_df = pd.DataFrame(index=country_mobility["ds"].unique(), dtype=float)
    for pid in country_pop_data:
        if pid in country_mobility["polygon_id"]:
            cur_data = country_mobility.filter(pl.col("polygon_id") == pid)
            mob_series = pd.Series(index=cur_data["ds"].unique(), data=cur_data[data_col]).dropna()
            region_pop = country_pop_data[pid]
            regional_df[pid] = mob_series
            country_mob_series += (
                mob_series.reindex(country_mob_series.index, method="nearest") * region_pop
            )
            total_pop += region_pop
        else:
            logger.warn(f"PID {pid} not contained in Facebook data for {iso3}/{gadm_level}")
    weighted_country_mob = country_mob_series / total_pop
    return regional_df, weighted_country_mob


def infer_gadm_level(country_mobility):
    poly_id_ref = country_mobility["polygon_id"][0]
    return len(poly_id_ref.split(".")) - 1


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

    def build_mobility(
        self,
        iso3: str,
        gadm_level: Optional[int] = None,
        write_csv=True,
        geom_method=shp.GeometryType.POINT,
        force_rebuild=False,
        process_gadm_func=None,
    ) -> pd.Series:
        """Build a population-weighted mobility series for the given country (or load cached version if already present)

        Args:
            iso3: ISO3 country code
            gadm_level: GADM level (1 or 2); if not supplied, will be inferred from data
            write_csv: Write resulting data to CSV.
            geom_method: Type of shapely geometry to intersect (only POINT supported at present)
            force_rebuild: Rebuild existing data
            process_gadm_function: Callable to remap gadm series (unused)

        Returns:
            Weighted timeseries

        Notes
        -----
        For each geographic region included in the Facebook data, 
        we estimated a population by intersecting polygons
        with the centroid of the population data grids, 
        and then weighted the resulting series 
        by these calculated populations.
        For the small proportion of 
        (low-population) timeseries that had missing data, 
        we imputed population estimates
        by nearest neighbour interpolation.
        In general, these series were found 
        to have a negligible contribution to the final outputs.
        """
        mobility_csv_path = DATA_PATH / f"mobility/{iso3}_fbmob_data.csv"

        if iso3 not in self.fb_data["country"]:
            raise Exception(f"No Facebook data for {iso3}")

        country_mobility = self.fb_data.filter(pl.col("country") == iso3)

        # USA doesn't use GADM - we hack in a county level FIPS shapefile for this elsewhere
        if gadm_level is None and iso3 != "USA":
            gadm_level = infer_gadm_level(country_mobility)
            logger.info(f"Inferred GADM level {gadm_level} for {iso3}")

        gpids = set(polyids_from_gadm(iso3, gadm_level))
        fbpids = set(country_mobility["polygon_id"].unique())

        if len(gpids.intersection(fbpids)) == 0:
            logger.error(f"No matching polygons found for {iso3}")
            raise Exception(f"No matching polygons found for {iso3}")

        pop_dict = population_from_gadm(
            iso3,
            gadm_level,
            self.pop_ds,
            poly_ids=fbpids,  # fbpid_updated
            force_rebuild=force_rebuild,
            geom_method=geom_method,
            process_gadm_func=process_gadm_func,
        )
        pop_info = (
            f"Total population for {iso3} is {round(sum(pop_dict.values()) / 1e6, 3)} million"
        )
        logger.info(pop_info)

        regional_df, weighted_country_mob = mobility_from_population(
            country_mobility, iso3, gadm_level, pop_dict
        )

        if write_csv:
            weighted_country_mob.to_csv(mobility_csv_path)

        return weighted_country_mob
