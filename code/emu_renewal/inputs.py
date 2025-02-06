import pandas as pd
import numpy as np
import shapely as shp
from pathlib import Path
import os
import pycountry
from typing import Tuple, List, Dict
from datetime import datetime, timedelta
import yaml as yml
import geopandas as gp
from numpyro import distributions as dist
from xarray import DataArray


BASE_PATH = Path(__file__).parent.parent.parent

OUTPUTS_PATH = BASE_PATH / "outputs"
DATA_PATH = BASE_PATH / "data"
RAW_MOB_PATH = DATA_PATH / "mobility_raw"  # Will not push these larger original files
VAR_MAP = {
    "ba1": "21K.Omicron",
    "ba2": "21L.Omicron",
    "ba5": "22B.Omicron",
    "eu1": "20A.EU1",
    "eu2": "20A.EU2",
    "alpha": "20I.Alpha.V1",
}
ANALYSIS_TYPES = [
    "no_mob",
    "google_nonresi_linear",
    "google_nonresi_square",
    "fb_linear",
    "fb_square",
]


def get_indicator_series_from_who_data(
    indicator: str,
    country: str,
) -> pd.Series:
    """Get WHO estimates for single indicator from the original raw data.

    Args:
        indicator: Name of the indicator
        country: Name of the country

    Returns:
        The data
    """
    who_data = pd.read_csv(DATA_PATH / "who/WHO-COVID-19-global-data_21_8_24.csv")
    iso2 = pycountry.countries.lookup(country).alpha_2
    select_data = who_data.loc[who_data["Country_code"] == iso2]
    select_data.index = pd.to_datetime(select_data["Date_reported"], format="%d/%m/%Y")
    return select_data[indicator].interpolate(method="linear").fillna(0.0)


def get_who_targets(
    country: str,
    analysis_start: datetime,
    analysis_end: datetime,
    init_duration: int,
    analysis_to_data_delay: int,
) -> Tuple[pd.Series]:
    """Get all the WHO indicators relevant to the country of interest.
    Indicators are reported weekly, targets remain weekly
    but initialisation incidence is converted to daily.

    Args:
        country: Name of the country of interest
        analysis_start: Start date of the analysis
        analysis_end: End date of the analysis
        init_duration: Duration of the initialisation period
        analysis_to_data_delay: Time from starting the analysis to comparing against targets

    Returns:
        Cases target, deaths target, initialisation incidence values
    """
    cases_data = get_indicator_series_from_who_data("New_cases", country)
    deaths_data = get_indicator_series_from_who_data("New_deaths", country)
    data_start = analysis_start + timedelta(analysis_to_data_delay)
    cases_target = cases_data.loc[data_start:analysis_end]
    deaths_target = deaths_data.loc[data_start:analysis_end]
    init_start = analysis_start - timedelta(init_duration)
    init_end = analysis_start - timedelta(1)
    init_smooth_period = 7.0  # To correct values after resampling from weekly to daily
    init_data = (
        cases_data.resample("D").asfreq().interpolate().loc[init_start:init_end]
        / init_smooth_period
    )
    return cases_target, deaths_target, init_data


def get_hosp_series_from_owid_data(
    indicator: str,
    iso2: str,
) -> pd.Series:
    """Get OWID hospitalisation-related estimates for single indicator
    from the original raw data.

    Args:
        indicator: Name of the indicator
        country: ISO2 code for the country

    Returns:
        The data
    """
    hosp = pd.read_csv(DATA_PATH / "owid/owid_hosp.csv", index_col="date")
    hosp.index = pd.to_datetime(hosp.index)
    country = pycountry.countries.lookup(iso2).name
    data = hosp[hosp["entity"] == country]
    return data.loc[data["indicator"] == indicator, "value"]


def get_hosp_target(
    country: str,
    analysis_start: datetime,
    analysis_end: datetime,
    analysis_to_data_delay: int,
    indicator: str,
) -> pd.Series:
    """Get hospitalisation target, the data for which
    comes from OWID because not available from WHO.
    Series is converted from daily to weekly to harmonise with WHO targets.

    Args:
        country: Name of the country of interest
        analysis_start: Start date of the analysis
        analysis_end: End date of the analysis
        analysis_to_data_delay: Time from starting the analysis to comparing against targets

    Returns:
        Hospital occupancy target
    """
    data_start = analysis_start + timedelta(analysis_to_data_delay)
    hosp_data = get_hosp_series_from_owid_data(indicator, country)
    return hosp_data[data_start: analysis_end: 7]


def get_var_country_data(
    var: str,
    iso2: str,
) -> pd.Series:
    """Get data for the number of isolates attributable to
    a particular variant in a certain country.

    Args:
        var: Nextclade name for the variant
        country: ISO2 code for the country

    Returns:
        The data
    """
    # Countries needing official name for Nextclade data
    offic_countries = ["CZ"]
    pycountry_obj = pycountry.countries.lookup(iso2)
    country_name = pycountry_obj.official_name if iso2 in offic_countries else pycountry_obj.name
    data = pd.read_json(DATA_PATH / f"nextclade/{var}.json")[country_name]
    dates = pd.to_datetime(data["week"])
    return pd.Series(data["cluster_sequences"], index=dates)


def get_multivars_country_data(
    country: str,
) -> pd.DataFrame:
    """Get data for multiple variants using mapping from
    our names for the variants to Nextclade

    Args:
        country: Name of the country

    Returns:
        The data
    """
    return pd.DataFrame({k: get_var_country_data(v, country) for k, v in VAR_MAP.items()})


def get_row_proportions(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Normalise the rows of a dataframe over its columns.

    Args:
        df: The input dataframe containing numeric values

    Returns:
        The result
    """
    return df.divide(df.sum(axis=1), axis=0).fillna(0.0)


def get_european_var_props(
    country: str,
    start_date: datetime,
    end_date: datetime,
    var_names: List[str],
) -> pd.Series:
    """Get the variant proportions applicable to the early waves
    of the European epidemics.

    Args:
        country: Name of the country of interest
        start_date: Start date for variant proportion comparisons
        end_date: End date for variant proportion comparisons

    Returns:
        Variant proportions data
    """
    data = get_multivars_country_data(country)
    data["eu"] = data["eu1"] + data["eu2"]
    select_data = data[var_names]
    select_props = get_row_proportions(select_data)
    return select_props.loc[(start_date < select_data.index) & (select_data.index < end_date), "eu"]


def process_raw_google_mobility(
    country: str,
) -> pd.DataFrame:
    """Load raw Google mobility data and process for storing.

    Args:
        country: Name of country of interest

    Returns:
        The data
    """
    years = range(2020, 2023)
    iso2 = pycountry.countries.get(name=country).alpha_2
    data_files = [
        pd.read_csv(RAW_MOB_PATH / f"{y}_{iso2}_Region_Mobility_Report.csv", index_col="date")
        for y in years
    ]
    all_data = pd.concat(data_files)
    all_data.index = pd.to_datetime(all_data.index)
    national_data = all_data.loc[pd.isna(all_data["sub_region_1"]) & pd.isna(all_data["metro_area"])]
    national_data = national_data[
        [c for c in national_data.columns if "change_from_baseline" in c]
    ]  # Extract the mobility columns
    national_data = national_data.rename(
        lambda c: c.replace("_percent_change_from_baseline", ""), axis=1
    )  # Simplify column naming
    national_data = 1.0 + national_data / 100.0  # Convert to relative change
    return national_data.sort_index()


def get_all_seroprev(
    lag: int = 14,
) -> pd.Series:
    """Get all the seroprevalence data,
    including calculating midpoint for survey date
    and lagging by 14 days.

    Args:
        lag: Days to lag for antibody development

    Returns:
        All SeroTracker data
    """
    data = pd.read_csv(DATA_PATH / "seroprevalence/serotracker.csv")
    data["start"] = pd.to_datetime(data["sampling_start_date"])
    data["end"] = pd.to_datetime(data["sampling_end_date"])
    data.index = (data["end"] - data["start"]) / 2 + data["start"] - timedelta(lag)
    data.index = data.index.normalize()
    return data.sort_index()


def get_filtered_seroprev(
    country: str,
    start: datetime,
    end: datetime,
) -> pd.Series:
    """Filter the SeroTracker data according
    to our choices about what constitutes good
    enough data for including in the calibration targets.

    Args:
        country: Name of the country of interest
        start: Start date of analysis
        end: End date of analysis

    Returns:
        Filtered data to use as target
    """
    data = get_all_seroprev()
    country = pycountry.countries.lookup(country).name
    country_filt = data["country"] == country
    time_filt = (start < data.index) & (data.index < end)
    nat_filt = data["estimate_grade"] == "National"
    type_filt = (data["subgroup_var"] == "Primary Estimate") & (
        data["is_unity_aligned"] == "Unity-Aligned"
    )
    data = data.loc[time_filt & country_filt & nat_filt & type_filt]
    return data["serum_pos_prevalence"]


def get_standard_priors() -> Dict[str, dist.Distribution]:
    """Load the priors from the yml and combine with
    standard hard-coded priors.

    Returns:
        The prior distributions
    """
    loaded_priors = yml.safe_load(open(DATA_PATH / "config/priors.yml", "r"))
    duration_priors = {
        k: dist.TruncatedNormal(v["mean"], v["sd"], low=1.0, high=v["mean"] * 2.5)
        for k, v in loaded_priors["durations"].items()
    }
    beta_priors = {k: dist.Beta(v["alpha"], v["beta"]) for k, v in loaded_priors["beta"].items()}
    other_priors = {
        "alpha_relinfect": dist.TruncatedNormal(1.25, 0.1, low=1.0, high=1.5),
        "rt_init": dist.Normal(0.0, 0.5),
        "shared_dispersion": dist.HalfNormal(0.5),
    }
    return duration_priors | beta_priors | other_priors


def get_worldbank_national_pop(
    iso3,
) -> float:
    """Read population data downloaded from the World Bank
    at https://databank.worldbank.org/source/population-estimates-and-projections#
    on 28th January 2025 and return population size in 2020
    for country of interest.

    Args:
        iso3: ISO3 code for country

    Returns:
        Population data by country ISO3 code
    """
    path = DATA_PATH / "population/6f450edc-f8ef-4d8c-bb2b-dbb1864d88c8_Data.csv"
    dtype = {"2020 [YR2020]": float}
    col = "Country Code"
    data = pd.read_csv(path, index_col=col, na_values=[".."], dtype=dtype)["2020 [YR2020]"].dropna()
    return data[iso3]


def raster_to_polydf(
    raster_ds: DataArray,
    data_name: str,
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
                out_data[valid_idx] = cell_data
                valid_idx += 1

    data = data.flatten()
    return gp.GeoDataFrame({data_name: out_data}, geometry=geoms)


def get_latest_analyses(
    country: str,
    analyses: List[str],
    date_format="%Y%m%d_%H%M",
) -> Dict[str, str]:
    """Get the most recent analysis time string
    for each of the requested analysis types
    for a particular country.

    Args:
        country: Name of the country
        analyses: The names of the mobility analysis types requested
        date_format: String format to represent the date

    Returns:
        The requested information
    """
    last_analyses = {}
    for analysis in analyses:
        path = OUTPUTS_PATH / country / analysis
        dates = [datetime.strptime(d, date_format) for d in os.listdir(path)]
        last_analyses[analysis] = datetime.strftime(max(dates), date_format)
    return last_analyses


def get_country_mobility(
    iso3: str,
) -> pd.DataFrame:
    """Get all the different types of mobility
    for a particular country.

    Args:
        iso3: ISO3 code representing the country

    Returns:
        The mobility estimates
    """
    g_mob = pd.read_csv(DATA_PATH / f"mobility/{iso3}_gmob_data.csv", index_col=0)
    g_mob.index = pd.to_datetime(g_mob.index)
    nonresi_g_mob = g_mob.loc[:, g_mob.columns!="residential"].mean(axis=1).rolling(7).mean().dropna()
    
    fb_mob = pd.read_csv(DATA_PATH / f"mobility/{iso3}_fbmob_data.csv", index_col=0)["0"]
    fb_mob.index = pd.to_datetime(fb_mob.index)
    fb_mob = 1.0 + fb_mob.rolling(7).mean().dropna()
    
    collated_mob = pd.DataFrame(
        {
            "google_nonresi_linear": nonresi_g_mob,
            "google_nonresi_square": nonresi_g_mob ** 2.0,
            "fb_linear": fb_mob,
            "fb_square": fb_mob ** 2.0,
        },
    )
    collated_mob["no_mob"] = 1.0
    return collated_mob


def get_standard_targets(
    country: str, 
    start: datetime,
    end: datetime, 
    init_duration: int,
    hosp_indicator: str, 
    data_delay: int=14,
) -> Tuple[pd.DataFrame]:
    """Get the standard epidemiological targets for a model run.

    Args:
        country: The country name
        start: Analysis start time
        end: Analysis end time
        init_duration: Time for initialisation before analysis starts
        data_delay: Delay from analysis starting to comparing against data

    Returns:
        Case, hospitalisation, death and seroprevalence targets and initialisation data
    """
    cases_target, deaths_target, init_data = get_who_targets(country, start, end, init_duration, data_delay)
    hosp_target = get_hosp_target(country, start, end, data_delay, hosp_indicator)
    seroprev_target = get_filtered_seroprev(country, start, end)
    return cases_target, hosp_target, deaths_target, seroprev_target, init_data


def get_euro_var_inputs(
    country: str,
    strains: List[str],
    analysis_start: datetime,
    seed_duration: int,
    var_target_start_date: datetime,
    var_target_end_date: datetime,
    val: float=0.5,
    lag: int=80,
) -> tuple:
    """Get information relevant to variants for European countries.

    Args:
        country: Name of the country
        strains: The strains being implemented (always "eu" and "alpha")
        analysis_start: Start date of the analysis
        seed_duration: Duration for seeding alpha variant
        var_target_start_date: Start time of window for calibrating to variant proportions
        var_target_end_date: End time of window for calibrating to variant proportions
        val: Proportion to reach for alpha variant
        lag: Time before the proportion reached to start seeding

    Returns:
        The variant target and the seeding times for the alpha variant
    """
    var_target = get_european_var_props(country, var_target_start_date, var_target_end_date, strains)
    before_prop_time = (var_target - val).abs().idxmin() - timedelta(lag)
    alpha_seed_start = max([before_prop_time, analysis_start])
    alpha_seed_times = [alpha_seed_start, alpha_seed_start + timedelta(seed_duration)]
    seed_times = [alpha_seed_times]
    return var_target, seed_times


def get_country_vacc_data(
    iso3 :str,
) -> pd.DataFrame:
    """Get the initial course cumulative vaccination coverage
    data for a specific country.

    Args:
        iso3: ISO3 code for country

    Returns:
        The data
    """
    country_name = pycountry.countries.lookup(iso3).name
    data = pd.read_csv(DATA_PATH / "owid/share-of-people-who-completed-the-initial-covid-19-vaccination-protocol.csv", index_col="Day")
    data.index = pd.to_datetime(data.index)
    return data.loc[data["Entity"] == country_name, "People fully vaccinated (cumulative, per hundred)"]


def get_first_date_above_cov(
    iso3 :str,
    coverage_val: float,
) -> datetime:
    """Find the first time vaccination coverage exceeds a threshold.

    Args:
        iso3: ISO3 code for country
        coverage_val: The coverage threshold as a proportion

    Returns:
        The time
    """
    vacc_data = get_country_vacc_data(iso3)
    return vacc_data[vacc_data.gt(coverage_val)].idxmin()
