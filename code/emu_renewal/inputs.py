from pathlib import Path
from datetime import datetime
import pandas as pd
import geopandas as gpd
import arviz as az
import pycountry
from typing import Tuple, List, Dict
import re

from emu_renewal.constants import (
    DATA_PATH,
    RAW_MOB_PATH,
    POP_YEAR,
    OC_POP_YEAR,
    SUB_DEU_COUNTRIES,
    SUB_GBR_COUNTRY,
    ASSUMED_HIGH_INCOME,
    G_MOB_LOCATION_CMAP,
    MOBILITY_SMOOTH_PERIOD,
    OXCGRT_DTYPES,
    OXCGRT_IND_MAX,
)
from emu_renewal.utils import get_cont_of_country


def get_country_pop(
    iso3: str,
) -> float:
    """Get the population size to use for a specified country.

    Args:
        iso3: The country identifier

    Returns:
        The population size

    Notes
    -----
    We used the total population size estimated
    by the World Bank where a population size was available
    from this source,
    and used the estimate provided by the United Nations
    Department of Economic and Social Affairs otherwise.
    """
    try:
        return get_worldbank_national_pop(iso3)
    except:
        return get_undesa_national_pop(iso3)


def get_worldbank_national_pop(
    iso3: str,
) -> float:
    """Get the population size of a country.

    Args:
        iso3: Country identifier

    Returns:
        Population size

    Notes
    -----
    Population data was downloaded from
    [the World Bank](https://databank.worldbank.org/source/population-estimates-and-projections#)
    on 1 April 2025.
    The population size for {POP_YEAR} was used for all countries
    except for Singapore and countries of Oceania, 
    for which the population size in {OC_POP_YEAR} was used
    (because of the later analysis period for these countries).
    """
    path = DATA_PATH / "population/173b86cf-b697-4715-8bd5-cbb5a6cc3885_Data.csv"
    year = OC_POP_YEAR if get_cont_of_country(iso3) == "OC" else POP_YEAR
    year_str = f"{year} [YR{year}]"
    return pd.read_csv(path, index_col="Country Code", na_values=[".."]).loc[iso3, year_str]


def get_undesa_national_pop(
    iso3: str,
) -> float:
    """Get UN-DESA population estimate for a single country, for 2020

    Args:
        iso3: ISO3 country code

    Returns:
        2020 UNDESA population total for country

    Notes
    -----
    [UN DESA population data](https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/EXCEL_FILES/2_Population/WPP2024_POP_F02_1_POPULATION_5-YEAR_AGE_GROUPS_BOTH_SEXES.xlsx)
    was downloaded on 18 March 2025 and 
    where UN DESA population was needed, 
    data for {POP_YEAR} was used.
    """
    csv_path = DATA_PATH / f"population/undesa_pops_{POP_YEAR}.csv"
    data = pd.read_csv(csv_path, index_col=["ISO3 Alpha-code"])
    return data.loc[iso3, "population"]


def get_owid_hosp_series(
    indicator: str,
    iso3: str,
) -> pd.Series:
    """Get OWID hospitalisation-related estimates for single indicator
    from the original raw data.

    Args:
        indicator: Name of the indicator
        iso3: Country identifier

    Returns:
        The data
    """
    hosp = pd.read_csv(DATA_PATH / "owid/owid_hosp.csv", index_col="date")
    hosp.index = pd.to_datetime(hosp.index)
    iso3 = pycountry.countries.lookup(iso3).alpha_3
    data = hosp[hosp["iso_code"] == iso3]
    return data.loc[data["indicator"] == indicator, "value"]


def process_raw_google_mobility(
    iso3: str,
) -> pd.DataFrame:
    """Load raw Google mobility data for single country.

    Args:
        iso3: Country identifier

    Returns:
        The data

    Notes
    -----
    We obtained mobility data from
    [the Google Community Mobility Reports](https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip)
    on 14 January 2025 and extracted national mobility estimates
    by Google mobility domain.
    """
    mob_col_identifier = "_percent_change_from_baseline"
    years = range(2020, 2023)
    iso2 = pycountry.countries.lookup(iso3).alpha_2
    file_end = f"_{iso2}_Region_Mobility_Report.csv"
    data_files = [
        pd.read_csv(RAW_MOB_PATH / (str(y) + file_end), index_col="date", low_memory=False)
        for y in years
    ]
    all_data = pd.concat(data_files)
    all_data.index = pd.to_datetime(all_data.index)

    # The rows at the national level for the country
    c_data = all_data.loc[pd.isna(all_data["sub_region_1"]) & pd.isna(all_data["metro_area"])]

    # Get the mobility columns and drop the end of their names
    c_data = c_data[[c for c in c_data.columns if mob_col_identifier in c]]
    c_data = c_data.rename(lambda c: c.replace(mob_col_identifier, ""), axis=1)

    return c_data.sort_index()


def get_google_mobility(
    iso3: str,
) -> pd.DataFrame:
    """Get all fields of the Google mobility data.

    Args:
        iso3: Country identifier

    Returns:
        The data

    Notes
    -----
    For this analysis,
    we used the raw value interpreted as a percentage
    plus one to scale the transmission rate.
    """
    filename = f"mobility/{iso3}_gmob_data.csv"
    g_mob = pd.read_csv(DATA_PATH / filename, index_col=0)
    g_mob.index = pd.to_datetime(g_mob.index)
    return 1.0 + g_mob / 100.0


def get_cont_g_mob(
    countries: List[str],
) -> Dict[str, pd.DataFrame]:
    """Collate the Google mobility data
    for a list of countries.

    Args:
        countries: The country identifiers

    Returns:
        The mobility data
    """
    mob = {}
    no_mob_countries = []
    for iso3 in countries:
        try:
            c_mob = get_google_mobility(iso3)
            # Don't include country (Guinea-Bissau) with locations missing
            if c_mob.isnull().all().any():
                no_mob_countries.append(iso3)
            else:
                mob[iso3] = c_mob
        except FileNotFoundError:
            no_mob_countries.append(iso3)

    return mob, no_mob_countries


def get_fb_visited_mobility(
    iso3,
) -> pd.Series:
    """Get the single field of the Facebook mobility data.

    Args:
        iso3: Country identifier

    Returns:
        The data

    Notes
    -----
    We used the `all_day_bing_tiles_visited_relative_change`
    for the first Facebook mobility analysis,
    and scaled transmission transmission according
    to one plus this mobility metric.
    """
    filename = f"mobility/{iso3}_fbmob_data.csv"
    mob = pd.read_csv(DATA_PATH / filename, index_col=0)["0"]
    mob.index = pd.to_datetime(mob.index)
    return 1.0 + mob


def get_fb_singletile_mobility(
    iso3,
) -> pd.Series:
    """Get the single field of the Facebook mobility data.

    Args:
        iso3: Country identifier

    Returns:
        The data

    Notes
    -----
    For the second Facebook mobility analysis,
    we used the `all_day_ratio_single_tile_users`
    estimate and scaled transmission according to
    one minus this mobility metric.
    """
    filename = f"mobility/{iso3}_fbsingletile_data.csv"
    mob = pd.read_csv(DATA_PATH / filename, index_col=0)["0"]
    mob.index = pd.to_datetime(mob.index)
    return 1.0 - mob


def get_oxcgrt(
    iso3: str, 
    field: str,
) -> pd.Series:
    """Get a named field for a single country
    from the Oxford CGRT database.

    Args:
        iso3: The country identifier
        field: The name of the field/column

    Returns:
        The data
    """
    mob = pd.read_csv(DATA_PATH / f"restrictions/oxcgrt.csv", dtype=OXCGRT_DTYPES)
    mob.index = pd.to_datetime(mob["Date"], format="%Y%m%d")
    mob = mob.loc[mob["CountryCode"] == iso3, field]
    return 1.0 - mob / 100.0


def get_requested_mob(
    iso3: str, 
    mob_source: str, 
    mob_location: str,
) -> pd.DataFrame:
    """Get the mobility data based on the type
    source and type of data being requested.

    Args:
        iso3: The country identifier
        mob_source: The source, either Google or a Facebook type
        mob_location: The Google location

    Returns:
        The mobility data
    """
    if mob_source == "g_mob":
        return get_google_mobility(iso3)[mob_location]
    elif mob_source == "fb_visited_mob":
        return get_fb_visited_mobility(iso3)
    elif mob_source == "fb_singletile_mob":
        return get_fb_singletile_mobility(iso3)
    

def get_country_vacc_data(
    iso3: str,
) -> pd.DataFrame:
    """Get the initial course cumulative vaccination coverage
    data for a specific country.

    Args:
        iso3: Country identifier

    Returns:
        The data

    Notes
    -----
    We substituted Germany's data for 
    {SUB_DEU_COUNTRIES} because these two
    countries had almost identical profiles of vaccine doses
    administered per person in the earliest phases of the roll-out
    and vaccination data were not available for these two countries
    through the period at which coverage approached the 
    analysis start threshold.
    Similarly, we substituted Great Britain for {SUB_GBR_COUNTRY}
    based on the same rationale.
    """
    sub_deu = [pycountry.countries.lookup(c).alpha_3 for c in SUB_DEU_COUNTRIES.split(" and ")]
    if iso3 == "KOR":
        country = pycountry.countries.lookup(iso3).common_name
    elif iso3 in sub_deu:
        country = pycountry.countries.lookup("DEU").name
    elif iso3 == SUB_GBR_COUNTRY:
        country = "GBR"
    else:
        country = pycountry.countries.lookup(iso3).name
    filename = "owid/share-of-people-who-completed-the-initial-covid-19-vaccination-protocol.csv"
    data = pd.read_csv(DATA_PATH / filename, index_col="Day")
    data.index = pd.to_datetime(data.index)
    col_name = "People fully vaccinated (cumulative, per hundred)"
    return data.loc[data["Entity"] == country, col_name]


def find_decreasing_groups(
    data: pd.Series,
) -> Tuple[pd.DatetimeIndex]:
    """Find the indices at which a series
    (which is supposed to be generally increasing)
    is decreasing.

    Args:
        data: The data

    Returns:
        Two lists of indices with the same length
            representing the starts and the ends of the
            decreasing sections of the series
    """
    dec_elements = (data.diff() < 0.0).astype(int)
    group_limits = dec_elements.diff().shift(-1).fillna(0.0)
    if dec_elements.iloc[-1] == 1:
        group_limits.iloc[-1] = -1.0
    starts = group_limits[group_limits == 1.0].index
    ends = group_limits[group_limits == -1.0].index
    return starts, ends


def pool_totals(
    starts: pd.DatetimeIndex,
    ends: pd.DatetimeIndex,
    data: pd.DataFrame,
    var_name: str = "prealpha",
) -> pd.DataFrame:
    """Replace periods of the pre-Alpha data
    that are increasing over time with averages
    over the period of increase.

    Args:
        starts: Starts of the increasing periods,
            the output from find_decreasing_groups
        ends: Ends of the increasing periods,
            the output from find_decreasing_groups
        data: The unadjusted data

    Returns:
        The adjusted data
    """
    period_sums = pd.DataFrame(columns=[var_name, "totals", f"{var_name}_prop"])
    idx_to_remove = []
    for limits in zip(starts, ends):
        period = data.loc[limits[0] : limits[1]]
        average_date = period.index.mean()
        period_sums.loc[average_date] = period.sum()
        idx_to_remove += list(period.index)
    new_data = pd.concat([period_sums, data.drop(index=idx_to_remove)])

    # Redo proportion calculations, which will now be wrong
    new_data[f"{var_name}_prop"] = new_data[var_name] / new_data["totals"]

    # Make sure indices fall on the start of a date
    new_data.index = new_data.index.round("D")
    return new_data.sort_index()


def get_incr_pooled_totals(
    data: pd.DataFrame,
    var_name: str,
) -> pd.DataFrame:
    """Combines the two preceding functions
    to get the totals after pooling to remove
    decreases in the data.

    Args:
        data: The unadjusted data

    Returns:
        The adjusted data
    """
    while not data[f"{var_name}_prop"].is_monotonic_increasing:
        group_starts, group_ends = find_decreasing_groups(data[f"{var_name}_prop"])
        data = pool_totals(group_starts, group_ends, data, var_name)
    return data


def get_income_group(
    iso3: str,
) -> str:
    """Get World Bank income group for a country.

    Args:
        iso3: Country identifier

    Returns:
        World Bank income classification

    Notes
    -----
    Country income classifications were obtained from
    [the World Bank](https://datacatalogapi.worldbank.org/ddhxext/ResourceDownload?resource_unique_id=DR0090755).

    """
    assumed_high_income = [pycountry.countries.lookup(c).alpha_3 for c in ASSUMED_HIGH_INCOME.split(", ")]
    if iso3 in assumed_high_income:
        return "High income"
    data = pd.read_excel(DATA_PATH / "income/CLASS.xlsx", index_col="Code")
    return data.loc[iso3, "Income group"]


def get_gdps(
    year: int,
) -> pd.Series:
    """Get GDP estimates for each country.

    Args:
        year: The year of interest

    Returns:
        The GDPs

    Notes
    -----
    We obtained data on GDP for each country from
    [the World Bank](https://data.worldbank.org/indicator/NY.GDP.PCAP.CD?most_recent_year_desc=true.)
    """
    filename = "API_NY.GDP.PCAP.CD_DS2_en_excel_v2_85284.xls"
    data = pd.read_excel(DATA_PATH / "income" / filename, header=3, index_col=1)
    return data[str(year)]


def get_world_shp():
    """Get shapefile for countries of the world.

    Returns:
        The cleaned shapefile

    Notes
    -----
    We obtained a shapefile for the countries of the world from
    [Natural Earth](https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip).
    """
    world = gpd.read_file(DATA_PATH / "mapping/ne_110m_admin_0_countries.shp")
    for c in ["FRA", "NOR"]:
        country = pycountry.countries.lookup(c).name
        world.loc[world["ADMIN"] == country, "ISO_A3"] = c
    return world


def get_smoothed_trunc_g_mob(
    iso3: str, 
    start: datetime, 
    finish: datetime,
) -> pd.DataFrame:
    """Get the smoothed, truncated Google mobility data

    Args:
        iso3: The country identifier
        start: The start time of the period of interest
        finish: The end time of the period of interest

    Returns:
        The mobility data
    """
    mob = get_google_mobility(iso3)
    smoothed_mob = mob.rolling(MOBILITY_SMOOTH_PERIOD, center=True).mean().dropna()
    return smoothed_mob[(start < smoothed_mob.index) & (smoothed_mob.index < finish)]


def get_g_mob_weight_posts(
    c_path: Path,
) -> pd.DataFrame:
    """Get a dataframe of the mobility weights
    applied to the Google data.

    Args:
        c_path: The country path for the analyses

    Returns:
        The mobility weights
    """
    idata = az.from_netcdf(c_path / "g_mob/idata_filtered.nc")
    params = idata.posterior["mob_weights"].to_dataframe().unstack(level=-1)
    params.columns = G_MOB_LOCATION_CMAP
    return params


def get_g_mob_quants(
    smoothed_mob: pd.DataFrame, 
    params: pd.DataFrame, 
    n_samples: int,
) -> pd.DataFrame:
    """Get the quantiles of the weighted Google
    mobility time series.

    Args:
        smoothed_mob: The mobility data
        params: The output of get_g_mob_weight_posts above
        n_samples: The number of samples to estimate quantiles

    Returns:
        The quantiles of the weighted series
    """
    weights = params.sample(n_samples)
    norm_weights = weights.div(weights.sum(axis=1), axis=0)
    mob_results = (norm_weights @ smoothed_mob.T).T
    return mob_results.quantile([0.025, 0.5, 0.975], axis=1).T


def get_oxcgrt_data() -> pd.DataFrame:
    """Get and process the OXCGRT policy data.

    Returns:
        The processed policy data
    """
    mob = pd.read_csv(DATA_PATH / f"restrictions/oxcgrt.csv", dtype=OXCGRT_DTYPES)
    mob.index = pd.to_datetime(mob["Date"], format="%Y%m%d")
    drop_strings = ["Index", "Vaccinated", "Confirmed", "Notes", "Unnamed", "Date", "Region", "CountryName", "Jurisdiction", "Flag"]
    cols_to_keep = [col for col in mob.columns if not any(s in col for s in drop_strings)]
    mob = mob[cols_to_keep]
    mob.columns = [col.split("_")[0] for col in mob.columns]
    return mob


def find_oxcgrt_country_data(
    iso3: str, 
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Get the country-specific data from the OxCGRT dataset
    produced by get_oxcgrt_raw_data.

    Args:
        iso3: The country identifier
        data: The full OxCGRT data

    Returns:
        The country-specific data
    """
    data = data[data["CountryCode"] == iso3]
    return data.drop("CountryCode", axis=1)


def get_oxcgrt_country_indicators(
    iso3: str,
) -> pd.DataFrame:
    """Get and process the OXCGRT policy data.

    Args:
        iso3: Country identifier

    Returns:
        The processed policy data
    """
    mob = pd.read_csv(DATA_PATH / f"restrictions/oxcgrt.csv", dtype=OXCGRT_DTYPES)
    mob.index = pd.to_datetime(mob["Date"], format="%Y%m%d")
    mob = mob[mob["CountryCode"] == iso3]
    # Note we can only drop region because this is assumed to be the national data
    drop_strings = ["Index", "Vaccinated", "Confirmed", "Notes", "Unnamed", "Date", "Region", "Country", "Jurisdiction", "Flag"]
    cols_to_keep = [col for col in mob.columns if not any(s in col for s in drop_strings)]
    mob = mob[cols_to_keep]
    mob.columns = [col.split("_")[0] for col in mob.columns]
    return mob


def get_rel_oxcgrt_cols(
    vacc_status: str, 
    pol_data: pd.DataFrame,
) -> List[str]:
    """Get the relevant columns for the OXCGRT data
    given a particular vaccination status request.

    Args:
        vacc_status: Must be V, NV, M or E
        pol_data: The OXCGRT data

    Returns:
        The relevant columns
    """
    matches = []
    regex_match = r"^([A-Z]+\d+)([A-Z]*)$"
    for col in pol_data.columns:
        match = re.match(regex_match, col)
        base, suffix = match.groups()
        match_strings = [vacc_status, "", "EV"]
        # Include all V columns, because V2E violates above assumptions
        if suffix in match_strings or base.startswith("V"):
            matches.append(col)
    return matches


def scale_oxcgrt_pols(
    pol_data: pd.DataFrame,
) -> pd.DataFrame:
    """Scale OXCGRT data relative to maximum
    value for indicator.

    Args:
        pol_data: The policy data

    Returns:
        The scaled data for indicators listed in OXCGRT_IND_MAX 
    """
    scaled_data = pd.DataFrame(index=pol_data.index)
    for col in OXCGRT_IND_MAX:
        scaled_data[col] = pol_data[next((c for c in pol_data.columns if c.startswith(col)))] / OXCGRT_IND_MAX[col]
    return scaled_data


def find_pol_countries(
    countries: List[str],
) -> List[str]:
    """Filter a list of countries according to whether
    OxCGRT data exists for them.

    Args:
        countries: The countries to consider

    Returns:
        The countries with data
    """
    data = pd.read_csv(DATA_PATH / f"restrictions/oxcgrt.csv", dtype=OXCGRT_DTYPES)
    return [iso3 for iso3 in countries if any(data["CountryCode"] == iso3)]
