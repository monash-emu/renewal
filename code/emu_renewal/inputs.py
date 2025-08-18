import pandas as pd
import geopandas as gpd
import json
import pycountry
from typing import Tuple
from emu_renewal.constants import (
    DATA_PATH,
    RAW_MOB_PATH,
    POP_YEAR,
    AUST_POP_YEAR,
    SUB_DEU_COUNTRIES,
    SUB_GBR_COUNTRY,
    ASSUMED_HIGH_INCOME,
)
from os import listdir as ls


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
    except for Australia, for which the population size in {AUST_POP_YEAR} was used
    (because of the later analysis period for this country).
    """
    path = DATA_PATH / "population/173b86cf-b697-4715-8bd5-cbb5a6cc3885_Data.csv"
    year = AUST_POP_YEAR if iso3 == "AUS" else POP_YEAR
    year_str = f"{year} [YR{year}]"
    return pd.read_csv(path, index_col="Country Code", na_values=[".."]).loc[iso3, year_str]


def get_undesa_national_pop(iso3: str) -> float:
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


def get_ordered_countries_by_cont(countries_by_cont, conts):
    ordered_countries = {}
    for cont in conts:
        pops = {c: get_country_pop(c) for c in countries_by_cont[cont]}
        ordered_countries[cont] = pd.Series(pops).sort_values(ascending=False).index
    return ordered_countries


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


def get_all_var_data() -> dict:
    """Get the downloaded NextClade data
    for all strains listed in VAR_NAMES.

    Returns:
        Data in raw form
    """
    var_names = [v.split(".json")[0] for v in ls(DATA_PATH / "nextclade") if v.startswith("2")]
    return {v: json.load(open(DATA_PATH / f"nextclade/{v}.json", "r")) for v in var_names}


def find_increasing_groups(
    data: pd.Series,
) -> Tuple[pd.DatetimeIndex]:
    """Find the indexes at which a series
    (which is supposed to be generally decreasing)
    is increasing.

    Args:
        data: The data

    Returns:
        Two lists of indexes with the same length
            representing the starts and the ends of the
            increasing sections of the series
    """
    inc_elements = (data.diff() > 0.0).astype(int)
    group_limits = inc_elements.diff().shift(-1).fillna(0.0)
    if inc_elements.iloc[-1] == 1:
        group_limits.iloc[-1] = -1.0
    starts = group_limits[group_limits == 1.0].index
    ends = group_limits[group_limits == -1.0].index
    return starts, ends


def find_decreasing_groups(
    data: pd.Series,
) -> Tuple[pd.DatetimeIndex]:
    """Find the indices at which a series
    (which is supposed to be generally increasing)
    is decreasing.

    Args:
        data: The data

    Returns:
        Two lists of indexes with the same length
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
            the output from find_increasing_groups
        ends: Ends of the increasing periods,
            the output from find_increasing_groups
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


def get_pooled_totals(
    data: pd.DataFrame,
    var_name: str = "prealpha",
) -> pd.DataFrame:
    """Combines the two preceding functions

    to get the totals after pooling for increases in the data.

    Args:
        data: The unadjusted data

    Returns:
        The adjusted data
    """
    while not data[f"{var_name}_prop"].is_monotonic_decreasing:
        starts, ends = find_increasing_groups(data[f"{var_name}_prop"])
        data = pool_totals(starts, ends, data, var_name)
    return data


def get_incr_pooled_totals(
    data: pd.DataFrame,
    var_name: str = "prealpha",
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
    if iso3 == pycountry.countries.lookup(ASSUMED_HIGH_INCOME).alpha_3:
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
    world = gpd.read_file(DATA_PATH / "mapping/ne_10m_admin_0_countries.shp")
    for c in ["FRA", "NOR"]:
        country = pycountry.countries.lookup(c).name
        world.loc[world["ADMIN"] == country, "ISO_A3"] = c
    world["geometry"] = world.simplify(tolerance=0.1)
    return world
