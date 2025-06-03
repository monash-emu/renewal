from typing import Union
import pandas as pd
import geopandas as gpd
from jax import numpy as jnp
import json
import pycountry
from typing import Dict, Tuple, Union
from datetime import datetime, timedelta
import yaml as yml
from numpyro import distributions as dist
from emu_renewal.utils import get_beta_params_from_mean_var
from emu_renewal.constants import (
    VAR_NAMES,
    DATA_PATH,
    RAW_MOB_PATH,
    ALREADY_WEEKLY_ADMIT_COUNTRIES,
    ALREADY_WEEKLY_OCCUP_COUNTRIES,
    PREV_KEY,
    ANTIBODY_DELAY,
)


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


def get_owid_hosps(
    country: str,
    start: datetime,
    end: datetime,
) -> Tuple[Union[pd.Series, None], str]:
    """Get only one hospitalisation target for a specified country,
    hierarchically choosing the preferred target,
    or returning None and empty string if nothing available.
    The "best" indicator is chosen hierarchically,
    such that a hospital indicator beats a ICU indicator
    and daily admissions beat occupancy.
    Croatia admissions from OWID is reported weekly,
    so no need to apply rolling average
    (even though it is weekly data, it is reported
    each day for most countries).
    Japan and Bulgaria occupancy from OWID is reported weekly,
    so no need to apply rolling average.

    Args:
        country: Country identifier
        start: Data comparison start time
        end: Analysis end time

    Returns:
        Tuple of two elements:
            - The calibration data for comparison
            - The name of the indicator for comparison
    """
    admits = get_owid_hosp_series("Weekly new hospital admissions", country)
    filt_admits = admits[(start < admits.index) & (admits.index < end) & (admits > 0.0)]
    occup = get_owid_hosp_series("Daily hospital occupancy", country)
    filt_occup = occup[(start < occup.index) & (occup.index < end)]
    icu_admits = get_owid_hosp_series("Weekly new ICU admissions", country)
    filt_icu_admits = icu_admits[(start < icu_admits.index) & (icu_admits.index < end)]
    icu_occup = get_owid_hosp_series("Daily ICU occupancy", country)
    filt_icu_occup = icu_occup[(start < icu_occup.index) & (icu_occup.index < end)]
    if not filt_admits.empty and country in ALREADY_WEEKLY_ADMIT_COUNTRIES:
        weekly_admits = filt_admits.dropna()
        return weekly_admits, "weekly_admissions"
    elif not filt_admits.empty:
        weekly_admits = filt_admits.rolling(7).mean()[::7].dropna()
        return weekly_admits, "weekly_admissions"
    elif not filt_occup.empty and country in ALREADY_WEEKLY_OCCUP_COUNTRIES:
        weekly_occup = filt_occup.dropna()
        return weekly_occup, "occupancy"
    elif not filt_occup.empty:
        weekly_occup = filt_occup.rolling(7).mean()[::7].dropna()
        return weekly_occup, "occupancy"
    elif not filt_icu_admits.empty:
        weekly_icu_admits = filt_icu_admits.rolling(7).mean()[::7].dropna()
        return weekly_icu_admits, "icu_weekly_admissions"
    elif not filt_icu_occup.empty:
        weekly_icu_occup = filt_icu_occup.rolling(7).mean()[::7].dropna()
        return weekly_icu_occup, "icu_occupancy"
    else:
        return None, ""


def process_raw_google_mobility(
    iso3: str,
) -> pd.DataFrame:
    """Load raw Google mobility data and process for storing.

    Args:
        iso3: Country identifier

    Returns:
        The data
    """
    years = range(2020, 2023)
    iso2 = pycountry.countries.lookup(iso3).alpha_2
    file_end = f"_{iso2}_Region_Mobility_Report.csv"
    data_files = [pd.read_csv(RAW_MOB_PATH / (str(y) + file_end), index_col="date") for y in years]
    all_data = pd.concat(data_files)
    all_data.index = pd.to_datetime(all_data.index)

    # The rows at the national level
    nat_data = all_data.loc[pd.isna(all_data["sub_region_1"]) & pd.isna(all_data["metro_area"])]

    # The mobility columns
    nat_data = nat_data[[c for c in nat_data.columns if "change_from_baseline" in c]]

    # Simplify column naming
    nat_data = nat_data.rename(lambda c: c.replace("_percent_change_from_baseline", ""), axis=1)

    # Convert from percentage reduction to ratio
    nat_data = 1.0 + nat_data / 100.0
    return nat_data.sort_index()


def get_all_seroprev() -> pd.Series:
    """Get all the seroprevalence data.

    Returns:
        All SeroTracker data

    Notes
    -----
    Seroprevalence data was obtained from
    [SeroTracker](https://github.com/serotracker/sars-cov-2-data/raw/refs/heads/main/serotracker_dataset.csv)
    on 11 December 2024,
    with the date for each serosurvey calculated as the
    mid-point between the reported start and end dates of sampling.
    This date was then lagged earlier by {ANTIBODY_DELAY} for the purposes
    of calibration to allow for a delay between infection
    and the subsequent development of detectable antibodies.
    """
    data = pd.read_csv(DATA_PATH / "seroprevalence/serotracker.csv")
    data["start"] = pd.to_datetime(data["sampling_start_date"])
    data["end"] = pd.to_datetime(data["sampling_end_date"])
    data.index = (data["end"] - data["start"]) / 2 + data["start"]
    data.index -= timedelta(ANTIBODY_DELAY)
    data.index = data.index.normalize()
    return data.sort_index()


def get_standard_priors(
    n_strains: int,
    hosp_out_type: str,
    iso3: str,
) -> Dict[str, dist.Distribution]:
    """Load the priors from the yml and combine with
    standard hard-coded priors.

    Args:
        n_strains: The number of strains implemented
        hosp_out_type: The hospital-related indicator name
            Must be one of the keys to relevant_duration_priors below

    Returns:
        The prior distributions
    """
    loaded_priors = yml.safe_load(open(DATA_PATH / "config/priors.yml", "r"))

    # Durations
    duration_priors = {
        k: dist.TruncatedNormal(v["mean"], v["sd"], low=1.0, high=v["mean"] * 2.5)
        for k, v in loaded_priors["durations"].items()
    }
    universal_prior_names = [
        "gen_mean",
        "gen_sd",
        "report_mean",
        "report_sd",
        "death_mean",
        "death_sd",
    ]
    rel_durations_dict = {
        "weekly_admissions": ["admit_mean", "admit_sd"],
        "occupancy": ["admit_mean", "admit_sd", "stay_mean", "stay_sd"],
        "icu_admissions": ["icu_admit_mean", "icu_admit_sd"],
        "icu_occupancy": ["icu_admit_mean", "icu_admit_sd", "icu_stay_mean", "icu_stay_sd"],
        "": [],
    }
    duration_prior_names = rel_durations_dict[hosp_out_type] + universal_prior_names
    rel_durs = {k: v for k, v in duration_priors.items() if k in duration_prior_names}
    irrel_durs = {k: 1.0 for k in duration_priors if k not in rel_durs}

    # Proportions from summary statistics
    income = get_income_group(iso3)
    adjusters = {
        "Low income": 0.4,
        "Lower middle income": 0.6,
        "Upper middle income": 0.8,
        "High income": 1.0,
    }
    adjuster = 0.4 if iso3 == "VEN" else adjusters[income]
    beta_from_sum = loaded_priors["beta_from_summary"]
    beta_from_sum_dists = {}
    for k, v in beta_from_sum.items():
        a, b = get_beta_params_from_mean_var(v["mean"] * adjuster, v["std"] ** 2.0)
        beta_from_sum_dists[k] = dist.Beta(a, b)
    if hosp_out_type == "":
        beta_from_sum_dists["har"] = 1.0

    # Proportions
    beta_priors = {
        k: dist.Beta(v["alpha"], v["beta"])
        for k, v in loaded_priors["beta"].items()
        if k != "cross_immunity"
    }
    if "icu_" not in hosp_out_type:
        beta_priors["icu_ar"] = 1.0
    if hosp_out_type == "":
        beta_priors["har"] = 1.0
        beta_priors["icu_ar"] = 1.0

    # Variant-related
    seed_low_lim = jnp.repeat(jnp.log(1e-7), n_strains)
    seed_up_lim = jnp.repeat(jnp.log(5e-6), n_strains)
    seed_rate_priors = {"seed_rates": dist.Uniform(seed_low_lim, seed_up_lim)}
    seed_offsets_dist = dist.Uniform(
        jnp.repeat(4.0, n_strains - 1), jnp.repeat(90.0, n_strains - 1)
    )
    seed_offsets_priors = seed_offsets_dist if n_strains > 1 else None
    seed_priors = {"seed_offsets": seed_offsets_priors}
    relinfect_means = jnp.repeat(1.4, n_strains - 1)
    infect_dist_prior = dist.TruncatedNormal(relinfect_means, 0.2, low=1.0, high=2.0)
    infect_dist = infect_dist_prior if n_strains > 1 else None
    inf_priors = {"relinfect": infect_dist}
    imm = loaded_priors["beta"]["cross_immunity"]
    imm_prior = {"cross_immunity": dist.Beta(imm["alpha"], imm["beta"])} if n_strains > 0 else {}

    # Miscellaneous
    rt_prior = {"rt_init": dist.Normal(0.0, 0.5)}
    disp_prior = {"shared_dispersion": dist.HalfNormal(0.5)}
    prop_disp_prior = {"prop_shared_disp": 0.05}
    seroprev_disp = {"seroprev_disp": 0.2}

    return (
        rel_durs
        | irrel_durs
        | beta_priors
        | beta_from_sum_dists
        | seed_rate_priors
        | inf_priors
        | imm_prior
        | rt_prior
        | disp_prior
        | prop_disp_prior
        | seed_priors
        | seroprev_disp
    )


def get_country_pop(iso3):
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
    Population data were downloaded from
    [the World Bank](https://databank.worldbank.org/source/population-estimates-and-projections#)
    on 01/04/2025. From this data, the population size in 2020
    of country of interest was extracted. The exception was Australia,
    for which the population size in 2022 was used,
    because of its later analysis period.
    """
    path = DATA_PATH / "population/173b86cf-b697-4715-8bd5-cbb5a6cc3885_Data.csv"
    year = 2022 if iso3 == "AUS" else 2020
    year_str = f"{year} [YR{year}]"
    return pd.read_csv(path, index_col="Country Code", na_values=[".."]).loc[iso3, year_str]


def get_ordered_countries_by_cont(countries_by_cont, conts):
    ordered_countries = {}
    for cont in conts:
        pops = {c: get_country_pop(c) for c in countries_by_cont[cont]}
        ordered_countries[cont] = pd.Series(pops).sort_values(ascending=False).index
    return ordered_countries


def get_undesa_national_pop(iso3: str) -> float:
    """Get UN-DESA population estimate for a single country, for 2020

    Sourced from
    https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/EXCEL_FILES/2_Population/WPP2024_POP_F01_1_POPULATION_SINGLE_AGE_BOTH_SEXES.xlsx

    Args:
        iso3: ISO3 country code

    Returns:
        2020 UNDESA population total for country
    """
    csv_path = DATA_PATH / "population/undesa_pops_2020.csv"
    data = pd.read_csv(csv_path, index_col=["ISO3 Alpha-code"])
    return data.loc[iso3, "population"]


def get_google_mobility(
    iso3: str,
) -> pd.DataFrame:
    """Get all fields of the Google mobility data.

    Args:
        iso3: Country identifier

    Returns:
        The data
    """
    filename = f"mobility/{iso3}_gmob_data.csv"
    g_mob = pd.read_csv(DATA_PATH / filename, index_col=0)
    g_mob.index = pd.to_datetime(g_mob.index)
    return g_mob


def get_fb_mobility(
    iso3,
) -> pd.Series:
    """Get the single field of the Facebook mobility data.

    Args:
        iso3: Country identifier

    Returns:
        The data
    """
    filename = f"mobility/{iso3}_fbmob_data.csv"
    fb_mob = pd.read_csv(DATA_PATH / filename, index_col=0)["0"]
    fb_mob.index = pd.to_datetime(fb_mob.index)
    return 1.0 + fb_mob


def get_fb_withintile_mobility(
    iso3,
) -> pd.Series:
    """Get the single field of the Facebook mobility data.

    Args:
        iso3: Country identifier

    Returns:
        The data
    """
    filename = f"mobility/{iso3}_fbmob_data.csv"
    fb_mob = pd.read_csv(DATA_PATH / filename, index_col=0)["0"]
    fb_mob.index = pd.to_datetime(fb_mob.index)
    return 1.0 - fb_mob


def get_apple_mobility(
    iso3: str,
) -> pd.DataFrame:
    """Get all fields of the Apple mobility data.

    Args:
        iso3: Country identifier

    Returns:
        The data
    """
    filename = "mobility/apple-mobility-test_apple_latest_apple-mobility-trends-report.csv"
    all_data = pd.read_csv(DATA_PATH / filename, low_memory=False)
    national_data = all_data.loc[all_data["country"].isnull()]
    region_row = national_data["region"]
    type_row = national_data["transportation_type"]
    national_data.index = pd.MultiIndex.from_arrays([region_row, type_row])
    national_data = national_data.iloc[:, 6:].T
    national_data.index = pd.to_datetime(national_data.index)
    countries = national_data.columns.levels[0]
    crename_map = {
        "Republic of Korea": "South Korea",
        "Russia": "Russian Federation",
        "Turkey": "Türkiye",
    }
    reverse_lookup = {
        pycountry.countries.lookup(crename_map.get(c) or c).alpha_3: c for c in countries
    }
    country_df = national_data[reverse_lookup[iso3]].interpolate()
    country_df /= 100.0
    return country_df


def get_country_vacc_data(
    iso3: str,
) -> pd.DataFrame:
    """Get the initial course cumulative vaccination coverage
    data for a specific country.
    Have substituted Germany for Switzerland because these two
    countries had almost identical profiles of vaccine doses
    administered per person in the early phases of the roll-out.
    Substitute Germany for Ireland based on almost identical
    profiles, but late start for fully-vaccinated data in Ireland.


    Args:
        iso3: Country identifier

    Returns:
        The data
    """
    if iso3 == "KOR":
        country = pycountry.countries.lookup(iso3).common_name
    elif iso3 in ["CHE", "IRL"]:
        country = pycountry.countries.lookup("DEU").name
    elif iso3 == "QAT":
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
    return {v: json.load(open(DATA_PATH / f"nextclade/{v}.json", "r")) for v in VAR_NAMES}


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
    """Find the indexes at which a series
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


def pool_seroprev_totals(
    starts: datetime,
    ends: datetime,
    data: pd.Series,
) -> pd.Series:
    """Pool groups of seroprevalence data that are
    decreasing over time and were identified by
    find_decreasing_groups.

    Args:
        starts: The start indices for the decreasing groups
        ends: The end indices for the decreasing groups
        data: The data before processing

    Returns:
        The processed data
    """
    period_sums = pd.DataFrame()
    idx_to_remove = []
    for start, end in zip(starts, ends):
        period = data.loc[start:end]
        average_date = period.index.mean()
        prevs = period[PREV_KEY]
        denoms = period["denominator_value"]
        total_denoms = denoms.sum()
        new_prev = (prevs * denoms).sum() / total_denoms
        period_sums.loc[average_date, PREV_KEY] = new_prev
        period_sums.loc[average_date, "denominator_value"] = total_denoms
        idx_to_remove += list(period.index)
    new_data = pd.concat([period_sums, data.drop(index=idx_to_remove)])
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


def get_seroprev_pooled_totals(
    data: pd.Series,
) -> pd.Series:
    """Pool any sequences of seroprevalence data
    that are decreasing over time.
    Continue pooling until all estimates are monotonically
    increasing.

    Args:
        data: The raw seroprevalence data

    Returns:
        The data after pooling
    """
    while not data[PREV_KEY].is_monotonic_increasing:
        starts, ends = find_decreasing_groups(data[PREV_KEY])
        data = pool_seroprev_totals(starts, ends, data)
    return data[PREV_KEY]


def get_incr_pooled_totals(
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
    while not data[f"{var_name}_prop"].is_monotonic_increasing:
        group_starts, group_ends = find_decreasing_groups(data[f"{var_name}_prop"])
        data = pool_totals(group_starts, group_ends, data, var_name)
    return data


def get_income_group(
    iso3: str,
) -> str:
    """We obtained income groups from
    [the World Bank](https://datacatalogapi.worldbank.org/ddhxext/ResourceDownload?resource_unique_id=DR0090755).

    Args:
        iso3: Country identifier

    Returns:
        World Bank income classification
    """
    if iso3 == "GUF":
        return "High income"
    data = pd.read_excel(DATA_PATH / "income/CLASS.xlsx", index_col="Code")
    return data.loc[iso3, "Income group"]


def get_gdps(year):
    """
    https://data.worldbank.org/indicator/NY.GDP.PCAP.CD?most_recent_year_desc=true
    """
    data = pd.read_excel(
        DATA_PATH / "income/API_NY.GDP.PCAP.CD_DS2_en_excel_v2_85284.xls", header=3, index_col=1
    )
    return data[str(year)]


def get_world_shp():
    """Data obtained from:
    https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip

    Returns:
        The cleaned shapefile
    """
    world = gpd.read_file(DATA_PATH / "mapping/ne_10m_admin_0_countries.shp")
    for c in ["FRA", "NOR"]:
        country = pycountry.countries.lookup(c).name
        world.loc[world["ADMIN"] == country, "ISO_A3"] = c
    return world
