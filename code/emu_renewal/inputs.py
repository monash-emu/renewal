import pandas as pd
import json
from pathlib import Path
import pycountry
from typing import Tuple, List, Dict
from datetime import datetime, timedelta
import yaml as yml
from numpyro import distributions as dist


DATE_FORMAT = "%Y%m%d_%H%M"
TEXT_DATE_FORMAT = "%d/%m/%Y"

BASE_PATH = Path(__file__).parent.parent.parent

OUTPUTS_PATH = BASE_PATH / "outputs"
DATA_PATH = BASE_PATH / "data"
RAW_MOB_PATH = DATA_PATH / "mobility_raw"  # Will not push these larger original files

VAR_NAMES = [
    "21K.Omicron",
    "21L.Omicron",
    "22B.Omicron",
    "20A.EU1",
    "20A.EU2",
    "20I.Alpha.V1",
    "20A.S.126A",
    "20A.S.210T",
    "20B.S.732A",
    "20B.S.796H",
    "20H.Beta.V2",
    "20I.Alpha.V1",
    "20J.Gamma.V3",
    "21A.Delta.S.K417",
    "21A.Delta",
    "21B.Kappa",
    "21C.Epsilon",
    "21D.Eta",
    "21F.Iota",
    "21G.Lambda",
    "21H.Mu",
    "21I.Delta",
    "21J.Delta",
    "21K.Omicron",
    "21L.Omicron",
    "21L",
    "21M.Omicron",
    "22A.Omicron",
    "22B22E",
    "22C.Omicron",
    "22D.Omicron",
    "22E.Omicron",
    "22F.Omicron",
    "23A.Omicron",
    "23B.Omicron",
    "23C.Omicron",
    "23D.Omicron",
    "23E.Omicron",
    "23F.Omicron",
    "23G.Omicron",
    "23H.Omicron",
    "23I.Omicron",
    "24A.Omicron",
    "24B.Omicron",
    "24C.Omicron",
    "24D.Omicron",
    "24E.Omicron",
    "24F.Omicron",
    "24G.Omicron",
    "24H.Omicron",
    "24I.Omicron",
]

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
    select_data = who_data.loc[who_data["Country_code"] == pycountry.countries.lookup(country).alpha_2]
    select_data.index = pd.to_datetime(select_data["Date_reported"], format=TEXT_DATE_FORMAT)
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
        country: Code for the country of interest
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
    cases_target = cases_data.loc[data_start: analysis_end]
    deaths_target = deaths_data.loc[data_start: analysis_end]
    init_start = analysis_start - timedelta(init_duration)
    init_end = analysis_start - timedelta(1)
    init_smooth_period = 7.0  # To correct values after resampling from weekly to daily
    init_data = cases_data.resample("D").asfreq().interpolate().loc[init_start: init_end] / init_smooth_period
    return cases_target, deaths_target, init_data


def get_hosp_series_from_owid_data(
    indicator: str,
    country: str,
) -> pd.Series:
    """Get OWID hospitalisation-related estimates for single indicator
    from the original raw data.

    Args:
        indicator: Name of the indicator
        country: Country name or code

    Returns:
        The data
    """
    hosp = pd.read_csv(DATA_PATH / "owid/owid_hosp.csv", index_col="date")
    hosp.index = pd.to_datetime(hosp.index)
    country = pycountry.countries.lookup(country).name
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
    country: str,
) -> pd.Series:
    """Get data for the number of isolates attributable to
    a particular variant in a particular country.

    Args:
        var: Nextclade name for the variant
        country: Name or code for the country of interest

    Returns:
        The data
    """
    offic_countries = ["CZE"]  # Countries needing official name for Nextclade data
    pycountry_obj = pycountry.countries.lookup(country)
    country_name = pycountry_obj.official_name if country in offic_countries else pycountry_obj.name
    data = pd.read_json(DATA_PATH / f"nextclade/{var}.json")[country_name]
    dates = pd.to_datetime(data["week"])
    return pd.Series(data["cluster_sequences"], index=dates)


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
    iso2 = pycountry.countries.lookup(country).alpha_2
    data_files = [pd.read_csv(RAW_MOB_PATH / f"{y}_{iso2}_Region_Mobility_Report.csv", index_col="date") for y in years]
    all_data = pd.concat(data_files)
    all_data.index = pd.to_datetime(all_data.index)
    nat_data = all_data.loc[pd.isna(all_data["sub_region_1"]) & pd.isna(all_data["metro_area"])]  # The rows at the national level
    nat_data = nat_data[[c for c in nat_data.columns if "change_from_baseline" in c]]  # The mobility columns
    nat_data = nat_data.rename(lambda c: c.replace("_percent_change_from_baseline", ""), axis=1)  # Simplify column naming
    nat_data = 1.0 + nat_data / 100.0  # Convert from percentage reduction to ratio
    return nat_data.sort_index()


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
    type_filt = (data["subgroup_var"] == "Primary Estimate") & (data["is_unity_aligned"] == "Unity-Aligned")
    return data.loc[time_filt & country_filt & nat_filt & type_filt, "serum_pos_prevalence"]


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
    nonresi_g_mob = (g_mob.loc[:, g_mob.columns != "residential"].mean(axis=1).rolling(7).mean().dropna())

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
    data_delay: int = 14,
) -> Tuple[pd.DataFrame]:
    """Get the standard epidemiological targets for a model run.

    Args:
        country: The country code
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


def get_all_var_data() -> dict:
    """Get all the downloaded NextClade data
    for all strains listed in VAR_MAP.

    Returns:
        Data in raw form
    """
    return {v: json.load(open(DATA_PATH / f"nextclade/{v}.json", "r")) for v in VAR_NAMES}


def get_country_var_data(
    raw_data: dict, 
    country: str,
) -> pd.DataFrame:
    """Extract the NextClade data available
    for a particular country.

    Args:
        raw_data: Raw NextClade data returned by get_all_var_data above
        country: The country name

    Returns:
        The variant data relevant to the country
    """
    data = pd.DataFrame()
    for v in raw_data:
        if country in raw_data[v]:
            var_data = raw_data[v][country]
            data[v] = pd.Series(var_data["cluster_sequences"], index=var_data["week"])
    data.index = pd.to_datetime(data.index)
    return data


def get_country_vars(
    country: str,
) -> pd.DataFrame:
    """Get all the CoVariants data for a particular country.

    Args:
        country: The country name

    Returns:
        The data
    """
    data = pd.DataFrame()
    for var in VAR_NAMES:
        all_var_data = pd.read_json(DATA_PATH / f"nextclade/{var}.json")
        if country in all_var_data:
            raw_data = all_var_data[country]
            dates = pd.to_datetime(raw_data["week"])
            vals = raw_data["cluster_sequences"]
            data[var] = pd.Series(vals, index=dates)
    return data


def find_relevant_vars(
    data: pd.DataFrame,
    date_cutoff: datetime,
    threshold_seqs: int,
) -> List[str]:
    """Find the variants that have a significant number of
    sequences before a particular date.

    Args:
        data: The full country-specific data (returned by get_country_vars)
        threshold_seqs: The number of sequences to consider the variant relevant

    Returns:
        The names of the relevant variants
    """
    relevant_vars = []
    for var in data:
        if data.loc[data.index < date_cutoff, var].sum() > threshold_seqs:
            relevant_vars.append(var)
    return relevant_vars
