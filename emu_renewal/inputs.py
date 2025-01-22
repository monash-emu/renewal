import pandas as pd
import re
from pathlib import Path
import pycountry
from typing import Tuple, List, Dict
from datetime import datetime, timedelta
import yaml as yml
from numpyro import distributions as dist


BASE_PATH = Path(__file__).parent.parent
OUTPUTS_PATH =  BASE_PATH / "outputs"
DATA_PATH = BASE_PATH / "data"
RAW_MOB_PATH = DATA_PATH / "mobility_raw"  # Will not push these larger original files
VAR_MAP = {
    "ba1": "21K.Omicron",
    "ba2": "21L.Omicron",
    "ba5": "22B.Omicron",
    "eu1": "20A.EU1",
    "eu2": "20A.EU2",
    "alpha": "20I.Alpha.V1"
}


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
    select_data = who_data.loc[who_data["Country"] == country]
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
    cases_target = cases_data.loc[data_start: analysis_end]
    deaths_target = deaths_data.loc[data_start: analysis_end]
    init_start = analysis_start - timedelta(init_duration)
    init_end = analysis_start - timedelta(1)
    init_smooth_period = 7.0  # To correct values after resampling from weekly to daily
    init_data = cases_data.resample("D").asfreq().interpolate().loc[init_start: init_end] / init_smooth_period
    return cases_target, deaths_target, init_data


def get_multicountry_df_from_who_data(indicator, countries):
    """May delete later - only used for Australia analyses"""
    data_dict = {i: get_indicator_series_from_who_data(indicator, i) for i in countries}
    return pd.DataFrame(data_dict)


def get_hosp_series_from_owid_data(
    indicator: str, 
    country: str,
) -> pd.Series:
    """Get OWID hospitalisation-related estimates for single indicator 
    from the original raw data.

    Args:
        indicator: Name of the indicator
        country: Name of the country

    Returns:
        The data
    """
    hosp = pd.read_csv(DATA_PATH / "owid/owid_hosp.csv", index_col="date")
    hosp.index = pd.to_datetime(hosp.index)
    data = hosp[hosp["entity"] == country]
    return data.loc[data["indicator"] == indicator, "value"]


def get_hosp_target(
    country: str, 
    analysis_start: datetime,
    analysis_end: datetime,
    analysis_to_data_delay: int,
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
    hosp_data = get_hosp_series_from_owid_data("Daily hospital occupancy", country)
    return hosp_data[data_start: analysis_end: 7]


def get_var_country_data(
    var: str,
    country: str,
) -> pd.Series:
    """Get data for the number of isolates attributable to
    a particular variant in a certain country.

    Args:
        var: Nextclade name for the variant
        country: Name of the country

    Returns:
        The data
    """
    data = pd.read_json(DATA_PATH / f"nextclade/{var}.json")[country]
    dates = pd.to_datetime(data["week"])
    return pd.Series(data["cluster_sequences"], index=dates)


def get_multivars_country_data(
    var_map: Dict[str, str],
    country: str,
) -> pd.DataFrame:
    """Get data for multiple variants.

    Args:
        var_map: Mapping from our names for the variants to Nextclade
        country: Name of the country

    Returns:
        The data
    """
    return pd.DataFrame({k: get_var_country_data(v, country) for k, v in var_map.items()})


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
    data = get_multivars_country_data(VAR_MAP, country)
    data["eu"] = data["eu1"] + data["eu2"]
    select_data = data[var_names]
    select_props = get_row_proportions(select_data)
    return select_props.loc[(start_date < select_data.index) & (select_data.index < end_date), "eu"]


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


def get_country_mobility(
    country: str,
) -> pd.DataFrame:
    """_summary_

    Args:
        country: Name of country of interest

    Returns:
        The data
    """
    years = range(2020, 2023)
    iso2 = pycountry.countries.get(name=country).alpha_2
    data_files = [pd.read_csv(RAW_MOB_PATH / f"{y}_{iso2}_Region_Mobility_Report.csv", index_col="date") for y in years]
    all_data = pd.concat(data_files)
    all_data.index = pd.to_datetime(all_data.index)
    national_data = all_data.loc[pd.isna(all_data["sub_region_1"])]
    national_data = national_data[[c for c in national_data.columns if "change_from_baseline" in c]]  # Extract the mobility columns
    national_data = national_data.rename(lambda c: c.replace("_percent_change_from_baseline", ""), axis=1)  # Simplify column naming
    national_data = 1.0 + national_data / 100.0  # Convert to relative change
    return national_data.sort_index()


def get_google_mobility(
    country: str,
) -> pd.Series:
    """Load previously saved Google mobility data for a requested country.

    Args:
        country: Name of the country of interest

    Returns:
        The data
    """
    iso2 = pycountry.countries.get(name=country).alpha_2
    data = pd.read_csv(DATA_PATH / f"mobility/{iso2}_mob_data.csv", index_col=0)
    data.index = pd.to_datetime(data.index)
    return data


def get_fb_mobility(
    country: str,
) -> pd.Series:
    """Load previously saved Facebook mobility data for a requested country.
    This was saved in raw form, which is proportional reduction, so add one.

    Args:
        country: Name of the country of interest

    Returns:
        The data
    """
    iso2 = pycountry.countries.get(name=country).alpha_2
    fb_mob = pd.read_csv(DATA_PATH / f"mobility/{iso2}_fbmob_data.csv", index_col=0)["0"]
    fb_mob = 1.0 + fb_mob.rolling(7).mean().dropna()
    fb_mob.index = pd.to_datetime(fb_mob.index)
    return fb_mob


def get_all_seroprev(
    lag: int=14,
) -> pd.Series:
    """Get all the seroprevalence data,
    including lagging by 14 days.

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
    country_filt = data["country"] == country
    time_filt = (start < data.index) & (data.index < end)
    nat_filt = data["estimate_grade"] == "National"
    type_filt = (data["subgroup_var"] == "Primary Estimate") & (data["is_unity_aligned"] == "Unity-Aligned")
    data = data.loc[time_filt & country_filt & nat_filt & type_filt]
    return data["serum_pos_prevalence"]


def extract_country_from_fb_mobility(
    data: pd.DataFrame, 
    country: str,
) -> pd.DataFrame:
    """Extract the Facebook mobility data relevant
    to a particular country from the raw data.

    Args:
        data: The raw Facebook mobility data (by subnational region)
        country: Name of the country of interest

    Returns:
        The country-specific data
    """
    iso3_code = pycountry.countries.get(name=country).alpha_3
    country_data = data[data["country"] == iso3_code]
    return country_data[country_data.notna()]


def extract_country_from_euro_pop(
    data: pd.Series, 
    country: str,
) -> pd.Series:
    """Extract the region populations
    for a particular country from the raw data.

    Args:
        data: The raw Estat data for subnational populations
        country: Name of the country of interest

    Returns:
        The country-specific data
    """
    iso2_code = pycountry.countries.get(name=country).alpha_2
    pattern = re.compile(f"^{iso2_code}[A-Z0-9]{{1,3}}:.+$")
    pop_map = data[[i for i in data.index if pattern.match(i)]]
    pop_map.index = [i.split(":")[1] for i in pop_map.index]
    return pop_map


string_pairs = [
    ["Î", "-"],
    ["ô", "-"],
    ["é", "-"],
    ["’", "-"],
    ["'", "-"],
    [" — ", "-"],
    ["Ile de ", "-le-de-"],
    ["ó", "-"],
    ["í", "-"],
    ["ñ", "-"],
    ["è", "-"],
    ["Comunitat", "Comunidad"],
    ["Sicily", "Sicilia"],
    ["Valle d-Aosta/Vall-e d-Aoste", "Valle d-Aosta"],
    ["Apulia", "Puglia"],
]


def replace_chars_for_map(
    region_names: pd.Series,
) -> pd.Series:
    """Adjust subnational regional strings
    to be easier to match.

    Args:
        region_data: The unadjusted region names

    Returns:
        The adjusted region names
    """
    for pair in string_pairs:
        region_names = region_names.str.replace(*pair)
    return region_names


def get_weighted_average_from_df(
    mob: pd.DataFrame,
    val_col: str,
    weight_col: str,
) -> pd.Series:
    """Calculate a weighted average using 
    two columns of a dataframe.

    Args:
        val_col: Name of the column with the data
        weight_col: Name of the column with the weights

    Returns:
        The weighted averages
    """
    def weighted_average(data):
        weights = data[weight_col]
        vals = data[val_col]
        return (vals * weights).sum() / weights.sum()
    return mob.groupby(mob.index).apply(weighted_average)


def get_standard_priors() -> Dict[str, dist.Distribution]:
    """Load the priors from the yml and combine with 
    standard hard-coded priors.

    Returns:
        The prior distributions        
    """
    loaded_priors = yml.safe_load(open(BASE_PATH / "emu_renewal/priors.yml", "r"))
    duration_priors = {
        k: dist.TruncatedNormal(v["mean"], v["sd"], low=1.0, high=v["mean"] * 2.5) 
        for k, v in loaded_priors["durations"].items()
    }
    beta_priors = {
        k: dist.Beta(v["alpha"], v["beta"]) 
        for k, v in loaded_priors["beta"].items()
    }
    other_priors = {
        "alpha_relinfect": dist.TruncatedNormal(1.25, 0.1, low=1.0, high=1.5),
        "rt_init": dist.Normal(0.0, 0.5),
        "shared_dispersion": dist.HalfNormal(0.5),
    }
    return duration_priors | beta_priors | other_priors
