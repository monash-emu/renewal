import pandas as pd
import numpy as np
import json
from pathlib import Path
import pycountry
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import yaml as yml
from numpyro import distributions as dist
import pycountry_convert as pc
from scipy.optimize import curve_fit


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
    select_data = who_data.loc[
        who_data["Country_code"] == pycountry.countries.lookup(country).alpha_2
    ]
    select_data.index = pd.to_datetime(select_data["Date_reported"], format=TEXT_DATE_FORMAT)
    return select_data[indicator].interpolate(method="linear").fillna(0.0)


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
    iso3 = pycountry.countries.lookup(country).alpha_3
    data = hosp[hosp["iso_code"] == iso3]
    return data.loc[data["indicator"] == indicator, "value"]


def get_country_hosps(country, start, end):
    admits = get_hosp_series_from_owid_data("Weekly new hospital admissions", country)
    filt_admits = admits[(start < admits.index) & (admits.index < end)]
    occup = get_hosp_series_from_owid_data("Daily hospital occupancy", country)
    filt_occup = occup[(start < occup.index) & (occup.index < end)]
    icu_admits = get_hosp_series_from_owid_data("Weekly new ICU admissions", country)
    # filt_icu_admits = icu_admits[(start < icu_admits.index) & (icu_admits.index < end)]
    icu_occup = get_hosp_series_from_owid_data("Daily ICU occupancy", country)
    # filt_icu_occup = icu_occup[(start < icu_occup.index) & (icu_occup.index < end)]
    if not filt_admits.empty:
        return filt_admits[::7], "admissions"
    elif not filt_occup.empty:
        return filt_occup[::7], "occupancy"
    # elif not filt_icu_admits.empty:
    #     return filt_icu_admits[::7], "icu_admits"
    # elif not filt_icu_occup.empty:
    #     return filt_icu_occup[::7], "icu_occup"
    else:
        return None, ""


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
    data_files = [
        pd.read_csv(RAW_MOB_PATH / f"{y}_{iso2}_Region_Mobility_Report.csv", index_col="date")
        for y in years
    ]
    all_data = pd.concat(data_files)
    all_data.index = pd.to_datetime(all_data.index)
    nat_data = all_data.loc[
        pd.isna(all_data["sub_region_1"]) & pd.isna(all_data["metro_area"])
    ]  # The rows at the national level
    nat_data = nat_data[
        [c for c in nat_data.columns if "change_from_baseline" in c]
    ]  # The mobility columns
    nat_data = nat_data.rename(
        lambda c: c.replace("_percent_change_from_baseline", ""), axis=1
    )  # Simplify column naming
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
    type_filt = data["subgroup_var"] == "Primary Estimate"
    unity_filt = data["is_unity_aligned"] == "Unity-Aligned"
    n_filt = data["denominator_value"] > 599
    all_filt = time_filt & country_filt & nat_filt & type_filt & unity_filt & n_filt
    return data.loc[all_filt, "serum_pos_prevalence"]


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
        "first_seed_rate": dist.Uniform(1.0, 100.0),
        "other_seed_rate": dist.Uniform(1.0, 100.0),
    }
    return duration_priors | beta_priors | other_priors


def get_worldbank_national_pop(
    iso3: str,
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


def get_apple_mobility(iso3: str) -> pd.DataFrame:
    all_data = pd.read_csv(
        DATA_PATH / "mobility/apple-mobility-test_apple_latest_apple-mobility-trends-report.csv",
        low_memory=False,
    )
    national_data = all_data.loc[all_data["country"].isnull()]
    national_data.index = pd.MultiIndex.from_arrays(
        [national_data["region"], national_data["transportation_type"]]
    )
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
    return country_df.rolling(7, center=True).mean().dropna()


def get_google_mobility(
    iso3,
) -> pd.DataFrame:
    """Get all the Google mobility fields.

    Args:
        iso3: Country identifier

    Returns:
        The mobility data
    """
    g_mob = pd.read_csv(DATA_PATH / f"mobility/{iso3}_gmob_data.csv", index_col=0)
    g_mob.index = pd.to_datetime(g_mob.index)
    return g_mob.rolling(7, center=True).mean().dropna()


def get_fb_mobility(iso3):
    fb_mob = pd.read_csv(DATA_PATH / f"mobility/{iso3}_fbmob_data.csv", index_col=0)["0"]
    fb_mob.index = pd.to_datetime(fb_mob.index)
    return 1.0 + fb_mob.rolling(7, center=True).mean().dropna()


def get_country_vacc_data(
    iso3: str,
) -> pd.DataFrame:
    """Get the initial course cumulative vaccination coverage
    data for a specific country.
    *** No full dose vaccination coverage data available from OWID.
    Have substituted Germany for Switzerland because these two
    countries had almost identical profiles of vaccine doses
    administered per person in the early phases of the roll-out.

    Args:
        iso3: ISO3 code for country

    Returns:
        The data
    """
    if iso3 == "KOR":
        country_name = pycountry.countries.lookup(iso3).common_name
    elif iso3 == "CHE":
        country_name = pycountry.countries.lookup("DEU").name
    else:
        country_name = pycountry.countries.lookup(iso3).name
    owid_vacc_filename = (
        "owid/share-of-people-who-completed-the-initial-covid-19-vaccination-protocol.csv"
    )
    data = pd.read_csv(DATA_PATH / owid_vacc_filename, index_col="Day")
    data.index = pd.to_datetime(data.index)
    col_name = "People fully vaccinated (cumulative, per hundred)"
    return data.loc[data["Entity"] == country_name, col_name]


def get_all_var_data() -> dict:
    """Get all the downloaded NextClade data
    for all strains listed in VAR_MAP.

    Returns:
        Data in raw form
    """
    return {v: json.load(open(DATA_PATH / f"nextclade/{v}.json", "r")) for v in VAR_NAMES}


def get_country_vars(
    iso3: str,
) -> pd.DataFrame:
    """Get all the CoVariants data for a particular country.

    Args:
        iso3: The country identifier

    Returns:
        The data
    """
    if iso3 == "CZE":
        country = pycountry.countries.lookup(iso3).official_name
    elif iso3 == "USA":
        country = pycountry.countries.lookup(iso3).alpha_3
    else:
        country = pycountry.countries.lookup(iso3).name
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


def get_prealpha_prop(iso3, min_var_samples):
    var_data = get_country_vars(iso3)
    var_data = var_data[var_data.sum(axis=1) >= min_var_samples]

    # Lithuania has no 20A.EU2
    prealpha_vars = ["20A.EU1"] if iso3 == "LTU" else ["20A.EU1", "20A.EU2"]
    prealpha_prop = var_data[prealpha_vars].sum(axis=1) / var_data.sum(axis=1)

    # Fluctuations in sample numbers in Portugal
    if iso3 == "PRT":
        prealpha_prop = prealpha_prop[prealpha_prop.index > datetime(2021, 1, 1)]

    return prealpha_prop


def get_pre_alpha_vars(
    country: str,
    min_samples: int = 5,
    end_date: datetime = datetime(2021, 6, 30),
    min_obs=5,
) -> pd.DataFrame:
    """Find the number of pre-Alpha variant samples
    and the total number of specimens, discarding
    data if zero or 100% pre-Alpha specimens.

    Args:
        country: The country identifier
        min_samples: Minimum number of samples for including a date
        end_date: End date for extracting the data
        min_obs: Threshold for the number of dates available
            before discarding all the data

    Returns:
        Number of pre-Alpha specimens, total specimens and
            proportion pre-Alpha by date
    """
    pre_alpha_vars = ["20A.EU1", "20A.EU2", "20B.S.732A", "21C.Epsilon"]
    var_data = get_country_vars(country)
    var_data = var_data[var_data.index < end_date]
    var_data = var_data[var_data.sum(axis=1) >= min_samples]
    avail_pre_alpha = [c for c in pre_alpha_vars if c in var_data.columns]
    pre_alpha_vals = var_data[avail_pre_alpha].sum(axis=1)
    totals = var_data.sum(axis=1)
    country_df = pd.DataFrame(
        {
            "pre_alpha": pre_alpha_vals,
            "totals": totals,
            "pre_alpha_prop": pre_alpha_vals / totals,
        }
    )
    out_df = country_df[(0.0 < country_df["pre_alpha_prop"]) & (country_df["pre_alpha_prop"] < 1.0)]
    if len(out_df) > min_obs:
        return out_df


def get_continent_data(
    continent: str,
) -> Dict[str, pd.DataFrame]:
    """Get the variant data for each country of
    a particular continent, ignoring the (small) pycountry
    countries that don't have a continent.

    Args:
        continent: The continent of interest

    Returns:
        The data by country of the continent of interest
    """
    invalid_countries = ["AQ", "TF", "EH", "PN", "SX", "TL", "UM", "VA"]
    all_countries = [c for c in pycountry.countries if c.alpha_2 not in invalid_countries]
    cont_data = {}
    for country in all_countries:
        if pc.country_alpha2_to_continent_code(country.alpha_2) == continent:
            iso3 = country.alpha_3
            cont_data[iso3] = get_pre_alpha_vars(iso3)
    return cont_data


def get_continent_pre_alpha_vars(
    data: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Get the overall pre-Alpha proportions for a continent
    from the country data for that continent.
    (Recalculate the proportions because these
    have been summed too.)

    Args:
        data: Data on variants by country for a continent,
            the output of get_continent_data

    Returns:
        The aggregated data for the continent
    """
    cont_data = pd.DataFrame()
    for d in data.values():
        if d is not None:
            cont_data = cont_data.add(d, fill_value=0.0)
    cont_data["pre_alpha_prop"] = cont_data["pre_alpha"] / cont_data["totals"]
    return cont_data


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
    starts = group_limits[group_limits == 1.0].index
    ends = group_limits[group_limits == -1.0].index
    return starts, ends


def pool_totals(
    starts: pd.DatetimeIndex,
    ends: pd.DatetimeIndex,
    data: pd.DataFrame,
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
    period_sums = pd.DataFrame(columns=["pre_alpha", "totals", "pre_alpha_prop"])
    indexes_to_remove = []
    for limits in zip(starts, ends):
        period = data.loc[limits[0] : limits[1]]
        average_date = period.index.mean()
        period_sums.loc[average_date] = period.sum()
        indexes_to_remove += list(period.index)
    new_data = pd.concat([period_sums, data.drop(index=indexes_to_remove)])
    new_data["pre_alpha_prop"] = new_data["pre_alpha"] / new_data["totals"]
    new_data.index = new_data.index.round("D")
    return new_data.sort_index()


def get_pooled_totals(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Combines the two preceding functions
    to get the totals after pooling for increases in the data.

    Args:
        var_data: The unadjusted data

    Returns:
        The adjusted data
    """
    group_starts, group_ends = find_increasing_groups(data["pre_alpha_prop"])
    return pool_totals(group_starts, group_ends, data)


def get_var_target(country):
    country_vars = get_pre_alpha_vars(country)
    index_iso2 = pycountry.countries.lookup(country).alpha_2
    continent = pc.country_alpha2_to_continent_code(index_iso2)
    if country_vars is not None:
        return get_pooled_totals(country_vars)["pre_alpha_prop"]
    elif continent != "AF":
        cont_data = get_continent_data(continent)
        country_vars = get_continent_pre_alpha_vars(cont_data)
        return get_pooled_totals(country_vars)["pre_alpha_prop"]


def cosine_function(t, start, end):
    period = end - start
    curve = lambda x: 0.5 * np.cos((x - start) * np.pi / period) + 0.5
    in_range = abs(t - start - period / 2.0) < period / 2.0
    conditions = [t <= start, in_range, start + period <= t]
    functions = [lambda x: 1.0, curve, lambda x: 0.0]
    return np.piecewise(t, conditions, functions)


def get_alpha_seed_time(var_prop):
    num_index = [t.timestamp() for t in var_prop.index]
    params, _ = curve_fit(cosine_function, num_index, var_prop, p0=[num_index[0], num_index[-1]])
    dt_ref_date = datetime(1970, 1, 1)
    date_num = params[0]
    date = (dt_ref_date + timedelta(seconds=date_num)).date()
    round_date = datetime.combine(date, datetime.min.time())
    return round_date, params
