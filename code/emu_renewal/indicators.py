from typing import List, Tuple, Dict, Union
from datetime import datetime, timedelta
import pandas as pd
import pycountry
from numpyro import distributions as dist
import pycountry_convert as pc
from os import listdir as ls

from emu_renewal.constants import (
    DATA_PATH,
    DEATHS_WEIGHT,
    CASES_START,
    SEROPREV_EXTREME,
    SEROPREV_WEIGHT,
    SEROPREV_MIN_SIZE,
    VAR_WEIGHT,
    ALPHA_DELTA_EXCEPTS,
    ALPHA_PERIOD_START,
    ALPHA_DELTA_TRANS,
    DELTA_INCLUSION_DATE,
    MIN_DELTA_PROP,
    DELTA_PERIOD_END,
    ALPHA_FULL_REPLACE_DATE,
    POST_SIM_DATE,
    POST_SIM_DATE,
    POST_SIM_DATE,
    BA2_PERIOD_START,
    BA2_PERIOD_END,
    BA5_PERIOD_START,
    BA5_PERIOD_END,
    ZERO_IND_REPLACEMENT,
    SEROPREV_START_DELAY,
    WHO_DATE_FORMAT,
    ALREADY_WEEKLY_ADMIT_COUNTRIES,
    ALREADY_WEEKLY_OCCUP_COUNTRIES,
    ANTIBODY_DELAY,
    PREV_KEY,
    CODE_DATE_FORMAT,
    LATE_DELTA_WEIGHT,
    LATE_DELTA_TIME,
    MIN_VAR_SEQS,
    MIN_VAR_DATES,
    PREALPHA_IDENTIFIERS,
    BA2_IDENTIFIER,
)
from emu_renewal.inputs import (
    get_income_group,
    get_incr_pooled_totals,
    get_owid_hosp_series,
    find_decreasing_groups,
)
from emu_renewal.targets import StandardDispTarget, UnivariateDispersionTarget, StandardPropTarget


def get_who_indicator(
    indicator: str,
    iso3: str,
) -> pd.Series:
    """Get WHO estimates for single indicator from the original raw data.

    Args:
        indicator: Name of the indicator
        iso3: Country identifier

    Returns:
        The data
    """
    who_data = pd.read_csv(DATA_PATH / "who/WHO-COVID-19-global-data_21_8_24.csv")
    iso2 = pycountry.countries.lookup(iso3).alpha_2
    select_data = who_data.loc[who_data["Country_code"] == iso2]
    select_data.index = pd.to_datetime(select_data["Date_reported"], format=WHO_DATE_FORMAT)
    return select_data[indicator]


def get_deaths_target(
    iso3: str,
    start: datetime,
    end: datetime,
) -> Tuple[int, Dict[str, StandardDispTarget]]:
    """Get the deaths calibration target.

    Args:
        iso3: The country identifier
        start: The calibration start time
        end: The calibration end time

    Returns:
        Number of observations in the deaths series
        The target

    Notes
    -----
    The number of deaths by week reported by WHO
    was used as the first calibration target for all countries.
    Any values of zero in this series were replaced with a
    value of {ZERO_IND_REPLACEMENT} to enable comparison to
    modelled outputs on the log scale.
    Deaths was the one of two indicators
    for which a common dispersion parameter was used
    for the distribution comparison of the modelled value.
    The value for weighting the deaths indicator time series
    was set to {DEATHS_WEIGHT}.
    """
    data = get_who_indicator("New_deaths", iso3)
    data = data.interpolate(method="linear").fillna(0.0)
    data[data == 0.0] = ZERO_IND_REPLACEMENT
    mask = (start < data.index) & (data.index < end)
    select_data = data.loc[mask]
    target = StandardDispTarget(select_data, weight=DEATHS_WEIGHT)
    return len(select_data), {"weekly_deaths": target}


def get_cases_target(
    iso3: str,
    start: datetime,
    end: datetime,
    n_deaths: int,
) -> Dict[str, StandardDispTarget]:
    """Get the cases calibration target.

    Args:
        iso3: The country identifier
        start: The calibration start time
        end: The calibration end time
        n_deaths: The number of deaths observations

    Returns:
        The target

    Notes
    -----
    The number of cases by week reported by WHO
    was used as the second calibration target for all countries.
    Linear interpolation was used to replace missing values, and
    as for deaths, any zero values were replaced with a value of 0.5.
    Cases was the second indicator for which
    a common dispersion parameter was applied.
    A target weight was applied to the series of cases
    such that the weight for each case observation point
    was the same as for each death observation.
    """
    data = get_who_indicator("New_cases", iso3)
    data = data.interpolate(method="linear").fillna(0.0)
    data[data == 0.0] = 0.5
    cases_start = max([CASES_START, start])
    mask = (cases_start < data.index) & (data.index < end)
    target = data.loc[mask]
    weight = DEATHS_WEIGHT * len(target) / n_deaths
    target = StandardDispTarget(target, weight=weight)
    return {"weekly_cases": target}


def get_owid_hosps(
    country: str,
    start: datetime,
    end: datetime,
) -> Tuple[Union[pd.Series, None], str]:
    """Select the hospitalisation calibration target.
    Code needs to account for some countries that already report
    their hospitalisation indicator weekly.

    Args:
        country: Country identifier
        start: Data comparison start time
        end: Analysis end time

    Returns:
        Tuple of two elements:
            - The calibration data for comparison
            - The name of the indicator for comparison

    Notes
    -----
    A single hospitalisation indicator used for calibration was
    chosen using a hierarchical approach.
    In selecting the indicator, the number of new admissions was preferred
    over estimates of total bed occupancy, and total hospital
    indicators were preferred over ICU indicators.
    The final hierarchy of indicators was:
    __RETURN__1. New hospital admissions
    __RETURN__2. Hospital occupancy
    __RETURN__3. New ICU admissions
    __RETURN__4. ICU occupancy
    __RETURN__5. No hospital indicator
    __RETURN____RETURN__
    That is, the highest ranked indicator was used based on data availability,
    and no hospital indicator was incorporated if none were available.
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


def get_hosp_target(
    iso3: str,
    start: datetime,
    end: datetime,
    n_deaths: int,
) -> Dict[str, StandardDispTarget]:
    """Get the hospitalisations calibration target.

    Args:
        iso3: The country identifier
        start: The calibration start time
        end: The calibration end time
        n_deaths: The number of deaths observations

    Returns:
        The hospitalisation calibration target

    Notes
    -----
    This indicator was the final calibration target for which
    a common dispersion parameter was applied.
    As for cases, a weight was applied to the hospitalisation series
    such that the weight for each observation point
    was the same as for each death observation.
    """
    data, output_name = get_owid_hosps(iso3, start, end)
    if data is None:
        return {}
    mask = (start < data.index) & (data.index < end)
    select_data = data.loc[mask]
    if select_data.empty:
        return {}
    weight = DEATHS_WEIGHT * len(select_data) / n_deaths
    target = StandardDispTarget(select_data, weight=weight)
    return {output_name: target}


def get_all_seroprev() -> pd.Series:
    """Get all the seroprevalence data.

    Returns:
        All SeroTracker data

    Notes
    -----
    Seroprevalence data was obtained from
    [SeroTracker](https://github.com/serotracker/sars-cov-2-data/raw/refs/heads/main/serotracker_dataset.csv)
    on 11 December 2024,
    with the date for each serosurvey estimate calculated as the
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


def filter_seroprev(
    iso3: str,
    data: pd.DataFrame,
) -> pd.Series:
    """Get and filter the seroprevalence data.

    Args:
        iso3: Country identifier
        data: The raw data from get_all_seroprev

    Returns:
        Filtered data to use as target

    Notes
    -----
    We filtered the SeroTracker data to include
    only the estimate reported as primary from
    Unity-aligned national-level surveys for
    which the number of participants was at least {SEROPREV_MIN_SIZE}.
    We also considered only a maximum of one seroprevalence
    value for any given date (keeping the largest
    estimate of three surveys done on the same day for Mexico).
    """
    country = pycountry.countries.lookup(iso3).name
    country_filt = data["country"] == country
    nat_filt = data["estimate_grade"] == "National"
    type_filt = data["subgroup_var"] == "Primary Estimate"
    unity_filt = data["is_unity_aligned"] == "Unity-Aligned"
    n_filt = data["denominator_value"] >= SEROPREV_MIN_SIZE
    all_filt = country_filt & nat_filt & type_filt & unity_filt & n_filt
    filt_data = data[all_filt]
    return filt_data[[not i for i in filt_data.index.duplicated()]]


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

    Notes
    -----
    For any consecutive estimates for which a lower estimate
    followed an immediately preceding higher estimate,
    we pooled these two estimates and placed the pooled
    estimate at the mid-point of the dates of the two estimates.
    We repeatedly applied this process until seroprevalence
    estimates were monotonically increasing over time.
    """
    while not data[PREV_KEY].is_monotonic_increasing:
        starts, ends = find_decreasing_groups(data[PREV_KEY])
        data = pool_seroprev_totals(starts, ends, data)
    return data[PREV_KEY]


def get_seroprev_target(
    iso3: str,
    start: datetime,
    end: datetime,
    continent: str,
) -> Dict[str, UnivariateDispersionTarget]:
    """Construct the seroprevalence calibration target.

    Args:
        iso3: The country identifier
        start: The calibration start time
        end: The calibration end time
        continent: The country's continent

    Returns:
        The seroprevalence calibration target

    Notes
    -----
    We compared the modelled
    proportion ever infected against the seroprevalence
    reported at least six months ({SEROPREV_START_DELAY} days)
    after the start of the simulation,
    because a comparison against early seroprevalence estimates
    would not account for waves of transmission prior to
    the start of the simulation.
    We discarded seroprevalence estimates that were less than
    {SEROPREV_EXTREME}% away from a value of zero or 100%.
    We also ignored seroprevalence estimates from
    low and lower middle income countries of Africa, because
    we were unable to obtain good fits for several of these countries
    while also maintaining plausible detection/severity parameters
    (i.e. case detection rate, hospital admission rate
    and infection fatality rate).
    That is, we applied much lower priors for these parameters
    in these countries, although the modelled attack rate
    still remained well below seroprevalence estimates for
    some countries.
    Last, we ignored seroprevalence estimates for Australia,
    for which the analysis was run largely through 2022,
    during which time seroprevalence values were much higher.
    For countries for which seroprevalence calibration targets
    were available, we assigned a target weight to this indicator
    of {SEROPREV_WEIGHT} (which is an arbitrary quantity,
    but can be interpreted with reference to the deaths indicator
    weight of {DEATHS_WEIGHT}).
    """
    income = get_income_group(iso3)
    if continent == "OC" or continent in "AF" and income in ["Lower middle income", "Low income"]:
        return {}
    seroprev = get_all_seroprev()
    seroprev = filter_seroprev(iso3, seroprev)
    if seroprev.empty:
        return {}
    data = get_seroprev_pooled_totals(seroprev)
    time_filt = (start + timedelta(SEROPREV_START_DELAY) < data.index) & (data.index < end)
    data = data[time_filt]
    seroprev_mask = (SEROPREV_EXTREME / 1e2 < data) & (data < 1.0 - SEROPREV_EXTREME / 1e2)
    data = data[seroprev_mask]
    if data.empty:
        return {}
    target = UnivariateDispersionTarget(data, dist.Normal, "seroprev_disp", weight=SEROPREV_WEIGHT)
    return {"seropos": target}


def get_country_vars(
    iso3: str,
) -> pd.DataFrame:
    """Get all the covariants data for a particular country.

    Args:
        iso3: The country identifier

    Returns:
        The data

    Notes
    -----
    Reports of the number of isolates of specific variants of SARS-CoV-2
    were obtained from the
    [covariants](https://github.com/hodcroftlab/covariants/raw/refs/heads/master/cluster_tables/)
    GitHub repository. Each variant-specific file was downloaded
    used to create country-specific tables of the variant-specific
    counts by date.
    """
    if iso3 == "CZE":
        country = pycountry.countries.lookup(iso3).official_name
    elif iso3 == "USA":
        country = iso3
    else:
        country = pycountry.countries.lookup(iso3).name
    data = pd.DataFrame()
    var_names = [v.split(".json")[0] for v in ls(DATA_PATH / "nextclade") if v.startswith("2")]
    for var in var_names:
        var_data = pd.read_json(DATA_PATH / f"nextclade/{var}.json")
        if country in var_data:
            raw_data = var_data[country]
            dates = pd.to_datetime(raw_data["week"])
            vals = raw_data["cluster_sequences"]
            data[var] = pd.Series(vals, index=dates)
    return data.astype(float)


def get_specific_var_props(
    data: pd.DataFrame,
    var_name: str,
    rel_cols: List[str],
    end_date: datetime,
) -> Union[pd.DataFrame, None]:
    """Get the total number and proportion
    of sequences attributable to a particular variant.

    Args:
        data: The country variant data
        var_name: Our name for the variant of interest
        rel_cols: The names of the relevant columns for the variant
        end_date: A date after which data are discarded

    Returns:
        The data for the variant of interest

    Notes
    -----
    Variant data was considered for dates on
    which at least {MIN_VAR_SEQS} sequences were available
    for the country considered.
    Further, we required at least {MIN_VAR_DATES} such dates be available
    for that country's variant data to be used as a 
    calibration target.
    """
    data = data[data.index < end_date]
    data = data[data.sum(axis=1) >= MIN_VAR_SEQS]
    rel_cols = [c for c in rel_cols if c in data.columns]
    vals = data[rel_cols].sum(axis=1)
    totals = data.sum(axis=1)
    country_df = pd.DataFrame(
        {
            var_name: vals,
            "totals": totals,
            f"{var_name}_prop": vals / totals,
        }
    )
    country_df
    if len(country_df) > MIN_VAR_DATES:
        return country_df


def extract_specific_var(
    var_data: pd.DataFrame,
    var_name: str,
) -> Union[pd.DataFrame, None]:
    """Find the proportion of variant sequences
    attributable to a specific variant type.

    Args:
        var_data: All the raw variant data
        var_name: The name of the variant of interest

    Returns:
        Data for the number of pre-Alpha specimens, total specimens and
            proportion pre-Alpha by date - where available

    Notes
    -----
    The identifiers used to identify variants prior to Alpha
    were: {PREALPHA_IDENTIFIERS}.
    The text "Delta" was used to identify Delta variants and
    the text {BA2_IDENTIFIER} was used to distinguish the BA.2
    variant from later variants (modelled as BA.5).
    """
    if var_name == "alpha":
        prealpha_cols = PREALPHA_IDENTIFIERS.split(", ")
        cols = [c for c in var_data.columns if c not in prealpha_cols]
    elif var_name == "delta":
        cols = [c for c in var_data.columns if "Delta" in c]
    elif var_name == "ba2":
        cols = [BA2_IDENTIFIER]
    elif var_name == "ba5":
        cols = [c for c in var_data.columns if c != BA2_IDENTIFIER]
    end_date = ALPHA_FULL_REPLACE_DATE if var_name == "alpha" else POST_SIM_DATE
    return get_specific_var_props(var_data, var_name, cols, end_date)


def get_continent_data(
    cont: str,
    var: str,
) -> Dict[str, pd.DataFrame]:
    """Get the variant data for each country of
    a particular continent, ignoring the (small) pycountry
    countries that don't have an associated continent.

    Args:
        continent: The continent identifier
        var: The variant of interest

    Returns:
        The data by country of the continent of interest
    """
    no_cont_countries = ["AQ", "TF", "EH", "PN", "SX", "TL", "UM", "VA"]
    countries = [c for c in pycountry.countries if c.alpha_2 not in no_cont_countries]
    cont_data = {}
    for c in countries:
        if pc.country_alpha2_to_continent_code(c.alpha_2) == cont:
            iso3 = c.alpha_3
            var_data = get_country_vars(iso3)
            cont_data[iso3] = extract_specific_var(var_data, var)
    return cont_data


def get_continent_vars(
    data: Dict[str, pd.DataFrame],
    var_name: str = "prealpha",
) -> Dict[str, pd.DataFrame]:
    """Get the overall variant proportions for a continent
    from the country data for that continent.
    (Recalculate the proportions because these
    have been summed too.)

    Args:
        data: Data on variants by country for a continent,
            the output of get_continent_data
        var_name: The variant of interest

    Returns:
        The aggregated data for the continent
    """
    cont_data = pd.DataFrame()
    for d in data.values():
        if d is not None:
            cont_data = cont_data.add(d, fill_value=0.0)
    cont_data[f"{var_name}_prop"] = cont_data[var_name] / cont_data["totals"]
    return cont_data


def get_var_target(
    var_data: pd.DataFrame,
    continent: str,
    var_name: str,
) -> pd.DataFrame:
    """Get the variant-specific data (for Alpha or Delta)
    for a country, using the continent data
    if not available for the country.

    Args:
        var_data: All variant data for the country
        continent: Continent identifier
        var_name: Variant identifier

    Returns:
        The data

    Notes
    -----
    To obtain data to use as calibration targets for
    both the Alpha and the Delta variants,
    we used the totals for the country analysed where available.
    If data were not available for the country,
    we used pooled data from all the other countries
    from the same continent where available.
    """
    data = extract_specific_var(var_data, var_name)
    if data is None:
        cont_data = get_continent_data(continent, var_name)
        return get_continent_vars(cont_data, var_name)
    else:
        return data


def get_alpha_info(
    iso3: str,
    var_data: pd.DataFrame,
    continent: str,
    end_time: datetime,
    delta_targ: Dict[str, StandardPropTarget],
) -> Tuple[List[str], Dict[str, StandardPropTarget], List[datetime]]:
    """_summary_

    Args:
        iso3: The country identifier
        var_data: All the variant data for the country
        continent: The continent identifier
        end_time: The analysis end date
        delta_targ: The Delta target, output of get_delta_info

    Returns:
        - A list containing the name of the variant (if included)
        - The calibration target for Alpha (if included)
        - A list containing the first identification date of Alpha (if included)

    Notes
    -----
    For countries of all continents other than those 
    in Oceania (Australia only) and Africa,
    a target for the Alpha variant 
    was included in our calibration algorithm.
    Calibration against data for Alpha started from the beginning
    of the simulation period (from {ALPHA_PERIOD_START}).
    The periods for calibration against the Alpha and the Delta
    variants were set so as to be mutually exclusive in time.
    Specifically, the date to transition from calibrating 
    against available data for the Alpha to calibrating 
    against data for Delta was set as {ALPHA_DELTA_TRANS}. 
    Exceptions were made for several Asian countries
    for which this transition date was set one month earlier and two countries
    of North America for which it was set six weeks later.
    If this date occurred after the end of the simulation,
    the Alpha calibration period continued to the end of the simulation.
    As with the other variants and for the seroprevalence target,
    decreasing values for the proportion of sequences attributable
    to Alpha were recursively pooled to ensure they were strictly increasing.
    The target weight for the Alpha target was set to be {VAR_WEIGHT}.
    """
    if continent in ["OC", "AF"]:
        return [], {}, []
    data = get_var_target(var_data, continent, "alpha")
    alpha_start = datetime.strptime(ALPHA_PERIOD_START, CODE_DATE_FORMAT)
    ad_trans_req = datetime.strptime(ALPHA_DELTA_TRANS, CODE_DATE_FORMAT)
    ad_trans = ALPHA_DELTA_EXCEPTS[iso3] if iso3 in ALPHA_DELTA_EXCEPTS else ad_trans_req
    alpha_end = min([ad_trans, end_time]) if delta_targ else end_time
    mask = (alpha_start < data.index) & (data.index < alpha_end)
    target = get_incr_pooled_totals(data[mask], "alpha")["alpha_prop"]
    var_start = target.index[0]
    return ["alpha"], {"prop_alpha": StandardPropTarget(target, weight=VAR_WEIGHT)}, [var_start]


def get_delta_info(
    iso3: str,
    var_data: pd.DataFrame,
    continent: str,
    end_time: datetime,
) -> Tuple[List[str], Dict[str, StandardPropTarget], List[datetime]]:
    """Get the required information relating
    to the Delta variant to run an analysis.

    Args:
        iso3: The country identifier
        var_data: All the variant data for the country
        continent: The continent identifier
        end_time: The analysis end date

    Returns:
        - A list containing the name of the variant (if included)
        - The calibration target for Delta (if included)
        - A list containing the first identification date of Delta (if included)

    Notes
    -----
    For all countries other than Australia,
    the Delta variant was included if the end date of the
    calibration fell later than {DELTA_INCLUSION_DATE}.
    Values were again pooled to ensure they were strictly increasing.
    The target weight for calibration to Delta was set to be {VAR_WEIGHT}
    for most countries. Exceptions were made if the target for data
    emerged towards the very end of the calibration last
    ({LATE_DELTA_WEIGHT} days), in which case a higher weight
    (of {LATE_DELTA_WEIGHT}) was needed to capture this late emergence
    with fewer data points.
    """
    delta_inc_date = datetime.strptime(DELTA_INCLUSION_DATE, CODE_DATE_FORMAT)
    delta_end_date = datetime.strptime(DELTA_PERIOD_END, CODE_DATE_FORMAT)
    ad_trans = datetime.strptime(ALPHA_DELTA_TRANS, CODE_DATE_FORMAT)
    if continent == "OC" or end_time < delta_inc_date:
        return [], {}, []
    data = get_var_target(var_data, continent, "delta")
    delta_start = ALPHA_DELTA_EXCEPTS[iso3] if iso3 in ALPHA_DELTA_EXCEPTS else ad_trans
    delta_end = min([delta_end_date, end_time])
    mask = (delta_start < data.index) & (data.index < delta_end)
    target = get_incr_pooled_totals(data[mask], "delta")["delta_prop"]
    if target is None or target.empty or max(target) < MIN_DELTA_PROP:
        return [], {}, []
    is_end_period = (end_time - target.index[0]).days < LATE_DELTA_TIME
    weight = LATE_DELTA_WEIGHT if is_end_period else VAR_WEIGHT
    var_start = target.index[0]
    return ["delta"], {"prop_delta": StandardPropTarget(target, weight=weight)}, [var_start]


def get_ba2_target(var_data, continent):
    if continent == "OC":
        ba2_data = extract_specific_var(var_data, "ba2")
        period_mask = (BA2_PERIOD_START < ba2_data.index) & (ba2_data.index < BA2_PERIOD_END)
        filt_data = ba2_data[period_mask]
        return filt_data["ba2_prop"]


def get_ba5_target(var_data, continent):
    if continent == "OC":
        ba5_data = extract_specific_var(var_data, "ba5")
        period_mask = (BA5_PERIOD_START < ba5_data.index) & (ba5_data.index < BA5_PERIOD_END)
        filt_data = ba5_data[period_mask]
        return filt_data["ba5_prop"]


def get_ba2_info(var_data, continent):
    data = get_ba2_target(var_data, continent)
    if data is None:
        return [], {}, []
    var_start = data.index[0]
    return ["ba2"], {"prop_ba2": StandardPropTarget(data, weight=VAR_WEIGHT)}, [var_start]


def get_ba5_info(var_data, continent):
    data = get_ba5_target(var_data, continent)
    if data is None:
        return [], {}, []
    var_start = data.index[0]
    return ["ba5"], {"prop_ba5": StandardPropTarget(data, weight=VAR_WEIGHT)}, [var_start]
