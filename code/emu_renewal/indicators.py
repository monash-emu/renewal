from typing import List, Tuple, Dict, Union
from datetime import datetime, timedelta
import pandas as pd
import pycountry
from numpyro import distributions as dist
import pycountry_convert as pc

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
    VAR_NAMES,
    BA2_PERIOD_START,
    BA2_PERIOD_END,
    BA5_PERIOD_START,
    BA5_PERIOD_END,
    ZERO_IND_REPLACEMENT,
    SEROPREV_START_DELAY,
    WHO_DATE_FORMAT,
)
from emu_renewal.inputs import (
    get_who_indicator,
    get_owid_hosps,
    get_owid_hosps,
    get_all_seroprev,
    get_seroprev_pooled_totals,
    get_income_group,
    get_incr_pooled_totals,
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
    One hospitalisation indicator was also used for
    each country, where available.
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


def get_filtered_seroprev(
    iso3: str,
) -> pd.Series:
    """Get and filter the seroprevalence data.

    Args:
        iso3: Country identifier

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
    data = get_all_seroprev()
    country = pycountry.countries.lookup(iso3).name
    country_filt = data["country"] == country
    nat_filt = data["estimate_grade"] == "National"
    type_filt = data["subgroup_var"] == "Primary Estimate"
    unity_filt = data["is_unity_aligned"] == "Unity-Aligned"
    n_filt = data["denominator_value"] >= SEROPREV_MIN_SIZE
    all_filt = country_filt & nat_filt & type_filt & unity_filt & n_filt
    filt_data = data[all_filt]
    return filt_data[[not i for i in filt_data.index.duplicated()]]


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
    proportion ever infected against the reported seroprevalence
    reported at least six months ({SEROPREV_START_DELAY} days)
    after the start of the simulation,
    because a comparison against early seroprevalence estimates
    would not account for waves of transmission prior to
    the start of the simulation.
    For any consecutive estimates for which a lower estimate
    followed an immediately preceding higher estimate,
    we pooled these two estimates and placed the pooled
    estimate at the mid-point of the dates of the two estimates.
    We repeatedly applied this process until seroprevalence
    estimates were monotonically increasing over time.
    We discarded seroprevalence estimates that were less than
    {SEROPREV_EXTREME}% away from a value of zero or 100%.
    We also ignored seroprevalence estimates from
    low and lower middle income countries of Africa, because
    we were unable to obtain good fits for several of these countries
    while also maintaining plausible detection parameters
    (e.g. case detection rate, hospital admission rate
    and infection fatality rate).
    Last, we ignored seroprevalence estimates for Australia,
    for which the analysis was run largely through 2022
    during which seroprevalence values were much higher.
    For countries for which seroprevalence calibration targets
    were available, we assigned a target weight to this indicator
    of {SEROPREV_WEIGHT}.
    """
    income = get_income_group(iso3)
    if continent == "OC" or continent in "AF" and income in ["Lower middle income", "Low income"]:
        return {}
    seroprev = get_filtered_seroprev(iso3)
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
    """Get all the CoVariants data for a particular country.

    Args:
        iso3: The country identifier

    Returns:
        The data
    """
    if iso3 == "CZE":
        country = pycountry.countries.lookup(iso3).official_name
    elif iso3 == "USA":
        country = iso3
    else:
        country = pycountry.countries.lookup(iso3).name
    data = pd.DataFrame()
    for var in VAR_NAMES:
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
    min_samples: int = 5,
    min_obs: int = 5,
    min_prop: float = 0.0,
) -> Union[pd.DataFrame, None]:
    """Get the total number and proportion
    of sequences attributable to a particular variant.

    Args:
        data: The country variant data
        var_name: Our name for the variant of interest
        rel_cols: The names of the relevant columns for the variant
        end_date: A date after which data are discarded
        min_samples: Minimum number of dates needed to use the data
        min_obs: Minimum number of sequences at a date for inclusion in data
        min_prop: Minimum proportion attributable to the variant
            or to other non-index variants for inclusion

    Returns:
        The data for the variant of interest
    """
    data = data[data.index < end_date]
    data = data[data.sum(axis=1) >= min_samples]
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
    above_min_prop = min_prop < country_df[f"{var_name}_prop"]
    below_max_prop = country_df[f"{var_name}_prop"] < 1.0 - min_prop
    out_df = country_df[above_min_prop & below_max_prop]
    if len(out_df) > min_obs:
        return out_df


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
    """
    prealpha_cols = ["20A.EU1", "20A.EU2", "20B.S.732A", "21C.Epsilon"]
    alpha_cols = [c for c in var_data.columns if c not in prealpha_cols]
    delta_cols = [c for c in var_data.columns if "Delta" in c]
    ba2_col = ["21L.Omicron"]
    ba5_cols = [c for c in var_data.columns if c not in ba2_col]
    rel_cols = {
        "alpha": alpha_cols,
        "delta": delta_cols,
        "ba2": ba2_col,
        "ba5": ba5_cols,
    }
    end_dates = {
        "alpha": ALPHA_FULL_REPLACE_DATE,
        "delta": POST_SIM_DATE,
        "ba2": POST_SIM_DATE,
        "ba5": POST_SIM_DATE,
    }
    return get_specific_var_props(var_data, var_name, rel_cols[var_name], end_dates[var_name])


def get_continent_data(
    continent: str,
    var: str,
) -> Dict[str, pd.DataFrame]:
    """Get the variant data for each country of
    a particular continent, ignoring the (small) pycountry
    countries that don't have an associated continent.

    Args:
        continent: The continent of interest
        var: The variant of interest

    Returns:
        The data by country of the continent of interest
    """
    no_continent_countries = ["AQ", "TF", "EH", "PN", "SX", "TL", "UM", "VA"]
    countries = [c for c in pycountry.countries if c.alpha_2 not in no_continent_countries]
    cont_data = {}
    for country in countries:
        if pc.country_alpha2_to_continent_code(country.alpha_2) == continent:
            var_data = get_country_vars(country.alpha_3)
            iso3 = country.alpha_3
            cont_data[iso3] = extract_specific_var(var_data, var)
    return cont_data


def get_var_target(var_data, continent, var_name):
    data = extract_specific_var(var_data, var_name)
    if data is None:
        cont_data = get_continent_data(continent, var_name)
        return get_continent_vars(cont_data, var_name)
    else:
        return data


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


def get_alpha_info(iso3, var_data, continent, end_time, delta_targ):
    if continent in ["OC", "AF"]:
        return [], {}, []
    data = get_var_target(var_data, continent, "alpha")
    alpha_start = ALPHA_DELTA_EXCEPTS[iso3] if iso3 in ALPHA_DELTA_EXCEPTS else ALPHA_DELTA_TRANS
    alpha_end = end_time if delta_targ else min([alpha_start, end_time])
    mask = (ALPHA_PERIOD_START < data.index) & (data.index < alpha_end)
    pooled_data = get_incr_pooled_totals(data[mask], "alpha")
    target = pooled_data["alpha_prop"]
    var_start = target.index[0]
    return ["alpha"], {"prop_alpha": StandardPropTarget(target, weight=VAR_WEIGHT)}, [var_start]


def get_delta_info(iso3, var_data, continent, end_time):
    if continent or end_time < DELTA_INCLUSION_DATE:
        return [], {}, []
    data = get_var_target(var_data, continent, "delta")
    delta_start = ALPHA_DELTA_EXCEPTS[iso3] if iso3 in ALPHA_DELTA_EXCEPTS else ALPHA_DELTA_TRANS
    delta_end = min([DELTA_PERIOD_END, end_time])
    mask = (delta_start < data.index) & (data.index < delta_end)
    pooled_data = get_incr_pooled_totals(data[mask], "delta")["delta_prop"]
    target = pooled_data["delta_prop"]
    if target is None or target.empty or max(target) < MIN_DELTA_PROP:
        return [], {}, []
    # Need extra weight for Delta target if emergence is right at end of simulation
    weight = 25.0 if (end_time - target.index[0]).days < 87 else VAR_WEIGHT
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
