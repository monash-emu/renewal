from typing import Tuple, Dict
from datetime import datetime, timedelta
import pandas as pd
import pycountry
from numpyro import distributions as dist

from emu_renewal.inputs import DEATHS_WEIGHT, CASES_START, SEROPREV_EXTREME, SEROPREV_WEIGHT, VAR_WEIGHT, \
    ALPHA_DELTA_EXCEPTS, ALPHA_PERIOD_START, ALPHA_DELTA_TRANS, \
    get_who_indicator, get_owid_hosps, get_owid_hosps, get_all_seroprev, get_all_seroprev, \
    get_seroprev_pooled_totals, get_income_group, get_alpha_data, get_var_target, \
    get_dec_pooled_totals
from emu_renewal.targets import StandardDispTarget, UnivariateDispersionTarget, StandardPropTarget


def get_deaths_target(
    iso3: str,
    start: datetime, 
    end: datetime,
) -> Tuple[int, Dict[str, StandardDispTarget]]:
    """The number of deaths by week reported by WHO 
    was used as the first calibration target for all countries.
    Any values of zero in this series were replaced with a
    value of 0.5 to enable comparison to modelled outputs
    on the log scale. Deaths was the one of two indicators
    for which a common dispersion parameter was used
    for the distribution comparison of the modelled value.

    Args:
        iso3: The country identifier
        start: The calibration start time
        end: The calibration end time

    Returns:
        Number of observations in the deaths series
        The deaths calibration target
    """
    data = get_who_indicator("New_deaths", iso3)
    data = data.interpolate(method="linear").fillna(0.0)
    data[data == 0.0] = 0.5
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
    """The number of cases by week reported by WHO 
    was used as the second calibration target for all countries.
    Linear interpolation was used to replace missing values, and
    as for deaths, any zero values were replaced with a value of 0.5.
    Cases was the second indicator for which 
    a common dispersion parameter was applied.
    A target weight was applied to the series of cases 
    such that the weight for each case observation point
    was the same as for each death observation.

    Args:
        iso3: The country identifier
        start: The calibration start time
        end: The calibration end time
        n_deaths: The number of deaths observations

    Returns:
        The cases calibration target
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
    """One hospitalisation indicator was also used for
    each country, where available.
    This indicator was the final calibration target for which 
    a common dispersion parameter was applied.
    As for cases, a weight was applied to the hospitalisation series 
    such that the weight for each observation point
    was the same as for each death observation.

    Args:
        iso3: The country identifier
        start: The calibration start time
        end: The calibration end time
        n_deaths: The number of deaths observations

    Returns:
        The hospitalisation calibration target
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
    start: datetime,
    end: datetime,
    africa_lic: bool = False,
) -> pd.Series:
    """
    Filter the SeroTracker data according to our choices
    about what constitutes good enough data
    for including in the calibration targets.
    Don't use seroprevalence for Australia,
    because it had reached high levels
    by the time of analysis.

    Args:
        iso3: Country identifier
        start: Start date of analysis
        end: End date of analysis

    Returns:
        Filtered data to use as target
    """
    if iso3 == "AUS" or africa_lic:
        return pd.Series([])
    data = get_all_seroprev()
    country = pycountry.countries.lookup(iso3).name
    country_filt = data["country"] == country
    time_filt = (start < data.index) & (data.index < end)
    nat_filt = data["estimate_grade"] == "National"
    type_filt = data["subgroup_var"] == "Primary Estimate"
    unity_filt = data["is_unity_aligned"] == "Unity-Aligned"
    n_filt = data["denominator_value"] > 599
    all_filt = time_filt & country_filt & nat_filt & type_filt & unity_filt & n_filt
    filt_data = data[all_filt]
    # Drop 2 of 3 estimates for Mexico on the same date (keeping the first and largest)
    return filt_data[[not i for i in filt_data.index.duplicated()]]


def get_seroprev_target(
    iso3: str,        
    start: datetime,
    end: datetime,
    continent: str,
) -> Dict[str, UnivariateDispersionTarget]:
    """For seroprevalence, we compared the modelled
    proportion ever infected against the reported seroprevalence
    reported at least six months after the start of the simulation,
    because a comparison against early seroprevalence estimates
    would not account for waves of transmission prior to 
    the start of the simulation.

    Args:
        iso3: The country identifier
        start: The calibration start time
        end: The calibration end time
        continent: The country's continent

    Returns:
        The seroprevalence calibration target
    """
    income = get_income_group(iso3)
    africa_reporting = continent == "AF" and income in ["Lower middle income", "Low income"]
    seroprev = get_filtered_seroprev(iso3, start, end, africa_reporting)
    if seroprev.empty or continent == "OC":
        return {}
    data = get_seroprev_pooled_totals(seroprev)
    data = data[start + timedelta(183) < data.index]
    seroprev_mask = (SEROPREV_EXTREME < data) & (data < 1.0 - SEROPREV_EXTREME)
    data = data[seroprev_mask]
    if data.empty:
        return {}
    target = UnivariateDispersionTarget(data, dist.Normal, "seroprev_disp", weight=SEROPREV_WEIGHT)
    return {"seropos": target}


def get_alpha_target(iso3, var_data, continent, end_time, delta_targ):
    if continent in ["OC", "AF"]:
        return {}
    alpha_data = get_var_target(var_data, continent, "alpha")
    alpha_delta_trans = ALPHA_DELTA_EXCEPTS[iso3] if iso3 in ALPHA_DELTA_EXCEPTS else ALPHA_DELTA_TRANS
    end_alpha_time = end_time if delta_targ is None else min([alpha_delta_trans, end_time])
    period_mask = (ALPHA_PERIOD_START < alpha_data.index) & (alpha_data.index < end_alpha_time)
    pooled_data = get_dec_pooled_totals(alpha_data[period_mask], "alpha")
    alpha_targ = pooled_data["alpha_prop"]
    return {"prop_alpha": StandardPropTarget(alpha_targ, weight=VAR_WEIGHT)}
