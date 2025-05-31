from typing import Union, Tuple
from datetime import datetime, timedelta
import pycountry
import pycountry_convert as pc
import logging
from numpyro import distributions as dist
from numpyro import infer
from jax import random
from typing import Dict
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import git

from emu_renewal.inputs import (
    DATE_FORMAT,
    BASE_PATH,
    DEFAULT_END_TIME,
    CASES_START,
    DEFAULT_START_TIME,
    MIN_DELTA_PROP,
    DELTA_INCLUSION_DATE,
    END_VACC_THRESHOLD,
    START_VACC_THRESHOLD_AUS,
    DEATHS_WEIGHT,
    DEATHS_START_THRESHOLD,
    SEROPREV_EXTREME,
    SEROPREV_WEIGHT,
    get_indicator_series_from_who_data,
    get_country_vacc_data,
    get_worldbank_national_pop,
    get_standard_priors,
    get_google_mobility,
    get_apple_mobility,
    get_fb_mobility,
    get_all_seroprev,
    get_country_hosps,
    get_country_vars,
    get_alpha_target,
    get_delta_target,
    get_ba2_target,
    get_ba5_target,
    get_seroprev_pooled_totals,
    get_income_group,
)
from emu_renewal.targets import StandardDispTarget, StandardPropTarget, UnivariateDispersionTarget
from emu_renewal.process import CosineMultiCurve
from emu_renewal.renew import MultiStrainModel
from emu_renewal.distributions import GammaDens
from emu_renewal.calibration import StandardCalib
from emu_renewal.outputs import store_outputs
from emu_renewal import mobility


class MobilityException(Exception):
    pass


def jax_config_cpu_only():
    import jax
    jax.config.update("jax_platform_name", "cpu")


def find_run_start_time(
    pop: float,
    iso3: str,
) -> datetime:
    """For all countries except Australia,
    the start of the calibration period was
    set to be the time at which the per capita
    daily rate of deaths passed a specified threshold.
    However, if this threshold was not reached by 
    a default start date, the simulation commenced at this default time.
    For Australia, the simulation commenced from
    the time that vaccination reached a proportion of its final value.

    Args:
        pop: Population size
        iso3: The country identifier

    Returns:
        The date to start the analysis
    """
    deaths_data = get_indicator_series_from_who_data("New_deaths", iso3)
    per_capita_deaths = deaths_data / pop
    start = per_capita_deaths.index[per_capita_deaths.gt(DEATHS_START_THRESHOLD)].min()
    if iso3 == "AUS":
        vacc_data = get_country_vacc_data("AUS")
        norm_vacc_data = vacc_data / vacc_data.iloc[-1]
        return norm_vacc_data[norm_vacc_data.gt(START_VACC_THRESHOLD_AUS)].idxmin()
    elif pd.isna(start) or start > DEFAULT_START_TIME:
        return DEFAULT_START_TIME
    else:
        return start


def find_run_end_time(iso3: str) -> datetime:
    """For all countries but Australia,
    the end time for the analysis was calculated as 
    the time that the population vaccination coverage
    passed a specific threshold value, 
    provided that the vaccination coverage did reach this
    value by the default end time.
    Otherwise, a default end date is used.
    For Australia, the latest date for which
    the Google mobility data was available was used.

    Args:
        iso3: The country identifier

    Returns:
        The date at which to end the analysis period
    """
    thresh_perc = END_VACC_THRESHOLD * 100
    vacc_data = get_country_vacc_data(iso3)
    if iso3 == "AUS":
        mob = get_google_mobility(iso3)
        return mob.index[-1].to_pydatetime()
    elif vacc_data.empty or vacc_data.max() < thresh_perc:
        return DEFAULT_END_TIME
    else:
        return min([DEFAULT_END_TIME, vacc_data[vacc_data.gt(thresh_perc)].idxmin()])


def collate_targets(
    end: datetime,
    alpha_targ: Union[pd.Series, None],
    delta_targ: Union[pd.Series, None],
    ba2_targ: Union[pd.Series, None],
    ba5_targ: Union[pd.Series, None],
) -> Dict[str, StandardDispTarget]:
    """Collate the targets gathered in the previous function
    into the appropriate structure for the calibration algorithm.

    Returns:
        All targets, either four or five, depending on whether there are seroprevalence estimates
    """

    # Alpha proportion
    var_weight = 5.0
    if alpha_targ is None:
        alpha_targ_dict = {}
    else:
        alpha_targ_dict = {"prop_alpha": StandardPropTarget(alpha_targ, weight=var_weight)}

    # Delta proportion
    if delta_targ is None or delta_targ.empty or max(delta_targ) < MIN_DELTA_PROP:
        delta_targ_dict = {}
    else:
        # Need extra weight for Delta target if emergence is right at end of simulation
        delta_weight = 25.0 if (end - delta_targ.index[0]).days < 87 else var_weight
        delta_targ_dict = {"prop_delta": StandardPropTarget(delta_targ, weight=delta_weight)}

    # BA.2 proportion
    if ba2_targ is None:
        ba2_targ_dict = {}
    else:
        ba2_targ_dict = {"prop_ba2": StandardPropTarget(ba2_targ, weight=var_weight)}

    # BA.5 proportion
    if ba5_targ is None:
        ba5_targ_dict = {}
    else:
        ba5_targ_dict = {"prop_ba5": StandardPropTarget(ba5_targ, weight=var_weight)}

    # Collate together
    return (
        alpha_targ_dict
        | delta_targ_dict
        | ba2_targ_dict
        | ba5_targ_dict
    )


def get_logger(log_file: Path = None):
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root_logger = logging.getLogger()

    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    root_logger.setLevel(logging.INFO)

    return root_logger


def get_mobility_provider(
    iso3: str,
    mob_type: str,
) -> mobility.MobilityProvider:
    """Get the appropriate mobility provider object.

    Args:
        iso3: Country identifier
        mob_type: Mobility approach

    Returns:
        The mobility provider
    """

    # Data processing
    if mob_type == "no_mob":
        mob = pd.Series([])
    elif mob_type == "g_mob":
        mob = get_google_mobility(iso3)
    elif mob_type == "fb_mob":
        mob = get_fb_mobility(iso3)
    elif mob_type == "a_mob":
        mob = get_apple_mobility(iso3)
    n_domains = len(mob.columns) if isinstance(mob, pd.DataFrame) else None
    smoothed_mob = mob.rolling(7, center=True).mean().dropna()

    # Priors
    exp_prior = {"mob_exp": dist.Uniform(0.0, 2.0)}
    if mob_type == "no_mob":
        return mobility.NoMobilityProvider()
    elif mob_type == "g_mob":
        weight_prior = {"mob_weights": dist.Uniform(np.zeros(n_domains), np.ones(n_domains))}
        return mobility.WeightedExpMobilityProvider(smoothed_mob, weight_prior | exp_prior)
    elif mob_type == "fb_mob":
        return mobility.SingleSeriesExpMobilityProvider(smoothed_mob, exp_prior)
    elif mob_type == "a_mob":
        weight_prior = {"mob_weights": dist.Uniform(np.zeros(n_domains), np.ones(n_domains))}
        return mobility.WeightedExpMobilityProvider(smoothed_mob, weight_prior | exp_prior)
    else:
        raise Exception(f"No provider available for analysis type {mob_type}")


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
    data = get_indicator_series_from_who_data("New_deaths", iso3)
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
    data = get_indicator_series_from_who_data("New_cases", iso3)
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
    data, output_name = get_country_hosps(iso3, start, end)
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


def run_single_country(
    country,
    proc_update_freq,
    init_duration,
    mob_analysis_type,
    n_iters,
    run_data_delay,
    analysis_name,
    seed_duration: int = 10,
    n_chains=4,
    prog_bar=False,
    logger=None,
):

    # Country identifiers
    iso3 = pycountry.countries.lookup(country).alpha_3
    iso2 = pycountry.countries.lookup(country).alpha_2
    continent = pc.country_alpha2_to_continent_code(iso2)
    
    # Logging
    logger = logger or logging.getLogger()
    logger.info(f"\n________________________\nRunning job at {analysis_name}")
    logger.info(f"Country: {iso3}")
    logger.info(f"Mobility approach: {mob_analysis_type}")
    commit = git.Repo(search_parent_directories=True).head
    logger.info(f"Git commit hash: {commit.object.hexsha}")
    logger.info(f"Commit message: {commit.reference.commit.message}")

    # Population size and analysis time
    pop = get_worldbank_national_pop(iso3)
    data_start = find_run_start_time(pop, iso3)
    end_time = find_run_end_time(iso3)
    run_start = data_start - timedelta(run_data_delay)
    start_str = run_start.strftime(DATE_FORMAT)
    end_str = data_start.strftime(DATE_FORMAT)
    logger.info(f"Running from {start_str} with data starting from {end_str}")
    logger.info(f"Running to {end_time.strftime(DATE_FORMAT)}")

    # Targets
    n_deaths, deaths_targ = get_deaths_target(iso3, data_start, end_time)
    cases_targ = get_cases_target(iso3, data_start, end_time, n_deaths)
    hosp_targ = get_hosp_target(iso3, data_start, end_time, n_deaths)
    seroprev_targ = get_seroprev_target(iso3, data_start, end_time, continent)



    # Old targets code
    var_data = get_country_vars(iso3)
    delta_targ = (
        None
        if continent == "OC" or end_time < DELTA_INCLUSION_DATE
        else get_delta_target(var_data, iso3, continent, end_time)
    )
    alpha_targ = (
        None
        if continent in ["OC", "AF"]
        else get_alpha_target(var_data, iso3, continent, end_time, delta_targ)
    )
    ba2_targ = get_ba2_target(var_data, continent)
    ba5_targ = get_ba5_target(var_data, continent)

    targets = collate_targets(
        end_time,
        alpha_targ,
        delta_targ,
        ba2_targ,
        ba5_targ,
    )

    targets = deaths_targ | cases_targ | hosp_targ | seroprev_targ | targets



    var_names = ["eu"]
    seed_times = []
    if alpha_targ is not None:
        var_names.append("alpha")
        alpha_seed_time = alpha_targ.index[0]
        seed_times.append(alpha_seed_time)
    if delta_targ is not None and not delta_targ.empty and max(delta_targ) > MIN_DELTA_PROP:
        var_names.append("delta")
        delta_seed_time = delta_targ.index[0]
        seed_times.append(delta_seed_time)
    if continent == "OC":
        var_names = ["ba1", "ba2", "ba5"]
        ba2_seed_time = ba2_targ.index[0]
        ba5_seed_time = ba5_targ.index[0]
        seed_times = [ba2_seed_time, ba5_seed_time]

    # Mobility
    try:
        mob_provider = get_mobility_provider(iso3, mob_analysis_type)
    except Exception as e:
        msg = f"{mob_analysis_type} mobility not available"
        raise MobilityException(msg)
    if mob_provider.mob_end:
        end_time = min([end_time, mob_provider.mob_end])

    # Model construction
    vacc_effect = continent == "OC"
    model = MultiStrainModel(
        pop,
        run_start,
        end_time,
        proc_update_freq,
        CosineMultiCurve(),
        GammaDens(),
        init_duration,
        init_duration,
        GammaDens(),
        GammaDens(),
        var_names,
        var_names[0],
        seed_times,
        mob_provider,
        seed_duration,
        vacc_effect=vacc_effect,
    )

    # Calibration
    hosp_key = list(hosp_targ.keys())[0] if hosp_targ else ""
    priors = get_standard_priors(len(var_names), hosp_key, iso3) | mob_provider.get_priors()
    calib = StandardCalib(model, priors, targets, proc_dispersion=dist.HalfNormal(0.5))
    init = calib.custom_init(radius=0.1)
    kernel = infer.NUTS(calib.calibration, dense_mass=True, init_strategy=init)
    mcmc = infer.MCMC(kernel, num_chains=n_chains, num_samples=n_iters, num_warmup=n_iters, progress_bar=prog_bar)
    mcmc.run(random.PRNGKey(0), extra_fields=["potential_energy"])

    # Outputs
    out_path = BASE_PATH / "outputs" / analysis_name / country / mob_analysis_type
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing to: {out_path}")
    store_outputs(out_path, model, calib, mcmc)
    logger.info(f"Completed {analysis_name}/{country}/{mob_analysis_type}")
