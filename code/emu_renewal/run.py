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
    get_indicator_series_from_who_data,
    get_country_vacc_data,
    get_worldbank_national_pop,
    get_standard_priors,
    get_google_mobility,
    get_apple_mobility,
    get_fb_mobility,
    get_filtered_seroprev,
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


def find_run_start_time(
    pop: float,
    threshold: float,
    iso3: str,
) -> datetime:
    """Determine the time that the analysis should start comparing to data from.
    Calculated as the time until the per capita death rate reaches the
    specified threshold for most countries, unless they never reach
    that threshold or don't reach it by the default time.

    Args:
        deaths_data: Deaths time series for the country considered
        pop: Population size
        threshold: How many deaths to reach
        iso3: The country identifier

    Returns:
        The date to start the analysis
    """
    deaths_data = get_indicator_series_from_who_data("New_deaths", iso3)
    per_capita_deaths = deaths_data / pop
    start = per_capita_deaths.index[per_capita_deaths.gt(threshold)].min()
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
    hosp_data: pd.Series,
    hosp_output_name: str,
    seroprev_target: pd.Series,
    ext_prop: float,
    start: datetime,
    end: datetime,
    continent: str,
    alpha_targ: Union[pd.Series, None],
    delta_targ: Union[pd.Series, None],
    ba2_targ: Union[pd.Series, None],
    ba5_targ: Union[pd.Series, None],
    n_deaths,
) -> Dict[str, StandardDispTarget]:
    """Collate the targets gathered in the previous function
    into the appropriate structure for the calibration algorithm.

    Returns:
        All targets, either four or five, depending on whether there are seroprevalence estimates
    """

    # Hospitalisations
    if hosp_data is None:
        hosp_targ_dict = {}
    else:
        hosp_mask = (start < hosp_data.index) & (hosp_data.index < end)
        select_hosps = hosp_data.loc[hosp_mask]
        if select_hosps.empty:
            hosp_targ_dict = {}
        else:
            hosp_weight = 20.0 * len(select_hosps) / n_deaths
            hosp_targ = StandardDispTarget(select_hosps, weight=hosp_weight)
            hosp_targ_dict = {hosp_output_name: hosp_targ}

    # Seroprevalence
    seroprev_mask = (ext_prop < seroprev_target) & (seroprev_target < 1.0 - ext_prop)
    seroprev_target = seroprev_target[seroprev_mask]
    if seroprev_target.empty or continent == "OC":
        seroprev_targ_dict = {}
    else:
        seroprev_targ = UnivariateDispersionTarget(
            seroprev_target, dist.Normal, "seroprev_disp", weight=5.0
        )
        # seroprev_targ = StandardPropTarget(seroprev_target, weight=2.5)
        seroprev_targ_dict = {"seropos": seroprev_targ}

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
        seroprev_targ_dict
        | hosp_targ_dict
        | alpha_targ_dict
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
    data[data == 0.0] = 0.5
    period_mask = (start < data.index) & (data.index < end)
    select_data = data.loc[period_mask]
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
    As for deaths, any zero values were replaced with 0.5.
    Cases was the other indicator for which 
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
    case_data = get_indicator_series_from_who_data("New_cases", iso3)
    case_data[case_data == 0.0] = 0.5
    cases_start = max([CASES_START, start])
    case_mask = (cases_start < case_data.index) & (case_data.index < end)
    cases_targ = case_data.loc[case_mask]
    case_weight = DEATHS_WEIGHT * len(cases_targ) / n_deaths
    target = StandardDispTarget(cases_targ, weight=case_weight)
    return {"weekly_cases": target}


def run_single_country(
    country,
    proc_update_freq,
    init_duration,
    mob_analysis_type,
    n_iters,
    run_data_delay,
    analysis_name,
    most_extreme_prop: float = 0.05,
    death_start_threshold: float = 2e-6,
    seed_duration: int = 10,
    n_chains=4,
    prog_bar=False,
    logger=None,
):

    # Preliminaries
    logger = logger or logging.getLogger()
    logger.info(f"\n________________________\nRunning job at {analysis_name}")
    iso3 = pycountry.countries.lookup(country).alpha_3
    iso2 = pycountry.countries.lookup(country).alpha_2
    continent = pc.country_alpha2_to_continent_code(iso2)
    logger.info(f"Country: {iso3}")
    logger.info(f"Mobility approach: {mob_analysis_type}")
    repo = git.Repo(search_parent_directories=True)
    repo_head = repo.head
    logger.info(f"Git commit hash: {repo_head.object.hexsha}")
    msg = repo.head.reference.commit.message
    logger.info(f"Commit message: {msg}")
    pop = get_worldbank_national_pop(iso3)
    end_time = find_run_end_time(iso3)
    data_start = find_run_start_time(pop, death_start_threshold, iso3)

    # Targets
    hosp_target, hosp_out_type = get_country_hosps(iso3, data_start, end_time)
    income = get_income_group(iso3)
    africa_reporting = continent == "AF" and income in ["Lower middle income", "Low income"]
    seroprev = get_filtered_seroprev(country, data_start, end_time, africa_reporting)
    if seroprev.empty:
        seroprev_target = seroprev
    else:
        seroprev_target = get_seroprev_pooled_totals(seroprev)
        seroprev_target = seroprev_target[seroprev_target.index > data_start + timedelta(183)]

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

    deaths_targ, n_deaths = get_deaths_target(iso3, data_start, end_time)
    cases_targ = get_cases_target(iso3, data_start, end_time, n_deaths)

    targets = collate_targets(
        hosp_target,
        hosp_out_type,
        seroprev_target,
        most_extreme_prop,
        data_start,
        end_time,
        continent,
        alpha_targ,
        delta_targ,
        ba2_targ,
        ba5_targ,
        n_deaths,
    )


    targets = deaths_targ | cases_targ | targets

    run_start = data_start - timedelta(run_data_delay)
    start_str = run_start.strftime(DATE_FORMAT)
    end_str = data_start.strftime(DATE_FORMAT)
    logger.info(f"Running from {start_str} with data starting from {end_str}")
    logger.info(f"Running to {end_time.strftime(DATE_FORMAT)}")
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
    priors = get_standard_priors(len(var_names), hosp_out_type, iso3) | mob_provider.get_priors()
    calib = StandardCalib(model, priors, targets, proc_dispersion=dist.HalfNormal(0.5))
    init = calib.custom_init(radius=0.1)
    kernel = infer.NUTS(calib.calibration, dense_mass=True, init_strategy=init)
    mcmc = infer.MCMC(
        kernel, num_chains=n_chains, num_samples=n_iters, num_warmup=n_iters, progress_bar=prog_bar
    )
    mcmc.run(random.PRNGKey(0), extra_fields=["potential_energy"])

    # Outputs
    storage_path = BASE_PATH / "outputs" / analysis_name / country / mob_analysis_type
    storage_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing to: {storage_path}")
    store_outputs(storage_path, model, calib, mcmc)
    logger.info(f"Completed {analysis_name}/{country}/{mob_analysis_type}")


def jax_config_cpu_only():
    import jax

    jax.config.update("jax_platform_name", "cpu")
