from typing import Union
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
    DEFAULT_START_TIME,
    MIN_DELTA_PROP,
    DELTA_INCLUSION_DATE,
    END_VACC_THRESHOLD,
    START_VACC_THRESHOLD_AUS,
    DEATHS_START_THRESHOLD,
    get_who_indicator,
    get_country_vacc_data,
    get_worldbank_national_pop,
    get_standard_priors,
    get_google_mobility,
    get_apple_mobility,
    get_fb_mobility,
    get_country_vars,
    get_ba2_target,
    get_ba5_target,
)
from emu_renewal.targets import StandardDispTarget, StandardPropTarget
from emu_renewal.process import CosineMultiCurve
from emu_renewal.renew import MultiStrainModel
from emu_renewal.distributions import GammaDens
from emu_renewal.calibration import StandardCalib
from emu_renewal.outputs import store_outputs
from emu_renewal import mobility
from emu_renewal.indicators import get_deaths_target, get_cases_target, get_hosp_target, get_seroprev_target, get_alpha_target, get_delta_target


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
    deaths_data = get_who_indicator("New_deaths", iso3)
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

    # Variants
    var_weight = 5.0
    var_data = get_country_vars(iso3)

    delta_var, delta_targ, delta_seed = get_delta_target(iso3, var_data, continent, end_time)
    alpha_var, alpha_targ, alpha_seed = get_alpha_target(iso3, var_data, continent, end_time, delta_targ)

    # BA.2 proportion
    ba2_targ = get_ba2_target(var_data, continent)
    if ba2_targ is None:
        ba2_targ_dict = {}
    else:
        ba2_targ_dict = {"prop_ba2": StandardPropTarget(ba2_targ, weight=var_weight)}

    # BA.5 proportion
    ba5_targ = get_ba5_target(var_data, continent)
    if ba5_targ is None:
        ba5_targ_dict = {}
    else:
        ba5_targ_dict = {"prop_ba5": StandardPropTarget(ba5_targ, weight=var_weight)}

    targets = deaths_targ | cases_targ | hosp_targ | seroprev_targ | alpha_targ | delta_targ | ba2_targ_dict | ba5_targ_dict

    var_names = ["eu"] + alpha_var + delta_var
    seed_times = [] + alpha_seed + delta_seed
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
