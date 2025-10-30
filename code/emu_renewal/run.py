from typing import Dict, Tuple
from datetime import datetime, timedelta
from socket import gethostname
import pycountry
import logging
from numpyro import distributions as dist
from numpyro import infer
from jax import random
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import git

from emu_renewal.constants import (
    DATE_FORMAT,
    CODE_DATE_FORMAT,
    BASE_PATH,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    END_VACC_THRESHOLD,
    START_VACC_THRESHOLD_OC,
    DEATHS_START_THRESHOLD,
    MOBILITY_SMOOTH_PERIOD,
    EXP_PRIOR_LOWER,
    EXP_PRIOR_UPPER,
    N_ITERS,
    RUN_DATA_DELAY,
    N_CHAINS,
)
from emu_renewal.inputs import (
    get_country_vacc_data,
    get_country_pop,
    get_google_mobility,
    get_fb_visited_mobility,
    get_fb_singletile_mobility,
)
from emu_renewal.renew import MultiStrainModel
from emu_renewal.calibration import StandardCalib
from emu_renewal.priors import get_standard_priors
from emu_renewal.outputs import store_outputs
from emu_renewal import mobility
from emu_renewal.indicators import (
    get_who_indicator,
    get_deaths_target,
    get_cases_target,
    get_hosp_target,
    get_seroprev_target,
    get_alpha_info,
    get_delta_info,
    get_ba2_info,
    get_ba5_info,
    get_country_vars,
)
from emu_renewal.targets import Target
from emu_renewal.utils import get_cont_of_country


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


class MobilityException(Exception):
    pass


def jax_config_cpu_only():
    import jax

    jax.config.update("jax_platform_name", "cpu")


def find_run_start_time(
    pop: float,
    iso3: str,
) -> datetime:
    """Find the start time for the analysis.

    Args:
        pop: Population size
        iso3: The country identifier

    Returns:
        The date to start the analysis

    Notes
    -----
    For these countries,
    the start of the calibration period was
    set to be the time at which the per capita
    daily rate of deaths passed {DEATHS_START_THRESHOLD}
    deaths per million population.
    However, if this threshold was not reached by {DEFAULT_START_DATE},
    the simulation commenced at this default time instead.
    For Singapore and countries of Oceania, the simulation commenced from
    the time that vaccination coverage reached {START_VACC_THRESHOLD_OC}%
    of its final value.
    """
    deaths_data = get_who_indicator("New_deaths", iso3)
    per_capita_deaths = deaths_data / pop
    start = per_capita_deaths.index[per_capita_deaths.gt(DEATHS_START_THRESHOLD / 1e6)].min()
    cont = get_cont_of_country(iso3)
    if cont == "OC":
        vacc_data = get_country_vacc_data(iso3)
        norm_vacc_data = vacc_data / vacc_data.iloc[-1]
        return norm_vacc_data[norm_vacc_data.gt(START_VACC_THRESHOLD_OC / 100.0)].idxmin()
    elif pd.isna(start) or start > datetime.strptime(DEFAULT_START_DATE, CODE_DATE_FORMAT):
        return datetime.strptime(DEFAULT_START_DATE, CODE_DATE_FORMAT)
    else:
        return start


def find_run_end_time(
    iso3: str,
    mob_source: str,
) -> datetime:
    """Find the end time for the analysis.

    Args:
        iso3: The country identifier
        mob_source: The mobility approach

    Returns:
        The date at which to end the analysis period

    Notes
    -----
    For all countries other than Singapore 
    and those of Oceania,
    the end time for the analysis was calculated as
    the time that population vaccination coverage
    passed {END_VACC_THRESHOLD}%,
    provided that vaccination coverage did reach this
    value before the default end time of {DEFAULT_END_DATE}.
    Otherwise, this default end date was used instead.
    For Singapore and Oceania, the latest date for which
    Google mobility data was available was used.
    """
    cont = get_cont_of_country(iso3)
    try:
        if cont == "OC" and "fb_" in mob_source:
            mob = get_fb_visited_mobility(iso3)
            return mob.index[-1].to_pydatetime()
        elif cont == "OC":
            mob = get_google_mobility(iso3)
            return mob.index[-1].to_pydatetime()
    except Exception as e:
        msg = f"{mob_source} mobility not available"
        raise MobilityException(msg)
    vacc_data = get_country_vacc_data(iso3)
    default_end_time = datetime.strptime(DEFAULT_END_DATE, CODE_DATE_FORMAT)
    if vacc_data.empty or vacc_data.max() < END_VACC_THRESHOLD:
        return default_end_time
    else:
        return min([default_end_time, vacc_data[vacc_data.gt(END_VACC_THRESHOLD)].idxmin()])


def get_mobility_provider(
    iso3: str,
    mob_source: str,
) -> mobility.MobilityProvider:
    """Get the appropriate mobility provider object.

    Args:
        iso3: Country identifier
        mob_source: Mobility approach

    Returns:
        The mobility provider

    Notes
    -----
    For each country, we ran one analysis with
    no mobility scaling to the transmission rate.
    We further ran one analysis in which Google mobility
    was used to scale the transmission rate,
    if mobility data was available from Google.
    We also ran two analyses in which Facebook mobility
    was used to scale the transmission rate,
    if mobility data was available from Facebook.
    Although Apple mobility data was available
    and we were able to run analyses using this
    data source, Apple's terms of use indicate
    that this source of data cannot be used for this purpose.
    We contacted Apple, who declined to allow
    their data to be used for this project.
    For all mobility sources, we smoothed the raw
    data using a {MOBILITY_SMOOTH_PERIOD}-day centred
    rolling average.
    For all analyses incorporating mobility scaling,
    we used an exponential scaling parameter
    (described in more detail below) which
    was assigned a uniform prior over domain
    [{EXP_PRIOR_LOWER}, {EXP_PRIOR_UPPER}].
    """

    # Data processing
    if mob_source in ["no_mob", "fb_no_mob"]:
        return mobility.NoMobilityProvider()
    elif mob_source == "g_mob":
        mob = get_google_mobility(iso3)
    elif mob_source == "fb_visited_mob":
        mob = get_fb_visited_mobility(iso3)
    elif mob_source == "fb_singletile_mob":
        mob = get_fb_singletile_mobility(iso3)
    smoothed_mob = mob.rolling(MOBILITY_SMOOTH_PERIOD, center=True).mean().dropna()

    # Priors
    exp_prior = {"mob_exp": dist.Uniform(EXP_PRIOR_LOWER, EXP_PRIOR_UPPER)}
    if mob_source == "g_mob":
        n_domains = len(mob.columns)
        weight_prior = {"mob_weights": dist.Uniform(np.zeros(n_domains), np.ones(n_domains))}
        return mobility.WeightedExpMobilityProvider(smoothed_mob, weight_prior | exp_prior)
    elif mob_source in ["fb_visited_mob", "fb_singletile_mob"]:
        return mobility.SingleSeriesExpMobilityProvider(smoothed_mob, exp_prior)
    else:
        raise Exception(f"No provider available for analysis type {mob_source}")


def run_calibration(
    model: MultiStrainModel,
    priors: Dict[str, dist.Distribution],
    targets: Dict[str, Target],
    prog_bar: bool,
    n_iters: int,
) -> Tuple[StandardCalib, infer.MCMC]:
    """Run a calibration using a standard approach.

    Args:
        model: The renewal model
        priors: The parameter priors
        targets: The calibration targets
        prog_bar: Whether to display a progress bar

    Returns:
        The calibration and MCMC objects

    Notes
    -----
    We ran the calibration algorithm with a warm-up of 
    {N_ITERS} iterations for each of {N_CHAINS} chains
    followed by {N_ITERS} evaluated iterations 
    for the main analysis.
    We used a Hamiltonian Monte Carlo inference approach
    with the No U-Turn Sampler (NUTS) with adaptive path length 
    and mass matrix adaptation from `numpyro`.
    """
    calib = StandardCalib(model, priors, targets)
    init = calib.custom_init()
    kernel = infer.NUTS(calib.calibration, dense_mass=True, init_strategy=init)
    mcmc = infer.MCMC(
        kernel, num_chains=N_CHAINS, num_samples=n_iters, num_warmup=n_iters, progress_bar=prog_bar
    )
    mcmc.run(random.PRNGKey(0), extra_fields=["potential_energy"])
    return calib, mcmc


def run_single_country(
    country: str,
    mob_source: str,
    task_name: str,
    prog_bar=False,
    logger=None,
):
    """Run an analysis for a single country / mobility approach.

    Args:
        country: The country identifier
        mob_source: The mobility analysis type
        task_name: Identifier for the set of tasks
        prog_bar: Whether to display the progress bar
        logger: The logging object

    Raises:
        MobilityException: Error if unable to get the mobility
            provider (assuming this is because mobility is unavailable)

    Notes
    -----
    Each analysis begins from {RUN_DATA_DELAY} days before the
    first available calibration data point.
    """

    # Country identifiers
    iso3 = pycountry.countries.lookup(country).alpha_3
    continent = get_cont_of_country(iso3)

    # Logging
    logger = logger or logging.getLogger()
    logger.info(f"\n________________________\nRunning job at {task_name}")
    logger.info(f"Country: {iso3}")
    logger.info(f"Mobility approach: {mob_source}")
    commit = git.Repo(search_parent_directories=True).head
    logger.info(f"Git commit hash: {commit.object.hexsha}")
    logger.info(f"Commit message: {commit.reference.commit.message}")
    logger.info(f"Hostname: {gethostname()}")

    # Population size and analysis time
    pop = get_country_pop(iso3)
    data_start = find_run_start_time(pop, iso3)
    end_time = find_run_end_time(iso3, mob_source)
    run_start = data_start - timedelta(RUN_DATA_DELAY)
    start_str = run_start.strftime(DATE_FORMAT)
    end_str = data_start.strftime(DATE_FORMAT)
    logger.info(f"Running from {start_str} with data starting from {end_str}")
    logger.info(f"Running to {end_time.strftime(DATE_FORMAT)}")

    # Targets
    n_deaths, deaths_targ = get_deaths_target(iso3, data_start, end_time)
    cases_targ = get_cases_target(iso3, data_start, end_time, n_deaths)
    hosp_targ = get_hosp_target(iso3, data_start, end_time, n_deaths)
    seroprev_targ = get_seroprev_target(iso3, continent, data_start, end_time)

    # Variants
    var_data = get_country_vars(iso3)
    delta_var, delta_targ, delta_seed = get_delta_info(iso3, var_data, continent, end_time)
    alpha_var, alpha_targ, alpha_seed = get_alpha_info(
        iso3, var_data, continent, end_time, delta_targ
    )
    ba2_var, ba2_targ, ba2_seed = get_ba2_info(var_data, continent)
    ba5_var, ba5_targ, ba5_seed = get_ba5_info(var_data, continent)
    start_var = "ba1" if continent == "OC" else "eu"
    var_names = [start_var] + alpha_var + delta_var + ba2_var + ba5_var
    seed_times = [] + alpha_seed + delta_seed + ba2_seed + ba5_seed
    var_targs = alpha_targ | delta_targ | ba2_targ | ba5_targ

    # Mobility
    try:
        mob_provider = get_mobility_provider(iso3, mob_source)
    except Exception as e:
        msg = f"{mob_source} mobility not available"
        raise MobilityException(msg)
    if mob_provider.mob_end:
        end_time = min([end_time, mob_provider.mob_end])

    # Model construction
    vacc_effect = continent == "OC"
    model = MultiStrainModel(
        pop,
        run_start,
        end_time,
        var_names,
        seed_times,
        mob_provider,
        vacc_effect,
    )

    # Calibration
    hosp_key = list(hosp_targ.keys())[0] if hosp_targ else ""
    priors = get_standard_priors(len(var_names), hosp_key, iso3) | mob_provider.get_priors()
    targets = deaths_targ | cases_targ | hosp_targ | seroprev_targ | var_targs
    calib, mcmc = run_calibration(model, priors, targets, prog_bar, N_ITERS)

    # Outputs
    out_path = BASE_PATH / "outputs" / task_name / country / mob_source
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing to: {out_path}")
    store_outputs(out_path, model, calib, mcmc)
    logger.info(f"Completed {task_name}/{country}/{mob_source}")
