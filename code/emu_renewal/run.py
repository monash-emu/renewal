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
    CASES_START,
    DEFAULT_START_TIME,
    MIN_DELTA_PROP,
    DELTA_INCLUSION_DATE,
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
    deaths_data: pd.Series,
    vacc_data: pd.Series,
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
        vacc_data: Two-dose vaccination coverage data
        pop: Population size
        threshold: How many deaths to reach
        iso3: The country identifier

    Returns:
        The date to start the analysis
    """
    per_capita_deaths = deaths_data / pop
    start = per_capita_deaths.index[per_capita_deaths.gt(threshold)].min()
    if iso3 == "AUS":
        aust_vacc_prop_to_start = 0.9
        norm_vacc_data = vacc_data / vacc_data.iloc[-1]
        return norm_vacc_data[norm_vacc_data.gt(aust_vacc_prop_to_start)].idxmin()
    elif pd.isna(start) or start > DEFAULT_START_TIME:
        return DEFAULT_START_TIME
    else:
        return start


def find_run_end_time(
    vacc_data: pd.Series,
    cov_threshold: float,
    iso3: str,
) -> datetime:
    """Find the time that the analysis should finish.
    Calculated as the time that the population vaccination coverage
    passes the requested threshold for all countries but Australia,
    provided that the vaccination coverage does reach this
    value by the default end time. Otherwise return the end date
    for Google mobility data for Australia,
    or return a default value for other countries.

    Args:
        vacc_data: The vaccination data for the country considered
        cov_threshold: The threshold
        iso3: The country identifier

    Returns:
        The date at which to end the analysis period
    """
    cov_thresh_perc = cov_threshold * 100
    if iso3 == "AUS":
        mob = get_google_mobility(iso3)
        return mob.index[-1].to_pydatetime()
    elif vacc_data.empty or vacc_data.max() < cov_thresh_perc:
        return DEFAULT_END_TIME
    else:
        return min([DEFAULT_END_TIME, vacc_data[vacc_data.gt(cov_thresh_perc)].idxmin()])


def collate_targets(
    cases_data: pd.Series,
    deaths_data: pd.Series,
    hosp_data: pd.Series,
    hosp_output_name: str,
    seroprev_target: pd.Series,
    ext_prop: float,
    start: datetime,
    end: datetime,
    iso3: str,
    continent: str,
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

    # Deaths
    death_mask = (start < deaths_data.index) & (deaths_data.index < end)
    select_deaths = deaths_data.loc[death_mask]
    deaths_targ = StandardDispTarget(select_deaths, weight=20.0)

    # Cases
    pre_test_scaleup = cases_data.index > CASES_START
    case_mask = (start < cases_data.index) & (cases_data.index < end) & pre_test_scaleup
    cases_targ = cases_data.loc[case_mask]
    case_weight = 20.0 * len(cases_targ) / len(select_deaths)
    cases_targ = StandardDispTarget(cases_targ, weight=case_weight)

    # Hospitalisations
    if hosp_data is None:
        hosp_targ_dict = {}
    else:
        hosp_mask = (start < hosp_data.index) & (hosp_data.index < end)
        select_hosps = hosp_data.loc[hosp_mask]
        if select_hosps.empty:
            hosp_targ_dict = {}
        else:
            hosp_weight = 20.0 * len(select_hosps) / len(select_deaths)
            hosp_targ = StandardDispTarget(select_hosps, weight=hosp_weight)
            hosp_targ_dict = {hosp_output_name: hosp_targ}

    # Seroprevalence
    seroprev_mask = (ext_prop < seroprev_target) & (seroprev_target < 1.0 - ext_prop)
    seroprev_target = seroprev_target[seroprev_mask]
    if seroprev_target.empty or continent == "OC" or iso3 in ["PAK", "ZMB", "NGA"]:
        seroprev_targ_dict = {}
    else:
        # seroprev_targ = UnivariateDispersionTarget(seroprev_target, "seroprev_disp", weight=2.5)
        seroprev_targ = StandardPropTarget(seroprev_target, weight=2.5)
        seroprev_targ_dict = {"seropos": seroprev_targ}

    # Alpha proportion
    if alpha_targ is None:
        alpha_targ_dict = {}
    else:
        alpha_targ_dict = {"prop_alpha": StandardPropTarget(alpha_targ, weight=5.0)}

    # Delta proportion
    if delta_targ is None or delta_targ.empty or max(delta_targ) < MIN_DELTA_PROP:
        delta_targ_dict = {}
    else:
        # Need extra weight for Delta target if emergence is right at end of simulation
        delta_weight = 40.0 if (end - delta_targ.index[0]).days < 90 else 5.0
        delta_targ_dict = {"prop_delta": StandardPropTarget(delta_targ, weight=delta_weight)}

    # BA.2 proportion
    if ba2_targ is None:
        ba2_targ_dict = {}
    else:
        ba2_targ_dict = {"prop_ba2": StandardPropTarget(ba2_targ, weight=5.0)}

    # BA.5 proportion
    if ba5_targ is None:
        ba5_targ_dict = {}
    else:
        ba5_targ_dict = {"prop_ba5": StandardPropTarget(ba5_targ, weight=5.0)}

    # Collate together
    core_targs = {"weekly_cases": cases_targ, "weekly_deaths": deaths_targ}
    return (
        core_targs
        | seroprev_targ_dict
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
    pop_year = 2022 if continent == "OC" else 2020
    pop = get_worldbank_national_pop(iso3, pop_year)
    vacc_data = get_country_vacc_data(iso3)
    end_time = find_run_end_time(vacc_data, most_extreme_prop, iso3)

    # Targets
    case_data = get_indicator_series_from_who_data("New_cases", country)
    death_data = get_indicator_series_from_who_data("New_deaths", country)
    data_start = find_run_start_time(death_data, vacc_data, pop, death_start_threshold, iso3)
    hosp_target, hosp_out_type = get_country_hosps(iso3, data_start, end_time)
    seroprev = get_filtered_seroprev(country, data_start, end_time)
    seroprev_target = seroprev if seroprev.empty else get_seroprev_pooled_totals(seroprev)

    # # Unable to get good fit to Bangladesh seroprevalence
    # if iso3 in ["BGD", "CAN"]:
    #     seroprev_target = pd.Series([])

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
        case_data,
        death_data,
        hosp_target,
        hosp_out_type,
        seroprev_target,
        most_extreme_prop,
        data_start,
        end_time,
        iso3,
        continent,
        alpha_targ,
        delta_targ,
        ba2_targ,
        ba5_targ,
    )
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
