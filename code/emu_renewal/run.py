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

from emu_renewal.inputs import (
    DATE_FORMAT,
    BASE_PATH,
    AUST_END,
    OTHER_DEFAULT_END,
    CASES_START,
    get_indicator_series_from_who_data,
    get_country_vacc_data,
    get_worldbank_national_pop,
    get_standard_priors,
    get_google_mobility,
    get_apple_mobility,
    get_fb_mobility,
    get_filtered_seroprev,
    get_country_hosps,
    get_var_target,
    get_alpha_seed_time,
)
from emu_renewal.targets import StandardDispTarget
from emu_renewal.process import CosineMultiCurve
from emu_renewal.renew import MultiStrainModel
from emu_renewal.distributions import GammaDens
from emu_renewal.calibration import StandardCalib
from emu_renewal.outputs import store_outputs
from emu_renewal import mobility


class MobilityException(Exception):
    pass


def find_run_start_time(
    deaths_data,
    pop: float,
    threshold: float,
    default_start_time: datetime = datetime(2020, 6, 1),
) -> datetime:
    """Determine the time that the model should start running from.
    Calculated as the time until the per capita death rate reaches the
    specified threshold.

    Args:
        deaths_data: Deaths time series for the country considered
        pop: Population size
        death_start_threshold: How many deaths to reach
        latest_start_time: Default start time if deaths don't reach the threshold

    Returns:
        The date that the threshold is reached
    """
    per_capita_deaths = deaths_data / pop
    start = per_capita_deaths.index[per_capita_deaths.gt(threshold)].min()
    if pd.isna(start) or start > default_start_time:
        return default_start_time
    else:
        return start


def find_aust_start_times(vacc_data):
    norm_vacc_data = vacc_data / vacc_data.iloc[-1]
    start_time = norm_vacc_data[norm_vacc_data.gt(0.9)].idxmin()
    data_start_time = norm_vacc_data[norm_vacc_data.gt(0.9)].idxmin()
    return start_time, data_start_time


def find_run_end_time(
    vacc_data: pd.Series,
    cov_threshold: float,
    continent: str,
) -> datetime:
    """Find the time that the analysis should finish.
    Calculated as the time that the population vaccination coverage
    passes the requested threshold for all countries but Australia,
    provided that the vaccination coverage does reach this
    value by the 1st of June 2021.
    Otherwise return the end date for Google mobility data
    for Australia (12th of October 2022),
    or return the 1st of June 2021 for other countries.

    Args:
        vacc_data: The vaccination data for the country considered
        cov_threshold: The threshold
        continent: Two-character code for the continent of the analysis country

    Returns:
        The date at which to end the analysis period
    """
    cov_thres_perc = cov_threshold * 100
    if continent == "OC":
        return AUST_END
    elif vacc_data.empty or vacc_data.max() < cov_thres_perc:
        return OTHER_DEFAULT_END
    else:
        return vacc_data[vacc_data.gt(cov_thres_perc)].idxmin()


def collate_targets(
    cases_target: pd.Series,
    deaths_target: pd.Series,
    hosp_target: pd.Series,
    hosp_output_name: str,
    seroprev_target: pd.Series,
    ext_prop: float,
    prealpha_prop: pd.Series,
    start: datetime,
    end: datetime,
    continent: str,
) -> Dict[str, StandardDispTarget]:
    """Collate the targets gathered in the previous function
    into the appropriate structure for the calibration algorithm.

    Returns:
        All targets, either four or five, depending on whether there are seroprevalence estimates
    """
    pre_test_scaleup = cases_target.index > CASES_START
    case_mask = (start < cases_target.index) & (cases_target.index < end) & pre_test_scaleup
    select_cases = cases_target.loc[case_mask]

    death_mask = (start < deaths_target.index) & (deaths_target.index < end)
    select_deaths = deaths_target.loc[death_mask]

    if hosp_target is None:
        hosp_target_dict = {}
    else:
        hosp_mask = (start < hosp_target.index) & (hosp_target.index < end) & (hosp_target > 0.0)
        select_hosps = hosp_target.loc[hosp_mask]
        if select_hosps.empty:
            hosp_target_dict = {}
        else:
            hosp_weight = 20.0 * len(select_hosps) / len(select_deaths)
            hosp_target_dict = {
                hosp_output_name: StandardDispTarget(select_hosps, weight=hosp_weight)
            }

    seroprev_mask = (ext_prop < seroprev_target) & (seroprev_target < 1.0 - ext_prop)
    seroprev_target = seroprev_target[seroprev_mask]
    if seroprev_target.empty or continent == "OC":
        seroprev_target_dict = {}
    else:
        seroprev_target_dict = {"seropos": StandardDispTarget(seroprev_target, weight=10.0)}

    if prealpha_prop is not None:
        var_mask = (ext_prop < prealpha_prop) & (prealpha_prop < 1.0 - ext_prop)
        var_target_dict = {"prop_eu": StandardDispTarget(prealpha_prop[var_mask], weight=20.0)}
    else:
        var_target_dict = {}

    all_targets = (
        {
            "weekly_cases": StandardDispTarget(
                select_cases, weight=20.0 * len(select_cases) / len(select_deaths)
            ),
            "weekly_deaths": StandardDispTarget(select_deaths, weight=20.0),
        }
        | seroprev_target_dict
        | hosp_target_dict
        | var_target_dict
    )
    return all_targets


def find_variant_seeds(val, prealpha_prop, start_time):
    before_prop_time = (prealpha_prop - val).abs().idxmin() - timedelta(80)
    alpha_seed_start = max([before_prop_time, start_time])
    return [alpha_seed_start]


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


def get_mobility_provider(iso3: str, mob_analysis_type: str) -> mobility.MobilityProvider:

    if mob_analysis_type == "weighted_google_1exp":
        mob = get_google_mobility(iso3)
        nseries = len(mob.columns)
        priors = {
            "mob_weights": dist.Uniform(np.zeros(nseries), np.ones(nseries)),
            "mob_exp": dist.Uniform(0.0, 2.0),
        }
        return mobility.WeightedExpMobilityProvider(mob, priors)

    elif mob_analysis_type == "weighted_google_multiexp":
        mob = get_google_mobility(iso3)
        nseries = len(mob.columns)
        priors = {
            "mob_weights": dist.Uniform(np.zeros(nseries), np.ones(nseries)),
            "mob_exp": dist.Uniform(np.repeat(0.0, nseries), np.repeat(2.0, nseries)),
        }
        return mobility.WeightedMultiExpMobilityProvider(mob, priors)

    elif mob_analysis_type == "fb_linear":
        mob = get_fb_mobility(iso3)
        return mobility.SingleSeriesMobilityProvider(mob)

    elif mob_analysis_type == "fb_exp":
        mob = get_fb_mobility(iso3)
        priors = {"mob_exp": dist.Uniform(0.0, 2.0)}
        return mobility.SingleSeriesExpMobilityProvider(mob, priors)

    elif mob_analysis_type == "weighted_apple_1exp":
        mob = get_apple_mobility(iso3)
        nseries = len(mob.columns)
        priors = {
            "mob_weights": dist.Uniform(np.zeros(nseries), np.ones(nseries)),
            "mob_exp": dist.Uniform(0.0, 2.0),
        }
        return mobility.WeightedExpMobilityProvider(mob, priors)

    elif mob_analysis_type == "weighted_apple_multiexp":
        mob = get_apple_mobility(iso3)
        nseries = len(mob.columns)
        priors = {
            "mob_weights": dist.Uniform(np.zeros(nseries), np.ones(nseries)),
            "mob_exp": dist.Uniform(np.repeat(0.0, nseries), np.repeat(2.0, nseries)),
        }
        return mobility.WeightedMultiExpMobilityProvider(mob, priors)

    elif mob_analysis_type == "all_source_multiexp":
        apple_mob = get_apple_mobility(iso3)
        fb_mob = get_fb_mobility(iso3)
        g_mob = get_google_mobility(iso3)
        all_df = pd.concat([apple_mob, fb_mob, g_mob], axis=1).bfill().ffill()
        nseries = len(all_df.columns)
        priors = {
            "mob_weights": dist.Uniform(np.zeros(nseries), np.ones(nseries)),
            "mob_exp": dist.Uniform(np.repeat(0.0, nseries), np.repeat(2.0, nseries)),
        }
        return mobility.WeightedMultiExpMobilityProvider(all_df, priors)

    elif mob_analysis_type == "no_mob":
        return mobility.NoMobilityProvider()

    else:
        raise Exception(f"No provider available for analysis type {mob_analysis_type}")


def run_single_country(
    country,
    proc_update_freq,
    init_duration,
    mob_analysis_type,
    iterations,
    run_data_delay,
    analysis_name,
    most_extreme_prop: float = 0.05,
    deaths_start_threshold: float = 2e-6,
    seed_duration: int = 10,
    num_chains=4,
    prog_bar=False,
    logger=None,
):
    logger = logger or logging.getLogger()

    logger.info(f"\n________________________\nRunning job at {analysis_name}")
    iso3 = pycountry.countries.lookup(country).alpha_3
    iso2 = pycountry.countries.lookup(iso3).alpha_2
    continent = pc.country_alpha2_to_continent_code(iso2)
    logger.info(f"Country: {iso3}")
    logger.info(f"Mobility approach: {mob_analysis_type}")
    pop_year = 2022 if continent == "OC" else 2020
    pop = get_worldbank_national_pop(iso3, pop_year)
    vacc_data = get_country_vacc_data(iso3)
    end_time = find_run_end_time(vacc_data, most_extreme_prop, continent)

    cases_data = get_indicator_series_from_who_data("New_cases", country)
    deaths_data = get_indicator_series_from_who_data("New_deaths", country)
    data_start = find_run_start_time(deaths_data, pop, deaths_start_threshold)
    hosp_target, hosp_out_name = get_country_hosps(country, data_start, end_time)
    seroprev_target = get_filtered_seroprev(country, data_start, end_time)
    prealpha_prop = get_var_target(iso3)

    targets = collate_targets(
        cases_data,
        deaths_data,
        hosp_target,
        hosp_out_name,
        seroprev_target,
        most_extreme_prop,
        prealpha_prop,
        data_start,
        end_time,
        continent,
    )
    run_start = data_start - timedelta(run_data_delay)
    logger.info(
        f"Running from {run_start.strftime(DATE_FORMAT)} with data starting from {data_start.strftime(DATE_FORMAT)}"
    )
    logger.info(f"Running to {end_time.strftime(DATE_FORMAT)}")
    if continent == "AF":
        vars = ["eu"]
        seed_times = [run_start]
    else:
        vars = ["eu", "alpha"]
        seed_times = find_variant_seeds(0.5, prealpha_prop, run_start)
        alpha_seed_time, _ = get_alpha_seed_time(prealpha_prop)
        seed_times = [run_start, alpha_seed_time]

    try:
        mob_provider = get_mobility_provider(iso3, mob_analysis_type)
    except Exception as e:
        msg = f"{mob_analysis_type} mobility not available"
        raise MobilityException(msg)
    if mob_provider.mob_end:
        end_time = min([end_time, mob_provider.mob_end])

    priors = get_standard_priors() | mob_provider.get_priors()

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
        vars,
        "eu",
        seed_times,
        mob_provider,
        seed_duration,
    )
    calib = StandardCalib(model, priors, targets, proc_dispersion=dist.HalfNormal(0.5))
    kernel = infer.NUTS(
        calib.calibration, dense_mass=True, init_strategy=calib.custom_init(radius=0.1)
    )
    mcmc = infer.MCMC(
        kernel,
        num_chains=num_chains,
        num_samples=iterations,
        num_warmup=iterations,
        progress_bar=prog_bar,
    )
    mcmc.run(random.PRNGKey(0), extra_fields=["potential_energy"])
    storage_path = BASE_PATH / "outputs" / analysis_name / country / mob_analysis_type
    storage_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing to: {storage_path}")
    store_outputs(storage_path, model, calib, mcmc)
    logger.info(f"Completed {analysis_name}/{country}/{mob_analysis_type}")
