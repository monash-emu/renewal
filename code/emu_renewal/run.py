from datetime import datetime, timedelta
import pycountry
import pycountry_convert as pc
from numpyro import distributions as dist
from numpyro import infer
from jax import random
from typing import Dict
import pandas as pd
import numpy as np
import sys

from emu_renewal.inputs import (
    DATE_FORMAT,
    BASE_PATH,
    get_indicator_series_from_who_data,
    get_country_vacc_data,
    get_worldbank_national_pop,
    get_standard_priors,
    get_google_mobility,
    get_apple_mobility,
    get_fb_mobility,
    get_prealpha_prop,
    get_filtered_seroprev,
    get_country_hosps,
)
from emu_renewal.targets import StandardDispTarget
from emu_renewal.process import CosineMultiCurve
from emu_renewal.renew import MultiStrainModel
from emu_renewal.distributions import GammaDens
from emu_renewal.calibration import StandardCalib
from emu_renewal.outputs import store_outputs
from emu_renewal import mobility


def find_run_start_time(
    deaths_data,
    pop: float,
    threshold: float,
) -> datetime:
    """Determine the time that the model should start running from.
    Calculated as the time until the per capita death rate reaches the
    specified threshold.

    Args:
        deaths_data: Deaths time series for the country considered
        pop: Population size
        death_start_threshold: How many deaths to reach

    Returns:
        The date that the threshold is reached
    """
    per_capita_deaths = deaths_data / pop
    return per_capita_deaths.index[per_capita_deaths.gt(threshold)].min()


def find_run_end_time(
    vacc_data: pd.Series,
    cov_threshold: float,
) -> datetime:
    """Find the time that the analysis should finish.
    Calculated as the time that the population vaccination coverage
    passes the requested threshold.

    Args:
        vacc_data: The vaccination data for the country considered
        cov_threshold: The threshold

    Returns:
        The date at which the threshold is reached
    """
    return vacc_data[vacc_data.gt(cov_threshold * 100)].idxmin()


def collate_targets(
    cases_target: pd.Series,
    deaths_target: pd.Series,
    hosp_target: pd.Series,
    hosp_output_name: pd.Series,
    seroprev_target: pd.Series,
    most_extreme_prop: pd.Series,
    prealpha_prop: pd.Series,
    start: pd.Series,
    end: pd.Series,
) -> Dict[str, StandardDispTarget]:
    """Collate the targets gathered in the previous function
    into the appropriate structure for the calibration algorithm.

    Returns:
        All targets, either four or five, depending on whether there are seroprevalence estimates
    """
    # Ignore initial cases before testing scaled up
    case_mask = (
        (start < cases_target.index)
        & (cases_target.index < end)
        & (cases_target > 0.0)
        & (cases_target.index > datetime(2020, 6, 1))
    )
    select_cases = cases_target.loc[case_mask]

    death_mask = (start < deaths_target.index) & (deaths_target.index < end) & (deaths_target > 0.0)
    select_deaths = deaths_target.loc[death_mask]

    hosp_mask = (start < hosp_target.index) & (hosp_target.index < end) & (hosp_target > 0.0)
    select_hosps = hosp_target.loc[hosp_mask]
    if select_hosps.empty:
        hosp_target_dict = {}
    else:
        hosp_weight = 20.0 * len(select_hosps) / len(select_deaths)
        hosp_target_dict = {hosp_output_name: StandardDispTarget(select_hosps, weight=hosp_weight)}

    seroprev_mask = (
        (most_extreme_prop < seroprev_target)
        & (seroprev_target < 1.0 - most_extreme_prop)
        & (seroprev_target > 0.0)
    )
    seroprev_target = seroprev_target[seroprev_mask]
    if seroprev_target.empty:
        seroprev_target_dict = {}
    else:
        {"seropos": StandardDispTarget(seroprev_target, weight=10.0)}

    var_mask = (most_extreme_prop < prealpha_prop) & (prealpha_prop < 1.0 - most_extreme_prop)
    prealpha_prop = prealpha_prop[var_mask]

    all_targets = (
        {
            "weekly_cases": StandardDispTarget(
                select_cases, weight=20.0 * len(select_cases) / len(select_deaths)
            ),
            "weekly_deaths": StandardDispTarget(select_deaths, weight=20.0),
            "prop_eu": StandardDispTarget(prealpha_prop, weight=20.0),
        }
        | seroprev_target_dict
        | hosp_target_dict
    )
    return all_targets


def find_variant_seeds(val, prealpha_prop, start_time):
    before_prop_time = (prealpha_prop - val).abs().idxmin() - timedelta(80)
    alpha_seed_start = max([before_prop_time, start_time])
    return [alpha_seed_start]


def log(log_str: str):
    print(log_str)
    sys.stdout.flush()


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
    min_var_threshold: int = 10,
    seed_duration: int = 10,
    num_chains=4,
    prog_bar=False,
):
    log(f"\n________________________\nRunning job at {analysis_name}")
    iso3 = pycountry.countries.lookup(country).alpha_3
    iso2 = pycountry.countries.lookup(iso3).alpha_2
    continent = pc.country_alpha2_to_continent_code(iso2)
    log(f"Country: {iso3}")
    log(f"Mobility approach: {mob_analysis_type}")
    pop = get_worldbank_national_pop(iso3)
    vacc_data = get_country_vacc_data(iso3)
    end_time = find_run_end_time(vacc_data, most_extreme_prop)

    cases_data = get_indicator_series_from_who_data("New_cases", country)
    deaths_data = get_indicator_series_from_who_data("New_deaths", country)
    data_start = find_run_start_time(deaths_data, pop, deaths_start_threshold)
    hosp_target, hosp_out_name = get_country_hosps(country, data_start, end_time)
    seroprev_target = get_filtered_seroprev(country, data_start, end_time)
    prealpha_prop = get_prealpha_prop(iso3, min_var_threshold)

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
    )
    run_start = data_start - timedelta(run_data_delay)
    log(
        f"Running from {run_start.strftime(DATE_FORMAT)} with data starting from {data_start.strftime(DATE_FORMAT)}"
    )
    log(f"Running to {end_time.strftime(DATE_FORMAT)}")
    seed_times = find_variant_seeds(0.5, prealpha_prop, run_start)
    seed_times = [run_start] + seed_times

    mob_provider = get_mobility_provider(iso3, mob_analysis_type)
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
        ["eu", "alpha"],
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
    log(f"Writing to: {storage_path}")
    store_outputs(storage_path, model, calib, mcmc)
