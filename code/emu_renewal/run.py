from datetime import datetime, timedelta
import pycountry
from numpyro import distributions as dist
from numpyro import infer
from jax import random
from typing import Tuple, Dict
import pandas as pd
import numpy as np
import sys

from emu_renewal.inputs import (
    DATE_FORMAT,
    BASE_PATH,
    DATA_PATH,
    get_indicator_series_from_who_data,
    get_country_vacc_data,
    get_standard_targets,
    get_country_vars,
    get_worldbank_national_pop,
    get_country_mobility,
    get_standard_priors,
)
from emu_renewal.targets import StandardDispTarget
from emu_renewal.process import CosineMultiCurve
from emu_renewal.renew import MultiStrainModel
from emu_renewal.distributions import GammaDens
from emu_renewal.calibration import StandardCalib
from emu_renewal.outputs import store_outputs
from emu_renewal import mobility


def find_run_start_time(
    iso3: str,
    pop: float,
    death_start_threshold: float,
) -> datetime:
    """Determine the time that the model should start running from.
    Calculated as the time until the per capita death rate reaches the
    specified threshold.

    Args:
        iso3: Country identifier
        pop: Population size
        death_start_threshold: How many deaths to reach

    Returns:
        The date that the threshold is reached
    """
    deaths_series = get_indicator_series_from_who_data("New_deaths", iso3)
    per_capita_deaths = deaths_series / pop
    return per_capita_deaths.index[per_capita_deaths.gt(death_start_threshold)].min()


def find_run_end_time(
    country: str,
    cov_threshold: float,
) -> datetime:
    """Find the time that the analysis should finish.
    Calculated as the time that the population vaccination coverage
    passes the requested threshold.

    Args:
        country: The name of the country
        cov_threshold: The threshold

    Returns:
        The date at which the threshold is reached
    """
    vacc_data = get_country_vacc_data(country)
    return vacc_data[vacc_data.gt(cov_threshold * 100)].idxmin()


def gather_targets(
    iso3: str,
    data_start: datetime,
    analysis_end: datetime,
    min_var_samples: int,
    hosp_out: str,
) -> Tuple[pd.Series]:
    """Get the targets as separate series, plus the initialisation series.

    Args:
        iso3: Country identifier
        start_time: Time that analysis starts
        end_time: Time that analysis ends
        min_var_samples: The minimum number of variant samples allowed
        hosp_out: The hospitalisation output required from the OWID data
            (either "Daily hospital occupancy" or "Weekly new hospital admissions")

    Returns:
        The various calibration targets and initialisation data
    """
    cases_target, hosp_target, deaths_target, seroprev_target = get_standard_targets(
        iso3, data_start, analysis_end, hosp_out
    )
    cases_target = cases_target[
        cases_target.index >= datetime(2020, 6, 1)
    ]  # Ignore initial cases before testing scaled up
    var_country_name = (
        pycountry.countries.lookup(iso3).official_name
        if iso3 in ["CZE"]
        else pycountry.countries.lookup(iso3).name
    )
    var_data = get_country_vars(var_country_name)
    var_data = var_data[var_data.sum(axis=1) >= min_var_samples]
    prealpha_vars = (
        ["20A.EU1"] if iso3 == "LTU" else ["20A.EU1", "20A.EU2"]
    )  # Lithuania has no 20A.EU2
    prealpha_prop = var_data[prealpha_vars].sum(axis=1) / var_data.sum(axis=1)
    if iso3 == "PRT":
        prealpha_prop = prealpha_prop[
            prealpha_prop.index > datetime(2021, 1, 1)
        ]  # Fluctuations in sample numbers in Portugal
    return cases_target, hosp_target, deaths_target, seroprev_target, prealpha_prop


def collate_targets(
    cases_target: pd.Series,
    deaths_target: pd.Series,
    hosp_target: pd.Series,
    hosp_output_name: pd.Series,
    seroprev_target: pd.Series,
    most_extreme_prop: pd.Series,
    prealpha_prop: pd.Series,
    start_time: pd.Series,
    end_time: pd.Series,
) -> Dict[str, StandardDispTarget]:
    """Collate the targets gathered in the previous function
    into the appropriate structure for the calibration algorithm.

    Returns:
        All targets, either four or five, depending on whether there are seroprevalence estimates
    """
    select_cases = cases_target.loc[
        (start_time < cases_target.index) & (cases_target.index < end_time)
    ]
    select_deaths = deaths_target.loc[
        (start_time < deaths_target.index) & (deaths_target.index < end_time)
    ]
    select_hosps = hosp_target.loc[
        (start_time < hosp_target.index) & (hosp_target.index < end_time)
    ]
    seroprev_target = seroprev_target[
        (most_extreme_prop < seroprev_target) & (seroprev_target < 1.0 - most_extreme_prop)
    ]
    seroprev_target_dict = (
        {"seropos": StandardDispTarget(seroprev_target, weight=10.0)}
        if any(seroprev_target)
        else {}
    )
    prealpha_prop = prealpha_prop[
        (most_extreme_prop < prealpha_prop) & (prealpha_prop < 1.0 - most_extreme_prop)
    ]
    all_targets = {
        "weekly_cases": StandardDispTarget(
            select_cases, weight=20.0 * len(select_cases) / len(select_deaths)
        ),
        "weekly_deaths": StandardDispTarget(select_deaths, weight=20.0),
        hosp_output_name: StandardDispTarget(
            select_hosps, weight=20.0 * len(select_hosps) / len(select_deaths)
        ),
        "prop_eu": StandardDispTarget(prealpha_prop, weight=20.0),
    } | seroprev_target_dict
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
        g_mob_df = pd.read_csv(DATA_PATH / f"mobility/{iso3}_gmob_data.csv", index_col=0)
        g_mob_df.index = pd.to_datetime(g_mob_df.index)
        g_mob_df = g_mob_df.rolling(7).mean().dropna()

        priors = {
            "mob_weights": dist.Uniform(np.zeros(6), np.ones(6)),
            "mob_exp": dist.Uniform(0.0, 2.0),
        }

        mob_provider = mobility.WeightedExpMobilityProvider(g_mob_df, priors)
        return mob_provider
    elif mob_analysis_type == "fb_linear":
        mob_df = get_country_mobility(iso3)
        # Additional dropna required since dataframe may extend beyond series validity
        mob_series = mob_df["fb_linear"].dropna()
        return mobility.SingleSeriesMobilityProvider(mob_series)
    elif mob_analysis_type == "fb_exp":
        mob_df = get_country_mobility(iso3)
        # Additional dropna required since dataframe may extend beyond series validity
        mob_series = mob_df["fb_linear"].dropna()
        priors = {"mob_exp": dist.Uniform(0.0, 2.0)}
        return mobility.SingleSeriesExpMobilityProvider(mob_series, priors)
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
    hosp_out,
    hosp_out_name,
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
    log(f"Country: {iso3}")
    log(f"Mobility approach: {mob_analysis_type}")
    pop = get_worldbank_national_pop(iso3)
    data_start = find_run_start_time(iso3, pop, deaths_start_threshold)
    end_time = find_run_end_time(country, most_extreme_prop)
    cases_target, hosp_target, deaths_target, seroprev_target, prealpha_prop = gather_targets(
        iso3, data_start, end_time, min_var_threshold, hosp_out
    )
    targets = collate_targets(
        cases_target,
        deaths_target,
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
