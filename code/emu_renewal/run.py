from datetime import datetime, timedelta
import pycountry
from numpyro import distributions as dist
from numpyro import infer
from jax import random

from emu_renewal.inputs import DATE_FORMAT, BASE_PATH, get_indicator_series_from_who_data, get_country_vacc_data, get_standard_targets, get_country_vars, \
    get_worldbank_national_pop, get_country_mobility, get_standard_priors
from emu_renewal.targets import StandardDispTarget
from emu_renewal.process import CosineMultiCurve
from emu_renewal.renew import MultiStrainModel
from emu_renewal.distributions import GammaDens
from emu_renewal.calibration import StandardCalib
from emu_renewal.outputs import store_outputs


def find_run_start_time(
    iso3: str,
    pop: float,
    death_start_threshold: float,
) -> datetime:
    deaths_series = get_indicator_series_from_who_data("New_deaths", iso3)
    per_capita_deaths = deaths_series / pop
    return per_capita_deaths.index[per_capita_deaths.gt(death_start_threshold)].min()


def find_run_end_time(country, cov_threshold):
    vacc_data = get_country_vacc_data(country)
    return vacc_data[vacc_data.gt(cov_threshold * 100)].idxmin()


def gather_targets(iso3, start_time, end_time, most_extreme_prop, min_var_samples, initial_countries):
    iso2 = pycountry.countries.lookup(iso3).alpha_2
    hosp_indicator, hosp_output = ("Weekly new hospital admissions", "weekly_admissions") if iso2 in initial_countries["admissions"] else ("Daily hospital occupancy", "occupancy")
    cases_target, hosp_target, deaths_target, seroprev_target, init_data = get_standard_targets(iso2, start_time, end_time, 50, hosp_indicator)
    cases_target = cases_target[cases_target.index >= datetime(2020, 6, 1)]  # Ignore initial cases before testing scaled up
    var_country_name = pycountry.countries.lookup(iso3).official_name if iso3 in ["CZE"] else pycountry.countries.lookup(iso3).name
    var_data = get_country_vars(var_country_name)
    var_data = var_data[var_data.sum(axis=1) >= min_var_samples]
    prealpha_vars = ["20A.EU1"] if iso3 == "LTU" else ["20A.EU1", "20A.EU2"]  # Lithuania has no 20A.EU2
    prealpha_prop = var_data[prealpha_vars].sum(axis=1) / var_data.sum(axis=1)
    prealpha_prop = prealpha_prop[(most_extreme_prop < prealpha_prop) & (prealpha_prop < 1.0 - most_extreme_prop)]
    return cases_target, hosp_target, hosp_output, deaths_target, seroprev_target, prealpha_prop, init_data


def collate_targets(cases_target, deaths_target, hosp_target, hosp_output, seroprev_target, most_extreme_prop, prealpha_prop, start_time, end_time):
    seroprev_target = seroprev_target[(most_extreme_prop < seroprev_target) & (seroprev_target < 1.0 - most_extreme_prop)]
    seroprev_target_dict = {"seropos": StandardDispTarget(seroprev_target, weight=10.0)} if any(seroprev_target) else {}
    select_cases = cases_target.loc[(start_time < cases_target.index) & (cases_target.index < end_time)]
    select_deaths = deaths_target.loc[(start_time < deaths_target.index) & (deaths_target.index < end_time)]
    select_hosps = hosp_target.loc[(start_time < hosp_target.index) & (hosp_target.index < end_time)]   
    all_targets = {
        "weekly_cases": StandardDispTarget(select_cases, weight=20.0 * len(select_cases) / len(select_deaths)),
        "weekly_deaths": StandardDispTarget(select_deaths, weight=20.0),
        hosp_output: StandardDispTarget(select_hosps, weight=20.0 * len(select_hosps) / len(select_deaths)),
        "prop_eu": StandardDispTarget(prealpha_prop, weight=20.0),
    } | seroprev_target_dict
    return all_targets


def find_variant_seeds(val, prealpha_prop, start_time, seed_duration):
    before_prop_time = (prealpha_prop - val).abs().idxmin() - timedelta(80)
    alpha_seed_start = max([before_prop_time, start_time])
    return [[alpha_seed_start, alpha_seed_start + timedelta(seed_duration)]]


def run_single_country(country, initial_countries, seed_duration, proc_update_freq, init_duration, mob_analysis_type, iterations):
    analysis_time = datetime.now().strftime(DATE_FORMAT)
    iso3 = pycountry.countries.lookup(country).alpha_3
    pop = get_worldbank_national_pop(iso3)
    start_time = find_run_start_time(iso3, pop, 2e-6)
    most_extreme_prop = 0.05
    end_time = find_run_end_time(country, most_extreme_prop)
    cases_target, hosp_target, hosp_output, deaths_target, seroprev_target, prealpha_prop, init_data = gather_targets(iso3, start_time, end_time, most_extreme_prop, 10, initial_countries)
    targets = collate_targets(cases_target, deaths_target, hosp_target, hosp_output, seroprev_target, most_extreme_prop, prealpha_prop, start_time, end_time)
    seed_times = find_variant_seeds(0.5, prealpha_prop, start_time, seed_duration)
    mobility = get_country_mobility(iso3)
    priors = get_standard_priors()
    model = MultiStrainModel(
        pop,
        start_time,
        end_time,
        proc_update_freq,
        CosineMultiCurve(),
        GammaDens(),
        init_duration,
        init_data,
        GammaDens(),
        GammaDens(),
        ["eu", "alpha"],
        "eu",
        seed_times,
        100.0,
        mobility[mob_analysis_type].dropna(),
    )
    calib = StandardCalib(model, priors, targets, proc_dispersion=dist.HalfNormal(0.5))
    kernel = infer.NUTS(calib.calibration, dense_mass=True, init_strategy=calib.custom_init(radius=0.1))
    mcmc = infer.MCMC(kernel, num_chains=4, num_samples=iterations, num_warmup=iterations)
    mcmc.run(random.PRNGKey(0), extra_fields=["potential_energy"])
    storage_path = BASE_PATH / "outputs" / country / mob_analysis_type / analysis_time
    storage_path.mkdir(parents=True, exist_ok=True)
    store_outputs(storage_path, model, calib, mcmc)
