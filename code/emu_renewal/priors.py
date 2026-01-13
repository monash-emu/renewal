from typing import Dict
from jax import numpy as jnp
import yaml as yml
from numpyro import distributions as dist
import pycountry

from emu_renewal.constants import (
    DATA_PATH,
    DUR_MIN,
    DUR_REL_MAX,
    SEVERITY_ADJS,
    EXTRA_LOW_INC,
    BETA_MIN,
    BETA_MAX,
    SEED_RATE_LOW,
    SEED_RATE_UP,
    SEED_OFF_LOW,
    SEED_OFF_UP,
    RELINF_MEAN,
    RELINF_LOW,
    RELINF_UP,
    RELINF_SD,
    SHARED_DISP_SD,
    PROP_DISP,
    SEROPREV_DISP,
)
from emu_renewal.inputs import get_income_group
from emu_renewal.utils import get_beta_params_from_mean_var, get_cont_of_country
from emu_renewal.document import get_exp_val_from_string, get_float_dict_from_str


def get_standard_priors(
    n_strains: int,
    hosp_out_type: str,
    iso3: str,
    continent: str,
    waning: bool,
) -> Dict[str, dist.Distribution]:
    """Load the priors from the yml and combine with
    standard hard-coded priors.

    Args:
        n_strains: The number of strains implemented
        hosp_out_type: The hospital-related indicator name
            Must be one of the keys to relevant_duration_priors below
        iso3: The country identifier
        continent: The continent identifier

    Returns:
        The prior distributions

    Notes
    -----
    For all prior distributions representing durations of time
    (including the generation interval, time from infection to reporting
    and time from infection to death),
    we used a truncated normal prior
    with the lower truncation limit set at {DUR_MIN} day
    and the upper truncation limit set at {DUR_REL_MAX} times
    the mean value of the prior.
    For each of the beta-distributed proportions
    of incidence resulting in the key outputs for calibration,
    we analytically calculated the parameters to the beta
    distribution from the desired mean and standard deviation
    determined from the literature.
    These parameters comprised the infection fatality rate,
    the case detection proportion and, where applicable,
    the hospital or ICU admission proportion.
    For high-income countries, the mean value used
    was as presented in the parameter choices section,
    whereas this was reduced for other countries.
    The reductions based on income status were: {SEVERITY_ADJS},
    with {EXTRA_LOW_INC} (for which a World Bank class was not available)
    considered a low-income country for this purpose.
    __RETURN__
    The starting transmissibility scaling parameter
    was assigned a uniform prior distribution on domain
    {BETA_MIN} to {BETA_MAX}.
    __RETURN__
    The seeding rate for each variant was calibrated from
    ${SEED_RATE_LOW}$ to ${SEED_RATE_UP}$ cases per capita per day
    using a uniform distribution in logarithmic space.
    The seeding offset (i.e. the time from modelled seeding
    to the first calibration data point)
    for each variant except for the first one simulated
    was calibrated from {SEED_OFF_LOW} to {SEED_OFF_UP} days
    using a uniform distribution in untransformed space.
    __RETURN__
    The relative infectiousness of each variant
    except for the first was calibrated using
    truncated normal distributions with mean {RELINF_MEAN},
    standard deviation {RELINF_SD},
    lower truncation limit {RELINF_LOW}
    and upper truncation limit {RELINF_UP}.
    __RETURN__
    The prior for the shared dispersion parameter
    for all time series data was a half-normal distribution
    with standard deviation {SHARED_DISP_SD}.
    The dispersion parameter for
    all variant incidence proportions
    was set at {PROP_DISP}.
    The dispersion parameter for
    all seroprevalence targets
    was set at {SEROPREV_DISP}.
    """
    loaded_priors = yml.safe_load(open(DATA_PATH / "evidence/priors.yml", "r"))

    # Durations
    duration_priors = {
        k: dist.TruncatedNormal(v["mean"], v["sd"], low=DUR_MIN, high=v["mean"] * DUR_REL_MAX)
        for k, v in loaded_priors["durations"].items()
        if k != "immune"
    }
    if get_cont_of_country(iso3) == "OC":
        duration_priors["gen_mean"] = duration_priors["gen_mean_oc"]
    universal_prior_names = [
        "gen_mean",
        "gen_sd",
        "report_mean",
        "report_sd",
        "death_mean",
        "death_sd",
    ]
    rel_durations_dict = {
        "weekly_admissions": ["admit_mean", "admit_sd"],
        "occupancy": ["admit_mean", "admit_sd", "stay_mean", "stay_sd"],
        "icu_admissions": ["icu_admit_mean", "icu_admit_sd"],
        "icu_occupancy": ["icu_admit_mean", "icu_admit_sd", "icu_stay_mean", "icu_stay_sd"],
        "": [],
    }
    duration_prior_names = rel_durations_dict[hosp_out_type] + universal_prior_names
    rel_durs = {k: v for k, v in duration_priors.items() if k in duration_prior_names}
    irrel_durs = {k: 1.0 for k in duration_priors if k not in rel_durs}

    # Proportions
    income = get_income_group(iso3)
    adjusters = get_float_dict_from_str(SEVERITY_ADJS)
    extra_low_inc = pycountry.countries.lookup(EXTRA_LOW_INC).alpha_3
    adjuster = adjusters["Low income"] if iso3 == extra_low_inc else adjusters[income]
    beta_reqs = loaded_priors["beta"]
    betas_to_drop = [] if n_strains > 1 else ["cross_immunity"]
    rel_betas = {k: v for k, v in beta_reqs.items() if k not in betas_to_drop}
    irrel_betas = {k: 1.0 for k in beta_reqs if k in betas_to_drop}
    beta_dists = {}
    for k, v in rel_betas.items():
        a, b = get_beta_params_from_mean_var(v["mean"] * adjuster, v["std"])
        beta_dists[k] = dist.Beta(a, b)
    if "icu_" not in hosp_out_type:
        beta_dists["icuar"] = 1.0
    if hosp_out_type == "":
        beta_dists["har"] = 1.0

    # Variant-related
    seed_rate_low = get_exp_val_from_string(SEED_RATE_LOW)
    seed_low_lim = jnp.repeat(jnp.log(float(seed_rate_low)), n_strains)
    seed_rate_up = get_exp_val_from_string(SEED_RATE_UP)
    seed_up_lim = jnp.repeat(jnp.log(float(seed_rate_up)), n_strains)
    seed_rate_priors = {"seed_rates": dist.Uniform(seed_low_lim, seed_up_lim)}
    seed_off_low_lim = jnp.repeat(SEED_OFF_LOW, n_strains - 1)
    seed_off_up_lim = jnp.repeat(SEED_OFF_UP, n_strains - 1)
    seed_offsets_dist = dist.Uniform(seed_off_low_lim, seed_off_up_lim)
    seed_offsets_priors = seed_offsets_dist if n_strains > 1 else None
    seed_priors = {"seed_offsets": seed_offsets_priors}
    relinf_mean = jnp.repeat(RELINF_MEAN, n_strains - 1)
    infect_dist_prior = dist.TruncatedNormal(relinf_mean, RELINF_SD, low=RELINF_LOW, high=RELINF_UP)
    infect_dist = infect_dist_prior if n_strains > 1 else None
    inf_priors = {"relinfect": infect_dist}
    if continent == "OC":
        severity_dist = jnp.ones(n_strains - 1)
    elif n_strains == 1:
        severity_dist = jnp.empty(0)
    else:
        relseverity_mean = jnp.repeat(1.5, n_strains - 1)
        severity_dist = dist.TruncatedNormal(relseverity_mean, 0.2, low=1.0, high=2.0)
    severity_priors = {"relseverity": severity_dist}

    # Miscellaneous
    fixed_params = loaded_priors["fixed"]
    vacc_protect_hosp = {"vacc_protect_hosp": fixed_params["vacc_protect_hosp"]["value"]}
    vacc_protect_death = {"vacc_protect_death": fixed_params["vacc_protect_death"]["value"]}
    beta = {"beta": dist.Uniform(BETA_MIN, BETA_MAX)}
    disp_prior = {"shared_dispersion": dist.HalfNormal(SHARED_DISP_SD)}
    prop_disp_prior = {"prop_disp": PROP_DISP}
    seroprev_disp = {"seroprev_disp": SEROPREV_DISP}
    imm_mean = loaded_priors["durations"]["immune"]["mean"]
    imm_sd = loaded_priors["durations"]["immune"]["sd"]
    wane_dist = dist.TruncatedNormal(imm_mean, imm_sd, low=30.0) if waning else 0.0
    waning_prior = {"imm_time": wane_dist}

    return (
        rel_durs
        | irrel_durs
        | beta_dists
        | irrel_betas
        | seed_rate_priors
        | inf_priors
        | severity_priors
        | beta
        | disp_prior
        | prop_disp_prior
        | seed_priors
        | seroprev_disp
        | vacc_protect_hosp
        | vacc_protect_death
        | waning_prior
    )
