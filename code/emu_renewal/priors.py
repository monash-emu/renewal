from typing import Dict
from jax import numpy as jnp
import yaml as yml
from numpyro import distributions as dist

from emu_renewal.constants import DATA_PATH
from emu_renewal.inputs import get_income_group
from emu_renewal.utils import get_beta_params_from_mean_var


def get_standard_priors(
    n_strains: int,
    hosp_out_type: str,
    iso3: str,
) -> Dict[str, dist.Distribution]:
    """Load the priors from the yml and combine with
    standard hard-coded priors.

    Args:
        n_strains: The number of strains implemented
        hosp_out_type: The hospital-related indicator name
            Must be one of the keys to relevant_duration_priors below

    Returns:
        The prior distributions
    """
    loaded_priors = yml.safe_load(open(DATA_PATH / "config/priors.yml", "r"))

    # Durations
    duration_priors = {
        k: dist.TruncatedNormal(v["mean"], v["sd"], low=1.0, high=v["mean"] * 2.5)
        for k, v in loaded_priors["durations"].items()
    }
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

    # Proportions from summary statistics
    income = get_income_group(iso3)
    adjusters = {
        "Low income": 0.4,
        "Lower middle income": 0.6,
        "Upper middle income": 0.8,
        "High income": 1.0,
    }
    adjuster = 0.4 if iso3 == "VEN" else adjusters[income]
    beta_from_sum = loaded_priors["beta_from_summary"]
    beta_from_sum_dists = {}
    for k, v in beta_from_sum.items():
        a, b = get_beta_params_from_mean_var(v["mean"] * adjuster, v["std"] ** 2.0)
        beta_from_sum_dists[k] = dist.Beta(a, b)
    if hosp_out_type == "":
        beta_from_sum_dists["har"] = 1.0

    # Proportions
    beta_priors = {
        k: dist.Beta(v["alpha"], v["beta"])
        for k, v in loaded_priors["beta"].items()
        if k != "cross_immunity"
    }
    if "icu_" not in hosp_out_type:
        beta_priors["icu_ar"] = 1.0
    if hosp_out_type == "":
        beta_priors["har"] = 1.0
        beta_priors["icu_ar"] = 1.0

    # Variant-related
    seed_low_lim = jnp.repeat(jnp.log(1e-7), n_strains)
    seed_up_lim = jnp.repeat(jnp.log(5e-6), n_strains)
    seed_rate_priors = {"seed_rates": dist.Uniform(seed_low_lim, seed_up_lim)}
    seed_offsets_dist = dist.Uniform(
        jnp.repeat(4.0, n_strains - 1), jnp.repeat(90.0, n_strains - 1)
    )
    seed_offsets_priors = seed_offsets_dist if n_strains > 1 else None
    seed_priors = {"seed_offsets": seed_offsets_priors}
    relinfect_means = jnp.repeat(1.4, n_strains - 1)
    infect_dist_prior = dist.TruncatedNormal(relinfect_means, 0.2, low=1.0, high=2.0)
    infect_dist = infect_dist_prior if n_strains > 1 else None
    inf_priors = {"relinfect": infect_dist}
    imm = loaded_priors["beta"]["cross_immunity"]
    imm_prior = {"cross_immunity": dist.Beta(imm["alpha"], imm["beta"])} if n_strains > 0 else {}

    # Miscellaneous
    rt_prior = {"rt_init": dist.Normal(0.0, 0.5)}
    disp_prior = {"shared_dispersion": dist.HalfNormal(0.5)}
    prop_disp_prior = {"prop_shared_disp": 0.05}
    seroprev_disp = {"seroprev_disp": 0.2}

    return (
        rel_durs
        | irrel_durs
        | beta_priors
        | beta_from_sum_dists
        | seed_rate_priors
        | inf_priors
        | imm_prior
        | rt_prior
        | disp_prior
        | prop_disp_prior
        | seed_priors
        | seroprev_disp
    )
