import numpy as np
import numpyro
from jax import numpy as jnp
from numpyro import distributions as dist
from numpyro import infer
from functools import partial
from warnings import warn

from emu_renewal.renew import MultiStrainModel
from emu_renewal.targets import Target
from emu_renewal.constants import INIT_RADIUS, PROC_DISP_SD


ParamDict = dict[str, dist.Distribution | float]


def custom_init(
    site=None, 
    n_proc: int=0,
):
    """Initialize a numpyro MCMC run, 
    returning 0.0 for "proc" (random process values),
    otherwise defaulting to init_to_uniform(radius).

    Args:
        site: 
        n_proc: Number of updates in the variable process

    Returns:
        The initialisation for the calibration

    Notes
    -----
    To initialise the model parameters,
    we started all the updates to the variable
    process from a value of zero (in logarithmic space)
    to represent no update, such that the 
    initialisation commenced with the variable 
    process being constant over time.
    For all other parameters,
    we used `numpyro`'s `init_to_uniform` method,
    with a radius of {INIT_RADIUS}.
    """
    if site is None:
        return partial(custom_init, n_proc=n_proc)

    if (
        site["type"] == "sample"
        and not site["is_observed"]
        and not site["fn"].support.is_discrete
    ):
        if site["value"] is not None:
            return site["value"]
        else:
            if site["name"] == "proc":
                return jnp.zeros(n_proc)
            else:
                return infer.init_to_uniform(site, INIT_RADIUS)


class StandardCalib:
    def __init__(
        self,
        epi_model: MultiStrainModel,
        params: ParamDict,
        targets: dict[str, Target],
    ):
        """Set up calibration object with epi model and data.

        Args:
            epi_model: The renewal model
            params: Parameter inputs, including both priors and fixed parameters
            targets: The data targets
        
        Notes
        -----
        For all epidemiological parameters,
        the priors were set as described in the parameters table.
        The dispersion parameter for the variable process
        was set to a half normal distribution with 
        standard deviation {PROC_DISP_SD}.
        """
        self.epi_model = epi_model
        self.n_proc_periods = len(self.epi_model.x_proc_data.points)
        self.custom_init = custom_init(n_proc=self.n_proc_periods)
        analysis_idx = self.epi_model.epoch.index_to_dti(self.epi_model.model_times)
        self.targets = targets
        self.common_idx = {}
        self.active_targets = []
        for ind in targets.keys():
            self.targets[ind].set_key(ind)
            ind_data = targets[ind].data
            common_dates_idx = ind_data.index.intersection(analysis_idx)
            if len(common_dates_idx) == 0:
                warn(f"Model dates exclude all data for target {ind}, disabling target")
            else:
                self.targets[ind].set_calibration_data(jnp.array(ind_data.loc[common_dates_idx]))
                common_abs_idx = np.array(self.epi_model.epoch.dti_to_index(common_dates_idx).astype(int))
                self.common_idx[ind] = common_abs_idx - self.epi_model.model_times[0]
                self.active_targets.append(ind)

        self.params = params
        # Compile transformed dists first to avoid memory leaks from
        # numpyro/jax buggy interaction
        _ = [p.mean for p in self.params.values() if isinstance(p, dist.Distribution)]

        # Separate parameters to sample vs fixed values
        self.sampled_params = {k: v for k, v in self.params.items() if isinstance(v, dist.Distribution)}
        self.fixed_params = {k: v for k, v in self.params.items() if not isinstance(v, dist.Distribution)}

        self.proc_dispersion = dist.HalfNormal(PROC_DISP_SD)

    def calibration(self):
        """Master calibration function.
        """
        params = self.sample_calib_params() | self.fixed_params
        result = self.epi_model.renewal_func(**params)
        for ind in self.active_targets:
            self.add_factor(result, ind, params)

    def sample_calib_params(self):
        """See describe_params below.

        Returns:
            Calibration parameters
        
        Notes
        -----
        The calibration process calibrates parameters for each
        consecutive update to the variable process in logarithmic space.
        The prior distribution for the update for each period 
        of the variable process is a normal distribution
        centred at a value of zero to represent no change
        from the previous value.
        The standard deviation of each normal distribution
        is provided by the (single) dispersion parameter
        of the variable process introduced above.
        """
        params = {k: numpyro.sample(k, v) for k, v in self.sampled_params.items()}
        proc_disp = numpyro.sample("dispersion_proc", self.proc_dispersion)
        proc_dist = dist.Normal(jnp.repeat(0.0, self.n_proc_periods), proc_disp)
        return params | {"proc": numpyro.sample("proc", proc_dist)}

    def add_factor(self, result, ind: str, parameters):
        """Add output target to calibration algorithm.

        Args:
            result: Output from model
            ind: Name of indicator

        Notes
        -----
        The log-likelihood of a simulation run
        is calculated as the sum of the log-likelihood
        associated with each target introduced above.
        """
        modelled = result[ind][self.common_idx[ind]]
        like_component = self.targets[ind].loglikelihood(modelled, parameters)
        numpyro.factor(f"{ind}_ll", like_component)
