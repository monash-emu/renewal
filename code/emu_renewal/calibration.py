from jax import numpy as jnp
import numpy as np
import numpyro
from numpyro import distributions as dist
from numpyro import infer
from functools import partial
from warnings import warn

from emu_renewal.renew import MultiStrainModel
from emu_renewal.targets import Target


ParamDict = dict[str, dist.Distribution | float]


def custom_init(
    site=None, 
    radius: float=2.0, 
    n_proc: int=0,
):
    """Initialize a numpyro MCMC run, 
    returning 0.0 for "proc" (random process values),
    otherwise defaulting to init_to_uniform(radius)
    """
    if site is None:
        return partial(custom_init, radius=radius, n_proc=n_proc)

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
                return infer.init_to_uniform(site, radius)


class StandardCalib:
    def __init__(
        self,
        epi_model: MultiStrainModel,
        params: ParamDict,
        targets: dict[str, Target],
        proc_dispersion: dist.Distribution=dist.HalfNormal(0.1),
    ):
        """Set up calibration object with epi model and data.

        Args:
            epi_model: The renewal model
            params: Parameter inputs, including both priors and fixed parameters
            targets: The data targets
            proc_dispersion: Distribution used for the dispersion of the random process
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

        self.proc_dispersion = proc_dispersion

    def get_description(self) -> str:
        description = self.describe_params()
        for ind in self.targets.keys():
            description += self.describe_like_contribution(ind)
        return description

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
        """
        params = {k: numpyro.sample(k, v) for k, v in self.sampled_params.items()}
        proc_disp = numpyro.sample("dispersion_proc", self.proc_dispersion)
        proc_dist = dist.Normal(jnp.repeat(0.0, self.n_proc_periods), proc_disp)
        return params | {"proc": numpyro.sample("proc", proc_dist)}

    def describe_params(self):
        return (
            f"The calibration process calibrates parameters for {self.n_proc_periods} "
            "values for periods of the variable process to the data. "
            "The relative values pertaining to each period of the variable process "
            "are estimated from normal prior distributions centred at no "
            "change from the value of the previous stage of the process. "
            "The dispersion of the variable process is calibrated, "
            "using a half-normal distribution "
            f"with standard deviation {self.proc_dispersion.scale}. "
        )

    def add_factor(self, result, ind: str, parameters):
        """Add output target to calibration algorithm.

        Args:
            result: Output from model
            ind: Name of indicator
        """
        modelled = result[ind][self.common_idx[ind]]
        like_component = self.targets[ind].loglikelihood(modelled, parameters)
        numpyro.factor(f"{ind}_ll", like_component)

    def describe_like_contribution(self, indicator):
        return (
            f"The log of the modelled {indicator} values for each parameter set "
            "is compared against the corresponding data "
            "from the end of the run-in phase through to the end of the analysis. "
            "The dispersion parameter for this comparison of log values is also calibrated, "
            "with the dispersion parameter prior using a half-normal distribution, "
            "and shared between the calibration indicators. "
        )
