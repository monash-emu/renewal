from typing import Dict
from jax import numpy as jnp
import numpy as np
import pandas as pd
import numpyro
from numpyro import distributions as dist

pd.options.plotting.backend = "plotly"

from emu_renewal.renew import RenewalModel, ModelResult
from emu_renewal.utils import custom_init


class Calibration:
    def __init__(
        self,
        epi_model: RenewalModel,
        priors: dict[str, dist.Distribution],
        data: dict[str, pd.Series],
    ):
        """Set up calibration object with epi model and data.

        Args:
            epi_model: The renewal model
            data: The data targets
        """
        self.epi_model = epi_model
        self.n_proc_periods = len(self.epi_model.x_proc_data.points)

        self.custom_init = custom_init(n_proc=self.n_proc_periods)

        analysis_indices = self.epi_model.epoch.index_to_dti(self.epi_model.model_times)
        self.targets = {}
        self.common_indices = {}
        for ind in data.keys():
            ind_data = data[ind]
            common_dates_idx = ind_data.index.intersection(analysis_indices)
            self.targets[ind] = jnp.array(ind_data.loc[common_dates_idx])
            common_abs_indices = np.array(
                self.epi_model.epoch.dti_to_index(common_dates_idx).astype(int)
            )
            self.common_indices[ind] = common_abs_indices - self.epi_model.model_times[0]

        self.priors = priors
        _ = [
            p.mean for p in self.priors.values()
        ]  # Compile transformed dists first to avoid memory leaks

    def get_model_indicator(self, result: ModelResult, indicator: str):
        """Get the modelled values for a particular epidemiological indicator
        for a given set of epi parameters.

        Args:
            params: All renewal model parameters

        Returns:
            Modelled time series of the indicator over analysis period
        """
        return getattr(result, indicator)[self.common_indices[indicator]]

    def calibration(self):
        pass

    def describe_params(self):
        pass

    def describe_like_contribution(self, ind: str):
        pass

    def get_description(self) -> str:
        description = self.describe_params()
        for ind in self.targets.keys():
            description += self.describe_like_contribution(ind)
        return description


class StandardCalib(Calibration):
    def __init__(
        self,
        epi_model: RenewalModel,
        priors: Dict[str, dist.Distribution],
        targets: Dict[str, pd.Series],
        target_sds: Dict[str, float],
    ):
        """Set up calibration object with epi model and data.

        Args:
            epi_model: The renewal model
            targets: The data targets
            target_sds: Standard deviation for the prior to the dispersion parameter for each indicator
        """
        super().__init__(epi_model, priors, targets)
        self.proc_disp_sd = 0.1
        assert targets.keys() == target_sds.keys(), "One standard deviation required for each target"
        self.target_sds = target_sds

    def calibration(self, fixed_params={}):
        """Main calibration function.

        Args:
            extra_params: Any parameters to be passed directly to model
        """
        params = self.set_calib_params() | fixed_params
        result = self.epi_model.renewal_func(**params)
        for ind in self.targets.keys():
            self.add_factor(result, ind)

    def set_calib_params(self):
        """See describe_params below.

        Returns:
            Calibration parameters
        """
        params = {k: numpyro.sample(k, v) for k, v in self.priors.items()}
        proc_disp = numpyro.sample("dispersion_proc", dist.HalfNormal(self.proc_disp_sd))
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
            f"with standard deviation {self.proc_disp_sd}. "
        )

    def add_factor(
        self, 
        result, 
        ind: str,
    ):
        """Add output target to calibration algorithm.

        Args:
            result: Output from model
            ind: Name of indicator
        """
        log_result = jnp.log(self.get_model_indicator(result, ind))
        log_target = jnp.log(self.targets[ind])
        dispersion = numpyro.sample(f"dispersion_{ind}", dist.HalfNormal(self.target_sds[ind]))
        like_component = dist.Normal(log_result, dispersion).log_prob(log_target).sum()
        numpyro.factor(f"{ind}_ll", like_component)

    def describe_like_contribution(self, indicator):
        return (
            f"The log of the modelled {indicator} values for each parameter set "
            "is compared against the corresponding data "
            "from the end of the run-in phase through to the end of the analysis. "
            "The dispersion parameter for this comparison of log values is also calibrated, "
            "with the dispersion parameter prior using a half-normal distribution, "
            f"with a standard deviation of {self.target_sds[indicator]}. "
        )
