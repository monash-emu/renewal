from typing import Dict, Union
from functools import partial
from jax import numpy as jnp
import numpy as np
import pandas as pd
import numpyro
from numpyro import infer, distributions as dist

pd.options.plotting.backend = "plotly"

from emu_renewal.renew import RenewalModel
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
        self.n_process_periods = len(self.epi_model.x_proc_data.points)

        self.custom_init = custom_init(n_proc=self.n_process_periods)

        analysis_indices = self.epi_model.epoch.index_to_dti(self.epi_model.model_times)
        self.data = {}
        self.common_indices = {}
        for ind in data.keys():
            ind_data = data[ind]
            common_dates_idx = ind_data.index.intersection(analysis_indices)
            self.data[ind] = jnp.array(ind_data.loc[common_dates_idx])
            common_abs_indices = np.array(self.epi_model.epoch.dti_to_index(common_dates_idx).astype(int))
            self.common_indices[ind] = common_abs_indices - self.epi_model.model_times[0]

        self.priors = priors
        _ = [p.mean for p in self.priors.values()]  # Compile transformed dists first to avoid memory leaks

    def calibration(self):
        pass

    def get_description(self):
        pass


class StandardCalib(Calibration):
    def __init__(
        self,
        epi_model: RenewalModel,
        priors: Dict[str, dist.Distribution],
        data: Dict[str, pd.Series],
        data_sds: Dict[str, float],
        fixed_params: Dict[str, float]={},
    ):
        """Set up calibration object with epi model and data.

        Args:
            epi_model: The renewal model
            data: The data targets
            data_sds: Standard deviation for the prior to the dispersion parameter for each indicator     
            fixed_params: Any additional fixed parameters to be delivered to the renewal model
        """
        super().__init__(epi_model, priors, data)
        self.proc_disp_sd = 0.1
        self.data_sds = data_sds
        self.fixed_params = fixed_params

    def get_model_indicator(
        self, 
        params: Dict[str, Union[float, jnp.array]],
        indicator: str,
    ):
        """Get the modelled values for a particular epidemiological indicator 
        for a given set of epi parameters.

        Args:
            params: All renewal model parameters
        
        Returns:
            Modelled time series of the indicator over analysis period
        """
        result = self.epi_model.renewal_func(**params)
        return getattr(result, indicator)[self.common_indices[indicator]]

    def calibration(self, extra_params={}):
        """See get_description below.
        """
        params = self.set_calib_params()
        for ind in self.data.keys():
            self.add_factor(params | extra_params, ind)

    def set_calib_params(self):
        params = {k: numpyro.sample(k, v) for k, v in self.priors.items()}
        proc_dispersion = numpyro.sample("proc_dispersion", dist.HalfNormal(self.proc_disp_sd))
        proc_dist = dist.Normal(jnp.repeat(0.0, self.n_process_periods), proc_dispersion)
        params["proc"] = numpyro.sample("proc", proc_dist)
        params.update(self.fixed_params)
        return params
    
    def describe_params(self):
        return (
            f"The calibration process calibrates parameters for {self.n_process_periods} "
            "values for periods of the variable process to the data. "
            "The relative values pertaining to each period of the variable process "
            "are estimated from normal prior distributions centred at no "
            "change from the value of the previous stage of the process. "
            "The dispersion of the variable process is calibrated, "
            "using a half-normal distribution "
            f"with standard deviation {self.proc_disp_sd}. "
        )

    def add_factor(self, params, indicator):
        log_result = jnp.log(self.get_model_indicator(params, indicator))
        log_target = jnp.log(self.data[indicator])
        dispersion = numpyro.sample(f"dispersion_{indicator}", dist.HalfNormal(self.data_sds[indicator]))
        likelihood_contribution = dist.Normal(log_result, dispersion).log_prob(log_target).sum()
        numpyro.factor(f"{indicator}_ll", likelihood_contribution)

    def describe_like_contribution(self, indicator):
        return (
            f"The log of the modelled {indicator} values for each parameter set "
            "is compared against the corresponding data " 
            "from the end of the run-in phase through to the end of the analysis. "
            "The dispersion parameter for this comparison of log values is also calibrated, "
            "with the dispersion parameter prior using a half-normal distribution, "
            f"with a standard deviation of {self.data_sds[indicator]}. "
        )

    def get_description(self) -> str:
        description = self.describe_params()
        for ind in self.data.keys():
            description += self.describe_like_contribution(ind)
        return description
