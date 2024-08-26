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
        data: pd.Series,
    ):
        """Set up calibration object with epi model and data.

        Args:
            epi_model: The renewal model
            data: The data targets
        """
        self.epi_model = epi_model
        self.n_process_periods = len(self.epi_model.x_proc_data.points)

        self.custom_init = custom_init(n_proc=self.n_process_periods)

        analysis_dates_idx = self.epi_model.epoch.index_to_dti(self.epi_model.model_times)
        common_dates_idx = data.index.intersection(analysis_dates_idx)
        self.data = jnp.array(data.loc[common_dates_idx])
        common_abs_idx = np.array(self.epi_model.epoch.dti_to_index(common_dates_idx).astype(int))
        self.common_model_idx = common_abs_idx - self.epi_model.model_times[0]

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
        priors: dict[str, dist.Distribution],
        data: pd.Series,
        fixed_params: Dict[str, float]={},
        indicator: str="cases",
    ):
        """Set up calibration object with epi model and data.

        Args:
            epi_model: The renewal model
            data: The data targets
        """
        super().__init__(epi_model, priors, data)
        self.data_disp_sd = 0.1
        self.proc_disp_sd = 0.1
        self.fixed_params = fixed_params
        self.indicator = indicator

    def get_model_indicator(
        self, 
        params: Dict[str, Union[float, jnp.array]],
    ):
        """Get the modelled values for a particular epidemiological indicator 
        for a given set of epi parameters.

        Args:
            params: All renewal model parameters
        
        Returns:
            Modelled time series of the indicator over analysis period
        """
        result = self.epi_model.renewal_func(**params)
        return getattr(result, self.indicator)[self.common_model_idx]

    def calibration(self):
        """See get_description below.
        """
        params = self.set_calib_params()
        self.add_notif_factor(params)

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

    def add_notif_factor(self, params):
        notif_log_result = jnp.log(self.get_model_notifications(params))
        notif_log_target = jnp.log(self.data)
        notif_disp = numpyro.sample("dispersion", dist.HalfNormal(self.data_disp_sd))
        notif_like = dist.Normal(notif_log_result, notif_disp).log_prob(notif_log_target).sum()
        numpyro.factor("notifications_ll", notif_like)

    def describe_notif_contribution(self):
        return (
            "The log of the modelled notification rate for each parameter set "
            "is compared against the data from the end of the run-in phase "
            "through to the end of the analysis. "
            "Modelled notifications are calculated as the product of modelled incidence and the "
            "(constant through time) case detection proportion. "
            "The dispersion parameter for this comparison of log values is "
            "also calibrated using a half-normal distribution, "
            f"with standard deviation {self.data_disp_sd}. "
        )

    def get_description(self) -> str:
        description = self.describe_params()
        description += self.describe_notif_contribution()
        return description
