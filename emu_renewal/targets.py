import pandas as pd
from jax import Array, numpy as jnp
from numpyro import distributions as dist
import numpyro

class Target:
    data: pd.Series
    sd: float
    key: str
    calibration_data: Array

    def set_key(self, key):
        raise(NotImplementedError)
    
    def set_calibration_data(self, data):
        raise NotImplementedError
    
    def loglikelihood(self, modelled):
        raise NotImplementedError

class LogLogHalfNormalTarget(Target):
    def __init__(self, data: pd.Series, sd: float):
        self.data = data
        self.sd = sd
        
        self.key: str = None
        self.calibration_data: Array = None

    def set_key(self, key):
        """
        Called by StandardCalib, used to set the parameter name of this target
        """
        self.key = key

    def set_calibration_data(self, data):
        self.calibration_data = data
        self._log_data = jnp.log(data)

    def loglikelihood(self, modelled):
        log_result = jnp.log(modelled)
        log_target = self._log_data
        dispersion = numpyro.sample(f"dispersion_{self.key}", dist.HalfNormal(self.sd))
        like_component = dist.Normal(log_result, dispersion).log_prob(log_target).sum()
        return like_component
