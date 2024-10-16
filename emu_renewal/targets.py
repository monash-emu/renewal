import pandas as pd
from jax import Array, numpy as jnp
from numpyro import distributions as dist
from numpyro.distributions.distribution import DistributionMeta
import numpyro


class Target:
    data: pd.Series
    key: str
    calibration_data: Array

    def set_key(self, key):
        """
        Called by StandardCalib, used to set the parameter name of this target
        """
        self.key = key

    def set_calibration_data(self, data):
        raise NotImplementedError

    def loglikelihood(self, modelled):
        raise NotImplementedError


class UnivariateDispersionTarget(Target):
    def __init__(
        self,
        data: pd.Series,
        dist: DistributionMeta,
        dispersion_dist: dist.Distribution,
        log: bool,
    ):
        """Create a Target for any distribution which is parameterized by the
        modelled data and a single additional parameter
        e.g Normal(modelled, sd)

        Args:
            data: The target data series
            dist: The likelihood distribution
            dispersion_dist: A dispersion distribution
            log: Whether to apply log transform to both modelled and observed data
        """
        self.data = data
        self.dist = dist
        self.dispersion_dist = dispersion_dist

        self.log = log

        self.key: str = None
        self.calibration_data: Array = None

    def set_calibration_data(self, data):
        self.calibration_data = data
        if self.log:
            self._log_data = jnp.log(data)

    def loglikelihood(self, modelled):
        if self.log:
            result = jnp.log(modelled)
            target = self._log_data
        else:
            result = modelled
            target = self.calibration_data

        dispersion = numpyro.sample(f"dispersion_{self.key}", self.dispersion_dist)
        like_component = self.dist(result, dispersion).log_prob(target).sum()
        return like_component


class StandardTarget(UnivariateDispersionTarget):
    def __init__(self, data, dispersion_sd=0.1):
        super().__init__(data, dist.Normal, dist.HalfNormal(dispersion_sd), log=True)
