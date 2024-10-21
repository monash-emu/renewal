import pandas as pd
from jax import Array, numpy as jnp
from numpyro import distributions as dist
from numpyro.distributions.distribution import DistributionMeta
import numpyro


class Target:
    data: pd.Series
    key: str
    calibration_data: Array
    transform: callable

    def set_key(self, key):
        """
        Called by StandardCalib, used to set the parameter name of this target
        """
        self.key = key

    def set_calibration_data(self, data):
        self.calibration_data = self.transform(data)

    def loglikelihood(self, modelled):
        data = self.transform(self.calibration_data)
        result = self.transform(modelled)
        dispersion = numpyro.sample(f"dispersion_{self.key}", self.dispersion_dist)
        return self.dist(result, dispersion).log_prob(data).sum()


class UnivariateDispersionTarget(Target):
    def __init__(
        self,
        data: pd.Series,
        dist: DistributionMeta,
        dispersion_dist: dist.Distribution,
        transform: callable,
    ):
        """Create a Target with any distribution, which is parameterised by 
        the modelled data and parameters to the dispersion distribution.

        Args:
            data: The target data series
            dist: The likelihood distribution
            dispersion_dist: A dispersion distribution
        """
        self.data = data
        self.dist = dist
        self.dispersion_dist = dispersion_dist
        self.transform = transform
        self.key: str = None
        self.calibration_data: Array = None


class FlatTarget(UnivariateDispersionTarget):
    def __init__(self, data, dispersion_sd: float):
        super().__init__(data, dist.Normal, dist.HalfNormal(dispersion_sd), lambda x: x)


class StandardTarget(UnivariateDispersionTarget):
    def __init__(self, data, dispersion_sd: float):
        super().__init__(data, dist.Normal, dist.HalfNormal(dispersion_sd), jnp.log)


class UniformDispTarget(UnivariateDispersionTarget):
    def __init__(self, data, dispersion_range: list[float]):
        super().__init__(data, dist.Normal, dist.Uniform(dispersion_range), jnp.log)
