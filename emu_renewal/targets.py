from typing import Callable, Union
import pandas as pd
from jax import Array, numpy as jnp
from numpyro import distributions as dist
from numpyro.distributions.distribution import DistributionMeta
import numpyro


Transform = Union[Callable | None]
DispersionSpec = Union[dist.Distribution, float, str]


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
        self.calibration_data = data

    def loglikelihood(self, modelled, parameters):
        raise NotImplementedError


class TransformTarget(Target):
    _transform: Transform

    def __init__(self, data: pd.Series, transform: Transform = None):
        self.data = data
        self._transform = transform

    def set_calibration_data(self, data):
        self.calibration_data = self.transform(data)

    def transform(self, x):
        if self._transform is None:
            return x
        else:
            return self._transform(x)


class UnivariateDispersionTarget(TransformTarget):
    def __init__(
        self,
        data: pd.Series,
        dist: DistributionMeta,
        dispersion: DispersionSpec,
        transform: Transform = None,
    ):
        """Create a Target with any distribution, which is parameterised by
        the modelled data and parameters to the dispersion distribution.

        Args:
            data: The target data series
            dist: The likelihood distribution
            dispersion_dist: A dispersion distribution
        """

        super().__init__(data, transform)

        self.data = data
        self.dist = dist
        self.dispersion = dispersion
        self.transform = transform
        self.key: str = None
        self.calibration_data: Array = None

    def loglikelihood(self, modelled, parameters):
        result = self.transform(modelled)

        if isinstance(self.dispersion, dist.Distribution):
            dispersion = numpyro.sample(f"dispersion_{self.key}", self.dispersion)
        elif isinstance(self.dispersion, str):
            dispersion = parameters[self.dispersion]
        else:
            dispersion = self.dispersion

        return self.dist(result, dispersion).log_prob(self.calibration_data).mean()


class HalfNormalDispTarget(UnivariateDispersionTarget):
    def __init__(self, data: pd.Series, dispersion_sd: float):
        super().__init__(data, dist.Normal, dist.HalfNormal(dispersion_sd), jnp.log)


class UniformDispTarget(UnivariateDispersionTarget):
    def __init__(self, data: pd.Series, disp_low: float, disp_high: float):
        super().__init__(data, dist.Normal, dist.Uniform(disp_low, disp_high), jnp.log)


class FixedDispTarget(UnivariateDispersionTarget):
    def __init__(self, data, dispersion: float):
        super().__init__(data, dist.Normal, dispersion, jnp.log)
