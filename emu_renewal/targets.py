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

        Args:
            key: Parameter name referenced by the rest of the calibration
                 infrastructure
        """
        self.key = key

    def set_calibration_data(self, data: Array):
        """Called by StandardCalib

        Args:
            data: Filtered and pre-indexed data that shares coordinates with
                  the modelled data input to loglikelihood
        """
        self.calibration_data = data

    def loglikelihood(self, modelled: Array, parameters: dict[str, float]) -> float:
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
        dispersion: str,
        transform: Transform = None,
    ):
        """Create a Target with any distribution, which is parameterised by
        the modelled data and parameters to the dispersion distribution.

        Args:
            data: The target data series
            dist: The likelihood distribution
            dispersion: Key of sampled parameter to use as dispersion
            transform: Optional function to apply to both data and input
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
        dispersion = parameters[self.dispersion]

        return self.dist(result, dispersion).log_prob(self.calibration_data).sum()


class StandardDispTarget(UnivariateDispersionTarget):
    """
    Normal likelihood target over the log transformed indicator, with a
    default of shared dispersion
    """

    def __init__(self, data, dispersion: str = "shared_dispersion"):
        super().__init__(data, dist.Normal, dispersion, jnp.log)
