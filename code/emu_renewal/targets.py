from typing import Callable
import pandas as pd
from jax import Array, numpy as jnp
from numpyro import distributions as dist
from numpyro.distributions.distribution import DistributionMeta

from emu_renewal.constants import PROP_EXTREME


Transform = Callable | None
ParamValues = dict[str, Array | float]


def logit(x):
    """Logit transform, with clipping away from 0 and 1."""
    x = jnp.clip(x, PROP_EXTREME, 1.0 - PROP_EXTREME)
    return jnp.log(x) - jnp.log1p(-x)


class Target:
    data: pd.Series
    calibration_data: Array

    def set_calibration_data(self, data: Array):
        """Called by StandardCalib

        Args:
            data: Filtered and pre-indexed data that shares coordinates with
                  the modelled data input to loglikelihood
        """
        self.calibration_data = data

    def loglikelihood(self, modelled: Array, parameters: ParamValues) -> float:
        raise NotImplementedError


class UnivariateDispersionTarget(Target):
    def __init__(
        self,
        data: pd.Series,
        dist: DistributionMeta,
        dispersion: str,
        transform: Transform = None,
        weight: float = None,
    ):
        """Create a Target with any distribution, which is parameterised by
        the modelled data and parameters to the dispersion distribution.

        Args:
            data: The target data series
            dist: The likelihood distribution
            dispersion: Key of sampled parameter to use as dispersion
            transform: Optional function to apply to both data and input
            weight: Total series weight; defaults to the number of observations
        """
        self.data = data
        self.dist = dist
        self.dispersion = dispersion
        self._transform = transform
        self.weight = weight

    def set_calibration_data(self, data: Array):
        self.calibration_data = self.transform(data)
        if self.weight is None:
            self.weight = float(len(self.calibration_data))

    def transform(self, x):
        if self._transform is None:
            return x
        return self._transform(x)

    def loglikelihood(self, modelled: Array, parameters: ParamValues) -> float:
        result = self.transform(modelled)
        dispersion = parameters[self.dispersion]
        return self.dist(result, dispersion).log_prob(self.calibration_data).mean() * self.weight


class SharedDispTarget(UnivariateDispersionTarget):
    """Normal likelihood target over the log transformed indicator, with a
    shared dispersion parameter.
    """

    def __init__(self, data: pd.Series, weight: float):
        super().__init__(data, dist.Normal, "shared_dispersion", jnp.log, weight)


class SharedPropTarget(UnivariateDispersionTarget):
    """Normal likelihood target over the logit-transformed indicator,
    with a shared dispersion parameter.
    """

    def __init__(self, data: pd.Series, weight: float):
        super().__init__(data, dist.Normal, "prop_disp", logit, weight)
