from typing import Dict
from jax.scipy.stats import gamma as jaxgamma
from jax import numpy as jnp


class Dens:
    """Get a probability distribution for use in
    calculation generation times for the renewal model.
    """

    def __init__(self):
        pass

    def get_params():
        """Get the parameters for the distribution type
        """
        pass

    def get_densities():
        """The densities for each integer increment in the distribution.
        """
        pass


class GammaDens(Dens):
    """Density class for generating gamma-distributed denities.
    """

    def get_params(self, mean: float, sd: float,) -> Dict[str, float]:
        """See get_desc below.

        Args:
            mean: Requested mean
            sd: Requested standard deviation

        Returns:
            The parameters
        """
        var = sd ** 2.0
        scale = var / mean
        a = mean / scale
        return {"a": a, "scale": scale}

    def get_densities(self, window_len, mean, sd):
        return jnp.diff(jaxgamma.cdf(jnp.arange(window_len + 1), **self.get_params(mean, sd)))

    def get_desc(self):
        """Get the description of this code.

        Returns:
            The description in markdown format
        """
        return (
            "Distribution times for each day are calculated by "
            "first finding the parameters needed to construct "
            "a gamma distribution with mean and standard deviation "
            "equal to those specified by the submitted parameter values. "
            "The integrals of the probability density of this distribution "
            "between consecutive integer values are then calculated for "
            "later combination with the incidence time series. "
        )
    