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
        """Get the parameters for the distribution type"""
        pass

    def get_densities():
        """The densities for each integer increment in the distribution."""
        pass


class GammaDens(Dens):
    """Density class for generating gamma-distributed denities."""

    def get_params(
        self,
        mean: float,
        sd: float,
    ) -> Dict[str, float]:
        """Get parameters to a gamma distribution
        based on the summary statistics.

        Args:
            mean: Requested mean
            sd: Requested standard deviation

        Returns:
            The parameters

        Notes
        -----
        The parameters to each gamma distribution
        used in our anlaysis were parameterised by
        analytically calculating the "a" (shape) 
        and scale parameters
        from the mean and standard deviation
        determined by our literature review.
        """
        var = sd ** 2.0
        scale = var / mean
        a = mean / scale
        return {"a": a, "scale": scale}

    def get_densities(self, window_len, mean, sd):
        return jnp.diff(self.get_cum_dens(window_len, mean, sd))

    def get_cum_dens(self, window_len, mean, sd):
        return jaxgamma.cdf(jnp.arange(window_len + 1), **self.get_params(mean, sd))
