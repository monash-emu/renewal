from typing import Dict
from jax.scipy.stats import gamma as jaxgamma
from jax import numpy as jnp


class GammaDens:
    """Density class for generating gamma-distributed densities."""

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
        used in our analysis were parameterised by
        analytically calculating the "a" (shape)
        and scale parameters
        from the mean and standard deviation
        determined by our literature review.
        """
        var = sd ** 2.0
        scale = var / mean
        a = mean / scale
        return {"a": a, "scale": scale}

    def get_densities(
        self,
        window_len: float,
        mean: float,
        sd: float,
    ) -> jnp.Array:
        return jnp.diff(self.get_cum_dens(window_len, mean, sd))

    def get_cum_dens(
        self,
        window_len: int,
        mean: float, 
        sd: float,
    ) -> jnp.Array:
        return jaxgamma.cdf(jnp.arange(window_len + 1), **self.get_params(mean, sd))
