import pandas as pd
from datetime import datetime
from warnings import warn
from jax import Array, numpy as jnp
from numpyro.distributions import Distribution

Prior = Distribution | float
PriorDict = dict[str, Prior]


class MobilityProvider:
    def __init__(self):
        raise NotImplementedError

    def reconcile_times(self, start: datetime, end: datetime):
        """Do any appropriate slicing/extension required based on the supplied
        (model) start and end times.  Called once at start of calibration.
        Any further calls into the MobilityProvider will use index times rather
        than datetimes, where start=0

        Args:
            start (datetime): First datetime of model run (will be referenced as t=0)
            end (datetime): Final datetime of model run (t=-1)
        """
        raise NotImplementedError

    def get_priors(self) -> PriorDict:
        """Get a dict of priors for any parameters required for mobility transforms
        These values will be sampled during calibration, and returned to the
        MobilityProvider in the get_parameterised_mobility call

        Returns:
            PriorDict: A dict of [str, Prior] pairs specific to this MobilityProvider
        """
        raise NotImplementedError

    def get_parameterised_mobility(self, **kwargs) -> Array:
        """Called once per iteration; any parameterized transforms of the mobility
        should occur here; kwargs will be guaranteed to contain any values described
        in get_priors

        Returns:
            Array: _description_
        """
        raise NotImplementedError


class WeightedExpMobilityProvider(MobilityProvider):
    def __init__(self, mobility: pd.DataFrame, priors: PriorDict):
        """Provide a mobility array to a RenewalModel, which is the weighted
        sum of a DataFrame, which is then exponentiated.

        Args:
            mobility: The untransformed source data
            priors: Priors for the transform parameters
        """
        self.mobility_df = mobility
        assert set(priors.keys()) == set(["mob_weights", "mob_exp"])
        assert priors["mob_weights"].batch_shape == (len(self.mobility_df.columns),)
        self.priors = priors

    def get_priors(self) -> dict[str, Distribution | float]:
        return self.priors

    def reconcile_times(self, start: datetime, end: datetime):
        if start < self.mobility_df.index[0]:
            extend_mob_start = (self.mobility_df.index[0] - start).days
            warn(f"Mobility series starts later than model, extending by {extend_mob_start} days")
            extension = jnp.repeat(
                self.mobility_df.iloc[0].to_numpy()[:, None], extend_mob_start, 1
            ).T
            mob_array = jnp.concat([extension, self.mobility_df.to_numpy()])
        else:
            mob_array = jnp.array(self.mobility_df.loc[start:])
        if end > self.mobility_df.index[-1]:
            extend_mob_end = (end - self.mobility_df.index[-1]).days
            warn(f"Mobility series ends earlier than model, extending by {extend_mob_end} days")
            extension = jnp.repeat(
                self.mobility_df.iloc[-1].to_numpy()[:, None], extend_mob_end, 1
            ).T
            mob_array = jnp.concat([mob_array, extension])

        self.mobility_arr = mob_array

    def get_parameterised_mobility(self, mob_weights, mob_exp, **kwargs) -> Array:
        norm_mob_weights = mob_weights / mob_weights.sum()
        mobility = (self.mobility_arr * norm_mob_weights).sum(axis=1) ** mob_exp
        return mobility


class NoMobilityProvider(MobilityProvider):
    def __init__(self):
        pass

    def get_priors(self) -> dict[str, Distribution | float]:
        return {}

    def reconcile_times(self, start: datetime, end: datetime):
        self.mob_arr = jnp.ones((end - start).days + 1)

    def get_parameterised_mobility(self, **kwargs) -> Array:
        return self.mob_arr
