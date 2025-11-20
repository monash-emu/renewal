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
        (model) start and end times. Called once at start of calibration.
        Any further calls into the MobilityProvider will use index times rather
        than datetimes, where start is 0.

        Args:
            start: First datetime of model run (will be referenced as t=0)
            end: Final datetime of model run (referenced as t=-1)
        """
        raise NotImplementedError

    def get_priors(self) -> PriorDict:
        """Get the priors for any parameters required for mobility transforms
        These values will be sampled during calibration, and returned to the
        MobilityProvider in the get_parameterised_mobility call.

        Returns:
            PriorDict: A dict of [str, Prior] pairs specific to this MobilityProvider
        """
        raise NotImplementedError

    def get_parameterised_mobility(self, **kwargs) -> Array:
        """Called once per iteration; any parameterized transforms of the mobility
        should occur here; kwargs will be guaranteed to contain any values described
        in get_priors.

        Returns:
            The mobility values
        """
        raise NotImplementedError


class NoMobilityProvider(MobilityProvider):
    def __init__(self):
        """Allow for analyses with no scaling for mobility.

        Notes
        -----
        For the analysis without mobility, no empiric data
        was used to scale the transmission rate over time
        (which was implemented by setting the mobility
        scaling value to one throughout the simulation).
        """
        self.mob_end = None

    def get_priors(self) -> dict[str, Distribution | float]:
        return {}

    def reconcile_times(self, start: datetime, end: datetime):
        self.mob_arr = jnp.ones((end - start).days + 1)

    def get_parameterised_mobility(self, **kwargs) -> Array:
        return self.mob_arr


class WeightedMobilityProvider(MobilityProvider):
    def get_priors(self) -> dict[str, Distribution | float]:
        return self.priors
    
    def reconcile_times(self, start: datetime, end: datetime):
        if start < self.mobility_df.index[0]:
            extend_mob_start = (self.mobility_df.index[0] - start).days
            warn(f"Mobility series starts later than model, extending by {extend_mob_start} days")
            start_vals = self.mobility_df.iloc[0].to_numpy()[:, None]
            extension = jnp.repeat(start_vals, extend_mob_start, 1).T
            mob_array = jnp.concat([extension, self.mobility_df.to_numpy()])
        else:
            mob_array = jnp.array(self.mobility_df.loc[start:])
        if end > self.mobility_df.index[-1]:
            extend_mob_end = (end - self.mobility_df.index[-1]).days
            warn(f"Mobility series ends earlier than model, extending by {extend_mob_end} days")
            end_vals = self.mobility_df.iloc[-1].to_numpy()[:, None]
            extension = jnp.repeat(end_vals, extend_mob_end, 1).T
            mob_array = jnp.concat([mob_array, extension])

        self.mobility_arr = mob_array


class WeightedExpMobilityProvider(WeightedMobilityProvider):
    def __init__(self, mobility: pd.DataFrame, priors: PriorDict):
        """Provide a mobility array to a RenewalModel, which is the weighted
        sum of a DataFrame, which is then exponentiated.

        Args:
            mobility: The untransformed source data
            priors: Priors for the transform parameters

        Notes
        -----
        This mobility analysis type used
        a set of priors weighting each mobility domain,
        along with one further prior governing the overall
        effect of the weighted mobility estimate in scaling 
        the transmission rate.
        """
        self.mobility_df = mobility
        assert set(priors.keys()) == set(["mob_weights", "mob_exp"])
        assert priors["mob_weights"].batch_shape == (len(self.mobility_df.columns),)
        self.priors = priors
        self.mob_end = mobility.index[-1]

    def get_parameterised_mobility(self, mob_weights, mob_exp, **kwargs) -> Array:
        """See methods to parent class MobilityProvider.

        Args:
            mob_weights: The weights for each mobility domain
            mob_exp: The scaling factor for the weighted mobility estimate

        Returns:
            The mobility values
        
        Notes
        -----
        The weights for each mobility domain were normalised to sum to one,
        with the resulting weighted mobility profile exponentiated
        to the value specified by the mobility exponential parameter.
        """
        norm_mob_weights = mob_weights / mob_weights.sum()
        return (self.mobility_arr * norm_mob_weights).sum(axis=1) ** mob_exp


class WeightedFloorMobilityProvider(WeightedMobilityProvider):
    def __init__(self, mobility: pd.DataFrame, priors: PriorDict):
        self.mobility_df = mobility
        assert set(priors.keys()) == set(["mob_weights", "scale_floor"])
        assert priors["mob_weights"].batch_shape == (len(self.mobility_df.columns),)
        self.priors = priors
        self.mob_end = mobility.index[-1]

    def get_parameterised_mobility(self, mob_weights, scale_floor, **kwargs) -> Array:
        norm_mob_weights = mob_weights / mob_weights.sum()
        return scale_floor + (self.mobility_arr * norm_mob_weights).sum(axis=1) * (1.0 - scale_floor)



class SingleSeriesMobilityProvider(MobilityProvider):
    def __init__(self, mobility: pd.Series):
        """Provide a mobility array to a RenewalModel.

        Args:
            mobility: The untransformed source data
            priors: Priors for the transform parameters
        """
        self.mobility_series = mobility
        self.mob_end = mobility.index[-1]

    def get_priors(self) -> dict[str, Distribution | float]:
        return {}

    def reconcile_times(self, start: datetime, end: datetime):
        if start < self.mobility_series.index[0]:
            extend_mob_start = (self.mobility_series.index[0] - start).days
            warn(f"Mobility series starts later than model, extending by {extend_mob_start} days")
            start_val = self.mobility_series.iloc[0]
            extension = jnp.repeat(start_val, extend_mob_start)
            mob_array = jnp.concat([extension, jnp.array(self.mobility_series)])
        else:
            mob_array = jnp.array(self.mobility_series.loc[start:])
        if end > self.mobility_series.index[-1]:
            extend_mob_end = (end - self.mobility_series.index[-1]).days
            warn(f"Mobility series ends earlier than model, extending by {extend_mob_end} days")
            end_val = self.mobility_series.iloc[-1]
            extension = jnp.repeat(end_val, extend_mob_end)
            mob_array = jnp.concat([mob_array, extension])

        self.mobility_arr = mob_array

    def get_parameterised_mobility(self, **kwargs) -> Array:
        return self.mobility_arr


class SingleSeriesExpMobilityProvider(SingleSeriesMobilityProvider):
    def __init__(self, mobility: pd.Series, priors: PriorDict):
        """Provide a mobility array to a RenewalModel, from a single series
        that is exponentiated by a sampled value

        Args:
            mobility: The untransformed source data
            priors: Priors for the transform parameters
        
        Notes
        -----
        This mobility approach was used for each of the two 
        analyses that incorporated Facebook data,
        using both the tiles visited and the within tile 
        estimates.
        """
        self.mobility_series = mobility
        assert set(priors.keys()) == set(["mob_exp"])
        self.priors = priors
        self.mob_end = mobility.index[-1]

    def get_priors(self) -> dict[str, Distribution | float]:
        return self.priors

    def get_parameterised_mobility(self, mob_exp, **kwargs) -> Array:
        """See methods to parent class MobilityProvider.

        Args:
            mob_exp: The mobility exponent parameter

        Returns:
            The mobility values
        
        Notes
        -----
        One prior value was incorporated with each of these approaches, 
        which specifies the exponent parameter for 
        the effect of the mobility data in scaling the transmission rate.
        """
        return self.mobility_arr ** mob_exp
