import pandas as pd
from datetime import datetime
from warnings import warn
from jax import Array, numpy as jnp
from numpyro.distributions import Distribution

Prior = Distribution | float
PriorDict = dict[str, Prior]


class ScalerProvider:
    def __init__(self):
        raise NotImplementedError

    def reconcile_times(self, start: datetime, end: datetime):
        """Do any appropriate slicing/extension required based on the supplied
        (model) start and end times. Called once at start of calibration.
        Any further calls into the ScalerProvider will use index times rather
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

    def get_parameterised_scaler(self, **kwargs) -> Array:
        """Called once per iteration; any parameterized transforms of the mobility
        should occur here; kwargs will be guaranteed to contain any values described
        in get_priors.

        Returns:
            The mobility values
        """
        raise NotImplementedError


class NoScalerProvider(ScalerProvider):
    def __init__(self):
        """Allow for analyses with no scaling for time series data.

        Notes
        -----
        For the analysis without scaling, no empiric data
        was used to scale the transmission rate over time
        (which was implemented by setting the scaling value to 
        one throughout the simulation).
        """
        self.mob_end = None

    def get_priors(self) -> dict[str, Distribution | float]:
        return {}

    def reconcile_times(self, start: datetime, end: datetime):
        self.mob_arr = jnp.ones((end - start).days + 1)

    def get_parameterised_scaler(self, **kwargs) -> Array:
        return self.mob_arr


class WeightedScalerProvider(ScalerProvider):
    def get_priors(self) -> dict[str, Distribution | float]:
        return self.priors
    
    def reconcile_times(self, start: datetime, end: datetime):
        if start < self.ts_data.index[0]:
            extend_mob_start = (self.ts_data.index[0] - start).days
            warn(f"Mobility series starts later than model, extending by {extend_mob_start} days")
            start_vals = self.ts_data.iloc[0].to_numpy()[:, None]
            extension = jnp.repeat(start_vals, extend_mob_start, 1).T
            scaling_array = jnp.concat([extension, self.ts_data.to_numpy()])
        else:
            scaling_array = jnp.array(self.ts_data.loc[start:])
        if end > self.ts_data.index[-1]:
            extend_mob_end = (end - self.ts_data.index[-1]).days
            warn(f"Mobility series ends earlier than model, extending by {extend_mob_end} days")
            end_vals = self.ts_data.iloc[-1].to_numpy()[:, None]
            extension = jnp.repeat(end_vals, extend_mob_end, 1).T
            scaling_array = jnp.concat([scaling_array, extension])

        self.scaling_arr = scaling_array


class WeightedExpScalerProvider(WeightedScalerProvider):
    def __init__(self, ts_data: pd.DataFrame, priors: PriorDict):
        """Provide a scaling array to a RenewalModel, which is the weighted
        sum of a DataFrame, which is then exponentiated.

        Args:
            ts_data: The untransformed source data
            priors: Priors for the transform parameters

        Notes
        -----
        This approach to analysis used
        a set of priors weighting each time series component,
        along with one further prior governing the overall
        effect of the weighted scaling estimate in adjusting 
        the transmission rate.
        """
        self.ts_data = ts_data
        assert set(priors.keys()) == set(["ts_weights", "scale_exp"])
        assert priors["ts_weights"].batch_shape == (len(self.ts_data.columns),)
        self.priors = priors
        self.mob_end = ts_data.index[-1]

    def get_parameterised_scaler(self, ts_weights, scale_exp, **kwargs) -> Array:
        """See methods to parent class MobilityProvider.

        Args:
            ts_weights: The weights for each mobility domain
            scale_exp: The scaling factor for the weighted mobility estimate

        Returns:
            The scaling values
        
        Notes
        -----
        The weights for each time series component were normalised to sum to one,
        with the resulting weighted scaling profile exponentiated
        to the value specified by the scaling exponential parameter.
        """
        norm_ts_weights = ts_weights / ts_weights.sum()
        return (self.scaling_arr * norm_ts_weights).sum(axis=1) ** scale_exp


class WeightedFloorMobilityProvider(WeightedScalerProvider):
    def __init__(self, mobility: pd.DataFrame, priors: PriorDict):
        self.ts_data = mobility
        assert set(priors.keys()) == set(["ts_weights", "scale_floor", "scale_exp"])
        assert priors["ts_weights"].batch_shape == (len(self.ts_data.columns),)
        self.priors = priors
        self.mob_end = mobility.index[-1]

    def get_parameterised_scaler(self, ts_weights, scale_exp, scale_floor, **kwargs) -> Array:
        norm_ts_weights = ts_weights / ts_weights.sum()
        return (scale_floor + (self.scaling_arr * norm_ts_weights).sum(axis=1) * (1.0 - scale_floor)) ** scale_exp


class SingleSeriesMobilityProvider(ScalerProvider):
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

    def get_parameterised_scaler(self, **kwargs) -> Array:
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
        assert set(priors.keys()) == set(["scale_exp"])
        self.priors = priors
        self.mob_end = mobility.index[-1]

    def get_priors(self) -> dict[str, Distribution | float]:
        return self.priors

    def get_parameterised_scaler(self, scale_exp, **kwargs) -> Array:
        """See methods to parent class MobilityProvider.

        Args:
            scale_exp: The mobility exponent parameter

        Returns:
            The mobility values
        
        Notes
        -----
        One prior value was incorporated with each of these approaches, 
        which specifies the exponent parameter for 
        the effect of the mobility data in scaling the transmission rate.
        """
        return self.mobility_arr ** scale_exp


class SingleSeriesExpFloorScalerProvider(SingleSeriesExpMobilityProvider):
    def __init__(self, mobility: pd.Series, priors: PriorDict):
        self.mobility_series = mobility
        assert set(priors.keys()) == set(["scale_exp", "scale_floor"])
        self.priors = priors
        self.mob_end = mobility.index[-1]
    def get_parameterised_scaler(self, scale_exp, scale_floor, **kwargs) -> Array:
        return (scale_floor + self.mobility_arr * (1.0 - scale_floor)) ** scale_exp
    