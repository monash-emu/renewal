from jax import lax, numpy as jnp

from summer2.functions.interpolate import InterpolatorScaleData
from summer2.functions import interpolate as sinterp


def _get_cos_curve_at_x(
    x: float, 
    x_data: InterpolatorScaleData, 
    y_data: InterpolatorScaleData,
) -> float:
    """Get interpolated function value using half-cosine function.

    Args:
        x: Independent value to calculate result at
        x_data: Requested series of independent values
        y_data: Requested series of dependent values

    Returns:
        Interpolated value

    Notes
    -----
    The cosine function was obtained by translating 
    and scaling a half cosine function 
    (i.e. a cosine function with support $[0, \pi]$),
    such that it intersected the starting point
    $(t_{{1}}, y_{{1}})$ and finishing point $(t_{{2}}, y_{{2}})$
    with a gradient of zero at both of these points.
    This choice of fitting approach ensures that 
    the variable process function, its derivative 
    and its higher order derivatives are continuous.
    """
    idx = sinterp.binary_search_sum_ge(x, x_data.points) - 1
    offset = x - x_data.points[idx]
    relx = offset / x_data.ranges[idx]
    rely = 0.5 + 0.5 * -jnp.cos(relx * jnp.pi)
    return y_data.points[idx] + (rely * y_data.ranges[idx])


class MultiCurve:
    """Abstract class for fitting a curve to a series of data.
    """
    def get_multicurve(self):
        pass
    def get_description(self):
        pass


class CosineMultiCurve(MultiCurve):
    """Fit a cosine-based curve to a series of data.
    See get_description below for details.

    Args:
        MultiCurve: Abstract parent class
    """
    def get_multicurve(
        self,
        t: float, 
        x_data: InterpolatorScaleData, 
        y_data: InterpolatorScaleData,
    ) -> callable:
        """Construct a half-cosine-based multi-curve.

        Args:
            t: Model time
            x_data: Values of independent variable
            y_data: Values of dependent variable

        Returns:
            Curve fitting function
        """
        # Branch on whether t is in bounds
        bounds_state = sum(t > x_data.bounds)
        branches = [
            lambda _, __, ___: y_data.bounds[0],
            _get_cos_curve_at_x,
            lambda _, __, ___: y_data.bounds[1],
        ]
        return lax.switch(bounds_state, branches, t, x_data, y_data)
