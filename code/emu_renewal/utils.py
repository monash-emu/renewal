from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import pycountry
import pycountry_convert as pc
import arviz as az

from emu_renewal.inputs import get_google_mobility, get_fb_visited_mobility, get_fb_singletile_mobility


def get_col_increases(
    input_array: np.array,
) -> np.array:
    """Find the elements of a 2-dimensional
    numpy array for which the values
    represent an increase over the column.

    Args:
        input_array: The array to analyse

    Returns:
        The elements that represent an increase
    """
    col_diffs = np.diff(input_array, axis=0)
    row1_zeros = np.zeros(input_array.shape[1])
    diff_array = np.concatenate([[row1_zeros], col_diffs])
    return diff_array == 1.0


def get_reset_array_from_increases(
    input_array: np.array,
) -> np.array:
    """Find all the elements of an array
    for which there has not been a preceding
    increase in the column.

    Args:
        input_array: The output of get_col_increases

    Returns:
        The elements of the array that don't follow the increase
    """
    reset_array = np.zeros_like(input_array)
    for c in range(input_array.shape[1]):
        col = input_array[:, c]
        increases = np.where((col[:-1] == False) & (col[1:] == True))[0]
        last_increase = increases[-1] + 1 if increases.size > 0 else 0
        remaining = col.size - last_increase
        reset_array[:, c] = np.concatenate([np.ones(last_increase), np.zeros(remaining)])
    return reset_array.astype(bool)


def get_beta_params_from_mean_var(
    mu: float,
    sd: float,
) -> Tuple[float]:
    """Get the beta distribution priors
    in the format needed by numpyro
    from the mean and variance.

    Args:
        mu: Requested distribution mean
        sd: Requested distribution standard deviation

    Returns:
        The a and b parameters to the beta distribution
    """
    var = sd**2.0
    a = mu * (mu * (1.0 - mu) / var - 1.0)
    b = (1.0 - mu) * (mu * (1.0 - mu) / var - 1.0)
    return a, b


def get_param_dim(
    param: str,
    idata: az.InferenceData,
) -> int:
    """Find how many elements a parameter has
    from the calibration results.

    Args:
        param: Name of the parameter
        idata: Calibration results

    Returns:
        Number of elements
    """
    dims = idata.posterior[param].shape[2:]
    return dims[0] if dims else 1


def get_countries_by_continent(
    countries: List[str],
) -> Dict[str, List[str]]:
    """Organise a list of countries into a
    dictionary according to the continent they are from.

    Args:
        countries: List of ISO3 codes

    Returns:
        The dictionary with keys for each continent present
    """
    result = {}
    for c in countries:
        iso2 = pycountry.countries.lookup(c).alpha_2
        cont = pc.country_alpha2_to_continent_code(iso2)
        if cont in result:
            result[cont].append(c)
        else:
            result[cont] = [c]
    sorted_result = {cont: sorted(result[cont]) for cont in sorted(result)}
    return sorted_result


def count_repeat_nans(
    data: pd.Series,
) -> int:
    """Find the maximum number of consecutive NaNs
    in a row in the input data.

    Args:
        data: The data

    Returns:
        The number of NaNs
    """
    is_nan = data.isna()
    consecutive_nans = is_nan.groupby((is_nan != is_nan.shift()).cumsum()).cumsum()
    return consecutive_nans.max()


def get_mob_avail_countries(
    countries: List[str],
    mob_source: str,
) -> List[str]:
    """Find whether mobility is available for a group of countries.

    Args:
        countries: The countries to find if mobility is present for
        mob_source: The mobility analysis type

    Returns:
        The sub-set of the submitted countries for which mobility is available
    """
    avail_countries = []
    mob_func_map = {
        "g_mob": get_google_mobility,
        "fb_singletile_mob": get_fb_singletile_mobility,
        "fb_visited_mob": get_fb_visited_mobility,
    }
    for iso3 in countries:
        try:
            mob_func_map[mob_source](iso3)
            avail_countries.append(iso3)
        except:
            continue
    return avail_countries


def split_list_into_segments(
    l: list, 
    segment_length: int,
) -> List[list]:
    """Split a list into groups of equal size until
    the last one which contains whatever is left over.

    Args:
        l: The list to split
        segment_length: The size of the segments

    Returns:
        The list segments
    """
    return [l[i: i + segment_length] for i in range(0, len(l), segment_length)]
