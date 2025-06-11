from typing import List, Dict
import pandas as pd
from datetime import datetime
import itertools
import numpy as np
import pycountry
import pycountry_convert as pc
import arviz as az


def format_date_for_str(
    date: datetime,
    include_year: bool = True,
) -> str:
    """Get a markdown-ready string that could be included in
    paragraph text from a datetime object.

    Args:
        date: The datetime object

    Returns:
        The formatted string
    """
    ord_excepts = {1: "st", 2: "nd", 3: "rd"}
    ordinal = ord_excepts.get(date.day % 10, "th")
    if include_year:
        return f"{date.day}<sup>{ordinal}</sup> {date: %B} {date: %Y}"
    else:
        return f"{date.day}<sup>{ordinal}</sup> {date: %B}"


def round_sigfig(value: float, sig_figs: int) -> float:
    """
    Round a number to a certain number of significant figures,
    rather than decimal places.

    Args:
        value: Number to round
        sig_figs: Number of significant figures to round to
    """
    return (
        round(value, -int(np.floor(np.log10(abs(value)))) + sig_figs - 1) if value != 0.0 else 0.0
    )


def get_proc_period_from_index(
    idx: int,
    model,
) -> str:
    """Get markdown-formatted string for date of
    variable process period from its index number.

    Args:
        idx: The sequence of the process period
        model: The renewal model

    Returns:
        The formatted string
    """
    start = int(model.x_proc_vals[idx])
    end = start + model.proc_update_freq - 1
    start_date = format_date_for_str(model.epoch.number_to_datetime(start), include_year=False)
    end_date = format_date_for_str(model.epoch.number_to_datetime(end), include_year=False)
    return f"Variable process update, {start_date} to {end_date}"


map_dict = {
    "cdr": "Case detection proportion",
    "gen_mean": "Generation time, mean",
    "gen_sd": "Generation time, standard deviation",
    "dispersion_proc": "Variable process update dispersion",
    "dispersion_cases": "Cases comparison dispersion",
    "rt_init": "Rt starting value",
    "report_mean": "Reporting time, mean",
    "report_sd": "Reporting time, standard deviation",
}


def get_adjust_idata_index(
    model,
) -> callable:
    """Get function to adjust the dataframe index
    containing the model parameters.

    Args:
        model: The model

    Returns:
        The adjuster function
    """

    def adjust_idata_index(i):
        if i.startswith("proc["):
            i_proc = int(i[i.find("[") + 1 : i.find("]")])
            return get_proc_period_from_index(i_proc, model)
        elif i in map_dict:
            return map_dict[i]
        else:
            raise ValueError("Parameter not found")

    return adjust_idata_index


col_names_map = {
    "sd": "standard deviation",
    "hdi_3%": "high-density interval, 3%",
    "hdi_97%": "high-density interval, 97%",
    "ess_bulk": "effective sample size, bulk",
    "ess_tail": "effective sample size, tail",
    "r_hat": "_&#x0052;&#x0302;_",
}


def adjust_summary_cols(summary):
    summary = summary.rename(columns=col_names_map)
    summary = summary.drop(["mcse_mean", "mcse_sd"], axis=1)
    summary.columns = summary.columns.str.capitalize()
    return summary


def get_combs(n_cats: int) -> np.ndarray:
    """For a given set of categories, work out all the possible
    combinations of one of the categories being True or False.

    Example:
        Argument 2 would yield:
        [[False, False], [False, True], [True, False], [True, True]]

    Args:
        n_cats: Number of categories

    Returns:
        The combinations, with each list element having n_cats entries.
    """
    return np.array([list(i) for i in itertools.product([False, True], repeat=n_cats)]).T


def get_row_proportions(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Normalise the rows of a dataframe over its columns.

    Args:
        df: The input dataframe containing numeric values

    Returns:
        The result
    """
    return df.divide(df.sum(axis=1), axis=0).fillna(0.0)


def melt_df_except_first_level(df: pd.DataFrame) -> pd.DataFrame:
    """Melt (convert to long format)
    a multiindex dataframe retaining the first level.

    Args:
        df: The dataframe for conversion

    Returns:
        The melted dataframe
    """
    cols = set(df.columns.get_level_values(0))
    return pd.concat([df[c].melt()["value"] for c in cols], axis=1, keys=cols)


def group_countries_by_continent(
    countries: List[str],
) -> Dict[str, str]:
    """Group requested countries according to
    the continent they are located in.

    Args:
        countries: The countries to group

    Returns:
        The grouping
    """
    continents = ["AF", "EU", "AS", "SA", "NA"]
    cont_map = {cont: [] for cont in continents}
    for c in countries:
        iso2 = pycountry.countries.lookup(c).alpha_2
        continent = pc.country_alpha2_to_continent_code(iso2)
        cont_map[continent].append(c)
    return cont_map


def get_col_increases(input_array):
    col_diffs = np.diff(input_array, axis=0)
    row1_zeros = np.zeros(input_array.shape[1])
    diff_array = np.concatenate([[row1_zeros], col_diffs])
    return diff_array == 1.0


def get_reset_array_from_increases(input_array):
    reset_array = np.zeros_like(input_array)
    for c in range(input_array.shape[1]):
        col = input_array[:, c]
        increases = np.where((col[:-1] == False) & (col[1:] == True))[0]
        last_increase = increases[-1] + 1 if increases.size > 0 else 0
        remaining = col.size - last_increase
        reset_array[:, c] = np.concatenate([np.ones(last_increase), np.zeros(remaining)])
    return reset_array.astype(bool)


def get_beta_params_from_mean_var(mu, var):
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


def get_countries_by_continent(countries):
    result = {}
    for c in countries:
        iso2 = pycountry.countries.lookup(c).alpha_2
        cont = pc.country_alpha2_to_continent_code(iso2)
        if cont in result:
            result[cont].append(c)
        else:
            result[cont] = [c]
    return result


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
