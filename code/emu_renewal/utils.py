from typing import List, Dict, Tuple
from pathlib import Path
import os
from os import listdir as ls
import json
import pandas as pd
import numpy as np
import pycountry
import pycountry_convert as pc
import arviz as az

from emu_renewal.constants import ANALYSIS_TYPES, ANALYSIS_NAMES


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
    var = sd ** 2.0
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


def sort_countries_by_name(
    countries: List[str],
) -> List[str]:
    """Sort a list of country ISO3 codes according
    to the alphabetical order of the country name.

    Args:
        countries: The to sort

    Returns:
        The sorted list of country codes
    """
    sorted_country_names = sorted([pycountry.countries.lookup(c).name for c in countries])
    return [pycountry.countries.lookup(c).alpha_3 for c in sorted_country_names]


def get_countries_by_continent(
    countries: List[str],
) -> Dict[str, List[str]]:
    """Organise a list of countries into a
    dictionary according to the continent they are from.

    Args:
        countries: List of ISO3 identifiers

    Returns:
        The dictionary with keys for each continent present
    """
    result = {}
    for c in countries:
        cont = get_cont_of_country(c)
        if cont in result:
            result[cont].append(c)
        else:
            result[cont] = [c]
    return {cont: sort_countries_by_name(result[cont]) for cont in sorted(result)}


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


def get_cont_of_country(
    iso3: str,
) -> str:
    """Use pycountry_convert to get the continent code
    for a country without producing an error if none is available.

    Args:
        iso3: The country identifier

    Returns:
        The continent identifier

    Notes
    -----
    Given the profile and timing of its epidemic,
    Singapore was included with the Oceania countries
    throughout the following analyses.
    """
    if iso3 == "SGP":
        return "OC"
    try:
        iso2 = pycountry.countries.lookup(iso3).alpha_2
        return pc.convert_country_alpha2_to_continent_code.country_alpha2_to_continent_code(iso2)
    except KeyError:
        return "NOCONT"
    

def get_subdirs(
    path: Path,
) -> List[str]:
    """Get the subdirectories of a folder.

    Args:
        path: The path to the folder

    Returns:
        The names (only) of the subdirectories
    """
    return [d.name for d in os.scandir(path) if d.is_dir()]


def get_countries_with_mob_source(
    job_path: Path, 
    mob_source: str,
) -> List[str]:
    """Find all the countries in a path that have an analysis
    available for a particular analysis type.

    Args:
        job_path: Path for the runs
        mob_source: The mobility analysis type

    Returns:
        The list of countries
    """
    all_countries = ls(job_path)
    mob_countries = []
    for c in all_countries:
        if mob_source in [i.parts[-1] for i in (job_path / c).iterdir()]:
            mob_countries.append(c)
    return mob_countries


def get_country_short_name(
    iso3: str,
) -> str:
    """Get a shorter name for countries with
    long names to facilitate some plots.

    Args:
        iso3: The country identifier

    Returns:
        The abbreviated name
    """
    info = pycountry.countries.lookup(iso3)
    abbrevs = {
        "GBR": "UK",
        "ARE": "UAE",
        "RUS": "Russia",
        "DOM": "Domin. Rep.",
        "BIH": "Bosnia Herz",
        "AFG": "Afghan.",
    }
    if iso3 in abbrevs:
        return abbrevs[iso3]
    elif hasattr(info, "common_name"):
        return info.common_name
    else:
        return info.name


def get_country_name(
    iso3: str,
) -> str:
    """Safely get name of a country, returning
    the original ISO3 request if not availble.

    Args:
        iso3: The country identifier

    Returns:
        The name of the country
    """
    try:
        return pycountry.countries.lookup(iso3).name
    except:
        return iso3


def get_analysis_commits(
    job_path: Path, 
    iso3: str,
) -> Dict[str, str]:
    """Gather together the commit IDs for
    each analysis type of a given country's run.

    Args:
        iso3: The path to the job
        country: The country identifer

    Returns:
        Dictionary with keys for each analysis type
            and values short commit SHA
    """
    commits = {}
    for analysis in ANALYSIS_TYPES:
        a_path = job_path / iso3 / analysis
        if os.path.isdir(a_path):
            commit = json.load(open(a_path / "gitinfo.json", "r"))["sha"][:7]
        else:
            commit = "no analysis"
        commits[analysis] = commit
    return commits


def get_job_commits_df(
    job_path: Path, 
    countries: List[str],
) -> pd.DataFrame:
    """Use the preceding function to create a
    dataframe of the commits used for each analysis.

    Args:
        job_path: The path to the job
        countries: The country identifiers

    Returns:
        The dataframe with index countries and 
            columns for each analysis type
    """
    commits = pd.DataFrame(index=countries, columns=ANALYSIS_TYPES)
    for iso3 in countries:
        commits.loc[iso3, :] = get_analysis_commits(job_path, iso3)
    commits.rename(columns=ANALYSIS_NAMES, inplace=True)
    commits.rename(index=lambda c: pycountry.countries.lookup(c).name, inplace=True)
    return commits.sort_index()
