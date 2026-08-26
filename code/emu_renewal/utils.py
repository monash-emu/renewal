from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os
from os import listdir as ls
import shutil
import json
import pandas as pd
import numpy as np
import pycountry
import pycountry_convert as pc
import arviz as az

from emu_renewal.constants import ANALYSIS_TYPES, ANALYSIS_NAMES, OUTPUTS_PATH, DATA_PATH, UNOFFICIAL_COUNTRIES


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
    name_to_iso3 = {get_country_name(c): c for c in countries}
    return [name_to_iso3[name] for name in sorted(name_to_iso3)]


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
    return [l[i : i + segment_length] for i in range(0, len(l), segment_length)]


def to_iso3(
    country: str,
) -> str:
    """Resolve a country name or code to this project's ISO3 identifier.

    Args:
        country: Name or ISO code

    Returns:
        The ISO3 code used elsewhere in the project (e.g. RKS for Kosovo)
    """
    if country in UNOFFICIAL_COUNTRIES:
        return country
    for iso3, info in UNOFFICIAL_COUNTRIES.items():
        if country.lower() == info["name"].lower():
            return iso3
        if country in (info.get("alpha_2"), info.get("wb_code")):
            return iso3
    return pycountry.countries.lookup(country).alpha_3


def wb_iso3(
    iso3: str,
) -> str:
    """ISO3 code used by World Bank and UN population/income datasets.

    Args:
        iso3: The project country identifier

    Returns:
        The dataset code (XKX for Kosovo)
    """
    unofficial = UNOFFICIAL_COUNTRIES.get(iso3)
    if unofficial and "wb_code" in unofficial:
        return unofficial["wb_code"]
    return iso3


def iso3_to_iso2(
    iso3: str,
) -> Optional[str]:
    """Map an ISO3 code to the ISO2 country code used by WHO.

    Args:
        iso3: The country identifier

    Returns:
        The ISO2 code, or None if no mapping is available.

    Notes
    -----
    WHO surveillance data are keyed by ISO2 "Country_code".
    ISO 3166-1 does not include Kosovo, which appears as "RKS" in OxCGRT
    and as "XK" in the WHO dataset. We map these user-assigned codes
    explicitly so that Kosovo is joined to WHO indicators in the same
    way as other countries, rather than dropped as a special case.
    """
    try:
        return pycountry.countries.lookup(iso3).alpha_2
    except LookupError:
        unofficial = UNOFFICIAL_COUNTRIES.get(iso3)
        return unofficial["alpha_2"] if unofficial else None


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
    if iso3 in UNOFFICIAL_COUNTRIES:
        return UNOFFICIAL_COUNTRIES[iso3]["continent"]
    try:
        iso2 = pycountry.countries.lookup(iso3).alpha_2
        return pc.convert_country_alpha2_to_continent_code.country_alpha2_to_continent_code(iso2)
    except (KeyError, LookupError):
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
    if iso3 in UNOFFICIAL_COUNTRIES:
        return UNOFFICIAL_COUNTRIES[iso3]["name"]
    info = pycountry.countries.lookup(iso3)
    if hasattr(info, "common_name"):
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
    if iso3 in UNOFFICIAL_COUNTRIES:
        return UNOFFICIAL_COUNTRIES[iso3]["name"]
    try:
        return pycountry.countries.lookup(iso3).name
    except LookupError:
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


def get_analysis_commits_df(
    analysis_paths: Dict[str, Dict[str, Path]],
) -> pd.DataFrame:
    """New approach to getting commits used in running each job
    now based on analysis path dictionary produced by get_analysis_paths.

    Args:
        analysis_paths: The outputs of get_analysis_paths

    Returns:
        The dataframe for display
    """
    countries = analysis_paths.keys()
    commits = pd.DataFrame(index=countries, columns=ANALYSIS_TYPES)
    for iso3 in countries:
        for analysis in ANALYSIS_TYPES:
            c_paths = analysis_paths[iso3]
            if analysis in c_paths:
                a_path = c_paths[analysis]
                sha = (
                    json.load(open(a_path / "gitinfo.json", "r"))["sha"][:7]
                    if os.path.isdir(a_path)
                    else "no analysis"
                )
            else:
                sha = "no analysis"
            commits.loc[iso3, analysis] = sha
    commits.rename(columns=ANALYSIS_NAMES, inplace=True)
    commits.rename(index=get_country_name, inplace=True)
    return commits.sort_index()


def get_job_commits_df_new(
    analysis_paths: Dict[str, Dict[str, Path]],
) -> pd.DataFrame:
    """New approach to getting commits used in running each job
    now based on analysis path dictionary produced by get_analysis_paths.

    Args:
        analysis_paths: The outputs of get_analysis_paths

    Returns:
        The dataframe for display
    """
    countries = analysis_paths.keys()
    commits = pd.DataFrame(index=countries, columns=ANALYSIS_TYPES)
    for iso3 in countries:
        for analysis in ANALYSIS_TYPES:
            c_paths = analysis_paths[iso3]
            if analysis in c_paths:
                a_path = c_paths[analysis]
                sha = json.load(open(a_path / "gitinfo.json", "r"))["sha"][:7] if os.path.isdir(a_path) else "no analysis"
            else:
                sha = "no analysis"
            commits.loc[iso3, analysis] = sha
    commits.rename(columns=ANALYSIS_NAMES, inplace=True)
    commits.rename(index=get_country_name, inplace=True)
    return commits.sort_index()


def copy_analysis_type_to_run(
    src_id: str,
    dest_id: str, 
    analysis_type: str,
):
    """Copy all the runs of a particular type
    (e.g. no_mob, g_mob, oxcgrt) from one run ID folder
    to another.

    Args:
        src_id: The source run ID
        dest_id: The destination run ID
        analysis_type: The type of analysis to copy
    """
    for iso3 in ls(OUTPUTS_PATH / src_id):
        src = OUTPUTS_PATH / src_id / iso3 / analysis_type
        dest = OUTPUTS_PATH / dest_id / iso3 / analysis_type
        if src.exists() and not dest.exists():
            shutil.copytree(src, dest)





def get_analysis_paths(
    job_ids: List[str],
    countries: List[str],
) -> Dict[str, Dict[str, Path]]:
    """Find analysis output directories for each country.
    Job IDs are searched in the order provided.
    For each (country, analysis type) pair,
    the first matching analysis directory 
    found in that hierarchy is populated.
    If an analysis type is not found in any job directory 
    for a country, that analysis type is 
    omitted from the country's result dictionary.

    Returns:
        Nested dictionary mapping:
            country -> analysis type -> directory path
    """
    job_paths = [OUTPUTS_PATH / p for p in job_ids]
    analysis_paths = {}
    for c in countries:
        analysis_paths[c] = {}
        for a in ANALYSIS_TYPES:
            for j in job_paths:
                analysis_path = j / c / a
                if analysis_path.is_dir():
                    analysis_paths[c][a] = analysis_path
                    break
    return analysis_paths


def get_analysis_status(
    run_id: str,
) -> pd.DataFrame:
    """Find out what happened for each possible country-analysis combination.

    Args:
        run_id: The run ID

    Returns:
        Whether the analysis is complete, skipped, not run or no log available
    """
    countries = json.load(open(DATA_PATH / "config/included.json", "r"))
    analysis_status = pd.DataFrame("blank", index=countries, columns=ANALYSIS_TYPES, dtype=str)
    run_path = OUTPUTS_PATH / run_id
    for iso3 in countries:
        c_path = run_path / iso3
        if not (run_path / iso3 / "run.log").exists():
            analysis_status.loc[iso3, :] = "no log"
            continue
        text = open(run_path / iso3 / "run.log", "r").read()
        for a in ANALYSIS_TYPES:
            if (c_path / a / "updates.h5").exists():
                out = "complete"
            elif f"{a} mobility not available" in text:
                out = "skipped"
            else:
                out = "not run"
            analysis_status.loc[iso3, a] = out
    return analysis_status


def move_idata_full_to_bin(
    run_id: str,
    countries: List[str],
):
    """Move full idatas to bin if issues with storage.
    Usually we are working with idata_filtered,
    so can often discard these outputs,
    which are nearly half the storage.

    Args:
        run_id: The run ID to clear
        countries: The full list of countries for analysis
    """
    analysis_paths = get_analysis_paths([run_id], countries)
    for iso3 in analysis_paths:
        for analysis_type in analysis_paths[iso3]:
            src_path = analysis_paths[iso3][analysis_type]
            filename = "idata_full.nc"
            src_filename = src_path / filename

            dest_path = Path.home() / ".Trash" / run_id / iso3 / analysis_type
            dest_path.mkdir(parents=True, exist_ok=True)
            dest_filename = dest_path / filename

            try:
                shutil.move(src_filename, dest_filename)
            except FileNotFoundError:
                print(f"{src_filename} not found")
