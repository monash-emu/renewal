from typing import List, Tuple
import pandas as pd
from os import listdir as ls
from datetime import datetime

from emu_renewal.constants import (
    OUTLIER_THRESHOLD, 
    N_REPEATS, 
    DATA_QUALITY_START_TIME, 
    DATA_QUALITY_START_TIME_OC, 
    CODE_DATE_FORMAT, 
    VARIATION_THRESHOLD,
)
from emu_renewal.document import get_exp_val_from_string
from emu_renewal.inputs import DATA_PATH, get_oxcgrt_data
from emu_renewal.indicators import get_who_indicator
from emu_renewal.outputs import add_bool_row_to_table
from emu_renewal.run import find_run_end_time
from emu_renewal.utils import count_repeat_nans, get_cont_of_country


def get_data_avail_countries() -> Tuple[List[str], pd.DataFrame]:
    """Find the countries for which either mobility
    or policy data is available.

    Returns:
        - Countries for which at least one mobility domain or OxCGRT data is available
        - Table with indices for countries and Yes/No status for each mobility domain
        - The countries with data available for each source
    """
    g_avail = [c[:3] for c in ls(DATA_PATH / "mobility") if "gmob" in c]
    fb_avail = [c[:3] for c in ls(DATA_PATH / "mobility") if "fbmob" in c]
    pol_data = get_oxcgrt_data()
    pol_avail = [iso3 for iso3 in set(pol_data["CountryCode"]) if iso3 != "RKS"]
    any_data_avail = list(set(g_avail + fb_avail + pol_avail))
    summary = pd.DataFrame(index=any_data_avail)
    add_bool_row_to_table(summary, g_avail, "Google available")
    add_bool_row_to_table(summary, fb_avail, "FB available")
    add_bool_row_to_table(summary, pol_avail, "OxCGRT")
    return any_data_avail, summary, g_avail, fb_avail, pol_avail


def get_mob_avail_countries() -> Tuple[List[str], pd.DataFrame]:
    """Find the countries for which either
    Google or Facebook mobility is available.

    Returns:
        - Countries for which at least one mobility domain is available
        - Table with indices for countries and Yes/No status for each mobility domain
        - The countries with data available for each source

    Notes
    -----
    To select countries for inclusion in our analysis, we first identified all countries
    for which either Google or Facebook mobility data was available.
    """
    g_avail = [c[:3] for c in ls(DATA_PATH / "mobility") if "gmob" in c]
    fb_avail = [c[:3] for c in ls(DATA_PATH / "mobility") if "fbmob" in c]
    either_mob_avail = list(set(g_avail + fb_avail))
    summary = pd.DataFrame(index=either_mob_avail)
    add_bool_row_to_table(summary, g_avail, "Google available")
    add_bool_row_to_table(summary, fb_avail, "FB available")
    return either_mob_avail, summary, g_avail, fb_avail


def gather_who_data(
    countries: List[str],
) -> Tuple[pd.Series]:
    """Get the two main WHO indicators for a set of countries.

    Args:
        countries: The countries for which data is required

    Returns:
        The data

    Notes
    -----
    Next, we considered the quality of the data for our two main WHO indicators
    which we required for inclusion in the analysis: `New_cases` and `New_deaths`
    from the start of data availability through to the end time
    of the analysis for each country.
    We considered data quality from {DATA_QUALITY_START_TIME_OC}
    for countries of Oceania and from {DATA_QUALITY_START_TIME}
    for the countries of all other continents.
    """
    death_data = {}
    case_data = {}
    for c in countries:

        # Find data quality start and analysis end times
        cont = get_cont_of_country(c)
        start_time = DATA_QUALITY_START_TIME_OC if cont == "OC" else DATA_QUALITY_START_TIME
        start = datetime.strptime(start_time, CODE_DATE_FORMAT)
        try:
            end_time = find_run_end_time(c, "g_mob")
        except:
            try:
                end_time = find_run_end_time(c, "fb_visited_mob")
            except:
                end_time = datetime(2021, 12, 31)

        # Get deaths and cases data
        deaths = get_who_indicator("New_deaths", c)
        filter = (start < deaths.index) & (deaths.index < end_time)
        death_data[c] = deaths[filter]

        cases = get_who_indicator("New_cases", c)
        filter = (start < cases.index) & (cases.index < end_time)
        case_data[c] = cases[filter]

    return death_data, case_data


def find_absent_inds(
    deaths: pd.Series, 
    cases: pd.Series, 
    summary: pd.DataFrame,
) -> Tuple[List[str]]:
    """Find the countries for which there is no data
    available for either of the two main indicators.

    Args:
        deaths, cases: Output of gather_who_data
        summary: Second output of get_mob_avail_countries

    Returns:
        - The countries with no reported deaths
        - The countries with no reported cases

    Notes
    -----
    Using this data, we excluded any countries 
    for which no deaths or cases were reported
    throughout this data availability period.
    """
    no_deaths = [c for c, d in deaths.items() if d.empty or d.max() == 0.0 or all(d.isna())]
    no_cases = [c for c, d in cases.items() if d.empty or d.max() == 0.0 or all (d.isna())]
    add_bool_row_to_table(summary, no_deaths, "No death data")
    add_bool_row_to_table(summary, no_cases, "No case data")
    return no_deaths, no_cases


def find_neg_inds(
    deaths: pd.Series, 
    cases: pd.Series, 
    summary: pd.DataFrame,
) -> Tuple[List[str]]:
    """Find the countries with negative values
    for either of the two main indicators.

    Args:
        deaths, cases: Output of gather_who_data
        summary: Second output of get_mob_avail_countries

    Returns:
        - The countries with negative death values
        - The countries with negative case values

    Notes
    -----
    We also excluded any countries for which any negative values were 
    present within the available data.
    """
    neg_deaths = [c for c, d in deaths.items() if d.min() < 0.0]
    neg_cases = [c for c, d in cases.items() if d.min() < 0.0]
    add_bool_row_to_table(summary, set(neg_deaths + neg_cases), "Negative values present")
    return neg_deaths, neg_cases


def find_outliers(
    deaths: pd.Series, 
    cases: pd.Series, 
    summary: pd.DataFrame,
) -> Tuple[List[str]]:
    """Find the countries with outlier values
    for either of the two main indicators.

    Args:
        deaths, cases: Outputs of gather_who_data
        summary: Second output of get_mob_avail_countries

    Returns:
        - The countries with outlier death values
        - The countries with outlier case values

    Notes
    -----
    We further excluded any countries for which single marked outliers were present,
    which we defined as a single value that was more than {OUTLIER_THRESHOLD}
    times greater than the next highest estimate present during the analysis period.
    """
    death_outliers = [c for c, d in deaths.items() if has_outlier(d, OUTLIER_THRESHOLD)]
    case_outliers = [c for c, d in cases.items() if has_outlier(d, OUTLIER_THRESHOLD)]
    add_bool_row_to_table(summary, set(death_outliers + case_outliers), "Outlier values present")
    return death_outliers, case_outliers


def find_nans_repeats(
    deaths: pd.Series, 
    cases: pd.Series, 
    summary: pd.DataFrame,
) -> Tuple[List[str]]:
    """Find the countries to excluded based on
    consecutive NaN or repeated values.

    Args:
        deaths, cases: Output of gather_who_data
        summary: Second output of get_mob_avail_countries

    Returns:
        The countries with too many consecutive NaNs for deaths
        The countries with too many consecutive NaNs for cases
        The countries with repeated values for deaths
        The countries with repeated values for cases

    Notes
    -----
    Last we excluded any countries for which multiple consecutive
    missing values were present in the surveillance data
    because it was unclear whether these should be interpreted as
    unavailable or as zeroes.
    Similarly, we excluded any countries (which affected only one country)
    for which repeated identical values were present
    (defined as changes within ${VARIATION_THRESHOLD}$).
    For both consecutive missing values and consecutive repeated values,
    we set the threshold for the number of observations for exclusion
    to be {N_REPEATS}.
    """
    death_nans = [c for c, d in deaths.items() if count_repeat_nans(d) > N_REPEATS]
    case_nans = [c for c, d in cases.items() if count_repeat_nans(d) > N_REPEATS]
    
    # Excludes Nicaragua that has many ones at the end (so not worth a separate function)
    thresh = get_exp_val_from_string(VARIATION_THRESHOLD)
    death_reps = [c for c, d in deaths.items() if has_reps(d, N_REPEATS, thresh)]
    case_reps = [c for c, d in cases.items() if has_reps(d, N_REPEATS, thresh)]

    exclusions = set(death_nans + case_nans + case_reps + death_reps)
    add_bool_row_to_table(summary, exclusions, "Absent or repeat values")
    return death_nans, case_nans, death_reps, case_reps


def has_reps(
    data: pd.Series,
    n_repeats: int,
    threshold=1e-10,
) -> bool:
    """Find if an indicator series either has the
    same value repeated several times.

    Args:
        data: The indicator data
        n_repeats: The number of repeats to identify
        threshold: The threshold to define the same or the same change

    Returns:
        Whether the data has more than the number of repeats
    """
    repeat_change = data.diff().abs() < threshold
    is_repeat = repeat_change.astype(int)
    multirepeat = is_repeat.rolling(n_repeats).sum()
    return (multirepeat == float(n_repeats)).any()


def has_outlier(
    data: pd.Series,
    threshold: float,
) -> bool:
    """Determine if an indicator has an outlier.

    Args:
        data: The indicator data

    Returns:
        Whether an outlier is present in the data
    """
    if len(data) > 1:
        largest, second = data.nlargest(2)
        return second == 0.0 or largest / second > threshold
    else:
        return False


def find_pol_countries(
    countries: List[str],
) -> List[str]:
    """Filter a list of countries according to whether
    OxCGRT data exists for them.

    Args:
        countries: The countries to consider

    Returns:
        The countries with data
    """
    data = get_oxcgrt_data()
    return [iso3 for iso3 in countries if any(data["CountryCode"] == iso3)]
