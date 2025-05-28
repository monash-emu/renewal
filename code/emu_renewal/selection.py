from typing import List, Dict
import pandas as pd


def is_mostly_zeros(
    data: pd.Series,
) -> bool:
    """Return True if most values of a series are zero,
    otherwise False.

    Args:
        data: The indicator data

    Returns:
        Whether mostly zeroes
    """
    if len(data) > 0:
        return sum((data == 0.0).astype(int)) / len(data) > 0.5
    else:
        return False


def has_repeats(
    data: pd.Series,
    n_repeats: int,
    threshold=1e-10,
) -> bool:
    """Find if an indicator series either has the
    same value repeated several times,
    or if the absolute change in the indicators is the same
    (i.e. a straight diagonal line).

    Args:
        data: The indicator data
        n_repeats: The number of repeats to identify
        threshold: The threshold to define the same or the same change

    Returns:
        Whether the data has more than the number of repeats
    """
    repeat_change = (data.diff().diff().abs() < threshold) & (data > 0.0)
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
