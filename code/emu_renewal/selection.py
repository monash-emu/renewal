from typing import List, Dict
import pandas as pd


def has_repeats(
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
