from typing import Tuple, Dict
from datetime import datetime

from emu_renewal.inputs import DEATHS_WEIGHT, get_indicator_series_from_who_data
from emu_renewal.targets import StandardDispTarget


def get_deaths_target(
    iso3: str,
    start: datetime, 
    end: datetime,
) -> Tuple[int, Dict[str, StandardDispTarget]]:
    """The number of deaths by week reported by WHO 
    was used as the first calibration target for all countries.
    Any values of zero in this series were replaced with a
    value of 0.5 to enable comparison to modelled outputs
    on the log scale. Deaths was the one of two indicators
    for which a common dispersion parameter was used
    for the distribution comparison of the modelled value.

    Args:
        iso3: The country identifier
        start: The calibration start time
        end: The calibration end time

    Returns:
        Number of observations in the deaths series
        The deaths calibration target
    """
    data = get_indicator_series_from_who_data("New_deaths", iso3)
    data = data.interpolate(method="linear").fillna(0.0)
    data[data == 0.0] = 0.5
    mask = (start < data.index) & (data.index < end)
    select_data = data.loc[mask]
    target = StandardDispTarget(select_data, weight=DEATHS_WEIGHT)
    return len(select_data), {"weekly_deaths": target}