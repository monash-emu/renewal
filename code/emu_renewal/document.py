from typing import Dict
import re
from emu_renewal import constants


def get_exp_val_from_string(
    num_str: str,
) -> float:
    """Convert a number as an exponentially notated string to
    the actual value of the number for use in code.

    Args:
        num_str: The string for the number in exponential markdown notation
    
    Returns:
        The value of the number as a float
    """
    exp_elements = num_str.split("\\times10^{")
    return float(exp_elements[0]) * 10.0 ** float(exp_elements[1][:-1])


def get_float_dict_from_str(
    float_info: str,
) -> Dict[str, float]:
    """Get a dictionary of floats from a single string.

    Args:
        float_info: The raw string containing the information

    Returns:
        The dictionary
    """
    c_dict = {}
    for i in float_info.split(", "):
        str_parts = i.split(": ")
        c_date = float(str_parts[1])
        c_dict[str_parts[0]] = c_date
    return c_dict


def get_func_blurb(
    function: callable,
) -> str:
    """Get appropriately formatted text
    based on the Notes section (followed by five dashes)
    of the function argument.

    Args:
        function: The function

    Returns:
        The formatted text
    """
    docstring = function.__doc__
    txt = re.split("-----", docstring)[1]
    txt = re.sub(r"\s+", " ", txt)
    txt = re.sub(r"__RETURN__", "\n\n", txt)
    return txt.format(**constants.__dict__)
