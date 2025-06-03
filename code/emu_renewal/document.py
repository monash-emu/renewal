import re
from emu_renewal import constants


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
    blurb = re.split("-----", docstring)[1]
    blurb_str = re.sub(r"\s+", " ", blurb)
    return blurb_str.format(**constants.__dict__)
