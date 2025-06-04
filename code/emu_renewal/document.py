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
    txt = re.split("-----", docstring)[1]
    txt = re.sub(r"\s+", " ", txt)
    txt = re.sub(r"__RETURN__", "\n", txt)
    return txt.format(**constants.__dict__)
