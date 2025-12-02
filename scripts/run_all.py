import json
import sys
import time

from emu_renewal.inputs import DATA_PATH
from emu_renewal.constants import ANALYSIS_TYPES, BASE_PATH
from emu_renewal.run import (
    run_single_country,
    MobilityException,
    get_logger,
    jax_config_cpu_only,
)
from emu_renewal.utils import get_cont_of_country


if __name__ == "__main__":
    jax_config_cpu_only()
    countries = json.load(open(DATA_PATH / f"config/included.json", "r"))
    if len(sys.argv) < 3:
        task = str(int(time.time()))
    else:
        task = sys.argv[1]
    for c in countries:
        country_path = BASE_PATH / "outputs" / task / c
        country_path.mkdir(parents=True, exist_ok=True)
        logger = get_logger(country_path / "run.log")
        cont = get_cont_of_country(c)
        analyses = (
            ANALYSIS_TYPES + ["fb_no_mob"]
            if cont == "OC" and c != "SGP"
            else ANALYSIS_TYPES
        )
        for mob_type in analyses:
            try:
                run_single_country(c, mob_type, task, logger=logger)
            except MobilityException as e:
                logger.warning(e)
