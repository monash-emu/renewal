import json
import sys
 
from emu_renewal.inputs import DATA_PATH
from emu_renewal.constants import ANALYSIS_TYPES, BASE_PATH
from emu_renewal.run import run_single_country, MobilityException, get_logger, jax_config_cpu_only


if __name__ == "__main__":
    jax_config_cpu_only()
    countries = ['BHS', 'GNQ', 'SEN', 'ABW', 'LBR', 'STP', 'LSO', 'MLT']
    # countries = json.load(open(DATA_PATH / f"config/included.json", "r"))
    task = sys.argv[1]
    array_task_id = int(sys.argv[2])
    c = countries[array_task_id - 1]  # Convert to Python indexing
    country_path = BASE_PATH / "outputs" / task / c
    country_path.mkdir(parents=True, exist_ok=True)
    logger = get_logger(country_path / "run.log")
    for mob_type in ANALYSIS_TYPES:
        try:
            run_single_country(c, mob_type, task, logger=logger)
        except MobilityException as e:
            logger.warning(e)
