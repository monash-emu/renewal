import json
import sys

from emu_renewal.inputs import DATA_PATH, ANALYSIS_TYPES, BASE_PATH
from emu_renewal.run import run_single_country, MobilityException, get_logger


if __name__ == "__main__":
    countries = json.load(open(DATA_PATH / f"config/included.json", "r"))
    task_name = sys.argv[1]
    array_task_id = int(sys.argv[2])
    c = countries[array_task_id - 1]  # Convert to Python indexing

    country_path = BASE_PATH / "outputs" / task_name / c
    country_path.mkdir(parents=True, exist_ok=True)

    logger = get_logger(country_path / "run.log")

    for mob_type in ANALYSIS_TYPES:
        try:
            run_single_country(
                c, 7, 50, mob_type, 1000, 50, task_name, num_chains=8, logger=logger
            )
        except MobilityException as e:
            logger.warning(e)
