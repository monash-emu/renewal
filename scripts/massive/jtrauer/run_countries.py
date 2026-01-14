import json
import sys

from emu_renewal.inputs import DATA_PATH
from emu_renewal.constants import BASE_PATH
from emu_renewal.run import run_single_country, get_logger, jax_config_cpu_only


if __name__ == "__main__":
    jax_config_cpu_only()
    countries = json.load(open(DATA_PATH / "config/included.json", "r"))
    analyses = json.load(open(DATA_PATH / "config/analyses.json", "r"))
    task = sys.argv[1]
    array_task_id = int(sys.argv[2])
    iso3 = countries[array_task_id - 1]  # Convert to Python indexing
    country_path = BASE_PATH / "outputs" / task / iso3
    country_path.mkdir(parents=True, exist_ok=True)
    logger = get_logger(country_path / "run.log")
    for mob_type in analyses:
        run_single_country(iso3, mob_type, task, False, logger=logger)
