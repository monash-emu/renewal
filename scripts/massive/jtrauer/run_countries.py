import json
import sys

from emu_renewal.inputs import DATA_PATH, ANALYSIS_TYPES
from emu_renewal.run import run_single_country, MobilityException, log


if __name__ == "__main__":
    countries = json.load(open(DATA_PATH / f"config/countries.json", "r"))
    all_countries = countries["admissions"] + countries["occupancy"]
    task_name = sys.argv[1]
    array_task_id = int(sys.argv[2])
    c = all_countries[array_task_id - 1]  # Convert to Python indexing
    for mob_type in ANALYSIS_TYPES:
        try:
            run_single_country(c, 7, 50, mob_type, 1000, 50, task_name, num_chains=8)
        except MobilityException as e:
            log(e)
