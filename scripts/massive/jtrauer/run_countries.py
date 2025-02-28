import json
import sys

from emu_renewal.inputs import DATA_PATH, ANALYSIS_TYPES
from emu_renewal.run import run_single_country


if __name__=="__main__":
    initial_countries = json.load(open(DATA_PATH / f"config/countries.json", "r"))
    all_countries = initial_countries["admissions"] + initial_countries["occupancy"]
    array_task_id = int(sys.argv[2])
    country = all_countries[array_task_id - 1]  # Convert to Python indexing
    hosp_out, hosp_out_name = ("Daily hospital occupancy", "occupancy") if \
        country in initial_countries["occupancy"] else ("Weekly new hospital admissions", "admissions")
    for mob_analysis_type in ANALYSIS_TYPES:
        run_single_country(country, 7, 50, mob_analysis_type, 1000, hosp_out, hosp_out_name, 50, sys.argv[1], num_chains=8)
