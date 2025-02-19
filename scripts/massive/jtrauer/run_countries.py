import json
import sys
from datetime import datetime

from emu_renewal.inputs import DATA_PATH, ANALYSIS_TYPES, DATE_FORMAT
from emu_renewal.run import run_single_country


initial_countries = json.load(open(DATA_PATH / f"config/countries.json", "r"))
all_countries = initial_countries["admissions"] + initial_countries["occupancy"]
analysis_name = f"run_{datetime.now().strftime(DATE_FORMAT)}"

if __name__=="__main__":
    array_task_id = int(sys.argv[2])
    country = all_countries[array_task_id - 1]  # Convert to Python indexing
    hosp_out, hosp_out_name = ("Daily hospital occupancy", "occupancy") if \
        country in initial_countries["occupancy"] else ("Weekly new hospital admissions", "admissions")
    for mob_analysis_type in ANALYSIS_TYPES:
        run_single_country(country, 10, 7, 50, mob_analysis_type, 2000, hosp_out, hosp_out_name, analysis_name, num_chains=8)
