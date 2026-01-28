import numpy as np
import json
from datetime import datetime
import pycountry
import arviz as az

from emu_renewal.constants import DATA_PATH, RERUNS, FULL_RUN, TIMEOUTS, DATE_FORMAT
from emu_renewal.inputs import get_cont_of_country
from emu_renewal.run import run_identifiability
from emu_renewal.utils import get_analysis_paths


def run_identifiability_analyses():
    n_iters = 10000
    n_analyses = 12
    rng = np.random.default_rng(2)

    all_countries = json.load(open(DATA_PATH / "config/included.json", "r"))
    analysis_paths = get_analysis_paths(RERUNS + FULL_RUN + TIMEOUTS, all_countries)
    eligible_countries = [c for c in analysis_paths if get_cont_of_country(c) != "OC"]
    sample_countries = [eligible_countries[i] for i in rng.integers(0, len(eligible_countries), n_analyses)]
    analysis_time = datetime.now().strftime(DATE_FORMAT)
    analyses = []
    for iso3 in sample_countries:
        analysis_options = list(analysis_paths[iso3].keys())
        analyses.append(analysis_options[rng.choice(len(analysis_options), 1)[0]])

    scalar_params = {}
    multi_params = {}
    for a in range(n_analyses):
        iso3 = sample_countries[a]
        mob_source = analyses[a]
        print(f"\n country: {pycountry.countries.lookup(iso3).name}")
        print(f"mobility approach: {mob_source}")
        print(analysis_time)

        # Select random set of scalar parameters from calibration posterior
        path = analysis_paths[iso3][mob_source]
        idata = az.from_netcdf(path / "idata_filtered.nc")
        post = idata.posterior
        chain = rng.integers(0, post.sizes["chain"])
        draw = rng.integers(0, post.sizes["draw"])
        draw_params = post.isel(chain=chain, draw=draw)
        scalar_params[iso3] = {p: float(v) for p, v in draw_params.items() if v.ndim == 0}
        multi_params[iso3] = {p: v.values for p, v in draw_params.items() if v.ndim > 0}
        
        # Run analysis
        run_identifiability(iso3, mob_source, analysis_time, scalar_params[iso3], multi_params[iso3], n_iters)


if __name__ == "__main__":
    run_identifiability_analyses()
