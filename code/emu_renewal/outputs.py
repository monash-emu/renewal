from typing import List
from pathlib import Path
import git
import json
import numpy as np
import pandas as pd
from jax import jit
from typing import Dict
import matplotlib.pyplot as plt
import arviz as az
import pickle
from numpyro import infer
from os import listdir as ls
import os
from geopandas import GeoDataFrame
import pycountry

from estival.sampling.tools import SampleIterator
from estival.sampling import tools as esamp

from emu_renewal.constants import MOB_SOURCE_COLOURS, N_SAMPLES
from emu_renewal.calibration import StandardCalib
from emu_renewal.renew import MultiStrainModel
from emu_renewal.utils import get_subdirs

plt.style.use("ggplot")

TARGET_KEY = "target_"


def run_for_spaghetti(
    calib: StandardCalib,
    params: SampleIterator,
) -> Dict[str, pd.DataFrame]:
    """Run parameters through the model to get epidemiological outputs.

    Args:
        calib: The calibration object, which includes the model
        params: The parameter sets to feed through the model

    Returns:
        Dictionary with keys strings containing the chain and draw numbers
            and values dataframes with columns for each output.
    """
    model = calib.epi_model
    times = model.epoch.index_to_dti(model.model_times)

    @jit
    def get_full_result(**params):
        return model.renewal_func(**params | calib.fixed_params)

    spagh_dict = {}
    for i, p in params.iterrows():
        epi_params = {k: v for k, v in p.items() if "dispersion" not in k}
        spagh = pd.DataFrame(get_full_result(**epi_params))
        spagh.index = times
        spagh_dict[str(i)] = spagh

    return spagh_dict


def get_spagh_df_from_dict(
    spagh_dict: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Process the dictionaries produced
    by run_for_spaghetti into dataframe format.

    Args:
        spagh_dict: The output of run_for_spaghetti

    Returns:
        Dataframe with model times as index and multiindexed columns,
            with first level being the output name and second the parameter set
            by chain and iteration
    """
    outputs = list(spagh_dict.values())[0].columns  # Arbitrarily chosen output dataframe
    column_names = pd.MultiIndex.from_product([spagh_dict.keys(), outputs])
    spaghetti = pd.DataFrame(columns=column_names)
    for i in spagh_dict:
        spaghetti[i] = spagh_dict[i]
    spaghetti.columns = spaghetti.columns.swaplevel()
    return spaghetti.sort_index(axis=1, level=0)


def load_targets(
    outdir: Path,
) -> Dict[str, pd.Series]:
    """Load previously saved data for calibration targets.

    Args:
        outdir: Directory

    Returns:
        The targets' data
    """
    targets = {}
    for file in outdir.iterdir():
        filename = file.name
        if filename.startswith(TARGET_KEY):
            targ_name = file.stem[len(TARGET_KEY) :]
            targets[targ_name] = pd.read_hdf(outdir / filename)
    return targets


def get_table_df_from_priors_dict(
    priors_dict: pd.DataFrame,
) -> pd.DataFrame:
    """Convert format of priors from raw format
    to format prepared for visualising with Quarto.

    Args:
        priors_dict: Priors in raw format

    Returns:
        The priors in the revised format
    """
    priors_df = pd.DataFrame.from_dict(priors_dict).T
    priors_df = priors_df.set_index("param_name")
    priors_df.index.name = None
    keep_cols = [c for c in priors_df if c != "short_name"]
    priors_df = priors_df[keep_cols]
    priors_df.columns = priors_df.columns.str.capitalize()
    return priors_df.rename(columns={"Sd": "SD", "Std": "SD"})


def get_gitinfo() -> Dict[str, str]:
    """Get the git-related information for the current
    local repository state.

    Returns:
        The branch name and commit SHA
    """
    import emu_renewal

    rpath = Path(emu_renewal.__path__[0]).parent.parent
    repo = git.Repo(rpath)
    git_info = {"branch": str(repo.active_branch), "sha": repo.head.object.hexsha}
    return git_info


def store_outputs(
    out_dir: Path,
    model: MultiStrainModel,
    calib: StandardCalib,
    mcmc: infer.MCMC,
):
    """Store model and calibration characteristics and results in standard formats.

    Args:
        out_dir: Location to store the outputs
        model: Renewal model
        calib: Calibration object
        mcmc: MCMC object
    """
    idata_full = az.from_numpyro(mcmc)
    idata_full.to_netcdf(out_dir / "idata_full.nc")

    energy = pd.DataFrame(mcmc.get_extra_fields(True)["potential_energy"]).T
    likelihood = 0.0 - energy
    likelihood.to_hdf(out_dir / "likelihood.h5", key="likelihood")

    ll_chain_mean = likelihood.mean()
    max_diff = likelihood[ll_chain_mean.idxmax()].std()

    bad_idx_table = (ll_chain_mean.max() - ll_chain_mean) > max_diff
    good_chains = list(bad_idx_table.index[~bad_idx_table])

    print(f"Selected chains {good_chains}")

    idata_filtered = idata_full.sel(chain=good_chains)
    idata_filtered.to_netcdf(out_dir / "idata_filtered.nc")

    idata_sampled = az.extract(idata_filtered, num_samples=N_SAMPLES)
    sample_params = esamp.xarray_to_sampleiterator(idata_sampled)
    spaghetti = get_spagh_df_from_dict(run_for_spaghetti(calib, sample_params))
    spaghetti.to_hdf(out_dir / "spaghetti.h5", key="spaghetti")
    update_idx = model.epoch.index_to_dti(model.x_proc_vals)
    updates = pd.DataFrame(sample_params.components["proc"], columns=update_idx).T
    updates.to_hdf(out_dir / "updates.h5", key="updates")

    pickle.dump(calib.sampled_params, open(out_dir / "priors.pkl", "wb"))

    for t, target in calib.targets.items():
        target.data.to_hdf(out_dir / f"{TARGET_KEY}{t}.h5", key=t)

    json.dump(get_gitinfo(), open(out_dir / "gitinfo.json", "w"))


def get_country_procs(
    job_path: Path,
    countries: List[str],
) -> Dict[str, pd.DataFrame]:
    """Get dataframes containing the variable process
    values for a combination of countries
    and analysis types.

    Args:
        path: Parent path for all runs
        countries: The names of the countries of interest

    Returns:
        The variable process dataframes
    """
    procs = {}
    for c in countries:
        c_path = job_path / c
        c_procs = []
        analyses = get_subdirs(c_path)
        for a in analyses:
            c_procs.append(pd.read_hdf(c_path / a / "spaghetti.h5")["process"])
        procs[c] = pd.concat(c_procs, keys=analyses, axis=1)
    return procs


def get_param_vals_by_analysis(
    param: str,
    c_path: Path,
) -> pd.DataFrame:
    """Get dataframe of accepted parameter values
    by analysis for a particular parameter and country.

    Args:
        param_name: Name of the parameter
        country_path: Location of the country analyses

    Returns:
        The posterior estimates
    """
    param_df = []
    analyses = get_subdirs(c_path)
    for a in analyses:
        idata = az.from_netcdf(c_path / a / "idata_filtered.nc")
        param_df.append(idata.posterior[param].to_series())
    result = pd.concat(param_df, axis=1, keys=analyses)
    ordered_cols = [c for c in MOB_SOURCE_COLOURS if c in result.columns]
    return result[ordered_cols]


def add_bool_row_to_table(
    table: pd.DataFrame,
    bool_list: List[bool],
    col_name: str,
):
    """Add a column to a table based on its index
    and a list of indexes for interpretation as a boolean
    for whether to mark as yes or no for each index element.

    Args:
        table: The existing table
        bool_list: The indexes to be interpreted as yes
        col_name: The name for the new column
    """
    table[col_name] = table.index.isin(bool_list)
    table[col_name] = table[col_name].map({True: "Yes", False: "No"})


def add_mob_avail_to_world(
    world: GeoDataFrame,
    g_avail: List[str],
    fb_avail: List[str],
):
    """Add columns to world geopandas dataframe
    for whether Google and Facebook mobility are present.

    Args:
        world: The world geopandas dataframe
        g_avail: Whether Google mobility available
        fb_avail: Whether Facebook mobility available
    """
    world["g_avail"] = world["ISO_A3"].isin(g_avail)
    world["fb_avail"] = world["ISO_A3"].isin(fb_avail)
    world["mob"] = "neither"
    world.loc[world["g_avail"] == True, "mob"] = "Google"
    world.loc[world["fb_avail"] == True, "mob"] = "FB"
    world.loc[(world["fb_avail"] == True) & (world["g_avail"] == True), "mob"] = "both"


def get_prop_improve(
    disp_posts: Dict[str, pd.DataFrame],
    mob_source: str,
) -> Dict[str, float]:
    """Find the proportion of results from a particular run that
    have a lower dispersion parameter than 
    the median value of the no mobility analysis.

    Args:
        disp_posts: The posteriors of the dispersion parameter by country and analysis
        mob_type: The mobility analysis of interest

    Returns:
        The proportions by country
    """
    prop_improve_median = {}
    for c in disp_posts:
        c_posts = disp_posts[c]
        no_mob_median = c_posts["no_mob"].median()
        
        if mob_source in c_posts:
            mob_posts = c_posts[mob_source]
            prop_improve_median[c] = (mob_posts < no_mob_median).sum() / len(mob_posts)
    return prop_improve_median


def get_idatas_for_mob_type(
    job_path: Path,
    countries: List[str],
    mob_source: str,
) -> Dict[str, az.InferenceData]:
    """Collate all the inference data objects for
    a requested group of countries.

    Args:
        job_path: Path for the runs
        countries: Countries identifiers
        mob_source: Mobility type considered

    Returns:
        The inference data objects
    """
    country_idatas = {}
    unavailable_countries = []
    for iso3 in countries:
        country = pycountry.countries.lookup(iso3).name
        try:
            path = job_path / iso3 / mob_source / "idata_filtered.nc"
            country_idatas[iso3] = az.from_netcdf(path)
        except FileNotFoundError:
            unavailable_countries.append(country)
    return country_idatas, unavailable_countries


def get_param_mean_by_country(
    job_path: Path, 
    param: str, 
    mob_source: str,
) -> Dict[str, float]:
    """Get the mean of the parameter posterior for each
    country analysed under a particular mobility approach.

    Args:
        job_path: Path for the runs
        param: Name of the parameter
        mob_source: Mobility analysis type

    Returns:
        The parameter mean by country
    """
    countries = ls(job_path)
    i_datas, _ = get_idatas_for_mob_type(job_path, countries, mob_source)
    return {c: az.summary(i_datas[c], var_names=param, kind="stats")["mean"].values[0] for c in i_datas}


def get_ratios_from_disps(
    disp_posts: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Find the ratio of the variable process dispersion parameters
    under the mobility analyses compared to the relevant baseline.

    Args:
        disp_posts: Output of get_param_vals_by_analysis 
            with "dispersion_proc" as first argument
            (i.e. process dispersion samples by analysis)

    Returns:
        The ratios by country
    """
    ratios = {}
    for c in disp_posts:
        disp_post = disp_posts[c]
        ratio_df = pd.DataFrame()
        if "g_mob" in disp_post:
            ratio_df["g_mob"] = disp_post["g_mob"] / disp_post["no_mob"]
        if "fb_visited_mob" in disp_post:
            fb_ref = "fb_no_mob" if "fb_no_mob" in disp_post else "no_mob"
            ratio_df["fb_visited_mob"] = disp_post["fb_visited_mob"] / disp_post[fb_ref]
        if "fb_singletile_mob" in disp_post:
            fb_ref = "fb_no_mob" if "fb_no_mob" in disp_post else "no_mob"
            ratio_df["fb_singletile_mob"] = disp_post["fb_singletile_mob"] / disp_post[fb_ref]        
        ratios[c] = ratio_df
    return ratios


def get_median_ratios(
    dists: Dict[str, pd.DataFrame], 
    mob_source: str,
) -> Dict[str, float]:
    """Get the median ratio of the variable process
    dispersion parameter sample under a mobility 
    analysis to the equivalent baseline.

    Args:
        dists: The output from get_ratios_from_disps
        mob_source: The mobility type

    Returns:
        The ratio values by country
    """
    median_ratios = {}
    for c in dists:
        c_ratios = dists[c]
        if mob_source in c_ratios:
            median_ratios[c] = c_ratios.median()[mob_source]
    return median_ratios
