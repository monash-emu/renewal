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

from estival.sampling.tools import SampleIterator
from estival.sampling import tools as esamp

from emu_renewal.calibration import StandardCalib
from emu_renewal.renew import MultiStrainModel
from emu_renewal.plotting import MOB_COLOURS

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


def get_gitinfo() -> dict[str, str]:
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
    n_samples=50,
):
    """Store model and calibration characteristics and results in standard formats.

    Args:
        out_dir: Location to store the outputs
        model: Renewal model
        calib: Calibration object
        mcmc: MCMC object
        n_samples: Number of samples to extract for spaghetti
    """
    idata_full = az.from_numpyro(mcmc)
    idata_full.to_netcdf(out_dir / "idata_full.nc")

    likelihood = pd.DataFrame(mcmc.get_extra_fields(True)["potential_energy"]).T
    likelihood = 0.0 - likelihood
    likelihood.to_hdf(out_dir / "likelihood.h5", key="likelihood")

    ll_chain_mean = likelihood.mean()
    max_diff = likelihood[ll_chain_mean.idxmax()].std()

    bad_idx_table = (ll_chain_mean.max() - ll_chain_mean) > max_diff
    good_chains = list(bad_idx_table.index[~bad_idx_table])

    print(f"Selected chains {good_chains}")

    idata_filtered = idata_full.sel(chain=good_chains)
    idata_filtered.to_netcdf(out_dir / "idata_filtered.nc")

    idata_sampled = az.extract(idata_filtered, num_samples=n_samples)
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


def get_country_posteriors(
    path: Path,
    countries: List[str],
) -> Dict[str, pd.DataFrame]:
    """Get dataframes containing the posterior
    values for a combination of countries
    and analysis types.

    Args:
        path: Parent path for all runs
        countries: The names of the countries of interest

    Returns:
        The posterior dataframes
    """
    likes = {}
    for c in countries:
        country_path = path / c
        c_likes = []
        analyses = ls(country_path)
        for a in analyses:
            idata = az.from_netcdf(country_path / a / "idata_filtered.nc")
            c_likes.append(idata["sample_stats"]["lp"].to_pandas().T)
        likes[c] = -pd.concat(c_likes, keys=analyses, axis=1)
    return likes


def get_all_like_comps(
    country_path: Path,
) -> pd.DataFrame:
    """Get the likelihood components and
    the overall likelihood for a set of runs.

    Args:
        country_path: Location of the country analyses

    Returns:
        The collated likelihood components by analysis
    """
    analyses = ls(country_path)
    likes_by_analysis = []
    for analysis in analyses:
        idata = az.from_netcdf(country_path / analysis / "idata_filtered.nc")
        all_likes = idata["log_likelihood"].to_dataframe()
        all_likes["total_ll"] = -idata["sample_stats"]["lp"].to_dataframe()
        likes_by_analysis.append(all_likes)
    return pd.concat(likes_by_analysis, axis=1, keys=analyses)


def get_country_procs(
    path: Path,
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
        c_path = path / c
        c_procs = []
        analyses = [i[1] for i in os.walk(c_path)][0]
        for a in analyses:
            c_procs.append(pd.read_hdf(c_path / a / "spaghetti.h5")["process"])
        procs[c] = pd.concat(c_procs, keys=analyses, axis=1)
    return procs


def get_param_vals_by_analysis(
    param_name: str,
    country_path: Path,
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
    analyses = [d.name for d in os.scandir(country_path) if d.is_dir()]
    for a in analyses:
        idata = az.from_netcdf(country_path / a / "idata_filtered.nc")
        param_df.append(idata.posterior[param_name].to_series())
    result = pd.concat(param_df, axis=1, keys=analyses)
    ordered_cols = [c for c in MOB_COLOURS if c in result.columns]
    return result[ordered_cols]


def get_completed_chains(job_path):
    completion_lists = []
    countries = ls(job_path)
    for c in countries:
        c_path = job_path / c
        analyses = ls(c_path)
        c_complete = pd.DataFrame(columns=analyses)
        for a in analyses:
            a_path = c_path / a
            all_idata = az.from_netcdf(a_path / "idata_full.nc")
            filt_idata = az.from_netcdf(a_path / "idata_filtered.nc")
            all_chains = all_idata["posterior"]["chain"].data
            completed_chains = filt_idata["posterior"]["chain"].data
            c_complete[a] = [c in completed_chains for c in all_chains]
        completion_lists.append(c_complete)
    return pd.concat(completion_lists, keys=countries, axis=1)


def add_bool_row_to_table(
    table: pd.DataFrame,
    bool_list: List[str],
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
    """
    world["g_avail"] = world["ISO_A3"].isin(g_avail)
    world["fb_avail"] = world["ISO_A3"].isin(fb_avail)
    world["mob"] = "neither"
    world.loc[world["g_avail"] == True, "mob"] = "Google"
    world.loc[world["fb_avail"] == True, "mob"] = "FB"
    world.loc[(world["fb_avail"] == True) & (world["g_avail"] == True), "mob"] = "both"


def get_prop_improve(
    disp_posts: Dict[str, pd.DataFrame],
    mob_type: str,
) -> Dict[str, float]:
    """Find the proportion of results from a particular run that
    have a lower dispersion parameter than for the no mobility analysis.

    Args:
        disp_posts: The posteriors of the dispersion parameter by country and analysis
        mob_type: The mobility analysis of interest

    Returns:
        The proportions by country
    """
    prop_improve = {}
    for c in disp_posts:
        c_posts = disp_posts[c]
        no_mob_posts = c_posts["no_mob"]
        if mob_type in c_posts:
            mob_posts = np.random.permutation(c_posts[mob_type].values)
            prop_improve[c] = pd.Series(no_mob_posts > mob_posts).astype(int).mean()
    return prop_improve
