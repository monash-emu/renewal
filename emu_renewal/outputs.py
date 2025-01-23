from pathlib import Path
import numpy as np
import pandas as pd
from jax import jit
from typing import List, Dict
import matplotlib.pyplot as plt
import arviz as az
import pickle
from numpyro import infer

from estival.sampling.tools import SampleIterator
from estival.sampling import tools as esamp

from emu_renewal.inputs import OUTPUTS_PATH
from emu_renewal.calibration import StandardCalib
from emu_renewal.renew import MultiStrainModel

plt.style.use("ggplot")


def get_spaghetti(
    calib: StandardCalib,
    params: SampleIterator,
) -> pd.DataFrame:
    """Run parameters through the model to get outputs.

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
        res = get_full_result(**epi_params)
        spagh = pd.DataFrame(res)
        spagh.index = times
        #spagh.columns = res.keys
        spagh_dict[str(i)] = spagh

    return spagh_dict


def get_spagh_df_from_dict(
    spagh_dict: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Process the dictionaries produced by get_spaghetti into dataframe format.

    Args:
        spagh_dict: The output of get_spaghetti

    Returns:
        Dataframe with model times as index and multiindexed columns,
            with first level being the output name and second the parameter set
            by chain and iteration
    """
    outputs = list(spagh_dict.values())[0].columns  # Arbitrary output dataframe
    column_names = pd.MultiIndex.from_product([spagh_dict.keys(), outputs])
    spaghetti = pd.DataFrame(columns=column_names)
    for i in spagh_dict:
        spaghetti[i] = spagh_dict[i]
    spaghetti.columns = spaghetti.columns.swaplevel()
    return spaghetti.sort_index(axis=1, level=0)


def get_quant_df_from_spaghetti(
    spaghetti: pd.DataFrame,
    quantile_req: list[float],
) -> pd.DataFrame:
    """Calculate requested quantiles over spaghetti created
    in previous function.

    Args:
        spaghetti: Output of get_spaghetti
        quantiles: The quantiles at which to make the calculations

    Returns:
        Dataframe with index of model times and multiindexed columns,
            with first level being the output name and second the quantile
    """
    outputs = set(spaghetti.columns.get_level_values(0))
    column_names = pd.MultiIndex.from_product([outputs, quantile_req])
    quantile_df = pd.DataFrame(index=spaghetti.index, columns=column_names)
    for out in outputs:
        quantile_df[out] = spaghetti[out].quantile(quantile_req, axis=1).T
    return quantile_df


def get_model_recovered_locs(model):
    strain_map = model.strain_map
    strains = model.strains
    ever_infect_cols = {}
    for st, strain in enumerate(strains):
        locs = [f"sus_{su}" for su in range(strain_map.shape[1]) if strain_map[st, su]]
        ever_infect_cols[strain] = locs
    return ever_infect_cols


def get_recovered_df(spagh, model, locs):
    rec_cats = [f"rec_{s}" for s in model.strains]
    runs = spagh.columns.get_level_values(1).unique()
    new_cols = pd.MultiIndex.from_product([rec_cats, runs])
    rec_df = pd.DataFrame(index=spagh.index, columns=new_cols)
    for s in model.strains:
        rec_df[f"rec_{s}"] = spagh[locs[s]].T.groupby(level=[1]).sum().T
    return rec_df


def add_recovered_to_spaghetti(spagh, model):
    rec_locs = get_model_recovered_locs(model)
    rec_df = get_recovered_df(spagh, model, rec_locs)
    return spagh.join(rec_df)


def get_col_abs_dist_from_mean(
    results_df: pd.DataFrame,
) -> pd.Series:
    """For a given dataframe, find the divergence 
    of each value from the mean of that column,
    then find the average absolute value of this
    divergence across each row.

    Args:
        results_df: Spaghetti for the variable process

    Returns:
        The series over time described above
    """
    diff_from_mean = results_df - results_df.mean()
    return diff_from_mean.abs().mean(axis=1)


def get_df_from_3darray(
    array: np.ndarray,
    order: List[int],
) -> pd.DataFrame:
    """Convert numpy array to pandas dataframe
    with count index and count multi-indexing over columns.

    Args:
        array: 3-dimensional numpy array
        order: The order of the dimensions in 
            the output dataframe relative to the input array

    Returns:
        Dataframe with:
            Index:
                Count over first dimension of numpy array
            First level of column multiindexing: 
                Count over dimension of input array listed second in order
            Second level of column multiindexing:
                Count over dimension of input array listed last in order
    """
    vals = array.reshape(array.shape[order[0]], -1)
    level_1_cols = range(array.shape[order[1]])
    level_2_cols = range(array.shape[order[2]])
    cols = pd.MultiIndex.from_product([level_1_cols, level_2_cols])
    return pd.DataFrame(vals, columns=cols)


def get_combined_df(
    df0: pd.DataFrame, 
    df1: pd.DataFrame, 
    name0: str, 
    name1: str,
) -> pd.DataFrame:
    """Join two pandas dataframes left-to-right
    retaining original index but extending on columns.

    Args:
        df0: First dataframe to join
        df1: Second dataframe to join
        name0: Name for first dataset
        name1: Name for second dataset

    Returns:
        New dataframe with data joined by columns
            with a new level of the column index at the top
            to indicate where the data came from
    """
    col_names0 = pd.MultiIndex.from_product([[name0]] + df0.columns.levels)
    col_names1 = pd.MultiIndex.from_product([[name1]] + df1.columns.levels)
    return pd.concat([df0.set_axis(col_names0, axis=1), df1.set_axis(col_names1, axis=1)], axis=1)


def get_output_dir(
    country: str, 
    analysis: str, 
    time: str,
) -> Path:
    """Get path for outputs from a run and ensure directory exists.

    Args:
        country: Country name
        analysis: Mobility analysis approach
        time: Time that analysis was run

    Returns:
        The path
    """
    path = Path(OUTPUTS_PATH / country.lower() / analysis / time)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_targets(
    country: str,
    analysis: str,
    time: str,
) -> Dict[str, pd.Series]:
    """Load previously saved data for calibration targets.

    Args:
        country: Name of the country of interest
        analysis: Mobility analysis approach
        time: Date and time that analysis was run

    Returns:
        The targets' data
    """
    targets = {}
    targ_key = "target_"
    outputs_path = get_output_dir(country, analysis, time)
    for file in outputs_path.iterdir():
        filename = file.name
        if filename.startswith(targ_key):
            targ_name = file.stem[len(targ_key):]
            data = pd.read_hdf(outputs_path / filename)
            targets[targ_name] = data
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
    priors_df.columns = priors_df.columns.str.capitalize()
    priors_df = priors_df.rename(columns={"Sd": "SD"})
    return priors_df


def get_multianalysis_ind_spaghetti(country, indicator, analysis_times):
    out_dfs = [pd.read_hdf(get_output_dir(country, k, v) / "spaghetti.h5")[indicator] for k, v in analysis_times.items()]
    return pd.concat(out_dfs, keys=analysis_times.keys(), axis=1)


def get_multianalysis_procvals(country, analysis_times):
    out_dfs = [pd.read_hdf(get_output_dir(country, k, v) / "updates.h5") for k, v in analysis_times.items()]
    return pd.concat(out_dfs, keys=analysis_times.keys(), axis=1)


def get_multianalysis_likelihoods(country, analysis_times):
    out_dfs = [pd.read_hdf(get_output_dir(country, k, v) / "likelihood.h5") for k, v in analysis_times.items()]
    return pd.concat(out_dfs, keys=analysis_times.keys(), axis=1)


def melt_df_except_first_level(df):
    cols = set(df.columns.get_level_values(0))
    return pd.concat([df[c].melt()["value"] for c in cols], axis=1, keys=cols)


def get_multianalysis_procvals_from_idatas(idatas, ref_analysis="no_mob"):
    n_proc_vals = idatas[ref_analysis].posterior["proc"].shape[-1]
    n_chains = idatas[ref_analysis].posterior.chain.size
    multianalysis_proc_df = pd.DataFrame(columns=pd.MultiIndex.from_product([idatas.keys(), range(n_proc_vals), range(n_chains)]))
    for a in idatas.keys():
        idata = idatas[a]
        proc_vals = np.swapaxes(idata.posterior["proc"].to_numpy(), 0, 1)
        multianalysis_proc_df[a] = get_df_from_3darray(proc_vals, [0, 2, 1])
    return multianalysis_proc_df


def get_multianalysis_dispvals_from_idatas(idatas, ref_analysis="no_mob"):
    n_chains = idatas[ref_analysis].posterior.chain.size
    multianalysis_disp_df = pd.DataFrame(columns=pd.MultiIndex.from_product([idatas.keys(), range(n_chains)]))
    for a in idatas.keys():
        idata = idatas[a]
        multianalysis_disp_df[a] = pd.DataFrame(np.swapaxes(idata.posterior["dispersion_proc"].to_numpy(), 0, 1))
    return multianalysis_disp_df


def store_outputs(
    country: str, 
    mob_analysis_type: str,
    analysis_time: str,
    model: MultiStrainModel,
    calib: StandardCalib,
    mcmc: infer.MCMC,
    n_samples=50,
):
    """Store model and calibration characteristics and results in standard formats.

    Args:
        country: Name of the country of interest
        mob_analysis_type: Mobility analysis type
        analysis_time: Time that the calibration was started
        model: Renewal model
        calib: Calibration object
        mcmc: MCMC object
        n_samples: Number of samples to extract for spaghetti
    """
    out_dir = get_output_dir(country, mob_analysis_type, analysis_time)
    idata = az.from_dict(mcmc.get_samples(True))
    idata.to_netcdf(out_dir / "idata.nc")
    idata_sampled = az.extract(idata, num_samples=n_samples)
    sample_params = esamp.xarray_to_sampleiterator(idata_sampled)
    spaghetti = get_spagh_df_from_dict(get_spaghetti(calib, sample_params))
    spaghetti.to_hdf(out_dir / "spaghetti.h5", key="spaghetti")
    updates = pd.DataFrame(sample_params.components["proc"], columns=model.epoch.index_to_dti(model.x_proc_vals)).T
    updates.to_hdf(out_dir / "updates.h5", key="updates")
    likelihood = pd.DataFrame(mcmc.get_extra_fields(True)["potential_energy"]).T
    pickle.dump(calib.sampled_params, open(out_dir / "priors.pkl", "wb"))
    likelihood.to_hdf(out_dir / "likelihood.h5", key="likelihood")
    for t, target in calib.targets.items():
        target.data.to_hdf(out_dir / f"target_{t}.h5", key=t)
    pd.Series(model.mobility).to_hdf(out_dir / "mobility.h5", key="mobility")
