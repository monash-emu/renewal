from pathlib import Path
import numpy as np
import pandas as pd
from jax import jit
from typing import List, Dict
import matplotlib.pyplot as plt
import arviz as az
import pickle
from numpyro import infer
from datetime import datetime
import pycountry

from estival.sampling.tools import SampleIterator
from estival.sampling import tools as esamp

from emu_renewal.inputs import OUTPUTS_PATH, DATE_FORMAT, ANALYSIS_TYPES
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
        spagh = pd.DataFrame(get_full_result(**epi_params))
        spagh.index = times
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
    outputs = list(spagh_dict.values())[0].columns  # Arbitrarily chosen output dataframe
    column_names = pd.MultiIndex.from_product([spagh_dict.keys(), outputs])
    spaghetti = pd.DataFrame(columns=column_names)
    for i in spagh_dict:
        spaghetti[i] = spagh_dict[i]
    spaghetti.columns = spaghetti.columns.swaplevel()
    return spaghetti.sort_index(axis=1, level=0)


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


def get_multianalysis_ind_spaghetti(country_path, indicator, analysis_times):
    out_dfs = [pd.read_hdf(country_path / a / analysis_times[a] / "spaghetti.h5")[indicator] for a in ANALYSIS_TYPES]
    return pd.concat(out_dfs, keys=analysis_times.keys(), axis=1)


def get_multianalysis_procvals(country_path, analysis_times):
    proc_dfs__ = [pd.read_hdf(country_path / a / analysis_times[a] / "updates.h5") for a in ANALYSIS_TYPES]
    return pd.concat(proc_dfs__, keys=analysis_times.keys(), axis=1)


def get_multianalysis_likelihoods(country_path, analysis_times):
    out_dfs = [pd.read_hdf(country_path / a / analysis_times[a] / "likelihood.h5") for a in ANALYSIS_TYPES]
    return pd.concat(out_dfs, keys=analysis_times.keys(), axis=1)


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


def load_last_runs_from_path(path):
    spaghs = {}
    targets = {}
    countries = []
    for country_folder in path.iterdir():
        country = pycountry.countries.lookup(country_folder.parts[-1]).alpha_3
        countries.append(country)
        spaghs[country] = {}
        targets[country] = {}
        for analysis_folder in country_folder.iterdir():
            analysis_name = analysis_folder.parts[-1]
            avail_times = [datetime.strptime(d.parts[-1], DATE_FORMAT) for d in analysis_folder.iterdir()]
            last_time = datetime.strftime(max(avail_times), DATE_FORMAT)
            path = country_folder / analysis_folder / last_time
            spaghs[country][analysis_name] = pd.read_hdf(path / "spaghetti.h5")
            target_files = path.glob("target_*.h5")
            for targ in target_files:
                targets[country][targ.parts[-1][7:-3]] = pd.read_hdf(targ)
    return spaghs, targets, countries


def get_latest_analyses(
    country_path: Path,
    analyses: List[str],
) -> Dict[str, str]:
    """Get the most recent analysis time string
    for each of the requested analysis types
    for a particular country.

    Args:
        country: Name of the country
        analyses: The names of the mobility analysis types requested
        date_format: String format to represent the date

    Returns:
        The requested information
    """
    last_analyses = {}
    for analysis in analyses:
        path = country_path / analysis
        dates = [datetime.strptime(d.parts[-1], DATE_FORMAT) for d in path.iterdir()]
        last_analyses[analysis] = datetime.strftime(max(dates), DATE_FORMAT)
    return last_analyses
