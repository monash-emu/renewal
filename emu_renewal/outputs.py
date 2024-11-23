import pandas as pd
import numpy as np
from jax import jit
from matplotlib.pyplot import cm
from matplotlib import pyplot as plt
import arviz as az
from typing import List

from estival.sampling.tools import SampleIterator

from emu_renewal.calibration import StandardCalib


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


def plot_proc_comparison(
    idata_1: az.data.inference_data.InferenceData,
    idata_2: az.data.inference_data.InferenceData,
    panel_titles: List[str],
) -> plt.Figure:
    """Plot comparison of variable process updates
    from two inference data objects.

    Args:
        idata_1: First inference data
        idata_2: Second inference data
        panel_titles: Titles for subplots

    Returns:
        The figure
    """
    n_proc = idata_2.posterior["proc"]["proc_dim_0"].shape[0]
    no_mob_post_plot = az.plot_posterior(idata_1, var_names=["proc"])
    mob_post_plot = az.plot_posterior(idata_2, var_names=["proc"])
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colours = cm.rainbow(np.linspace(0.0, 1.0, n_proc))
    
    # Top panel without mobility
    no_mob_ax = axes[0]
    no_mob_ax.set_title(panel_titles[0])
    for a, ax in enumerate(no_mob_post_plot.flatten()[:n_proc]):
        line = ax.lines[0]
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        no_mob_ax.plot(xdata, ydata, color=colours[a], linewidth=0.4, label=a)
    axes[0].legend(ncol=2)

    # Bottom panel with mobility  
    mob_ax = axes[1]
    mob_ax.set_title(panel_titles[1])
    for a, ax in enumerate(mob_post_plot.flatten()[:n_proc]):
        line = ax.lines[0]
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        mob_ax.plot(xdata, ydata, color=colours[a], linewidth=0.4)
        
    return fig
