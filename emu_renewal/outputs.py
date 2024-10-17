import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import arviz as az
from numpyro import distributions as dist
from matplotlib import pyplot as plt
from jax import jit
from plotly.express.colors import qualitative as qual_colours

from estival.sampling.tools import SampleIterator

from emu_renewal.renew import RenewalModel
from emu_renewal.calibration import StandardCalib
from emu_renewal.utils import map_dict

PANEL_SUBTITLES = ["cases", "susceptibles", "R", "transmission potential"]
MARGINS = {m: 20 for m in ["t", "b", "l", "r"]}


def get_spaghetti(
    calib: StandardCalib,
    params: SampleIterator,
) -> pd.DataFrame:
    """Run parameters through the model to get outputs.

    Args:
        calib: The calibration object, which includes the model
        params: The parameter sets to feed through the model

    Returns:
        Dataframe with index of model times and multiindexed columns,
            with first level being the output name and second the parameter set
            by chain and iteration
    """
    model = calib.epi_model
    times = model.epoch.index_to_dti(model.model_times)

    @jit
    def get_full_result(**params):
        return model.renewal_func(**params | calib.fixed_params)

    # Get spaghetti for each output in a dictionary
    spagh_dict = {}
    for i, p in params.iterrows():
        res = get_full_result(**{k: v for k, v in p.items() if "dispersion" not in k})
        spagh = pd.DataFrame(np.array(res)).T
        spagh.index = times
        spagh.columns = res._fields
        spagh_dict[str(i)] = spagh

    # Wrangle into a dataframe with the desired format
    column_names = pd.MultiIndex.from_product([params.index.map(str), res._fields])
    spaghetti = pd.DataFrame(columns=column_names)
    for i in spagh_dict:
        spaghetti[i] = spagh_dict[i]
    spaghetti.columns = spaghetti.columns.swaplevel()
    return spaghetti.sort_index(axis=1, level=0)


def get_quant_df_from_spaghetti(
    spaghetti: pd.DataFrame,
    quantiles: list[float],
) -> pd.DataFrame:
    """Calculate requested quantiles over spaghetti created
    in previous function.

    Args:
        spaghetti: The output of get_spaghetti
        quantiles: The quantiles at which to make the calculations

    Returns:
        Dataframe with index of model times and multiindexed columns,
            with first level being the output name and second the quantile
    """
    outputs = set(spaghetti.columns.get_level_values(0))
    column_names = pd.MultiIndex.from_product([outputs, quantiles])
    quantiles_df = pd.DataFrame(index=spaghetti.index, columns=column_names)
    for col in outputs:
        quantiles_df[col] = spaghetti[col].quantile(quantiles, axis=1).T
    return quantiles_df


def get_standard_four_subplots() -> go.Figure:
    """Get a figure object with standard formatting of 2 x 2 subplots.

    Returns:
        The figure object
    """
    return make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        subplot_titles=PANEL_SUBTITLES,
    )


def plot_spaghetti(
    spaghetti: pd.DataFrame,
    targets: pd.Series,
) -> go.Figure:
    """Plot the outputs of the function that gets
    spaghetti outputs from parameters above.

    Args:
        spaghetti: The output of get_spaghetti
        targets: The target values of the calibration algorithm

    Returns:
        The figure object
    """
    fig = get_standard_four_subplots()
    fig.add_trace(go.Scatter(x=targets.index, y=targets, mode="markers"), row=1, col=1)
    for i in range(4):
        fig.add_traces(spaghetti[PANEL_SUBTITLES[i]].plot().data, rows=i // 2 + 1, cols=i % 2 + 1)
    return fig.update_layout(margin=MARGINS, height=600).update_yaxes(rangemode="tozero")


def get_area_from_df(
    df: pd.DataFrame,
    columns: list[float],
    colour: str,
) -> go.Scatter:
    """Get a patch object to add to a plotly graph from a dataframe
    that contains data for the upper and lower margins.

    Args:
        df: The data
        columns: The names of the columns containing the upper and lower margins
        colour: The colour request

    Returns:
        The patch
    """
    x_vals = df.index.to_list() + df.index[::-1].to_list()
    y_vals = df[columns[0]].to_list() + df[columns[1]][::-1].to_list()
    return go.Scatter(x=x_vals, y=y_vals, line={"width": 0.0, "color": colour}, fill="toself")


def add_ci_patch_to_plot(
    fig: go.Figure,
    df: pd.DataFrame,
    colour: str,
    row: int,
    col: int,
    quantiles: list[float],
):
    """Add a median line and confidence interval patch to a plotly figure object.

    Args:
        fig: The figure object
        df: The data to plot
        colour: The colour request
        row: The row of the subplot figure
        col: The column of the subplot figure
    """
    x_vals = df.index.to_list() + df.index[::-1].to_list()
    fig.add_trace(get_area_from_df(df, columns=quantiles, colour=colour), row=row, col=col)
    fig.add_trace(go.Scatter(x=x_vals, y=df[0.5], line={"color": colour}), row=row, col=col)


def plot_uncertainty_patches(
    calib: StandardCalib,
    quantile_df: list[float],
    req_outputs: list[str],
) -> go.Figure:
    """Create the main uncertainty output figure for a renewal analysis.

    Args:
        quantile_df: Output of get_quant_df_from_spaghetti
        targets: The target values of the calibration algorithm
        colour: The colour requests
        req_outputs: The names of the outputs of interest

    Returns:
        The figure object
    """
    colours = qual_colours.Plotly
    fig = get_standard_four_subplots()
    avail_quants = list(set(quantile_df.columns.get_level_values(1)))
    patch_lims = [avail_quants[0], avail_quants[-1]]
    for i, out in enumerate(req_outputs):
        row = i // 2 + 1
        col = i % 2 + 1
        add_ci_patch_to_plot(fig, quantile_df[req_outputs[i]], colours[i], row, col, patch_lims)
        if out in calib.targets:
            t = calib.targets[out].data
            fig.add_trace(go.Scatter(x=t.index, y=t, mode="markers"), row=row, col=col)
    fig.update_layout(margin=MARGINS, height=600, showlegend=False).update_yaxes(rangemode="tozero")
    return fig


def plot_3d_spaghetti(
    spaghetti: pd.DataFrame,
    column_req: list[str],
) -> go.Figure:
    """Plot to variables on y and z axes against index
    of a standard spaghetti dataframe.

    Args:
        spaghetti: Output of get_spaghetti
        column_req: The columns to plot against one-another

    Returns:
        The figure object
    """
    fig = go.Figure()
    col_1, col_2 = column_req
    for i in spaghetti.columns.get_level_values(1):
        x_data = spaghetti.index
        y_data = spaghetti[(col_1, i)]
        z_data = spaghetti[(col_2, i)]
        trace = go.Scatter3d(x=x_data, y=y_data, z=z_data, mode="lines", line={"width": 5.0})
        fig.add_trace(trace)
    axis_titles = {"xaxis": {"title": "time"}, "yaxis": {"title": col_1}, "zaxis": {"title": col_2}}
    return fig.update_layout(showlegend=False, scene=axis_titles, height=800)


def plot_post_prior_comparison(
    idata: az.InferenceData,
    req_vars: list[str],
    priors: list[dist.Distribution],
) -> plt.figure:
    """Plot comparison of model posterior outputs against priors.

    Args:
        idata: Arviz inference data from calibration
        req_vars: User-requested variables to plot
        priors: Numpyro prior objects

    Returns:
        The figure object
    """
    plot = az.plot_density(idata, var_names=req_vars, shade=0.3, grid=[1, len(req_vars)])
    for i_ax, ax in enumerate(plot.ravel()):
        ax_limits = ax.get_xlim()
        x_vals = np.linspace(*ax_limits, 50)
        y_vals = np.exp(priors[req_vars[i_ax]].log_prob(x_vals))
        y_vals *= ax.get_ylim()[1] / max(y_vals)
        ax.fill_between(x_vals, y_vals, color="k", alpha=0.2, linewidth=2)
    return plot


def plot_priors(
    priors: list[dist.Distribution],
) -> go.Figure:
    """Plot prior distributions with plotly.

    Args:
        priors: The priors

    Returns:
        The figure object
    """
    fig = make_subplots(1, len(priors), subplot_titles=[map_dict[i] for i in priors.keys()])
    for i, p in enumerate(priors):
        prior = priors[p]
        limit = 0.01 if isinstance(prior, [dist.Uniform | dist.Gamma]) else 0.0
        x_vals = np.linspace(prior.icdf(limit), prior.icdf(1.0 - limit), 50)
        y_vals = np.exp(prior.log_prob(x_vals))
        fig.add_trace(
            go.Scatter(x=x_vals, y=y_vals, name=map_dict[p], fill="tozeroy"), row=1, col=i + 1
        )
    return fig.update_layout(showlegend=False)


def plot_spaghetti_calib_comparison(spaghetti, calib_data):

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
    ).update_layout(height=800, width=800, showlegend=False)
    fig.add_traces(spaghetti["weekly_sum"].plot().data, rows=1, cols=1)
    fig.add_traces(go.Scatter(x=calib_data["weekly_sum"].data.index, y=calib_data["weekly_sum"].data / 7.0, mode="markers"), rows=1, cols=1)
    for col in spaghetti["weekly_deaths"].columns:
        fig.add_trace(go.Scatter(x=spaghetti.index, y=spaghetti["weekly_deaths"][col], line={"color": "black", "width": 0.5}), row=2, col=1)

    fig.add_traces(go.Scatter(x=calib_data["weekly_deaths"].data.index, y=calib_data["weekly_deaths"].data, mode="markers"), rows=2, cols=1)
    fig.add_traces(spaghetti["seropos"].plot().data, rows=3, cols=1)
    fig.add_traces(go.Scatter(x=calib_data["seropos"].data.index, y=calib_data["seropos"].data, mode="markers"), rows=3, cols=1)
    return fig
