import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import arviz as az
from numpyro import distributions as dist
from matplotlib import pyplot as plt
from plotly.express.colors import qualitative as qual_colours

from emu_renewal.calibration import StandardCalib
from emu_renewal.utils import map_dict

MARGINS = {m: 20 for m in ["t", "b", "l", "r"]}


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
        subplot_titles=["cases", "susceptibles", "R", "transmission potential"],
    )


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


def plot_spaghetti_calib_comparison(
    spaghetti: pd.DataFrame, 
    calib_data: StandardCalib,
    out_req: list[str],
) -> go.Figure:
    """Plot model outputs and compare against targets where available.

    Args:
        spaghetti: Output of get_spaghetti
        calib_data: _description_
        out_req: _description_

    Returns:
        The figure
    """
    fig = make_subplots(
        rows=len(out_req),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
    ).update_layout(height=300*len(out_req), width=800, showlegend=False)
    out_style = {"color": "black", "width": 0.5}
    targ_style = {"color": "red"}
    for o, out in enumerate(out_req):
        for col in spaghetti[out].columns:
            line = go.Scatter(x=spaghetti.index, y=spaghetti[out][col], line=out_style)
            fig.add_trace(line, row=o+1, col=1)
        if out in calib_data:
            target = calib_data[out].data
            target_scatter = go.Scatter(x=target.index, y=target, mode="markers", line=targ_style)
            fig.add_trace(target_scatter, row=o+1, col=1)
    return fig
