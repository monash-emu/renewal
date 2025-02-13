from typing import List, Dict
import numpy as np
from random import choice
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import arviz as az
from numpyro import distributions as dist
from matplotlib import pyplot as plt
from plotly.express.colors import qualitative as qual_colours
from matplotlib.pyplot import cm
from datetime import timedelta

from emu_renewal.calibration import StandardCalib
from emu_renewal.utils import map_dict
from emu_renewal.outputs import get_col_abs_dist_from_mean


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
        vertical_spacing=0.02,
        horizontal_spacing=0.05,
        subplot_titles=out_req,
    ).update_layout(height=300*len(out_req), width=800, showlegend=False)
    out_style = {"color": "black", "width": 0.5}
    targ_style = {"color": "red"}
    for o, out in enumerate(out_req):
        for col in spaghetti[out].columns:
            line = go.Scatter(x=spaghetti.index, y=spaghetti[out][col], line=out_style)
            fig.add_trace(line, row=o+1, col=1)
        if out in calib_data:
            target = calib_data[out]
            target_scatter = go.Scatter(x=target.index, y=target, mode="markers", line=targ_style)
            fig.add_trace(target_scatter, row=o+1, col=1)
    return fig


def plot_post_prior_comparison(
    idata: az.InferenceData,
    req_vars: list[str],
    priors: list[dist.Distribution],
    req_grid=None,
    req_size=None,
) -> plt.figure:
    """Plot comparison of model posterior outputs against priors.

    Args:
        idata: Arviz inference data from calibration
        req_vars: User-requested variables to plot
        priors: Numpyro prior objects
        req_grid: Requested subplot dimensions

    Returns:
        The figure object
    """
    grid = req_grid if req_grid else [1, len(req_vars)]
    size = req_size if req_size else None
    plot = az.plot_density(idata, var_names=req_vars, shade=0.3, grid=grid, figsize=size)
    for i_ax, ax in enumerate(plot.ravel()):
        ax_limits = ax.get_xlim()
        x_vals = np.linspace(*ax_limits, 50)
        y_vals = np.exp(priors[req_vars[i_ax]].log_prob(x_vals))
        y_vals *= ax.get_ylim()[1] / max(y_vals)
        ax.fill_between(x_vals, y_vals, color="k", alpha=0.2, linewidth=2)
    return plot


def plot_imm_props(
    spaghetti: pd.DataFrame, 
    n_strains: int,
) -> go.Figure:
    """Plot susceptible population proportions from randomly selected run.

    Args:
        spaghetti: Spaghetti
        n_strains: Number of modelled strains

    Returns:
        Figure
    """
    spagh = spaghetti[[f"sus_{i}" for i in range(2 ** n_strains)]]
    spagh.columns = spagh.columns.swaplevel()
    runs = list(set(spagh.columns.get_level_values(0)))
    return spagh[choice(runs)].plot.area()


def plot_process_comparison(spaghetti, analysis_names, colours, linewidth=0.2):
    fig, ax = plt.subplots(figsize=[12, 8])
    for i, analysis in enumerate(analysis_names):
        plot_data = spaghetti[analysis]
        for l, line in enumerate(plot_data.columns):
            label = analysis if l == 0 else ""
            ax.plot(spaghetti.index, plot_data[line], color=colours[i], alpha=0.5, linewidth=linewidth, label=label)
    return fig


def plot_updates_comparison(updates, analysis_times, colours, jitter_days=1.0):
    fig, ax = plt.subplots(figsize=[9, 5])
    for i, analysis in enumerate(analysis_times):
        adj = jitter_days if i == 0 else -jitter_days
        for run in updates[analysis].columns:
            ax.scatter(updates.index + timedelta(adj), updates[analysis, run], color=colours[i], alpha=0.2)


def plot_beta_priors(all_priors):
    beta_priors = {v["param_name"]: dist.Beta(v["alpha"], v["beta"]) for v in all_priors["beta"].values()}
    fig, axes = plt.subplots(2, 2)
    for i, dist_name, distri in [[i, d[0], d[1]] for i, d in enumerate(beta_priors.items())]:
        upper_lim = distri.icdf(0.999) if distri.icdf(0.999) < 0.3 else 1.0
        x_vals = np.linspace(0.0, upper_lim, 100)
        ax = axes.ravel()[i]
        ax.plot(x_vals, np.exp(distri.log_prob(x_vals)))
        ax.set_title(dist_name, size=12)
        ax.set_yticks([])
    return fig.tight_layout()


def plot_progress_priors(priors, xmax, leg=True):
    fig, axes = plt.subplots(2, 1)
    x_vals = np.linspace(0.0, xmax, 1000)
    for k, v in priors.items():
        row = 0 if "mean" in k else 1
        label = k.split("_")[0] if row == 0 else None
        y_vals = np.exp(v.log_prob(x_vals))
        axes[row].plot(x_vals, y_vals / max(y_vals), label=label)
    axes[0].set_title("Mean", size=12)
    axes[0].set_yticks([])
    axes[1].set_title("SD", size=12)
    axes[1].set_yticks([])
    if leg:
        fig.legend()
    return fig.tight_layout()


def plot_mob_update_comparison(idatas, xlim, fig_height=8):
    az_plots = {}
    for k, idata in idatas.items():
        az_plots[k] = az.plot_posterior(idata, var_names=["proc"])
        plt.close()
    fig, axes = plt.subplots(len(idatas), 1, figsize=(10, fig_height), sharex=True)
    n_proc_vals = idatas["no_mob"].posterior["proc"]["proc_dim_0"].shape[0]
    colours = cm.rainbow(np.linspace(0.0, 1.0, n_proc_vals))
    for an, analysis in enumerate(idatas):
        for a, ax in enumerate(az_plots[analysis].flatten()):
            line = ax.lines[0]
            axes[an].plot(line.get_xdata(), line.get_ydata(), color=colours[a], linewidth=0.4)
            axes[an].set_title(analysis)
        axes[an].set_xlim([-xlim, xlim])
    fig.tight_layout()
    plt.close()
    return fig
