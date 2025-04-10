from typing import List, Dict, Tuple
from pathlib import Path
import warnings
import numpy as np
from random import choice
import pandas as pd
import seaborn as sns
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import arviz as az
from numpyro import distributions as dist
from matplotlib import pyplot as plt
import pycountry
from os import listdir as ls

from emu_renewal.inputs import get_google_mobility, get_apple_mobility
from emu_renewal.calibration import StandardCalib


def plot_spaghetti_calib_comparison(
    spaghetti: pd.DataFrame,
    calib_data: StandardCalib,
    out_req: list[str],
) -> go.Figure:
    """Plot model outputs and compare against targets where available.

    Args:
        spaghetti: Output of run_for_spaghetti
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
    ).update_layout(height=300 * len(out_req), width=800, showlegend=False)
    out_style = {"color": "black", "width": 0.5}
    targ_style = {"color": "red"}
    for o, out in enumerate(out_req):
        for col in spaghetti[out].columns:
            line = go.Scatter(x=spaghetti.index, y=spaghetti[out][col], line=out_style)
            fig.add_trace(line, row=o + 1, col=1)
        if out in calib_data:
            target = calib_data[out]
            target_scatter = go.Scatter(x=target.index, y=target, mode="markers", line=targ_style)
            fig.add_trace(target_scatter, row=o + 1, col=1)
    return fig


def plot_post_prior_comparison(
    idata: az.InferenceData,
    req_vars: List[str],
    priors: List[dist.Distribution],
    req_grid=None,
    req_size=None,
) -> plt.figure:
    """Plot comparison of calibration posterior estimates
    for parameters against their prior distributions.

    Args:
        idata: Calibration inference data
        req_vars: Names of the parameters to plot
        priors: Prior distributions for the parameters
        req_grid: Dimensions of the subplot
        req_size: Figure size request

    Returns:
        The figure
    """
    grid = req_grid if req_grid else [1, len(req_vars)]
    size = req_size if req_size else None
    fig = az.plot_density(idata, var_names=req_vars, shade=0.3, grid=grid, figsize=size)
    for i_ax, ax in enumerate(fig.ravel()):
        ax_limits = ax.get_xlim()
        param = ax.title.get_text().split("\n")[0]
        if param:
            x_vals = np.linspace(*ax_limits, 50)
            distri = priors[param]
            if len(distri.batch_shape) == 0:
                y_vals = np.exp(distri.log_prob(x_vals))
            else:
                y_vals = np.exp(distri.log_prob(x_vals[:, None])[:, 0])
            y_vals *= ax.get_ylim()[1] / max(y_vals)
            ax.fill_between(x_vals, y_vals, color="k", alpha=0.2, linewidth=2)
    return ax.figure.tight_layout()


def plot_imm_props(
    spaghetti: pd.DataFrame,
) -> go.Figure:
    """Plot susceptible population proportions from randomly selected run.

    Args:
        spaghetti: Spaghetti

    Returns:
        Figure
    """
    n_strains = len([i for i in set(spaghetti.columns.get_level_values(0)) if "prop_" in i])
    spagh = spaghetti[[f"sus_{i}" for i in range(2**n_strains)]]
    spagh.columns = spagh.columns.swaplevel()
    runs = list(set(spagh.columns.get_level_values(0)))
    return spagh[choice(runs)].plot.area()


def plot_beta_priors(all_priors):
    beta_priors = {
        v["param_name"]: dist.Beta(v["alpha"], v["beta"]) for v in all_priors["beta"].values()
    }
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


def plot_multianalysis_fit(
    country: str,
    targets: Dict[str, pd.Series],
    spaghs: Dict[str, pd.DataFrame],
) -> plt.Figure:
    """Plot the fit of each of the analyses to data
    using spaghetti and calibration targets.

    Args:
        country: Name of the country
        targets: The calibration targets
        spaghs: The spaghettis

    Returns:
        The figure
    """
    msg = ".*axis already has a converter set*"
    warnings.filterwarnings("ignore", message=msg)
    pd.options.plotting.backend = "matplotlib"
    n_targs = len(targets)
    fig, axes = plt.subplots(n_targs, len(spaghs), figsize=[12, 15], sharex=True, sharey="row")
    country_name = pycountry.countries.lookup(country).name
    fig.suptitle(country_name, fontsize=30, y=1.0)
    for a, analysis in enumerate(spaghs):
        a_spaghs = spaghs[analysis]
        for o, out in enumerate(targets):
            ax = axes[o, a]
            o_spagh = a_spaghs[out]
            o_spagh.plot(ax=ax, legend=False, color="black", linewidth=0.15)
            target = targets[out]
            ax.plot(target.index, target, linewidth=0.0, marker=".")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
            if o == 0:
                ax.set_title(analysis, fontsize=22)
            if a == 0:
                ax.set_ylabel(out, fontsize=18)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05)
    return fig


def plot_proc_comparison(
    procs: Dict[str, pd.DataFrame],
    countries: List[str],
    colours: List[tuple],
    title: str,
    path: Path,
):
    """Plot the comparison of the variable processes
    across analysis types.

    Args:
        procs: Variable process data
        countries: Names of the countries
        colours: Colours to use for lines
        title: Title to go above the whole figure
        path: Path to the analyses
    """
    n_rows = 4
    proc_fig, axes = plt.subplots(n_rows, 4, figsize=[10, 10])
    proc_fig.suptitle(title, fontsize=15)
    flat_axes = axes.ravel()
    for c, country in enumerate(countries):
        c_ax = flat_axes[c]
        c_ax.set_title(pycountry.countries.lookup(country).name)
        analyses = ls(path / country)
        for a, analysis in enumerate(analyses):
            quants = procs[country][analysis].quantile([0.05, 0.5, 0.95], axis=1).T
            c_ax.plot(quants.index, quants[0.5], color=colours[a], label=analysis, linewidth=2.0)
            c_ax.fill_between(quants.index, quants[0.05], quants[0.95], alpha=0.2, color=colours[a])
        if c == 0:
            c_ax.legend()
        plt.setp(c_ax.xaxis.get_majorticklabels(), rotation=70)
        if c_ax.get_subplotspec().rowspan.stop != n_rows:
            c_ax.set_xticklabels([])
        c_ax.set_yticks([])
    proc_fig.tight_layout()
    proc_fig.savefig("proc_fig.svg")


def get_param_medians(
    param_vals: Dict[str, pd.DataFrame],
    countries: List[str],
) -> pd.DataFrame:
    """Get median values for a parameter
    for presentation as a table.

    Args:
        param_vals: The parameter values by country
        countries: Names of the countries

    Returns:
        The formatted table
    """
    medians = pd.DataFrame()
    for country in countries:
        medians[country] = param_vals[country].median()
    to_country_name = lambda c: pycountry.countries.lookup(c).name
    medians = medians.rename(columns=to_country_name)
    return medians.T


def plot_kde_comparison(
    data: Dict[str, pd.DataFrame],
    colours: Tuple[tuple],
    title: str,
    filename: str,
    alpha: float = 0.1,
):
    """Plot the comparison of the kernel density of some
    repeatedly sampled quantity (posterior or parameter)
    for each analysis type by country.

    Args:
        data: The values of interest for each country
        colours: The colours for shading (to allow consistency between plots)
        title: Title to go above the whole figure
        filename: Filename stem for saving
        alpha: Depth of the shading of the patches
    """
    kde_fig, axes = plt.subplots(4, 4, figsize=[10, 10])
    kde_fig.suptitle(title, fontsize=15)
    flat_axes = axes.ravel()
    for c, (country, c_likes) in enumerate(data.items()):
        country_name = pycountry.countries.lookup(country).name
        c_ax = flat_axes[c]
        c_ax.set_title(country_name)
        sns.kdeplot(c_likes, fill=True, ax=c_ax, palette=colours, alpha=alpha)
        c_ax.set_yticks([])
        c_ax.set_ylabel("")
        if c != 0:
            flat_axes[c].get_legend().remove()
    kde_fig.tight_layout()
    kde_fig.savefig(f"{filename}_fig.svg")


def plot_mob_weights_by_country(job_path, mob_type, normalise=False):
    fig, axes = plt.subplots(4, 4, figsize=[10, 10])
    flat_axes = axes.ravel()
    countries = ls(job_path)
    for c, country in enumerate(countries):
        c_path = job_path / country
        country_name = pycountry.countries.lookup(country).name
        idata = az.from_netcdf(c_path / f"weighted_{mob_type}_1exp/idata_filtered.nc")
        mob_weights = idata.posterior["mob_weights"].to_dataframe().unstack("mob_weights_dim_0")
        if normalise:
            mob_weights = mob_weights.div(mob_weights.sum(axis=1), axis=0)
        if mob_type == "google":
            mob_columns = get_google_mobility(country).columns
        elif mob_type == "apple":
            mob_columns = get_apple_mobility(country).columns
        else:
            raise ValueError("unavailable mobility type request")
        mob_weights.columns = mob_columns
        c_ax = flat_axes[c]
        sns.kdeplot(mob_weights, fill=True, alpha=0.1, linewidth=1.5, ax=c_ax)
        c_ax.set_yticks([])
        c_ax.set_ylabel("")
        c_ax.set_title(country_name)
        c_ax.get_legend().set_title("")
        if country != "FIN" and mob_type == "google":
            c_ax.get_legend().remove()
    fig.tight_layout()
    fig.savefig("mob_fig.svg")


def compare_proc_versus_mobility(proc_centiles, mob_types, mob_source="google"):
    mob_comparison_fig, axes = plt.subplots(4, 4, figsize=[15, 15], sharex=True)
    flat_axes = axes.ravel()
    for c, country in enumerate(proc_centiles):
        c_ax = flat_axes[c]
        country_name = pycountry.countries.lookup(country).name
        c_ax.set_title(country_name)
        centiles = proc_centiles[country]
        c_ax.plot(centiles.index, centiles[0.5], label="process", color="navy")
        c_ax.fill_between(centiles.index, centiles[0.05], centiles[0.95], alpha=0.2, color="navy")
        if mob_source == "google":
            mob = get_google_mobility(country)
        elif mob_source == "apple":
            mob = get_apple_mobility(country)
        mobility = mob.loc[mob.index < centiles.index[-1]]
        for mob_type in mob_types:
            c_ax.plot(mobility.index, mobility[mob_type], label=mob_type)
        if country == "FIN":
            c_ax.legend()
        plt.setp(c_ax.xaxis.get_majorticklabels(), rotation=70)
    mob_comparison_fig.tight_layout()
    return mob_comparison_fig
