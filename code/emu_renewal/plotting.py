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
import os
import yaml as yml

from emu_renewal.inputs import DATA_PATH
from emu_renewal.inputs import (
    get_google_mobility,
    get_apple_mobility,
    get_fb_mobility,
    ANALYSIS_TYPES,
)
from emu_renewal.calibration import StandardCalib
from emu_renewal.utils import get_param_dim

plt.style.use("ggplot")

ANALYSIS_NAMES = {
    "no_mob": "no mobility",
    "g_mob": "Google mobility",
    "fb_mob": "Facebook mobility",
    "a_mob": "Apple mobility",
}
AN_ABBREVS = {
    "no_mob": "none",
    "g_mob": "Google",
    "fb_mob": "Facebook",
    "a_mob": "Apple",
}
TARGET_TYPES = {
    "weekly_cases": "weekly cases",
    "weekly_deaths": "weekly deaths",
    "weekly_admissions": "weekly admissions",
    "occupancy": "hospital occupancy",
    "icu_weekly_admissions": "ICU weekly admissions",
    "icu_occupancy": "ICU occupancy",
    "prop_alpha": "proportion Alpha",
    "prop_delta": "proportion Delta",
    "prop_ba2": "proportion BA.2",
    "prop_ba5": "proportion BA.5",
    "seropos": "seroprevalence",
}
VAR_NAME_MAP = {
    "start": "starting strain",
    "alpha": "Alpha",
    "delta": "Delta",
    "ba2": "BA.2",
    "ba5": "BA.5",
}
MOB_COLOURS = {
    "no_mob": "black",
    "g_mob": "green",
    "fb_mob": "blue",
    "a_mob": "red",
}
MOB_DOMAIN_MAP = {
    "retail_and_recreation": "g_mob",
    "grocery_and_pharmacy": "g_mob",
    "parks": "g_mob",
    "transit_stations": "g_mob",
    "workplaces": "g_mob",
    "residential": "g_mob",
    "driving": "a_mob",
    "transit": "a_mob",
    "walking": "a_mob",
    "": "fb_mob",
}
MOB_SOURCE_MAP = {
    "g_mob": "Google",
    "fb_mob": "Facebook",
    "a_mob": "Apple",
}


def get_standard_subplot(n_subplots, n_cols):
    n_rows = int(np.ceil(n_subplots / n_cols))
    height = min([1.0 + n_rows * 2.5, 13])  # Ceiling stops Quarto adding blank pages
    return plt.subplots(n_rows, n_cols, figsize=[12, height])


def plot_analysis_fit(
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


def plot_multianalysis_fit(
    iso3: str,
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
    country = pycountry.countries.lookup(iso3).name
    msg = ".*axis already has a converter set*"
    warnings.filterwarnings("ignore", message=msg)
    pd.options.plotting.backend = "matplotlib"
    n_analyses = len(spaghs)
    n_targs = len(targets)
    ordered_analyses = [a for a in ANALYSIS_TYPES if a in spaghs]
    ordered_targets = [t for t in TARGET_TYPES if t in targets]
    fig, axes = plt.subplots(n_targs, n_analyses, figsize=[12, 13], sharey="row")
    fig.suptitle(f"Fit to data, {country}", fontsize=20, y=1.0)
    for a, analysis in enumerate(ordered_analyses):
        a_spaghs = spaghs[analysis]
        for o, out in enumerate(ordered_targets):
            ax = axes[o, a]
            a_spaghs[out].plot(ax=ax, legend=False, color="black", linewidth=0.15, alpha=0.5)
            target = targets[out]
            ax.plot(target.index, target, linewidth=0.0, marker=".")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
            if o == 0:
                ax.set_title(ANALYSIS_NAMES[analysis], fontsize=15)
            if a == 0:
                ax.set_ylabel(TARGET_TYPES[out], fontsize=15)
            ymax = ax.get_ylim()[1]
            targ_max = max(targets[out]) * 1.5
            if ymax > targ_max and out != "seropos" and "prop_" not in out:
                ylim = min([ymax, targ_max])
                ax.set_ylim(-ylim * 0.05, ylim)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05)
    plt.close()
    return fig


def plot_prior_post(
    idata: az.InferenceData,
    req_vars: List[str],
    priors: List[dist.Distribution],
    iso3: str,
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
    country = pycountry.countries.lookup(iso3).name
    grid = req_grid if req_grid else [1, len(req_vars)]
    size = req_size if req_size else None
    fig = az.plot_density(idata, var_names=req_vars, shade=0.3, grid=grid, figsize=size)
    for ax in fig.ravel():
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
    ax.figure.suptitle(country, fontsize=30, y=1.0)
    return ax.figure.tight_layout()


def plot_prior_multipost(
    idatas: Dict[str, az.InferenceData],
    n_cols: int,
    priors: Dict[str, dist.Distribution],
    var_names: List[str],
    iso3: str,
):
    """Plot comparison of parameter prior distribution
    to posterior from each mobility analysis type.

    Args:
        idatas: The calibration results for each analysis type
        n_cols: Number of columns for figure
        priors: The prior distributions
        var_names: Names of the variables to plot
        iso3: Country identifier
    """

    # Preparation
    country = pycountry.countries.lookup(iso3).name
    idata = idatas["no_mob"]
    prior_info = get_flat_priors()
    params = [p for p in prior_info if "proc" not in p and p in idata.posterior]
    n_params = sum([get_param_dim(p, idata) for p in params])
    n_rows = int(np.ceil(n_params / n_cols))
    height = 2.0 + n_rows * 2.5

    # Plotting
    fig, ax = plt.subplots(n_rows, n_cols, figsize=[12, height])
    fig.suptitle(f"Prior posterior comparison, {country}", fontsize=20, y=1.0)
    axes = ax.ravel()
    n_ax = 0
    for p in params:

        # Posteriors
        for a in idatas:
            post = idatas[a].posterior[p]
            az.plot_density(post, ax=axes[n_ax:], hdi_prob=0.99, colors=[MOB_COLOURS[a]])

        # Prior
        p_dim = get_param_dim(p, idata)
        for d in range(p_dim):
            axis = axes[n_ax]
            x_vals = np.linspace(*axis.get_xlim(), 100)
            y_vals = get_prior_vals_from_dist(x_vals, priors[p], d)
            axis.fill_between(x_vals, y_vals, color="k", alpha=0.2)
            display_name = get_param_display_name(p, p_dim, d, var_names, prior_info)
            axis.set_title(display_name)
            plt.setp(axis.xaxis.get_majorticklabels(), fontsize=10)
            n_ax += 1

    # Suppress unused axes
    for a in range(n_ax, len(axes)):
        axes[a].set_axis_off()

    fig.tight_layout()
    plt.close()
    return fig


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


def plot_beta_priors(
    priors,
) -> plt.figure:
    """Plot the beta-distributed priors.

    Args:
        priors: The raw priors dictionary

    Returns:
        The plot
    """
    beta_vals = priors["beta"].values()
    beta_priors = {v["param_name"]: dist.Beta(v["alpha"], v["beta"]) for v in beta_vals}
    fig, axes = plt.subplots(2, 2)
    flat_axes = axes.ravel()
    for i, dist_name, distri in [[i, d[0], d[1]] for i, d in enumerate(beta_priors.items())]:
        upper_lim = distri.icdf(0.999) if distri.icdf(0.999) < 0.3 else 1.0
        x_vals = np.linspace(0.0, upper_lim, 100)
        ax = flat_axes[i]
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


def plot_proc_comparison(
    procs: Dict[str, pd.DataFrame],
    countries: List[str],
    cont_name: str,
    path: Path,
    n_cols: int,
) -> plt.Figure:
    """Plot the comparison of the variable processes
    across analysis types.

    Args:
        procs: Variable process data
        countries: Names of the countries
        cont_name: Name of the continent considered
        path: Path to the analyses
    """
    fig, axes = get_standard_subplot(len(countries), n_cols)
    title = f"Comparisons of variable process scaling under each mobility assumption, {cont_name}"
    fig.suptitle(title, fontsize=15)
    flat_axes = axes.ravel()
    for c, iso3 in enumerate(countries):
        country = pycountry.countries.lookup(iso3).name
        ax = flat_axes[c]
        ax.set_title(country)
        analyses = [i.parts[-1] for i in (path / iso3).iterdir() if i.is_dir()]
        for a in analyses:
            colour = MOB_COLOURS[a]
            quants = procs[iso3][a].quantile([0.05, 0.5, 0.95], axis=1).T
            ax.plot(quants.index, quants[0.5], color=colour, label=AN_ABBREVS[a], linewidth=2.0)
            ax.fill_between(quants.index, quants[0.05], quants[0.95], alpha=0.2, color=colour)
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)

    # Switch off unused axes
    for ax in flat_axes[c + 1 :]:
        ax.set_axis_off()

    fig.tight_layout()
    plt.close()
    return fig


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
    title: str,
    alpha: float = 0.1,
):
    """Plot the comparison of the kernel density of some
    repeatedly sampled quantity (posterior or parameter)
    for each analysis type by country.

    Args:
        data: The values of interest for each country
        title: Title to go above the whole figure
        alpha: Depth of the shading of the patches
    """
    fig, axes = get_standard_subplot(len(data), 4)
    fig.suptitle(title, fontsize=15)
    flat_axes = axes.ravel()
    for c, (country, c_likes) in enumerate(data.items()):
        c_likes = c_likes.rename(columns=AN_ABBREVS)
        country_name = pycountry.countries.lookup(country).name
        ax = flat_axes[c]
        ax.set_title(country_name)
        colours = [MOB_COLOURS[a] for a in data[country].columns]
        sns.kdeplot(c_likes, fill=True, ax=ax, palette=colours, alpha=alpha)
        ax.set_yticks([])
        ax.set_ylabel("")

    # Switch off unused axes
    for ax in flat_axes[c + 1 :]:
        ax.set_axis_off()

    fig.tight_layout()
    plt.close()
    return fig


def plot_mob_weights_by_country(job_path, mob_type, normalise=False):
    fig, axes = plt.subplots(4, 4, figsize=[10, 10])
    flat_axes = axes.ravel()
    countries = ls(job_path)
    for c, country in enumerate(countries):
        c_path = job_path / country
        country_name = pycountry.countries.lookup(country).name
        idata = az.from_netcdf(c_path / f"{mob_type}/idata_filtered.nc")
        mob_weights = idata.posterior["mob_weights"].to_dataframe().unstack("mob_weights_dim_0")
        if normalise:
            mob_weights = mob_weights.div(mob_weights.sum(axis=1), axis=0)
        if mob_type == "g_mob":
            mob_columns = get_google_mobility(country).columns
        elif mob_type == "a_mob":
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


def get_prior_vals_from_dist(
    eval_points: List[float],
    dist: dist.Distribution,
    i_element: int,
) -> np.array:
    """Get the densities of a distribution,
    accounting for it potentially having multiple elements.

    Args:
        eval_points: Points at which to evaluate
        dist: Distribution
        i_element: The element of the distribution

    Returns:
        The values
    """
    multi_dist = len(dist.batch_shape) > 0
    logp = dist.log_prob
    log_vals = logp(eval_points[:, None])[:, i_element] if multi_dist else logp(eval_points)
    return np.exp(log_vals)


def get_flat_priors() -> Dict[str, str]:
    """Get the prior information,
    but without the top level of the dictionary,
    so that the names of all the priors are the first dictionary level.

    Returns:
        The flattened prior information
    """
    loaded_priors = yml.safe_load(open(DATA_PATH / "config/priors.yml", "r"))
    flat_priors = {}
    for v in loaded_priors.values():
        flat_priors.update(v)
    return flat_priors


def get_param_display_name(
    param: str,
    param_dim: int,
    param_idx: int,
    var_names: List[str],
    prior_info: Dict[str, str],
) -> str:
    """Get parameter name, accounting for parameters
    that are vectors over multiple strains.

    Args:
        param: Parameter name
        param_dim: Number of elements to parameter
        param_idx: Index of this parameter element
        var_names: Names of the modelled variants
        prior_info: Output of get_flat_priors

    Returns:
        The formatted parameter name
    """
    var_idx = param_idx + 1 if param == "relinfect" else param_idx
    var_ext = "" if param_dim == 1 else f", {VAR_NAME_MAP[var_names[var_idx]]}"
    return prior_info[param]["short_name"] + var_ext


def compare_proc_mob(
    job_path: Path,
    countries: List[str],
    n_cols: int,
    mob_type: str,
) -> plt.Figure:
    """Plot comparison of variable process to mobility domain.

    Args:
        job_path: Path for the runs
        countries: Requested countries to plot
        n_cols: Number of subplot columns for the figure
        mob_type: The name of the mobility domain (from MOB_DOMAIN_MAP above)

    Returns:
        The figure
    """
    fig, axes = get_standard_subplot(len(countries), n_cols)
    mob_source = MOB_DOMAIN_MAP[mob_type]
    title = (
        f"Modelled variable process (with no mobility scaling) "
        f"versus {mob_type.replace('_', ' ')} "
        f"{MOB_SOURCE_MAP[MOB_DOMAIN_MAP[mob_type]]} mobility data"
    )
    fig.suptitle(title, fontsize=14, y=1.0)
    flat_axes = axes.ravel()
    for c, iso3 in enumerate(countries):
        ax = flat_axes[c]
        country = pycountry.countries.lookup(iso3).name
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)

        # Variable process plotting
        proc_samples = pd.read_hdf(job_path / iso3 / "no_mob/spaghetti.h5")["process"]
        centiles = proc_samples.quantile([0.05, 0.5, 0.95], axis=1).T
        ax.plot(centiles.index, centiles[0.5], label="process", color="navy")
        ax.fill_between(centiles.index, centiles[0.05], centiles[0.95], alpha=0.2, color="navy")

        # Mobility overlay
        try:
            if mob_source == "g_mob":
                mob = get_google_mobility(iso3)[mob_type]
            elif mob_source == "fb_mob":
                mob = get_fb_mobility(iso3)
            elif mob_source == "a_mob":
                mob = get_apple_mobility(iso3)[mob_type]
            mobility = mob.loc[(centiles.index[0] < mob.index) & (mob.index < centiles.index[-1])]
            smoothed_mob = mobility.rolling(7, center=True).mean().dropna()
            ax.plot(smoothed_mob.index, smoothed_mob, color=MOB_COLOURS[mob_source])
            ax.set_title(country)
        except:
            ax.set_title(f"{country} (data unavailable)")

    # Switch off unused axes
    for ax in flat_axes[c + 1 :]:
        ax.set_axis_off()

    fig.tight_layout()
    plt.close()
    return fig
