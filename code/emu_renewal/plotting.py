from typing import List, Dict
from pathlib import Path
import warnings
from os import listdir as ls
import yaml as yml
import numpy as np
from random import choice
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import arviz as az
from numpyro import distributions as dist
from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots
from matplotlib.cm import ScalarMappable
from matplotlib.pyplot import get_cmap
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import pycountry
import pycountry_convert as pc
from geopandas import GeoDataFrame
from IPython.display import display, Markdown
from plotly import graph_objects as go

from emu_renewal.constants import (
    ANALYSIS_TYPES, 
    ANALYSIS_NAMES, 
    MOB_SOURCE_ABBREVS, 
    MOB_SOURCE_COLOURS,
    MOB_LOCATION_SOURCE_MAP,
    DUR_MIN, 
    DUR_REL_MAX,
    TARGET_TYPES,
    VAR_NAME_MAP,
    INCLUSION_COLOURS,
    MOB_LOCATION_NAME_MAP,
    G_MOB_LOCATION_CMAP,
    MOB_LOCATION_ABBREVS,
    SHORT_COUNTRY_NAMES,
)
from emu_renewal.inputs import (
    DATA_PATH,
    get_google_mobility,
    get_requested_mob,
    get_gdps,
    get_country_pop,
    get_world_shp,
    get_g_mob_weight_posts,
    get_g_mob_quants,
    get_smoothed_trunc_g_mob,
)
from emu_renewal.outputs import get_idatas_for_mob_type, get_median_ratios, get_param_vals_by_analysis
from emu_renewal.utils import get_param_dim, get_beta_params_from_mean_var, get_cont_of_country, get_country_short_name, get_country_name

plt.style.use("ggplot")


def get_standard_subplot(
    n_subplots: int, 
    n_cols: int,
) -> tuple:
    """Get a standard multi-panel figure, axes combination
    that works well with previewing in Quarto.

    Args:
        n_subplots: Total number of panels
        n_cols: Number of columns

    Returns:
        The figure and the axes
    """
    n_rows = int(np.ceil(n_subplots / n_cols))
    height = min([1.0 + n_rows * 2.5, 13])  # Ceiling stops Quarto adding blank pages
    return plt.subplots(n_rows, n_cols, figsize=[12, height])


def plot_analysis_fit(
    spaghetti: pd.DataFrame,
    targets: Dict[str, pd.Series],
    out_req: List[str],
) -> go.Figure:
    """Plot model outputs and compare against targets where available.

    Args:
        spaghetti: Output of run_for_spaghetti
        targets: Calibration targets
        out_req: Names of the outputs for plotting

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
        if out in targets:
            target = targets[out]
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
        line_width: Width of spaghetti's lines

    Returns:
        The figure
    """
    cont = get_cont_of_country(iso3)
    msg = ".*axis already has a converter set*"
    warnings.filterwarnings("ignore", message=msg)
    pd.options.plotting.backend = "matplotlib"
    n_analyses = len(spaghs)
    n_targs = len(targets)
    ordered_analyses = [a for a in ANALYSIS_TYPES if a in spaghs]
    if cont == "OC" and iso3 != "SGP":
        ordered_analyses += ["fb_no_mob"]
    ordered_targets = [t for t in TARGET_TYPES if t in targets]
    fig, axes = plt.subplots(n_targs, n_analyses, figsize=[12, 13], sharey="row")
    for a, analysis in enumerate(ordered_analyses):
        a_spaghs = spaghs[analysis]
        analysis_name = ANALYSIS_NAMES[analysis] if len(ordered_analyses) < 4 else MOB_SOURCE_ABBREVS[analysis]
        for o, out in enumerate(ordered_targets):
            ax = axes[0] if n_analyses == 1 else axes[o, a]
            a_spaghs[out].plot(ax=ax, legend=False, color="black", linewidth=0.1, alpha=0.1)
            target = targets[out]
            ax.plot(target.index, target, linewidth=0.0, marker=".")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
            if o == 0:
                ax.set_title(analysis_name, fontsize=15)
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
    iso3: str,
    req_vars: List[str],
    priors: List[dist.Distribution],
    idata: az.InferenceData,
) -> plt.figure:
    """Plot comparison of calibration posterior estimates
    for parameters against their prior distributions.

    Args:
        iso3: The country identifier
        req_vars: Names of the parameters to plot
        priors: Prior distributions for the parameters
        idata: Calibration inference data

    Returns:
        The figure
    """
    country = pycountry.countries.lookup(iso3).name
    n_rows = int(np.ceil(len(priors) / 2)) + 3
    grid = [n_rows, 2]
    fig = az.plot_density(idata, var_names=req_vars, shade=0.3, grid=grid, figsize=[10, 40])
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


def get_prior_vals_from_dist(
    eval_points: List[float],
    dist: dist.Distribution,
    i_element: int,
) -> np.array:
    """Get the densities of a distribution, 
    accounting for it potentially having multiple elements
    (i.e. being multiple distributions).

    Args:
        eval_points: Points at which to evaluate
        dist: The distribution(s)
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
    loaded_priors = yml.safe_load(open(DATA_PATH / "evidence/priors.yml", "r"))
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


def plot_prior_multipost(
    var_names: List[str],
    priors: Dict[str, dist.Distribution],
    idatas: Dict[str, az.InferenceData],
    n_cols: int,
):
    """Plot comparison of parameter prior distribution
    to posterior from each mobility analysis type.

    Args:
        var_names: Names of the variables to plot
        priors: The prior distributions
        idatas: The calibration results for each analysis type
        n_cols: Number of columns for figure
    """

    # Preparation
    idata = idatas["no_mob"]
    prior_info = get_flat_priors()
    params = [p for p in prior_info if "proc" not in p and p in idata.posterior]
    n_axes = sum([get_param_dim(p, idata) for p in params]) + 1
    n_rows = int(np.ceil(n_axes / n_cols))
    height = min(2.0 + n_rows * 2.0, 13.0)

    # Plotting
    fig, ax = plt.subplots(n_rows, n_cols, figsize=[12, height])
    axes = ax.ravel()
    n_ax = 0
    for p in params:

        # Posteriors
        analyses = [a for a in ANALYSIS_NAMES if a in idatas]
        for a in analyses:
            idata = idatas[a]
            colour =[MOB_SOURCE_COLOURS[a]]
            az.plot_density(idata, ax=axes[n_ax:], hdi_prob=0.99, colors=colour, var_names=p)
    
            # Legend
            if p == params[-1]:
                ax = axes[n_ax]
                line = ax.get_lines()[-3]
                line.set_label(ANALYSIS_NAMES[a])
                ax.legend()
                handles, labels = ax.get_legend_handles_labels()
                ax.get_legend().remove()

        # Prior
        p_dim = get_param_dim(p, idata)
        for d in range(p_dim):
            ax = axes[n_ax]
            x_vals = np.linspace(*ax.get_xlim(), 100)
            y_vals = get_prior_vals_from_dist(x_vals, priors[p], d)
            ax.fill_between(x_vals, y_vals, color="k", alpha=0.2)
            display_name = get_param_display_name(p, p_dim, d, var_names, prior_info)
            ax.set_title(display_name)
            plt.setp(ax.xaxis.get_majorticklabels(), fontsize=10)
            n_ax += 1

    # Show legend
    ax = axes[n_ax]
    ax.legend(handles=handles, labels=labels, loc="center")

    # Suppress unused axes
    for a in range(n_ax, len(axes)):
        axes[a].set_axis_off()

    fig.tight_layout()
    plt.close()
    return fig


def plot_imm_props(
    spaghetti: pd.DataFrame,
) -> go.Figure:
    """Plot susceptible population proportions from 
    a randomly selected run from calibratin spaghetti.

    Args:
        spaghetti: Spaghetti

    Returns:
        The figure
    """
    imm_groups = sorted([c for c in set(spaghetti.columns.get_level_values(0)) if c.startswith("sus_")])
    run = choice(spaghetti.columns.get_level_values(1))
    return spaghetti.xs(run, axis=1, level=1)[imm_groups].plot.area()


def plot_beta_priors(
    priors: Dict[str, dict],
):
    """Plot the beta-distributed prior distributions

    Args:
        priors: The beta-distributed priors
    """
    fig, axes = plt.subplots(len(priors), 1, figsize=(15, 15))
    for p, (param_name, param) in enumerate(priors.items()):
    
        # Get the distribution
        a, b = get_beta_params_from_mean_var(param["mean"], param["std"])
        distri = dist.Beta(a, b)
    
        # Calculate the values
        max_val = 1.0 if param_name in ["cdr", "cross_immunity"] else 0.05
        x_vals = np.linspace(0.0, max_val, 1000)
        y_vals = np.exp(distri.log_prob(x_vals))
    
        # Plot
        ax = axes[p]
        ax.set_title(param["param_name"], fontsize=24)
        ax.fill_between(x_vals, y_vals, color="0.8")
        ax.plot(x_vals, y_vals, color="k", linewidth=2.0)
        ax.tick_params(axis="both", labelsize=20) 
        ax.set_yticks([])
    
    fig.tight_layout()


def plot_duration_params(
    duration_params: Dict[str, dict],
):
    """Plot the duration parameter priors, including the generation time
    and the various delays used in convolutions.

    Args:
        duration_params: The duration parameters loaded from the priors yaml
    """
    mean_xmax = 30.0
    sd_xmax = 15.0
    dur_param_types = [p.rsplit("_", 1)[0] for p in duration_params if p != "gen_mean_oc"]
    dur_types = list(dict.fromkeys(dur_param_types))  # Using set() loses the ordering of this list
    
    fig, axes = plt.subplots(len(dur_types), 2, figsize=(15, 18), width_ratios=[2, 1])
    for d, dur in enumerate(dur_types):

        # Extract prior values
        mean_param = duration_params[dur + "_mean"]
        sd_param = duration_params[dur + "_sd"]
        mean_mean = mean_param["mean"]
        sd_mean = sd_param["mean"]

        # Get the distributions        
        mean_prior = dist.TruncatedNormal(mean_mean, mean_param["sd"], low=DUR_MIN, high=mean_mean * DUR_REL_MAX)
        sd_prior = dist.TruncatedNormal(sd_mean, sd_param["sd"], low=DUR_MIN, high=sd_mean * DUR_REL_MAX)
        
        # Calculate the values
        mean_x_vals = np.linspace(0.0, mean_xmax, 1000)
        sd_x_vals = np.linspace(0.0, sd_xmax, 1000)
        mean_y_vals = np.exp(mean_prior.log_prob(mean_x_vals))
        sd_y_vals = np.exp(sd_prior.log_prob(sd_x_vals))

        # Plot mean        
        mean_ax = axes[d, 0]
        mean_ax.fill_between(mean_x_vals, mean_y_vals, color="0.8")
        mean_ax.plot(mean_x_vals, mean_y_vals, color="k", linewidth=2.0)
        mean_ax.set_title(mean_param["param_name"].replace(" (days)", ""), fontsize=22)
        mean_ax.set_xlabel("days", fontsize=18)
        mean_ax.tick_params(axis="both", labelsize=18)
        mean_ax.set_yticks([])
        
        # Plot SD
        sd_ax = axes[d, 1]
        sd_ax.fill_between(sd_x_vals, sd_y_vals, color="0.8")
        sd_ax.plot(sd_x_vals, sd_y_vals, color="k", linewidth=2.0)
        sd_ax.set_title(sd_param["param_name"].replace(" (days)", ""), fontsize=22)
        sd_ax.set_xlabel("days", fontsize=18)
        sd_ax.tick_params(axis="both", labelsize=18) 
        sd_ax.set_yticks([])
        
    fig.tight_layout()


def plot_proc_comparison(
    procs: Dict[str, pd.DataFrame],
    countries: List[str],
    path: Path,
) -> plt.Figure:
    """Plot the comparison of 
    the transmission scaling process
    across analysis types.

    Args:
        procs: Transmission process data
        countries: Names of the countries
        path: Path to the analyses
        
    Returns:
        The figure
    """
    fig, axes = plt.subplots(3, 3, figsize=[12, 14])
    flat_axes = axes.ravel()

    # Plot results by country
    for c, iso3 in enumerate(countries):
        ax = flat_axes[c]
        ax.set_title(pycountry.countries.lookup(iso3).name)
        analyses = [i.parts[-1] for i in (path / iso3).iterdir() if i.is_dir()]
        sorted_analyses = [a for a in MOB_SOURCE_COLOURS if a in analyses]
        for a in sorted_analyses:
            colour = MOB_SOURCE_COLOURS[a]
            quants = procs[iso3][a].quantile([0.025, 0.5, 0.975], axis=1).T
            ax.plot(quants.index, quants[0.5], color=colour, label=MOB_SOURCE_ABBREVS[a], linewidth=2.0)
            ax.fill_between(quants.index, quants[0.025], quants[0.975], alpha=0.1, color=colour)
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)

    # Switch off unused axes
    for ax in flat_axes[c + 1 :]:
        ax.set_axis_off()

    fig.tight_layout()
    plt.close()
    return fig


def plot_kde_comparison(
    data: Dict[str, pd.DataFrame],
) -> plt.figure:
    """Plot the comparison of the kernel density of some
    repeatedly sampled quantity (posterior or parameter)
    for each analysis type by country.

    Args:
        data: The values of interest for each country

    Returns:
        The figure
    """
    fig, axes = get_standard_subplot(len(data), 4)
    flat_axes = axes.ravel()

    # Plot the density distribtion by country
    for c, (country, likes) in enumerate(data.items()):
        likes = likes.rename(columns=MOB_SOURCE_ABBREVS)
        ax = flat_axes[c]
        ax.set_title(pycountry.countries.lookup(country).name)
        colours = [MOB_SOURCE_COLOURS[a] for a in data[country].columns]
        sns.kdeplot(likes, fill=True, ax=ax, palette=colours, alpha=0.1, linewidth=1.5)
        ax.set_yticks([])
        ax.set_ylabel("")

    # Switch off unused axes
    for ax in flat_axes[c + 1 :]:
        ax.set_axis_off()

    fig.tight_layout()
    plt.close()
    return fig


def plot_mob_weights_by_country(
    job_path: Path, 
    countries: List[str],
) -> plt.figure:
    """Plot the mobility weight posteriors for each
    of the mobility domains implemented for the Google analysis.

    Args:
        job_path: Path for the runs
        countries: The countries identifiers

    Returns:
        The figure
    """
    fig, axes = get_standard_subplot(len(countries) + 1, 4)
    x_vals = np.linspace(-0.1, 1.1, 200)
    flat_axes = axes.ravel()
    for c, iso3 in enumerate(countries):
    
        # Get mobility
        mob = get_google_mobility(iso3)
    
        # Get weights
        idata = az.from_netcdf(job_path / iso3 / "g_mob/idata_filtered.nc")
        weights = idata.posterior["mob_weights"].to_dataframe().unstack("mob_weights_dim_0")
        weights.columns = mob.columns
    
        # Plot
        ax = flat_axes[c]
        for l in weights.columns:
            colour = G_MOB_LOCATION_CMAP[l]
            kde = gaussian_kde(weights[l])
            label = l.replace("_", " ")
            ax.plot(x_vals, kde(x_vals), linewidth=2.0, label=label, color=colour)
            ax.fill_between(x_vals, kde(x_vals), alpha=0.1, color=colour)
    
        # Extra cosmetics
        country_name = pycountry.countries.lookup(iso3).name
        ax.set_title(country_name)    
        ax.set_yticks([])
        legend = ax.legend()
        legend.set_visible(False)
    
    # Legend on blank axis
    handles, labels = flat_axes[0].get_legend_handles_labels()
    flat_axes[c + 1].legend(handles=handles, labels=labels)
    
    # Turn off unused axes
    for a in range(c + 1, len(flat_axes)):
        ax = flat_axes[a]
        ax.axis("off")

    fig.tight_layout()
    plt.close()
    return fig


def compare_proc_mob(
    job_path: Path,
    countries: List[str],
    n_cols: int,
    mob_location: str,
) -> plt.Figure:
    """Plot comparison of 
    transmission scaling to mobility location.

    Args:
        job_path: Path for the runs
        countries: Requested countries to plot
        n_cols: Number of subplot columns for the figure
        mob_location: The name of the mobility location

    Returns:
        The figure
    """
    fig, axes = get_standard_subplot(len(countries), n_cols)
    mob_source = MOB_LOCATION_SOURCE_MAP[mob_location]
    mob_name = MOB_LOCATION_NAME_MAP[mob_location]
    title = f"Estimated transmission scaling (without mobility) versus {mob_name} mobility"
    fig.suptitle(title, fontsize=14, y=1.0)
    flat_axes = axes.ravel()
    for c, iso3 in enumerate(countries):
        ax = flat_axes[c]
        country = pycountry.countries.lookup(iso3).name
        ax.set_title(country)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)

        # Transmission scaling process plotting
        ref_analysis = "fb_no_mob" if (job_path / iso3 / "fb_no_mob").exists() else "no_mob"
        proc_samples = pd.read_hdf(job_path / iso3 / ref_analysis / "spaghetti.h5")["process"]
        centiles = proc_samples.quantile([0.025, 0.5, 0.975], axis=1).T
        ax.plot(centiles.index, centiles[0.5], label="process", color="navy", linewidth=2.0)
        ax.fill_between(centiles.index, centiles[0.025], centiles[0.975], alpha=0.1, color="navy")

        # Mobility
        mob = get_requested_mob(iso3, mob_source, mob_location)
        mobility = mob.loc[(centiles.index[0] < mob.index) & (mob.index < centiles.index[-1])]
        if mobility.isna().sum() / len(mobility) > 0.5:
            mob_name = MOB_LOCATION_NAME_MAP[mob_location]
            msg = f"Note, {mob_name} largely missing for {country} during the analysis period."
            display(Markdown(msg))
        smoothed_mob = mobility.rolling(7, center=True).mean().dropna()
        colour = G_MOB_LOCATION_CMAP[mob_location] if mob_source == "g_mob" else MOB_SOURCE_COLOURS[mob_location]
        ax.plot(smoothed_mob.index, smoothed_mob, color=colour, linewidth=2.0)

    # Switch off unused axes
    for ax in flat_axes[c + 1 :]:
        ax.set_axis_off()

    fig.tight_layout()
    plt.close()
    return fig


def compare_proc_weighted_gmob(
    job_path: Path, 
    countries: List[str],
    n_samples: int,
    n_cols: int,
) -> plt.Figure:
    """Plot comparison of composite Google time series to 
    the transmission scaling process.

    Args:
        job_path: Path for the runs
        countries: Requested countries to plot
        n_samples: Number of samples from Google weights to create composite series
        n_cols: Number of subplot columns for the figure

    Returns:
        The figure
    """
    fig, axes = get_standard_subplot(len(countries), n_cols)
    flat_axes = axes.ravel()
    title = f"Estimated scaling for transmission (without mobility) versus composite Google mobility time series"
    fig.suptitle(title, fontsize=14, y=1.0)

    for c, iso3 in enumerate(countries):
    
        # Starting cosmetics
        ax = flat_axes[c]
        ax.set_title(pycountry.countries.lookup(iso3).name)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
    
        # Get the transmission scaling process
        proc_samples = pd.read_hdf(job_path / iso3 / "no_mob/spaghetti.h5")["process"]
        centiles = proc_samples.quantile([0.025, 0.5, 0.975], axis=1).T

        # Get the mobility data
        smoothed_mob = get_smoothed_trunc_g_mob(iso3, centiles.index[0], centiles.index[-1])
    
        # Get the Google mobility weight posteriors and quantiles of weighted series
        params = get_g_mob_weight_posts(job_path / iso3)
        mob_quants = get_g_mob_quants(smoothed_mob, params, n_samples)
    
        # Plot the weighted Google mobility distribution
        ax.plot(mob_quants[0.5], color="green", linewidth=2.0)
        ax.fill_between(mob_quants.index, mob_quants[0.025], mob_quants[0.975], alpha=0.1, color="green")
    
        # Residual transmission scaling plotting
        ax.plot(centiles.index, centiles[0.5], label="process", color="navy", linewidth=2.0)
        ax.fill_between(centiles.index, centiles[0.025], centiles[0.975], alpha=0.1, color="navy")
        ax.set_xlim([centiles.index[0], centiles.index[-1]])
        
    for ax in flat_axes[c + 1 :]:
        ax.set_axis_off()
    
    fig.tight_layout()
    plt.close()
    return fig


def plot_select_proc_mob(
    job_path: Path, 
    panels: List[List[List[str]]],
    n_samples: int,
) -> plt.figure:
    """Plot selected comparisons between mobility
    and the residual transmission scaling.

    Args:
        job_path: Path for the runs
        panels: The comparisons to plot

    Returns:
        The figure
    """
    fig, axes = plt.subplots(4, 9, figsize=(14, 8))
    for c, col in enumerate(panels):
        for r, row in enumerate(col):

            # Gather data
            mob_location, country = row
            iso3 = pycountry.countries.lookup(country).alpha_3
            country_name = SHORT_COUNTRY_NAMES[country] if country in SHORT_COUNTRY_NAMES else country
            mob_source = mob_location if mob_location.startswith("fb_") else "g_mob"
            mob_source_name = MOB_LOCATION_ABBREVS[mob_location]

            # Plot residual transmission scaling
            proc_samples = pd.read_hdf(job_path / iso3 / "no_mob/spaghetti.h5")["process"]
            centiles = proc_samples.quantile([0.025, 0.5, 0.975], axis=1).T
            ax = axes[r, c]
            ax.plot(centiles.index, centiles[0.5], label="process", color="navy")
            ax.fill_between(centiles.index, centiles[0.025], centiles[0.975], alpha=0.2, color="navy")

            if "weighted" in mob_location:
                
                # Get the mobility data
                smoothed_mob = get_smoothed_trunc_g_mob(iso3, centiles.index[0], centiles.index[-1])
                
                # Get the Google mobility weight posteriors and quantiles of weighted series
                params = get_g_mob_weight_posts(job_path / iso3)
                mob_quants = get_g_mob_quants(smoothed_mob, params, n_samples)
                
                # Plot the weighted Google mobility distribution
                ax.plot(mob_quants[0.5], color="green")
                ax.fill_between(mob_quants.index, mob_quants[0.025], mob_quants[0.975], alpha=0.2, color="green")

            else:
                mob = get_requested_mob(iso3, mob_source, mob_location)
                mobility = mob.loc[(centiles.index[0] < mob.index) & (mob.index < centiles.index[-1])]
                smoothed_mob = mobility.rolling(7, center=True).mean().dropna()
                colour = G_MOB_LOCATION_CMAP[mob_location] if mob_source == "g_mob" else MOB_SOURCE_COLOURS[mob_source]
                ax.plot(smoothed_mob.index, smoothed_mob, color=colour)

            # Finish cosmetics
            ax.set_title(f"{country_name}, {mob_source_name}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()

    plt.close()
    return fig


def plot_exponent_dispersion_comparison(
    job_path: Path, 
    ratio_dists: Dict[str, pd.DataFrame],
) -> plt.figure:
    """Scatter the mobility exponent against
    the change in the transmission scaling dispersion.

    Args:
        job_path: Path for the runs
        ratio_dists: The posteriors of the dispersion ratio

    Returns:
        The figure
    """
    fig, axs = plt.subplots(2, 2, figsize=[12, 9])
    flat_axes = axs.ravel()
    all_countries = ls(job_path)
    analyses = {k: v for k, v in ANALYSIS_NAMES.items() if "no_mob" not in k}
    for m, (mob_source, mob_name) in enumerate(analyses.items()):
        ax = flat_axes[m]
        ax.set_title(mob_name)
    
        # Gather data
        idatas, _ = get_idatas_for_mob_type(job_path, all_countries, mob_source)
        plot_df = pd.DataFrame(
            {
                "mobility exponent": {c: float(d.posterior["mob_exp"].median()) for c, d in idatas.items()},
                "dispersion ratio": get_median_ratios(ratio_dists, mob_source),
                "GDP per capita": get_gdps(2020),
                "population (millions)": {c: get_country_pop(c) / 1e6 for c in all_countries},
            }
        )
    
        # Plot
        sns.scatterplot(x="dispersion ratio", y="mobility exponent", hue="GDP per capita", size="population (millions)", data=plot_df, sizes=(25, 500), ax=ax, edgecolors="k", palette=sns.color_palette("Reds", as_cmap=True))
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
    
    # Sort out legend
    ax = flat_axes[-1]
    ax.legend(handles=handles, labels=labels, loc="center", ncol=2)
    ax.axis("off")
    
    fig.tight_layout()
    plt.close()
    return fig


def plot_exponent_dispersion_comparison_interactive(
    job_path: Path, 
    mob_source: str,
    ratio_dists: Dict[str, pd.DataFrame],
) -> go.Figure:
    """Equivalent plot to the previous function,
    but using plotly to create an interactive version.

    Args:
        job_path: Path for the runs
        mob_source: Mobility analysis type
        ratio_dists: The posteriors of the dispersion ratio

    Returns:
        The plotly interactive figure
    """
    countries = ls(job_path)
    idatas, _ = get_idatas_for_mob_type(job_path, countries, mob_source)
    plot_df = pd.DataFrame(
        {
            "mobility exponent": {c: float(d.posterior["mob_exp"].median()) for c, d in idatas.items()},
            "dispersion ratio": get_median_ratios(ratio_dists, mob_source),
            "GDP per capita": get_gdps(2020),
            "population (millions)": {c: get_country_pop(c) / 1e6 for c in countries},
        },
    )
    plot_df["country name"] = plot_df.index.to_series().apply(get_country_name)
    plot_df.loc[plot_df["population (millions)"].isna(), "population (millions)"] = 0.0
    
    fig = go.Figure(
        layout={
            "width": 750, 
            "height": 600, 
            "plot_bgcolor": "#F0F0F0", 
            "title": {"text": ANALYSIS_NAMES[mob_source], "xanchor": "center", "x": 0.5},
            "xaxis_title": "dispersion ratio",
            "yaxis_title": "mobility exponent",
        }
    )
    hover_template = "%{text}<br>disp ratio: %{x:.2f}<br>mob exp: %{y:.2f}<extra></extra>"
    fig.add_trace(
        go.Scatter(
            x=plot_df["dispersion ratio"],
            y=plot_df["mobility exponent"],
            mode="markers",
            text=plot_df["country name"],
            hoverinfo="text",
            marker=dict(
                size=plot_df["population (millions)"], 
                line=dict(width=1.0, color="black"),
                color=plot_df["GDP per capita"],
                sizemode="area",
                sizemin=4.0,
                sizeref=2.5,
                colorscale="reds",
            ),
            hovertemplate=hover_template
        ),
    )
    return fig


def plot_inclusion(
    world: GeoDataFrame,
) -> plt.figure:
    """Plot inclusion status of countries based on mobility.

    Args:
        world: Countries of the world Geopandas dataframe

    Returns:
        The figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.set_xticks([])
    ax.set_yticks([])
    world.plot(ax=ax, color=world["mob"].map(INCLUSION_COLOURS), edgecolor="black", linewidth=0.2)
    world[world["included"]].geometry.centroid.plot(ax=ax, color="red", marker="o", markersize=50)
    return fig


def plot_continent_grouping(
    world: GeoDataFrame
):
    """Plot the countries of the world shaded according to
    continent grouping.

    Args:
        world: The GeoPandas dataframe with the continent specified

    Returns:
        The figure
    """
    fig, ax = plt.subplots(1, 1, figsize=[12, 9])

    # Plot
    world.plot(ax=ax, column="continent", figsize=[20, 8], cmap="Pastel1", legend=True)
    world.boundary.plot(ax=ax, color="black", linewidth=0.4)

    # Tidy up cosmetics
    ax.set_xticks([])
    ax.set_yticks([])
    leg = ax.get_legend()
    leg.set_bbox_to_anchor((0.22, 0.55))

    plt.close()
    return fig


def plot_dispersion_analysis(
    disp_posts: Dict[str, pd.DataFrame],
    ratios: Dict[str, pd.DataFrame],
) -> plt.figure:
    """Plot the analysis of strength of evidence
    that including mobility is an improvement based on
    the dispersion posterior results.

    Args:
        disp_posts: The results for the dispersion posteriors
        ratios: The ratios of the dispersion samples
    Returns:
        The figure
    """
    marker_size = 15

    plt.style.use("default")
    world = get_world_shp()
    fig, axes = plt.subplots(2, 2, figsize=(20, 8), constrained_layout=True)
    flat_axes = axes.ravel()
    
    # Strength of evidence for each mobility type panels
    for a, (analysis, analysis_name) in enumerate(list(ANALYSIS_NAMES.items())[1: -1]):

        # Find median ratio of the mobility approach to the baseline
        median_ratios = get_median_ratios(ratios, analysis)

        world["disp_ratio"] = world["ISO_A3"].map(median_ratios)
        mob_avail = world[world["disp_ratio"].notna()]
        mob_unavail = world[world["disp_ratio"].isna()]
    
        # Plot the proportion improvements
        ax = flat_axes[a]
        ax.set_title(analysis_name)
        mob_avail.plot(column="disp_ratio", ax=ax, cmap="RdGy_r", legend=True, vmin=0.4, vmax=1.6)
        mob_unavail.plot(ax=ax, color="w", hatch="///", edgecolor="whitesmoke")
        world["small"] = world.geometry.area < 2.5
        world["centroid"] = world.geometry.centroid
        centroids = world[world["small"]].set_geometry("centroid")
        centroids.plot(ax=ax, markersize=marker_size, column="disp_ratio", cmap="RdGy_r", vmin=0.4, vmax=1.6, edgecolor="black", linewidth=0.5, zorder=3)
    
    # Best mobility approach
    best_mob = {c: disp_posts[c].mean().idxmin() for c in disp_posts}
    world["best_mob"] = world["ISO_A3"].map(best_mob)
    world["best_mob_colour"] = world["best_mob"].map(MOB_SOURCE_COLOURS | {"no_mob": "0.45"})
    mob_avail = world[world["best_mob_colour"].notna()]
    mob_unavail = world[world["best_mob_colour"].isna()]
   
    # Plot the best mobility approach
    ax = flat_axes[-1]
    ax.set_title("best analysis approach")
    
    # Dummy colour bar to get axis in right position with constrained layout
    sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1))
    cb = fig.colorbar(sm, ax=ax)
    cb.ax.set_visible(False)
    
    mob_avail.plot(ax=ax, color=mob_avail["best_mob_colour"])
    mob_unavail.plot(ax=ax, color="w", hatch="///", edgecolor="whitesmoke")
    centroids.plot(ax=ax, markersize=marker_size, color=mob_avail["best_mob_colour"], vmin=0.4, vmax=1.6, edgecolor="black", linewidth=0.5, zorder=3)
    
    # Cosmetics for all panels
    for ax in flat_axes:
        world.boundary.plot(ax=ax, color="black", linewidth=0.2)
        ax.set_xticks([])
        ax.set_yticks([])
    
    return fig


COUNTRY_GROUPINGS = {
    "North America": [
        "CAN", "USA", 
        "MEX", "SLV", "HND", "CRI", "PAN", 
        "DOM", "HTI", "JAM", "PRI", "ABW", "BHS"
    ],
    "South America": [
        "VEN", "COL", "GUY", "SUR", "GUF",
        "BRA", "PRY", "URY", "ARG", "CHL", "PER"
    ],
    "Western Europe": [
        "IRL", "GBR",
        "FRA", "BEL", "LUX", "NLD",
        "CHE"
    ],
    "Northern Europe": [
        "DNK", "NOR", "SWE", "FIN",
        "EST", "LVA", "LTU"
    ],
    "Southern Europe": [
        "PRT", "ESP", "ITA", "MLT",
        "SVN", "HRV", "BIH", "SRB",
        "MKD", "ALB", "GRC"
    ],
    "Eastern Europe": [
        "POL", "CZE", "SVK", "HUN",
        "ROU", "BGR",
        "BLR", "UKR", "MDA",
        "RUS"
    ],
    "Northern Africa": [
        "MAR", "DZA", "TUN", "LBY", "EGY"
    ],
    "West Africa": [
        "CPV", "SEN", "GNB", "GIN",
        "LBR", "CIV", "TGO", "BEN", "BFA", "MLI", "NGA"
    ],
    "Southern Africa": [
        "ZAF", "LSO", "ZWE", "ZMB", "MWI", "MOZ", "AGO", "MDG"
    ],
    "Central & East Africa": [
        "STP", "GNQ", "GAB", "COG",
        "CMR", 
        "RWA", "KEN", "ETH"
    ],
    "Western Asia": [
        "TUR", "GEO",
        "LBN", "ISR", "JOR",
        "IRQ", "YEM",
        "SAU", "KWT", "BHR", "QAT", "ARE", "OMN"
    ],
    "Southern Asia": [
        "AFG", "PAK", "IND", "NPL", "LKA", "BGD"
    ],
    "Eastern/South-eastern Asia": [
        "JPN", "KOR",
        "PHL", "MYS", "IDN"
    ],
    "Oceania, Singpore": [
        "AUS", "NZL", "FJI", "SGP",
    ],
}


def get_avail_groupings(
    avail_countries: List[str],
) -> Dict[str, List[str]]:
    """Get the countries from the above grouping scheme
    that are actually available.

    Args:
        avail_countries: The countries that are available

    Returns:
        The revised grouping
    """
    avail_grouping = {}
    for region, countries in COUNTRY_GROUPINGS.items():
        cs = [c for c in countries if c in avail_countries]

        # Don't include the region if there are no countries
        if cs:
            avail_grouping[region] = cs
    return avail_grouping


def plot_mob_exp_violins(
    mob_source: str, 
    mob_exp_df: pd.DataFrame, 
    ratios: Dict[str, float],
) -> plt.figure:
    """Plot the mobility exponent distributions 
    as violin plots, with shade of colouring 
    determined by the transmission scaling dispersion ratios.

    Args:
        mob_source: The mobility approach
        mob_exp_df: The mobility exponent parameter values
        ratios: The median ratios

    Returns:
        The figure
    """
    fig, axes = subplots(3, 5, figsize=[12, 12], sharey=True)
    flat_axes = axes.ravel()
    
    norm = Normalize(vmin=min(ratios.values()), vmax=max(ratios.values()))
    cmap = get_cmap(MOB_SOURCE_COLOURS[mob_source].capitalize() + "s")
    palette = {c: cmap(norm(v)) for c, v in ratios.items()}
    
    grouping = get_avail_groupings(mob_exp_df.columns)
    for r, (region, countries) in enumerate(grouping.items()):
        ax = flat_axes[r]
        ax.set_title(region)
        plot_df = mob_exp_df[countries]
        plot_df.columns = plot_df.columns.map(get_country_short_name)
        sns.violinplot(plot_df, ax=ax, palette=palette)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    
    ax = flat_axes[r + 1]
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, location="left")
    ax.remove()
    
    fig.tight_layout()
    plt.close()
    return fig


def plot_composite_calibrations(
    job_path: Path, 
    iso3: str, 
    analyses: List[str], 
    spaghs: Dict[str, pd.DataFrame], 
    targets: Dict[str, pd.Series],
) -> plt.figure:
    """Plot Figure 1, combining calibration target comparisons
    with two other plotting approaches.

    Args:
        job_path: Path to the job
        iso3: Country identifier
        analyses: Analyses to consider
        spaghs: Spaghetti results
        targets: Calibration target data

    Returns:
        The figure
    """

    c_path = job_path / iso3
    fig = plt.figure(figsize=[16, 10])
    gs = GridSpec(5, 6)
    
    # Calibration comparison
    msg = ".*axis already has a converter set*"
    warnings.filterwarnings("ignore", message=msg)
    ordered_analyses = [a for a in ANALYSIS_TYPES if a in spaghs]
    ordered_targets = [t for t in TARGET_TYPES if t in targets]
    for a, analysis in enumerate(ordered_analyses):
        a_spaghs = spaghs[analysis]
        for o, out in enumerate(ordered_targets):
            ax = fig.add_subplot(gs[o, a])
            a_spaghs[out].plot(ax=ax, legend=False, color="black", linewidth=0.8, alpha=0.1)
            target = targets[out]
            ax.plot(target.index, target, linewidth=0.0, marker=".")
            if o == 0:
                ax.set_title(MOB_SOURCE_ABBREVS[analysis], fontsize=15)
            if a == 0:
                ax.set_ylabel(TARGET_TYPES[out], fontsize=12)
            ymax = ax.get_ylim()[1]
            targ_max = max(targets[out]) * 1.5
            if ymax > targ_max and out != "seropos" and "prop_" not in out:
                ylim = min([ymax, targ_max])
                ax.set_ylim(-ylim * 0.05, ylim)
            ax.set_yticks([])
            if o < 4:
                ax.set_xticklabels([])
            else:
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
    fig.tight_layout()
    # fig.subplots_adjust(wspace=0.05)
    
    # Residual transmission scaling with credible intervals
    c_procs = [pd.read_hdf(c_path / a / "spaghetti.h5")["process"] for a in analyses]
    procs = pd.concat(c_procs, keys=analyses, axis=1)
    
    ax = fig.add_subplot(gs[0: 2, 4: 6])
    ax.set_title("Residual transmission scaling")
    ax.set_yticks([])
    ax.tick_params(axis="x", labelrotation=70)
    for a in analyses:
        colour = MOB_SOURCE_COLOURS[a]
        label = MOB_SOURCE_ABBREVS[a]
        quants = procs[a].quantile([0.025, 0.5, 0.975], axis=1).T
        ax.plot(quants.index, quants[0.5], color=colour, label=label, linewidth=2.0)
        ax.fill_between(quants.index, quants[0.025], quants[0.975], alpha=0.1, color=colour)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    
    # Residual transmission scaling dispersion posteriors
    param_posts = get_param_vals_by_analysis("dispersion_proc", c_path)
    
    ax = fig.add_subplot(gs[3: 5, 4: 6])
    colours = [MOB_SOURCE_COLOURS[a] for a in param_posts.columns]
    param_posts = param_posts.rename(columns=MOB_SOURCE_ABBREVS)

    sns.kdeplot(param_posts, fill=True, ax=ax, palette=colours, alpha=0.1, linewidth=1.5)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_title("dispersion posterior distributions")

    plt.close()
    return fig


def add_cont_to_world_geodf(
    world: GeoDataFrame,
):
    """Add a column to a GeoPandas dataframe
    to indicate the continent of the country as a string.

    Args:
        world: The GeoPandas dataframe
    """
    for iso3 in world["ISO_A3"]:
        try:
            iso2 = pycountry.countries.lookup(iso3).alpha_2
            cont = pc.convert_country_alpha2_to_continent_code.country_alpha2_to_continent_code(iso2)
            cont_name = pc.convert_continent_code_to_continent_name(cont)
            world.loc[world["ISO_A3"] == iso3, "continent"] = cont_name
        except KeyError:
            world.loc[world["ISO_A3"] == iso3, "continent"] = "none"
        except LookupError:
            world.loc[world["ISO_A3"] == iso3, "continent"] = "none"
