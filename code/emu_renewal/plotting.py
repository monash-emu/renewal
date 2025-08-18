from typing import List, Dict
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
import yaml as yml
import pycountry_convert as pc
from geopandas import GeoDataFrame

from emu_renewal.constants import ANALYSIS_TYPES, MOB_COLOURS, DUR_MIN, DUR_REL_MAX
from emu_renewal.inputs import (
    DATA_PATH,
    get_google_mobility,
    get_fb_visited_mobility,
    get_fb_singletile_mobility,
    get_gdps,
    get_country_pop,
    get_world_shp,
)
from emu_renewal.outputs import get_idatas_for_mob_type
from emu_renewal.calibration import StandardCalib
from emu_renewal.utils import get_param_dim, sort_countries_by_name, get_beta_params_from_mean_var
from IPython.display import display, Markdown
from matplotlib.lines import Line2D

plt.style.use("ggplot")

ANALYSIS_NAMES = {
    "no_mob": "no mobility",
    "g_mob": "Google mobility",
    "fb_visited_mob": "Facebook tiles visited mobility",
    "fb_singletile_mob": "Facebook single tile mobility",
}
AN_ABBREVS = {
    "no_mob": "none",
    "g_mob": "Google",
    "fb_visited_mob": "FB tiles visited",
    "fb_singletile_mob": "FB single tile",
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
INCLUSION_COLOURS = {
    "neither": "lightgrey",
    "Google": "green",
    "FB": "blue",
    "both": "purple"
}
MOB_ANALYSIS_MAP = {
    "retail_and_recreation": "g_mob",
    "grocery_and_pharmacy": "g_mob",
    "parks": "g_mob",
    "transit_stations": "g_mob",
    "workplaces": "g_mob",
    "residential": "g_mob",
    "fb_visited_mob": "fb_visited_mob",
    "fb_singletile_mob": "fb_singletile_mob",
}
MOB_NAME_MAP = {
    "retail_and_recreation": "Google retail and recreation",
    "grocery_and_pharmacy": "Google grocery and pharmacy",
    "parks": "Google parks",
    "transit_stations": "Google transit stations",
    "workplaces": "Google workplaces",
    "residential": "Google residential",
    "fb_visited_mob": "Facebook tiles visited",
    "fb_singletile_mob": "Facebook single tile",
}
MOB_SOURCE_MAP = {
    "g_mob": "Google",
    "fb_visited_mob": "Facebook tiles visited",
    "fb_singletile_mob": "Facebook single tile",
}
G_MOB_DOMAIN_CMAP = {
    "retail_and_recreation": "darkgoldenrod",
    "grocery_and_pharmacy": "darkblue",
    "parks": "darkgreen",
    "transit_stations": "dimgrey",
    "workplaces": "purple",
    "residential": "brown",
}
A_MOB_DOMAIN_CMAP = {
    "driving": "red",
    "transit": "blue",
    "walking": "green",
}
CONT_CMAP = {
    "AF": "black",
    "AS": "yellow",
    "EU": "blue",
    "NA": "green",
    "OC": "red",
    "SA": "purple",
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
            a_spaghs[out].plot(ax=ax, legend=False, color="black", linewidth=0.1, alpha=0.1)
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
        analyses = [a for a in ANALYSIS_NAMES if a in idatas]
        for a in analyses:
            idata = idatas[a]
            colour =[MOB_COLOURS[a]]
            az.plot_density(idata, ax=axes[n_ax:], hdi_prob=0.99, colors=colour, var_names=p)
    
            # Legend
            if p == params[-1]:
                ax = axes[n_ax]
                line = ax.get_lines()[-3]
                line.set_label(ANALYSIS_NAMES[a])
                ax.legend()

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


def plot_beta_priors(priors):
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
    fig, axes = plt.subplots(3, 3, figsize=[12, 14])
    title = f"Comparisons of variable process scaling under each mobility assumption, {cont_name}"
    fig.suptitle(title, fontsize=15)
    flat_axes = axes.ravel()
    for c, iso3 in enumerate(countries):
        country = pycountry.countries.lookup(iso3).name
        ax = flat_axes[c]
        ax.set_title(country)
        analyses = [i.parts[-1] for i in (path / iso3).iterdir() if i.is_dir()]
        sorted_analyses = [a for a in MOB_COLOURS if a in analyses]
        for a in sorted_analyses:
            colour = MOB_COLOURS[a]
            quants = procs[iso3][a].quantile([0.05, 0.5, 0.95], axis=1).T
            ax.plot(quants.index, quants[0.5], color=colour, label=AN_ABBREVS[a], linewidth=2.0)
            ax.fill_between(quants.index, quants[0.05], quants[0.95], alpha=0.1, color=colour)
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
):
    """Plot the comparison of the kernel density of some
    repeatedly sampled quantity (posterior or parameter)
    for each analysis type by country.

    Args:
        data: The values of interest for each country
    """
    fig, axes = get_standard_subplot(len(data), 4)
    flat_axes = axes.ravel()
    for c, (country, likes) in enumerate(data.items()):
        likes = likes.rename(columns=AN_ABBREVS)
        country_name = pycountry.countries.lookup(country).name
        ax = flat_axes[c]
        ax.set_title(country_name)
        colours = [MOB_COLOURS[a] for a in data[country].columns]
        sns.kdeplot(likes, fill=True, ax=ax, palette=colours, alpha=0.1, linewidth=1.5)
        ax.set_yticks([])
        ax.set_ylabel("")

    # Switch off unused axes
    for ax in flat_axes[c + 1 :]:
        ax.set_axis_off()

    fig.tight_layout()
    plt.close()
    return fig


def get_cont_mobility(cont, countries_by_cont, mob_type):
    mob = {}
    no_mob_countries = []
    for c in countries_by_cont[cont]:
        country = pycountry.countries.lookup(c).name
        try:
            c_mob = get_google_mobility(c)
            # Don't include one country (Guinea-Bissau) with locations missing
            if c_mob.isnull().all().any():
                no_mob_countries.append(country)
            else:
                mob[c] = c_mob
        except:
            no_mob_countries.append(country)
    if no_mob_countries:
        mob_str = AN_ABBREVS[mob_type]
        countries_str = ", ".join(no_mob_countries)
        display(Markdown(f"No {mob_str} mobility available for {countries_str}."))
    return mob


def plot_mob_weights_by_country(
    job_path: Path,
    mob_type: str,
    mobility: Dict[str, pd.DataFrame],
) -> plt.figure:
    """Plot the mobility weight posteriors for each
    of the mobility domains implemented for the Google
    or Apple mobility analyses.

    Args:
        job_path: Path for the runs
        mob_type: Mobility type considered
        mobility: The mobility data by country

    Returns:
        The figure
    """
    fig, axes = get_standard_subplot(len(mobility) + 1, 4)
    flat_axes = axes.ravel()
    linewidth = 2.0
    for c, iso3 in enumerate(mobility):
        c_path = job_path / iso3
        country = pycountry.countries.lookup(iso3).name
        idata = az.from_netcdf(c_path / f"{mob_type}/idata_filtered.nc")
        weights = idata.posterior["mob_weights"].to_dataframe().unstack("mob_weights_dim_0")
        weights.columns = mobility[iso3].columns
        ax = flat_axes[c]
        sns.kdeplot(weights, fill=True, alpha=0.05, linewidth=linewidth, ax=ax, palette=G_MOB_DOMAIN_CMAP)
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_title(country)
        ax.get_legend().remove()

    # Include axis just for legend
    labels = [l.replace("_", " ") for l in G_MOB_DOMAIN_CMAP]
    colours = [col.get_facecolor()[0] for col in ax.collections]
    handles = [Line2D([0], [0], color=col, linewidth=4.0, alpha=1.0, label=l) for col, l in zip(colours, labels)]
    flat_axes[c + 1].legend(handles=handles, loc="center", title="mobility locations")

    # Switch off unused axes
    for ax in flat_axes[c + 1:]:
        ax.set_axis_off()

    fig.tight_layout()
    plt.close()
    return fig


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
    mob_source = MOB_ANALYSIS_MAP[mob_type]
    mob_name = MOB_NAME_MAP[mob_type]
    title = f"Estimated variable process (without mobility scaling) versus {mob_name} mobility"
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
        if mob_source == "g_mob":
            mob = get_google_mobility(iso3)[mob_type]
        elif mob_source == "fb_visited_mob":
            mob = get_fb_visited_mobility(iso3)
        elif mob_source == "fb_singletile_mob":
            mob = get_fb_singletile_mobility(iso3)

        mobility = mob.loc[(centiles.index[0] < mob.index) & (mob.index < centiles.index[-1])]
        if mobility.isna().sum() / len(mobility) > 0.5:
            mob_name = MOB_NAME_MAP[mob_type]
            msg = f"Note, {mob_name} largely missing for {country} during the analysis period."
            display(Markdown(msg))
        smoothed_mob = mobility.rolling(7, center=True).mean().dropna()
        colour = G_MOB_DOMAIN_CMAP[mob_type] if mob_source == "g_mob" else MOB_COLOURS[mob_type]
        ax.plot(smoothed_mob.index, smoothed_mob, color=colour)
        ax.set_title(country)

    # Switch off unused axes
    for ax in flat_axes[c + 1 :]:
        ax.set_axis_off()

    fig.tight_layout()
    plt.close()
    return fig


def plot_param_posts_for_countries(
    param: str,
    idatas: Dict[str, az.InferenceData],
    fv_idatas: Dict[str, az.InferenceData],
    n_cols: int,
) -> plt.figure:
    """Plot the posteriors of a specified
    parameter from inference data objects by country.

    Args:
        idatas: The inference data objects, output of get_idatas_for_mob_type
        n_cols: Number of columns for figure

    Returns:
        The figure
        The high-density intervals
        The means
    """
    means = {}
    hdis = {}
    fig, axes = get_standard_subplot(len(idatas), n_cols)
    axes = axes.ravel()
    for c, iso3 in enumerate(idatas):
        ax=axes[c]
        fv_idata = fv_idatas[iso3]
        post_plot = az.plot_posterior(fv_idata, var_names=param, ax=ax, point_estimate=None, hdi_prob="hide")

    for c, iso3 in enumerate(idatas):
        idata = idatas[iso3]
        country = pycountry.countries.lookup(iso3).name
        ax = axes[c]
        post_plot = az.plot_posterior(idata, var_names=param, ax=ax, point_estimate=None, hdi_prob="hide")
        line_data = post_plot.get_lines()[0]
        post_plot.fill_between(line_data.get_xdata(), line_data.get_ydata(), alpha=0.2)
        ax.set_xlim([0.0, 2.0])
        ax.set_title(country)
        means[iso3] = az.summary(idata, var_names="mob_exp", kind="stats")["mean"].values[0]
        hdis[iso3] = az.hdi(idata, var_names="mob_exp", hdi_prob=0.95).to_pandas()[param]

    for a in range(c + 1, len(axes)):
        axes[a].set_axis_off()
    fig.tight_layout()
    plt.close()
    return fig, means, hdis


def get_mob_exp_gdp_df(
    job_path: Path,
    countries: List[str],
) -> pd.DataFrame:
    """Collate data on mob_exp parameter
    estimated by various mobility analysis types,
    GDP and population data by country.

    Args:
        job_path: Path for the runs
        countries: ISO3 codes for the ountries to include

    Returns:
        The data
    """
    quants = {}
    analyses = [a for a in ANALYSIS_NAMES if a != "no_mob"]
    for mob_type in analyses:
        idatas, _ = get_idatas_for_mob_type(job_path, countries, mob_type)
        quants[mob_type] = {
            c: float(idatas[c].posterior["mob_exp"].quantile([0.5]).data) for c in idatas
        }
    quants["pop"] = {c: get_country_pop(c) for c in countries}
    quants["gdp"] = get_gdps(2020)
    quants["gdp"]["VEN"] = (
        42.84e9 / quants["pop"]["VEN"]
    )  # Assume Venezuela's GDP was 42.84 billion in 2020
    quants["continent"] = {
        c: pc.convert_continent_code_to_continent_name(
            pc.country_alpha2_to_continent_code(pycountry.countries.lookup(c).alpha_2)
        )
        for c in countries
    }
    return pd.DataFrame(quants)


def plot_mob_exp_versus_gdp(
    quants_df: pd.DataFrame,
) -> plt.figure:
    """Plot the mobility exponent parameter
    against GDP with population determining marker size
    adn continent determining colour.

    Args:
        quants_df: The data, the output of get_mob_exp_gdp_df

    Returns:
        The figure
    """
    fig, axs = plt.subplots(2, 2, figsize=[12, 9])
    axes = axs.ravel()
    analyses = {k: v for k, v in ANALYSIS_NAMES.items() if k != "no_mob"}
    quants_df["population (millions)"] = quants_df["pop"] / 1e6
    quants_df["GDP per capita (thousand USD)"] = quants_df["gdp"] / 1e3
    cont_name_cmap = {
        pc.convert_continent_code_to_continent_name(k): v for 
        k, v in CONT_CMAP.items()
    }
    for m, (mob_type, mob_name) in enumerate(analyses.items()):
        plot_df = quants_df[["GDP per capita (thousand USD)", mob_type, "population (millions)", "continent"]].dropna()
        ax = axes[m]
        sns.scatterplot(
            x="GDP per capita (thousand USD)",
            y=mob_type,
            size="population (millions)",
            hue="continent",
            data=plot_df,
            sizes=(50, 1000),
            ax=ax,
            palette=cont_name_cmap,
            alpha=0.6,
        )
        ax.set_title(mob_name)
        ax.set_ylabel("mobility exponent")
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()

    # Last axis contains only the legend
    ax = axes[-1]
    ax.legend(handles=handles, labels=labels, loc="center", ncol=2)
    ax.axis("off")

    fig.tight_layout()
    plt.close()
    return fig


def plot_inclusion(
    world: GeoDataFrame,
):
    """Plot inclusion status of countries based on mobility.
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.set_xticks([])
    ax.set_yticks([])
    world.plot(ax=ax, color=world["mob"].map(INCLUSION_COLOURS), edgecolor="black", linewidth=0.2)
    world[world["included"]].geometry.centroid.plot(ax=ax, color="red", marker="o", markersize=50)
    return fig


def get_detailed_param_results(
    job_path: Path,
    countries: List[str],
    param: str,
) -> tuple:
    """Plot the distributions of a particular parameter
    (currently just used for the mob_exp) parameter,
    and collate its mean over mobility types.

    Args:
        job_path: Path for the runs
        countries: The countries to analyse
        param: The name of the parameter

    Returns:
        The figure and the table of means by mobility type
    """
    mob_types = [k for k in AN_ABBREVS if k != "no_mob"]
    i_datas = {}
    table_info = {}
    for mob_type in mob_types:
        idatas, _ = get_idatas_for_mob_type(job_path, countries, mob_type)
        i_datas[mob_type] = idatas
    all_countries = sort_countries_by_name({key for subdict in i_datas.values() for key in subdict})
    fig, axes = get_standard_subplot(len(all_countries), 4)
    flat_axes = axes.ravel()
    table_info = pd.DataFrame(columns=mob_types)
    for c, country in enumerate(all_countries):
        country_name = pycountry.countries.lookup(country).name
        c_ax = flat_axes[c]
        for mob_type in mob_types:
            m_idatas = i_datas[mob_type]
            colour = MOB_COLOURS[mob_type]
            if country in m_idatas:
                c_idata = m_idatas[country]
                post_plot = az.plot_posterior(c_idata, var_names=param, ax=c_ax, point_estimate=None, hdi_prob="hide", color=colour)
                line_data = post_plot.get_lines()[-1]
                post_plot.fill_between(line_data.get_xdata(), line_data.get_ydata(), alpha=0.1, color=MOB_COLOURS[mob_type])
                mean = az.summary(c_idata, var_names=param, kind="stats")["mean"].values[0]
                table_info.loc[country_name, mob_type] = mean
        c_ax.set_title(country_name)
        c_ax.set_xlim([0.0, 2.0])
        c_ax.set_xticks(np.linspace(0.0, 2.0, 5))
        c_ax.tick_params(labelsize=9)
    stats_table = pd.DataFrame(table_info)
    stats_table = stats_table.mask(stats_table.isna(), "no analysis")
    stats_table = stats_table.rename(columns=ANALYSIS_NAMES)
    for a in range(c + 1, len(flat_axes)):
        flat_axes[a].set_axis_off()
    fig.tight_layout()
    return fig, stats_table


def plot_vals_map(
    vals: Dict[str, float], 
    colour_map: str,
):
    """Plot the values provided by country
    onto a map of the world.

    Args:
        vals: The values by country
        colour_map: matplotlib colour map
    """
    world = get_world_shp()
    _, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    world.boundary.plot(ax=ax, color="black", linewidth=0.2)
    world["vals"] = world["ISO_A3"].map(vals)
    mob_avail = world[world["vals"].notna()]
    mob_unavail = world[world["vals"].isna()]
    mob_avail.plot(ax=ax, column=mob_avail["vals"], cmap=colour_map, legend=True, vmin=0.0, vmax=2.0)
    mob_unavail.plot(ax=ax, color="w", hatch="///", alpha=0.04)


SHORT_MOB_NAMES = {
    "retail_and_recreation": "retail/rec",
    "grocery_and_pharmacy": "groc/pharm",
    "parks": "parks",
    "transit_stations": "transit",
    "workplaces": "work",
    "residential": "resi",
    "fb_visited_mob": "FB visit",
    "fb_singletile_mob": "FB tile",
}
SHORT_COUNTRY_NAMES = {
    "Russian Federation": "Russian Fed",
    "Dominican Republic": "Dominican Rep"
}


def plot_select_proc_mob(
    job_path: Path, 
    panels: List[List[List[str]]],
) -> plt.figure:
    """Plot selected comparisons between mobility
    and modelled variable process.

    Args:
        job_path: Path for the runs
        panels: The comparisons to plot

    Returns:
        The figure
    """
    fig, axes = plt.subplots(4, 8, figsize=(14, 8))
    for c, col in enumerate(panels):
        for r, row in enumerate(col):  
    
            # Gather data
            mob_type, country = row
            iso3 = pycountry.countries.lookup(country).alpha_3
            country_name = SHORT_COUNTRY_NAMES[country] if country in SHORT_COUNTRY_NAMES else country
            mob_source = mob_type if mob_type.startswith("fb_") else "g_mob"
            mob_source_name = SHORT_MOB_NAMES[mob_type]
    
            # Plot variable process
            proc_samples = pd.read_hdf(job_path / iso3 / "no_mob/spaghetti.h5")["process"]
            centiles = proc_samples.quantile([0.05, 0.5, 0.95], axis=1).T
            ax = axes[r, c]
            ax.plot(centiles.index, centiles[0.5], label="process", color="navy")
            ax.fill_between(centiles.index, centiles[0.05], centiles[0.95], alpha=0.2, color="navy")
    
            # Plot mobility
            if mob_source == "g_mob":
                mob = get_google_mobility(iso3)[mob_type]
            elif mob_source == "fb_visited_mob":
                mob = get_fb_visited_mobility(iso3)
            elif mob_source == "fb_singletile_mob":
                mob = get_fb_singletile_mobility(iso3)
            mobility = mob.loc[(centiles.index[0] < mob.index) & (mob.index < centiles.index[-1])]
            smoothed_mob = mobility.rolling(7, center=True).mean().dropna()
            colour = G_MOB_DOMAIN_CMAP[mob_type] if mob_source == "g_mob" else MOB_COLOURS[mob_type]
            ax.plot(smoothed_mob.index, smoothed_mob, color=colour)
    
            # Finish cosmetics
            ax.set_title(f"{country_name}, {mob_source_name}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()

    plt.close()
    return fig
