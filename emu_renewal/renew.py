from typing import Union, List
from typing import NamedTuple
from jax import lax, vmap
from jax import numpy as jnp
from datetime import datetime
import pandas as pd
import numpy as np
from warnings import warn

from summer2.utils import Epoch

from emu_renewal.process import sinterp, MultiCurve
from emu_renewal.distributions import Dens, GammaDens
from emu_renewal.utils import format_date_for_str, round_sigfig


class RenewalState(NamedTuple):
    incidence: jnp.array
    suscept: float


class ModelResult(NamedTuple):
    incidence: jnp.array
    suscept: jnp.array
    r_t: jnp.array
    process: jnp.array
    cases: jnp.array
    weekly_sum: jnp.array
    seropos: jnp.array


class ModelDeathsResult(NamedTuple):
    incidence: jnp.array
    suscept: jnp.array
    r_t: jnp.array
    process: jnp.array
    cases: jnp.array
    weekly_sum: jnp.array
    seropos: jnp.array
    deaths: jnp.array


class RenewalModel:
    def __init__(
        self,
        population: float,
        start: Union[datetime, int],
        end: Union[datetime, int],
        proc_update_freq: int,
        proc_fitter: MultiCurve,
        dens_obj: Dens,
        window_len: int,
        init_series: Union[pd.Series, np.array],
        reporting_dist: Dens,
    ):
        """Standard renewal model object.

        Args:
            population: Starting population value
            start: Start time for the analysis period (excluding run-in)
            end: End time for the analysis period
            proc_update_freq: Frequency with with the vairable process is updated
            proc_fitter: The object containing the method for fitting to the variable process series
            dens_obj: Generation time distribution
            window_len: How far to look back in calculating the renewal process
            init_series: Initialisation series prior to analysis start
        """

        # Initialising series
        if len(init_series) < window_len:
            warn("Padding initialisation series with zeroes because shorter than window")
            self.init_series = np.concatenate(
                [np.zeros(window_len - len(init_series)), init_series]
            )
        elif len(init_series) > window_len:
            warn("Trimming initialisation series because longer than window")
            self.init_series = jnp.array(init_series[-window_len:])
        else:
            self.init_series = jnp.array(init_series)

        # Times
        self.epoch = Epoch(start) if isinstance(start, datetime) else None
        self.start = self.process_time_req(start)
        self.end = self.process_time_req(end)
        self.model_times = jnp.arange(self.start, self.end + 1)
        self.description = {
            "Fixed parameters": (
                f"The main analysis period runs from {format_date_for_str(start)} "
                f"to {format_date_for_str(end)}, "
                f"with a preceding initialisation period of {len(self.init_series)} days. "
            )
        }

        # Population
        self.pop = population
        self.description[
            "Fixed parameters"
        ] += f"The starting model population is {round_sigfig(population / 1e6, 2)} million persons. "

        # Process
        self.proc_update_freq = proc_update_freq
        self.x_proc_vals = jnp.arange(self.end, self.start, -self.proc_update_freq)[::-1]
        self.x_proc_data = sinterp.get_scale_data(self.x_proc_vals)
        self.proc_fitter = proc_fitter
        self.process_start = int(self.x_proc_vals[0])
        self.description["Variable process"] = (
            "Each x-value for the requested points in the variable process "
            "are set at evenly spaced intervals through the analysis period "
            f"spaced by {self.proc_update_freq} days and "
            "ending at the analysis end time. "
        )
        if self.start < self.process_start:
            self.description["Variable process"] = (
                "Because the analysis period is not an exact multiple "
                "of the duration of a process interval, "
                "the variable process only starts from day "
                f"{self.process_start} of the analysis. "
            )
        self.describe_process()

        # Generation interval
        self.dens_obj = dens_obj
        self.description["Generation times"] = self.dens_obj.get_desc()
        self.window_len = window_len
        self.description["Generation times"] += (
            "The generation interval for all calculations "
            f"is truncated from {window_len} days onwards "
            "on the assumption that the distribution's density "
            "has reached negligible values once this period has elapsed. "
        )

        # Renewal process
        self.describe_renewal()

        # Reporting delay
        self.report_dist = reporting_dist
        self.describe_reporting()
        self.description["Reporting"] += self.report_dist.get_desc()
        self.describe_weekly_sum()
        
    def process_time_req(
        self,
        req: Union[datetime, int],
    ) -> int:
        """Sort out a user requested date.

        Args:
            req: The request

        Raises:
            ValueError: If neither date nor int

        Returns:
            The request converted to int according to the model's epoch
        """
        msg = "Time data type not supported"
        if isinstance(req, int):
            return req
        elif isinstance(req, datetime):
            return int(self.epoch.dti_to_index(req))
        else:
            raise ValueError(msg)

    def fit_process_curve(
        self,
        y_proc_req: List[float],
        rt_init,
    ) -> jnp.array:
        """See describe_process below.

        Args:
            y_proc_req: The submitted log values for the variable process

        Returns:
            The values of the variable process at each model time
        """
        y_proc_vals = jnp.cumsum(jnp.concatenate([jnp.array((rt_init,)), y_proc_req]))
        y_proc_data = sinterp.get_scale_data(y_proc_vals)
        cos_func = vmap(self.proc_fitter.get_multicurve, in_axes=(0, None, None))
        return jnp.exp(cos_func(self.model_times, self.x_proc_data, y_proc_data))

    def describe_process(self):
        self.description["Variable process"] += self.proc_fitter.get_description()
        self.description["Variable process"] += (
            "The parameters for the variable process are explored as "
            "the update of each process value relative to the preceding value. "
            "Each of the parameters for the variable process is exponentiated, "
            "such that these parameters are explored in the log-transformed space. "
        )

    def get_output_from_inc(
        self, 
        full_inc: jnp.array, 
        report_mean: float, 
        report_sd: float, 
        cdr: float, 
        n_dens: int,
    ) -> jnp.array:
        """Apply an observation model as a convolution to calculate an epidemiological output series.

        Args:
            full_inc: The full incidence series including the initialisation
            report_mean: Mean delay to reporting
            report_sd: Standard deviation of delay to reporting
            cdr: Case detection/ascertainment proportion
            n_dens: How far to go back with the observation model

        Returns:
            Output from start of initialisation to end of model time
        """
        densities = self.dens_obj.get_densities(n_dens, report_mean, report_sd)
        convolved_cases = jnp.convolve(full_inc, densities) * cdr
        return convolved_cases[: len(full_inc)]

    def describe_reporting(self):
        self.description["Reporting"] = (
            "Notifications are calculated by first convoling "
            "the probability distribution representing the time from "
            "onset of an infection episode to reporting with the "
            "time series of incidence. "
            "This is then multiplied through by the modelled "
            "case detection rate to obtain the final time series "
            "for case notifications. "
        )

    def get_period_output_from_daily(
        self, 
        raw_series: jnp.array, 
        n_sum_times: int,
    ) -> jnp.array:
        """Sum over a preceding window period to get counts over a period of time.

        Args:
            raw_series: Observations before windowing applied
            n_sum_times: Duration of period for summing

        Returns:
            Summed series
        """
        windower = jnp.array([1.0] * n_sum_times)
        return jnp.convolve(raw_series, windower)[: len(raw_series)]

    def describe_weekly_sum(self):
        self.description["Reporting"] += (
            "Last, weekly case counts are then calculated from this "
            "time series of notifications. "
        )

    def renewal_func(
        self,
        gen_mean: float,
        gen_sd: float,
        proc: List[float],
        cdr: float,
        rt_init: float,
        report_mean: float,
        report_sd: float,
        prop_immune: float=0.0,
    ) -> ModelResult:
        """See describe_renewal

        Args:
            gen_mean: Generation time mean
            gen_sd: Generation time standard deviation
            y_proc_req: Values of the variable process

        Returns:
            Results of the model run
        """
        densities = self.dens_obj.get_densities(self.window_len, gen_mean, gen_sd)
        process_vals = self.fit_process_curve(proc, rt_init)
        init_inc = self.init_series / cdr
        start_pop = self.pop * (1.0 - prop_immune) - jnp.sum(init_inc)
        init_state = RenewalState(init_inc[::-1], start_pop)

        def state_update(state: RenewalState, t) -> tuple[RenewalState, jnp.array]:
            proc_val = process_vals[t - self.start]
            r_t = proc_val * state.suscept / self.pop
            renewal = (densities * state.incidence).sum() * r_t
            new_inc = jnp.where(renewal > state.suscept, state.suscept, renewal)
            suscept = state.suscept - new_inc
            incidence = jnp.zeros_like(state.incidence)
            incidence = incidence.at[1:].set(state.incidence[:-1])
            incidence = incidence.at[0].set(new_inc)
            out = {"incidence": new_inc, "suscept": suscept, "r_t": r_t, "process": proc_val}
            return RenewalState(incidence, suscept), out

        end_state, outputs = lax.scan(state_update, init_state, self.model_times)
        full_inc = jnp.concatenate([init_inc, jnp.array(outputs["incidence"])])
        full_cases = self.get_output_from_inc(full_inc, report_mean, report_sd, cdr, len(full_inc))
        full_weekly_cases = self.get_period_output_from_daily(full_cases, 7)
        outputs["cases"] = full_cases[len(init_inc):]
        outputs["weekly_sum"] = full_weekly_cases[len(init_inc):]
        outputs["seropos"] = (start_pop - outputs["suscept"]) / start_pop
        return ModelResult(**outputs)

    def describe_renewal(self):
        self.description["Renewal process"] = (
            "Calculation of the renewal process "
            "consists of multiplying the incidence values for the preceding days "
            "by the reversed generation time distribution values. "
            "This follows a standard formula, "
            "described elsewhere by several groups,[@cori2013; @faria2021] i.e. "
            "$$i_t = R_t\sum_{\\tau<t} i_\\tau g_{t-\\tau}$$\n"
            "$R_t$ is calculated as the product of the proportion "
            "of the population remaining susceptible "
            "and the non-mechanistic random process "
            "generated external to the renewal model. "
            "The susceptible population is calculated by "
            "subtracting the number of new incident cases from the "
            "running total of susceptibles at each iteration. "
            "If incidence exceeds the number of susceptible persons available "
            "for infection in the model, incidence is capped at the "
            "remaining number of susceptibles. "
        )

    def get_description(self) -> str:
        """Compile the description of model.

        Returns:
            Description
        """
        description = ""
        for title, text in self.description.items():
            description += f"\n\n### {title}\n"
            description += text
        return description


class RenewalDeathsModel(RenewalModel):
    
    def renewal_func(
        self,
        gen_mean: float,
        gen_sd: float,
        proc: List[float],
        cdr: float,
        rt_init: float,
        report_mean: float,
        report_sd: float,
        prop_immune: float=0.0,
    ) -> ModelResult:
        """See describe_renewal

        Args:
            gen_mean: Generation time mean
            gen_sd: Generation time standard deviation
            y_proc_req: Values of the variable process

        Returns:
            Results of the model run
        """
        densities = self.dens_obj.get_densities(self.window_len, gen_mean, gen_sd)
        process_vals = self.fit_process_curve(proc, rt_init)
        init_inc = self.init_series / cdr
        start_pop = self.pop * (1.0 - prop_immune) - jnp.sum(init_inc)
        init_state = RenewalState(init_inc[::-1], start_pop)

        def state_update(state: RenewalState, t) -> tuple[RenewalState, jnp.array]:
            proc_val = process_vals[t - self.start]
            r_t = proc_val * state.suscept / self.pop
            renewal = (densities * state.incidence).sum() * r_t
            new_inc = jnp.where(renewal > state.suscept, state.suscept, renewal)
            suscept = state.suscept - new_inc
            incidence = jnp.zeros_like(state.incidence)
            incidence = incidence.at[1:].set(state.incidence[:-1])
            incidence = incidence.at[0].set(new_inc)
            out = {"incidence": new_inc, "suscept": suscept, "r_t": r_t, "process": proc_val}
            return RenewalState(incidence, suscept), out

        end_state, outputs = lax.scan(state_update, init_state, self.model_times)
        full_inc = jnp.concatenate([init_inc, jnp.array(outputs["incidence"])])
        full_cases = self.get_output_from_inc(full_inc, report_mean, report_sd, cdr, len(full_inc))
        full_weekly_cases = self.get_period_output_from_daily(full_cases, 7)
        outputs["cases"] = full_cases[len(init_inc):]
        outputs["weekly_sum"] = full_weekly_cases[len(init_inc):]
        outputs["seropos"] = (start_pop - outputs["suscept"]) / start_pop
        return ModelResult(**outputs)
