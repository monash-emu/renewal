## Evaluating the effects of population mobility on the COVID-19 pandemic</b>

### Source code and data.

This software is designed to run on Python 3.10 and greater, and has been tested on MacOS, Windows 10, and various Linux distributions.

In order to run the code and reproduce outputs from this paper, we have included a pixi environment manifest and lockfile.

If you do not have pixi installed, you can obtain it from https://pixi.sh/latest/installation/

### Execution

Clone the repository and check out the main branch (or clone as a single branch)

`git clone https://github.com/monash-emu/renewal.git --single-branch --branch main`

To build the environment for this repository, run `pixi install` from within the repository root

This will typically take a few minutes to build and install the environment.  

Various aspects of this paper can be explored and reproduced using the notebooks in the notebooks folder, which can
be accessed via `pixi run jupyter-lab` (or using your preferred notebook viewer, e.g. VSCode).

In particular, individual countries can be run and then analysed using the notebooks:
- notebooks/running/02-run.ipynb
- notebooks/running/03-analyse-run.ipynb

### Reproducibility

A full set out of calibration outputs can be produced by running `pixi run python scripts/run_all.py`
These outputs may then be processed by the notebooks in notebooks/manuscript to produce publication figures.

Note that with the default number of iterations (20000), this run is expected to take many days (or longer) on a single desktop machine,
so this code is included for completeness rather than with the practical intention of being run in this context.

For the paper, these outputs were run as parallel (batch) supercomputing jobs, using the scripts included in the massive subfolder.
Specific git revisions for each country are discussed in the manuscript.
