## Evaluating the effects of population mobility on the COVID-19 pandemic</b>

### Source code and data.

This software is designed to run on Python 3.10 and greater, and has been tested on MacOS, Windows 10, and various Linux distributions.

In order to run the code and reproduce outputs from this paper, we have included a pixi environment manifest and lockfile.

If you do not have pixi installed, you can obtain it from https://pixi.sh/latest/installation/

### Execution

Clone the repository and check out the submission_1 branch (or clone as a single branch)

`git clone https://github.com/monash-emu/renewal.git --single-branch --branch submission_1`

To build the environment for this repository, run `pixi run python` from within the repository root

This will typically take a few minutes to build and install the environment.  

Exit the python session, after which various aspects of this paper can be explored and reproduced using the notebooks in the notebooks folder, which can
be accessed via `pixi run jupyter-lab` (or using your preferred notebook viewer, e.g. VSCode).

In particular, individual countries can be run and then analysed using the notebooks:
- notebooks/running/02-run.ipynb
- notebooks/running/03-analyse-run.ipynb
