"""Build run script

Using run_countries.sh as a template, count the number of active countries in
our config JSON,and configure slurm appropriately
"""

import json
import os
import stat
import subprocess
from emu_renewal.constants import DATA_PATH
from pathlib import Path


def count_countries():
    country_list = json.load(open(DATA_PATH / "config/included.json", "r"))
    return len(country_list)


def chmodx(f: Path):
    """Set execute bit on a file, ie mimic chmod +x"""
    cur_mode = os.stat(f).st_mode
    os.chmod(f, cur_mode | stat.S_IEXEC)


def build_active_run():
    ccount = count_countries()
    print(f"Building run script for {ccount} countries")
    scripts_path = Path(__file__).parent.resolve()
    active_script_path = scripts_path / "run_countries_active.sh"
    with open(scripts_path / "run_countries.sh", "r") as template_script:
        with open(active_script_path, "w") as out_script:
            for line in template_script.readlines():
                if line.startswith("#SBATCH --array="):
                    line = f"#SBATCH --array=1-{ccount}"
                out_script.write(line)
    chmodx(active_script_path)

    # Run the script and wait for it to complete
    result = subprocess.run(
        [scripts_path / "launch.sh"], capture_output=True, text=True
    )

    # Print the output (stdout and stderr)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    if result != 0:
        raise Exception(result)


if __name__ == "__main__":
    build_active_run()
