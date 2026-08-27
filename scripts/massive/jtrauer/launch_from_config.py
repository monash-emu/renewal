"""Submit a SLURM array sized from oxcgrt_included.json."""

import json
import subprocess
from pathlib import Path

from emu_renewal.constants import DATA_PATH

SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG = DATA_PATH / "config/oxcgrt_included.json"

n = len(json.load(open(CONFIG)))
print(f"Submitting array 1-{n} from {CONFIG.name}")
subprocess.run(
    [
        "sbatch",
        f"--array=1-{n}",
        f"--export=SCRIPT_DIR={SCRIPT_DIR}",
        str(SCRIPT_DIR / "run_countries.sh"),
    ],
    cwd=SCRIPT_DIR.parents[2],
    check=True,
)
