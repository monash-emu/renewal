from pathlib import Path
from datetime import datetime
import re

DATE_FORMAT = "%Y%m%d_%H%M"
WHO_DATE_FORMAT = "%d/%m/%Y"
TEXT_DATE_FORMAT = "%-d %B %Y"
CODE_DATE_FORMAT = "%d %B %Y"

BASE_PATH = Path(__file__).parent.parent.parent

OUTPUTS_PATH = BASE_PATH / "outputs"
DATA_PATH = BASE_PATH / "data"
RAW_MOB_PATH = DATA_PATH / "mobility_raw"  # Will not push these larger original files

ANALYSIS_TYPES = ["no_mob", "g_mob", "fb_visited_mob", "fb_singletile_mob"]

CASES_START = datetime(2020, 6, 1)

DEFAULT_START_DATE = "1 June 2020"
DEFAULT_END_DATE = "1 December 2021"

ALPHA_PERIOD_START = "1 January 2020"
ALPHA_DELTA_TRANS = "1 March 2021"
ALPHA_DELTA_EXCEPTS = {
    "AFG": datetime(2021, 2, 1),
    "IND": datetime(2021, 2, 1),
    "NPL": datetime(2020, 12, 1),
    "IDN": datetime(2021, 2, 1),
    "SAU": datetime(2021, 2, 1),
    "OMN": datetime(2021, 2, 1),
    "KWT": datetime(2021, 2, 1),
    "KOR": datetime(2021, 2, 1),
    "MYS": datetime(2021, 2, 1),
    "HND": datetime(2021, 4, 15),
    "PRI": datetime(2021, 4, 15),
}
DELTA_INCLUSION_DATE = "1 May 2021"
DELTA_PERIOD_END = "1 September 2021"
MIN_DELTA_PROP = 0.05

BA2_PERIOD_START = "1 January 2022"
BA2_PERIOD_END = "15 April 2022"
BA5_PERIOD_START = "1 April 2022"
BA5_PERIOD_END = "1 September 2022"
POST_SIM_DATE = datetime(2100, 1, 1)
ALPHA_FULL_REPLACE_DATE = datetime(2021, 6, 30)
ALREADY_WEEKLY_ADMIT_COUNTRIES = ["HRV", "ZAF", "IRL", "GRC", "SVN", "NOR"]
ALREADY_WEEKLY_OCCUP_COUNTRIES = ["JPN", "BGR"]
END_VACC_THRESHOLD = 5
START_VACC_THRESHOLD_AUS = 90
ZERO_IND_REPLACEMENT = 0.5
DEATHS_WEIGHT = 20
PREV_KEY = "serum_pos_prevalence"
DEATHS_START_THRESHOLD = 2
SEROPREV_EXTREME = 5
SEROPREV_WEIGHT = 5
SEROPREV_MIN_SIZE = 600
SEROPREV_START_DELAY = 183
ANTIBODY_DELAY = 14
VAR_WEIGHT = 5
LATE_DELTA_WEIGHT = 25
LATE_DELTA_TIME = 87
OUTLIER_THRESHOLD = 2
N_REPEATS = 8
START_TIME = "1 April 2020"
MIN_VAR_SEQS = 5
MIN_VAR_DATES = 5
PREALPHA_IDENTIFIERS = "20A.EU1, 20A.EU2, 20B.S.732A, 21C.Epsilon"
BA2_IDENTIFIER = "21L.Omicron"
MOBILITY_SMOOTH_PERIOD = 7
EXP_PRIOR_LOWER = 0
EXP_PRIOR_UPPER = 2
PROC_UPDATE_FREQ = 7
INIT_DURATION = 50
N_ITERS = 1000
RUN_DATA_DELAY = 50
N_CHAINS = 8
SEED_DURATION = 10
INIT_RADIUS = 0.1
