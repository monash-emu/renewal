from pathlib import Path
from datetime import datetime

# Date formats
DATE_FORMAT = "%Y%m%d_%H%M"
WHO_DATE_FORMAT = "%d/%m/%Y"
CODE_DATE_FORMAT = "%d %B %Y"

# Paths
BASE_PATH = Path(__file__).parent.parent.parent
OUTPUTS_PATH = BASE_PATH / "outputs"
DATA_PATH = BASE_PATH / "data"
RAW_MOB_PATH = DATA_PATH / "mobility_raw"

# Analysis types
ANALYSIS_TYPES = ["no_mob", "g_mob", "fb_visited_mob", "fb_singletile_mob"]

# Dates
CASES_START = "1 June 2020"
DEFAULT_START_DATE = "1 June 2020"
DEFAULT_END_DATE = "1 December 2021"
ALPHA_PERIOD_START = "1 January 2020"
ALPHA_DELTA_TRANS = "1 March 2021"
ALPHA_DELTA_EXCEPTS = (
    "Afghanistan: 1 February 2021, "
    "India: 1 February 2021, "
    "Nepal: 1 December 2020, "
    "Indonesia: 1 February 2021, "
    "Saudi Arabia: 1 February 2021, "
    "Oman: 1 February 2021, "
    "Kuwait: 1 February 2021, "
    "South Korea: 1 February 2021, "
    "Malaysia: 1 February 2021, "
    "Honduras: 15 April 2021, "
    "Puerto Rico: 15 April 2021"
)
# ALPHA_DELTA_EXCEPTS = {
#     "AFG": datetime(2021, 2, 1),
#     "IND": datetime(2021, 2, 1),
#     "NPL": datetime(2020, 12, 1),
#     "IDN": datetime(2021, 2, 1),
#     "SAU": datetime(2021, 2, 1),
#     "OMN": datetime(2021, 2, 1),
#     "KWT": datetime(2021, 2, 1),
#     "KOR": datetime(2021, 2, 1),
#     "MYS": datetime(2021, 2, 1),
#     "HND": datetime(2021, 4, 15),
#     "PRI": datetime(2021, 4, 15),
# }
DELTA_INCLUSION_DATE = "1 May 2021"
DELTA_PERIOD_END = "1 September 2021"
BA2_PERIOD_START = "1 January 2022"
BA2_PERIOD_END = "15 April 2022"
BA5_PERIOD_START = "1 April 2022"
BA5_PERIOD_END = "1 September 2022"
DATA_QUALITY_START_TIME = "1 April 2020"

# General indicator-related
ZERO_IND_REPLACEMENT = 0.5
OUTLIER_THRESHOLD = 2
N_REPEATS = 8
ROUND_THRESHOLD = 1e-10

# Deaths-related
DEATHS_WEIGHT = 20
DEATHS_START_THRESHOLD = 2

# Hospitalisation-related
ALREADY_WEEKLY_ADMIT_COUNTRIES = ["HRV", "ZAF", "IRL", "GRC", "SVN", "NOR"]
ALREADY_WEEKLY_OCCUP_COUNTRIES = ["JPN", "BGR"]

# Variant-related
MIN_VAR_SEQS = 5
MIN_VAR_DATES = 5
MIN_DELTA_PROP = 0.05
VAR_WEIGHT = 5
SEED_DURATION = 10
PREALPHA_IDENTIFIERS = "20A.EU1, 20A.EU2, 20B.S.732A, 21C.Epsilon"
BA2_IDENTIFIER = "21L.Omicron"
LATE_DELTA_WEIGHT = 25
LATE_DELTA_TIME = 87

# Seroprevalence-related
PREV_KEY = "serum_pos_prevalence"
SEROPREV_START_DELAY = 183
SEROPREV_MIN_SIZE = 600
SEROPREV_EXTREME = 5
SEROPREV_WEIGHT = 5
ANTIBODY_DELAY = 14

# Vaccination-related
END_VACC_THRESHOLD = 5
START_VACC_THRESHOLD_AUS = 90
VACC_DEATH_PROTECT = 0.8
VACC_HOSP_PROTECT = 0.6

# Mobility-related
MOBILITY_SMOOTH_PERIOD = 7
EXP_PRIOR_LOWER = 0
EXP_PRIOR_UPPER = 2

# Variable process-related
PROC_UPDATE_FREQ = 7
PROC_DISP_SD = 0.5

# Renewal model-related
INIT_DURATION = 50
RUN_DATA_DELAY = 50
GEN_TRUNC_POINT = 50
CONV_TRUNC_POINT = 50
DAYS_IN_WEEK = 7

# Calibration-related
N_ITERS = 1000
N_CHAINS = 8
INIT_RADIUS = 0.1
