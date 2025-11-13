from pathlib import Path


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
ANALYSIS_TYPES = ["no_mob", "g_mob", "fb_visited_mob", "fb_singletile_mob", "oxcgrt"]

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
DELTA_INCLUSION_DATE = "1 May 2021"
DELTA_PERIOD_END = "1 September 2021"
BA2_PERIOD_START = "1 January 2022"
BA2_PERIOD_END = "15 April 2022"
BA5_PERIOD_START = "1 April 2022"
BA5_PERIOD_END = "1 September 2022"
DATA_QUALITY_START_TIME = "1 April 2020"
DATA_QUALITY_START_TIME_OC = "1 November 2021"
NO_CONT_COUNTRIES = ["ATA", "ATF", "ESH", "PCN", "SXM", "TLS", "UMI", "VAT"]

# Population-related
POP_YEAR = 2020
OC_POP_YEAR = 2022
ASSUMED_HIGH_INCOME = "French Guiana, Martinique"

# General indicator-related
ZERO_IND_REPLACEMENT = 0.5
OUTLIER_THRESHOLD = 2
N_REPEATS = 15
ROUND_THRESHOLD = 1e-10
SEVERITY_ADJS = (
    "Low income: 0.4, Lower middle income: 0.6, Upper middle income: 0.8, High income: 1.0"
)
EXTRA_LOW_INC = "Venezuela"
VARIATION_THRESHOLD = "1\\times10^{-7}"

# Deaths-related
DEATHS_WEIGHT = 20
DEATHS_START_THRESHOLD = 2

# Hospitalisation-related
ALREADY_WEEKLY_ADMIT_COUNTRIES = ["HRV", "ZAF", "IRL", "GRC", "SVN", "NOR", "MLT", "EST", "LUX", "SVK", "LVA"]
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
START_VACC_THRESHOLD_OC = 90
SUB_DEU_COUNTRIES = "Switzerland and Ireland"
SUB_GBR_COUNTRY = "Qatar"

# Mobility-related
MOBILITY_SMOOTH_PERIOD = 7
EXP_PRIOR_LOWER = 0
EXP_PRIOR_UPPER = 2

# Transmission scaling-related
PROC_UPDATE_FREQ = 7
PROC_DISP_SD = 0.5

# Renewal model-related
INIT_DURATION = 50
RUN_DATA_DELAY = 50
GEN_TRUNC_POINT = 50
CONV_TRUNC_POINT = 50
DAYS_IN_WEEK = 7

# Calibration-related
N_ITERS = 10000
N_CHAINS = 8
INIT_RADIUS = 0.1
SEED_RATE_LOW = "1\\times10^{-7}"
SEED_RATE_UP = "5\\times10^{-6}"
SEED_OFF_LOW = 4.0
SEED_OFF_UP = 90.0
RELINF_MEAN = 1.4
RELINF_LOW = 1
RELINF_UP = 2
RELINF_SD = 0.2
BETA_MIN = 0.5
BETA_MAX = 3.5
SHARED_DISP_SD = 0.5
PROP_DISP = 0.05
SEROPREV_DISP = 0.2
N_SAMPLES = 50

# Prior-related
DUR_MIN = 1
DUR_REL_MAX = 2.5

# Plotting and outputs
MOB_SOURCE_COLOURS = {
    "no_mob": "black",
    "g_mob": "green",
    "fb_visited_mob": "red",
    "fb_singletile_mob": "blue",
    "fb_no_mob": "grey",
}
ANALYSIS_NAMES = {
    "no_mob": "no mobility",
    "g_mob": "Google mobility",
    "fb_visited_mob": "Facebook tiles visited mobility",
    "fb_singletile_mob": "Facebook single tile mobility",
    "fb_no_mob": "Facebook no mobility",
}
MOB_SOURCE_ABBREVS = {
    "no_mob": "none",
    "g_mob": "Google",
    "fb_visited_mob": "FB tiles visited",
    "fb_singletile_mob": "FB single tile",
    "fb_no_mob": "FB baseline",
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
MOB_LOCATION_SOURCE_MAP = {
    "retail_and_recreation": "g_mob",
    "grocery_and_pharmacy": "g_mob",
    "parks": "g_mob",
    "transit_stations": "g_mob",
    "workplaces": "g_mob",
    "residential": "g_mob",
    "fb_visited_mob": "fb_visited_mob",
    "fb_singletile_mob": "fb_singletile_mob",
}
MOB_LOCATION_NAME_MAP = {
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
G_MOB_LOCATION_CMAP = {
    "retail_and_recreation": "darkgoldenrod",
    "grocery_and_pharmacy": "teal",
    "parks": "darkgreen",
    "transit_stations": "dimgrey",
    "workplaces": "purple",
    "residential": "brown",
}
CONT_CMAP = {
    "AF": "black",
    "AS": "yellow",
    "EU": "blue",
    "NA": "green",
    "OC": "red",
    "SA": "purple",
}
MOB_LOCATION_ABBREVS = {
    "retail_and_recreation": "retail/rec",
    "grocery_and_pharmacy": "groc/pharm",
    "parks": "parks",
    "transit_stations": "transit",
    "workplaces": "work",
    "residential": "resi",
    "weighted_g_mob": "weighted",
    "fb_visited_mob": "FB visit",
    "fb_singletile_mob": "FB tile",
}
SHORT_COUNTRY_NAMES = {
    "Russian Federation": "Russian Fed",
    "Dominican Republic": "Domin Rep"
}

# Stringency
OXCGRT_DTYPES = {
    "V2B_Vaccine age eligibility/availability age floor (general population summary)": str,
    "V2C_Vaccine age eligibility/availability age floor (at risk summary)": str,
    "V4_Notes": str,
    "M1_Notes": str,
    "E3_Notes": str,
    "E4_Notes": str,
    "H4_Notes": str,
}
OXCGRT_IND_MAX = {
    "C1": 3,
    "C2": 3,
    "C3": 2,
    "C4": 4,
    "C5": 2,
    "C6": 3,
    "C7": 2,
    "C8": 4,
    "E1": 2,
    "E2": 2,
    "H1": 2,
    "H2": 3,
    "H3": 2,
    "H6": 4,
    "H7": 5,
    "H8": 3,
}
