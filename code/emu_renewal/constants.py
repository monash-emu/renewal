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

VAR_NAMES = [
    "21K.Omicron",
    "21L.Omicron",
    "22B.Omicron",
    "20A.EU1",
    "20A.EU2",
    "20I.Alpha.V1",
    "20A.S.126A",
    "20A.S.210T",
    "20B.S.732A",
    "20B.S.796H",
    "20H.Beta.V2",
    "20I.Alpha.V1",
    "20J.Gamma.V3",
    "21A.Delta.S.K417",
    "21A.Delta",
    "21B.Kappa",
    "21C.Epsilon",
    "21D.Eta",
    "21F.Iota",
    "21G.Lambda",
    "21H.Mu",
    "21I.Delta",
    "21J.Delta",
    "21K.Omicron",
    "21L.Omicron",
    "21L",
    "21M.Omicron",
    "22A.Omicron",
    "22B22E",
    "22C.Omicron",
    "22D.Omicron",
    "22E.Omicron",
    "22F.Omicron",
    "23A.Omicron",
    "23B.Omicron",
    "23C.Omicron",
    "23D.Omicron",
    "23E.Omicron",
    "23F.Omicron",
    "23G.Omicron",
    "23H.Omicron",
    "23I.Omicron",
    "24A.Omicron",
    "24B.Omicron",
    "24C.Omicron",
    "24D.Omicron",
    "24E.Omicron",
    "24F.Omicron",
    "24G.Omicron",
    "24H.Omicron",
    "24I.Omicron",
]

ANALYSIS_TYPES = ["no_mob", "g_mob", "fb_mob", "fb_withintile_mob"]

CASES_START = datetime(2020, 6, 1)

DEFAULT_START_DATE = "1 June 2020"
DEFAULT_END_DATE = "1 December 2021"

ALPHA_PERIOD_START = datetime(2020, 1, 1)
ALPHA_DELTA_TRANS = datetime(2021, 3, 1)
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
DELTA_INCLUSION_DATE = datetime(2021, 5, 1)
DELTA_PERIOD_END = datetime(2021, 9, 1)
MIN_DELTA_PROP = 0.05
BA2_PERIOD_START = datetime(2022, 1, 1)
BA2_PERIOD_END = datetime(2022, 4, 15)
BA5_PERIOD_START = datetime(2022, 4, 1)
BA5_PERIOD_END = datetime(2022, 9, 1)
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
SEROPREV_WEIGHT = 5.0
ANTIBODY_DELAY = 14
VAR_WEIGHT = 5.0


def get_func_blurb(function):
    docstring = function.__doc__
    blurb = re.split("(Args|Returns):", docstring)[0]
    blurb_str = re.sub(r"\s+", " ", blurb)
    return eval(f"f'{blurb_str}'")
