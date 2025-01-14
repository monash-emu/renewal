import pandas as pd
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent
OUTPUTS_PATH =  BASE_PATH / "outputs"
DATA_PATH = BASE_PATH / "data"
RAW_MOB_PATH = DATA_PATH / "mobility_raw"  # Will not push these larger original files
VAR_MAP = {
    "ba1": "21K.Omicron",
    "ba2": "21L.Omicron",
    "ba5": "22B.Omicron",
    "eu1": "20A.EU1",
    "eu2": "20A.EU2",
    "alpha": "20I.Alpha.V1"
}


def get_indicator_series_from_who_data(indicator, country):
    who_data = pd.read_csv(DATA_PATH / "who/WHO-COVID-19-global-data_21_8_24.csv")
    select_data = who_data.loc[who_data["Country"] == country]
    select_data.index = pd.to_datetime(select_data["Date_reported"], format="%d/%m/%Y")
    return select_data[indicator].interpolate(method="linear").fillna(0.0)


def get_multicountry_df_from_who_data(indicator, countries):
    data_dict = {i: get_indicator_series_from_who_data(indicator, i) for i in countries}
    return pd.DataFrame(data_dict)


def get_hosp_series_from_owid_data(indicator, country):
    hosp = pd.read_csv(DATA_PATH / "owid/owid_hosp.csv", index_col="date")
    hosp.index = pd.to_datetime(hosp.index)
    data = hosp[hosp["entity"] == country]
    return data.loc[data["indicator"] == indicator, "value"]


def get_var_country_data(var, country):
    data = pd.read_json(DATA_PATH / f"nextclade/{var}.json")[country]
    dates = pd.to_datetime(data["week"])
    return pd.Series(data["cluster_sequences"], index=dates)


def get_multivars_country_data(var_map, country):
    return pd.DataFrame({k: get_var_country_data(v, country) for k, v in var_map.items()})


def get_row_proportions(df):
    return df.divide(df.sum(axis=1), axis=0).fillna(0.0)


def get_country_mobility(country):
    years = range(2020, 2023)
    data_files = [pd.read_csv(RAW_MOB_PATH / f"{y}_{country}_Region_Mobility_Report.csv", index_col="date") for y in years]
    all_data = pd.concat(data_files)
    all_data.index = pd.to_datetime(all_data.index)
    national_data = all_data.loc[pd.isna(all_data["sub_region_1"])]
    national_data = national_data[[c for c in national_data.columns if "change_from_baseline" in c]]  # Extract the mobility columns
    national_data = national_data.rename(lambda c: c.replace("_percent_change_from_baseline", ""), axis=1)  # Simplify column naming
    national_data = 1.0 + national_data / 100.0  # Convert to relative change
    return national_data.sort_index()


def get_seroprev():
    data = pd.read_csv(DATA_PATH / "seroprevalence" / "serotracker.csv")
    data["start"] = pd.to_datetime(data["sampling_start_date"])
    data["end"] = pd.to_datetime(data["sampling_end_date"])
    data.index = (data["end"] - data["start"]) / 2 + data["start"]
    data.index = data.index.normalize()
    return data.sort_index()


def filter_seroprev(data, country, start_date, end_date):
    country_filt = data["country"] == country
    time_filt = (start_date < data.index) & (data.index < end_date)
    nat_filt = data["estimate_grade"] == "National"
    type_filt = (data["subgroup_var"] == "Primary Estimate") & (data["is_unity_aligned"] == "Unity-Aligned")
    data = data.loc[time_filt & country_filt & nat_filt & type_filt]
    return data["serum_pos_prevalence"]
