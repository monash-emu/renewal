import pandas as pd
from pathlib import Path

PROJECT_PATH = Path.cwd().resolve()
DATA_PATH = PROJECT_PATH.parent / "data"
RAW_MOB_PATH = DATA_PATH / "mobility_raw"
VAR_MAP = {
    "ba1": "21K.Omicron",
    "ba2": "21L.Omicron",
    "ba5": "22B.Omicron",
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
