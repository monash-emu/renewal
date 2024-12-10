import pandas as pd
from pathlib import Path

PROJECT_PATH = Path.cwd().resolve()
DATA_PATH = PROJECT_PATH.parent / "data"


def get_indicator_series_from_who_data(indicator, country):
    who_data = pd.read_csv(DATA_PATH / "who/WHO-COVID-19-global-data_21_8_24.csv")
    select_data = who_data.loc[who_data["Country"] == country]
    select_data.index = pd.to_datetime(select_data["Date_reported"], format="%d/%m/%Y")
    return select_data[indicator].interpolate(method="linear").fillna(0.0)


def get_multicountry_df_from_who_data(indicator, countries):
    data_dict = {i: get_indicator_series_from_who_data(indicator, i) for i in countries}
    return pd.DataFrame(data_dict)
